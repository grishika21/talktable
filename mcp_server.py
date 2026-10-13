"""TalkTable as a stdio MCP server, for Claude Desktop, Cursor, MCP Inspector and other clients.

    .venv/bin/python mcp_server.py

The client starts this process and talks MCP over its stdin/stdout, so all logging
goes to stderr. Setup steps for MCP Inspector and Claude Desktop: docs/mcp.md.

Tools: list_databases, get_schema, ask, run_sql, run_mongo. All of them are
read-only: hand-written queries pass through the same guardrails as generated ones.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypeVar

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field, WithJsonSchema

from talktable import tools
from talktable.config import get_settings
from talktable.descriptions import load_descriptions
from talktable.models import AskResult, DatabaseStatus, QueryResult, TableInfo
from talktable.pipeline import ProgressFn

logger = logging.getLogger("talktable.mcp")

T = TypeVar("T")

# How often `ask` sends a progress notification while it waits for the LLM. Clients
# that reset their request timeout on progress (MCP Inspector does) then never give up
# on a slow local model.
HEARTBEAT_SECONDS = 5.0

# Every tool only reads. `openWorldHint=False`: they touch TalkTable's own databases only.
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def _instructions() -> str:
    descriptions = load_descriptions()
    catalog = "\n".join(
        f"- {name}: {d.title} ({', '.join(sorted(d.objects))})" for name, d in descriptions.databases.items()
    )
    return f"""TalkTable answers questions about an online bookstore. Its data is spread over three read-only databases:
{catalog}
The same book_id identifies a book in all three.

Two ways to use it:
1. `ask` a question in plain English. TalkTable picks the database, writes and runs the query, and explains the
   result. With a local LLM this can take a minute or two.
2. Write the query yourself, which is faster: `list_databases` and `get_schema`, then `run_sql` (sqlite, postgres)
   or `run_mongo` (mongo). This also lets you combine data from several databases.

Everything is read-only: only SELECT queries and read-only aggregation stages run, and results are capped at
200 rows."""


def _warm_up() -> None:
    """Build the engine (embedding model, database connections, schema index) in the
    background, so the client's first tool call doesn't wait for it."""
    try:
        engine = tools.get_talktable()
        logger.info("TalkTable is ready. Online databases: %s", ", ".join(engine.online) or "none")
    except Exception:
        logger.exception("TalkTable failed to start; tool calls will report the error")


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    # By now the stdio transport holds its own handle on the real stdout. Point
    # sys.stdout at stderr, so a stray print() in some library can't corrupt the protocol.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    threading.Thread(target=_warm_up, name="talktable-warm-up", daemon=True).start()
    try:
        yield
    finally:
        sys.stdout = original_stdout


mcp = FastMCP("TalkTable", instructions=_instructions(), lifespan=_lifespan)


async def _in_thread(fn: Callable[[], T]) -> T:
    """Run blocking engine code in a worker thread, so the server stays responsive."""
    return await anyio.to_thread.run_sync(fn)


async def _with_progress(ctx: Context, work: Callable[[ProgressFn], T]) -> T:
    """Run `work(on_step)` in a worker thread. Each step it reports, plus a heartbeat
    every HEARTBEAT_SECONDS, is sent to the client as an MCP progress notification."""
    send, receive = anyio.create_memory_object_stream[str](max_buffer_size=16)
    started = anyio.current_time()
    outcome: dict[str, T | Exception] = {}

    def on_step(message: str) -> None:  # called from the worker thread
        try:
            anyio.from_thread.run(send.send, message)
        except Exception:  # the request was cancelled; progress is best-effort
            pass

    async def run_work() -> None:
        with send:  # closing it tells the loop below that the work is done
            try:
                outcome["result"] = await anyio.to_thread.run_sync(work, on_step)
            except Exception as exc:  # re-raised below, outside the task group
                outcome["error"] = exc

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_work)
        with receive:
            message, seconds = "Starting", 0.0
            while True:
                # Progress is the elapsed time in seconds; MCP requires it to keep increasing.
                seconds = max(round(anyio.current_time() - started, 1), seconds + 0.1)
                await ctx.report_progress(seconds, message=message)
                with anyio.move_on_after(HEARTBEAT_SECONDS):
                    try:
                        message = await receive.receive()
                    except anyio.EndOfStream:
                        break
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["result"]  # type: ignore[return-value]


DatabaseName = Annotated[str, Field(description="The database: sqlite, postgres or mongo.")]

# Clients send the pipeline either as an array or as JSON text, so accept both. The
# type is left open on purpose: the guardrails check the value and explain mistakes
# ("pipeline must be a JSON array of stages") better than a type error would.
MongoPipeline = Annotated[
    Any,
    WithJsonSchema({"anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "string"}]}),
    Field(description='The aggregation pipeline: an array of stages (or that array as JSON text), e.g. '
                      '[{"$match": {"rating": 1}}, {"$count": "n"}]'),
]


@mcp.tool(annotations=READ_ONLY)
async def list_databases() -> list[DatabaseStatus]:
    """List TalkTable's databases: name, kind (sql or mongo), whether it is online, what
    data it holds, and its tables or collections. Call this first to see what's available.
    """
    return await _in_thread(tools.list_databases)


@mcp.tool(annotations=READ_ONLY)
async def get_schema(database: DatabaseName) -> list[TableInfo]:
    """Get the tables (SQL) or collections (MongoDB) of one database: every column or
    field with its type and meaning, the row count, primary key and foreign keys. Use it
    before writing a query for run_sql or run_mongo. Mongo fields are dotted paths
    (e.g. events.type), inferred from sample documents.
    """
    return await _in_thread(lambda: tools.get_schema(database))


@mcp.tool(annotations=READ_ONLY)
async def ask(
    question: Annotated[str, Field(description="A question in plain English, e.g. "
                                               "'Which 5 books have the most 1-star reviews?'")],
    ctx: Context,
) -> AskResult:
    """Answer a question about the bookstore's data in plain English. TalkTable picks the
    right database, writes the SQL or MongoDB query, checks it against read-only
    guardrails, runs it (fixing it up to 2 times if it fails), and explains the result.

    Returns the answer, the database and the query used, the result rows, the number of
    retries and the latency. If `error` is set, no answer could be found and `answer`
    says why. With a local LLM this can take a minute or two; progress is reported.
    """
    return await _with_progress(ctx, lambda on_step: tools.ask(question, on_step=on_step))


@mcp.tool(annotations=READ_ONLY)
async def run_sql(
    database: Annotated[str, Field(description="sqlite (book catalog) or postgres (sales).")],
    query: Annotated[str, Field(description="One SELECT statement, in the database's dialect.")],
) -> QueryResult:
    """Run your own read-only SQL query on the sqlite (SQLite dialect) or postgres
    (PostgreSQL dialect) database, and get the rows back.

    Only a single SELECT (or WITH ... SELECT) statement is allowed. Anything that could
    change data or reach outside the database is rejected with an explanation. A LIMIT of
    at most 200 rows is enforced. The returned `query` is the SQL as actually executed,
    and `truncated` tells you whether the rows were capped.
    """
    return await _in_thread(lambda: tools.run_sql(database, query))


@mcp.tool(annotations=READ_ONLY)
async def run_mongo(
    collection: Annotated[str, Field(description="A collection of the mongo database, e.g. reviews.")],
    pipeline: MongoPipeline,
) -> QueryResult:
    """Run your own read-only MongoDB aggregation pipeline on a collection of the mongo
    database, and get the documents back.

    Allowed stages: $match, $project, $group, $sort, $limit, $skip, $unwind, $count,
    $lookup, $addFields, $facet. Write dates as Extended JSON, e.g.
    {"$date": "2025-01-01T00:00:00Z"}. Stages or operators that write data ($out, $merge)
    or run code ($function, $where, $accumulator) are rejected. Output is capped at 200
    documents; `truncated` tells you whether that happened.
    """
    return await _in_thread(lambda: tools.run_mongo(collection, pipeline))


def main() -> None:
    settings = get_settings()
    # Replace the handler FastMCP installed, keeping everything on stderr.
    logging.basicConfig(level=settings.log_level.upper(), stream=sys.stderr, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub", "mcp.server.lowlevel.server"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
