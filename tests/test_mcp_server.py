"""MCP server tests: a real MCP client session talks to the server.

Most tests connect in memory to a server whose engine uses the real databases and a
scripted fake LLM. The last one starts `mcp_server.py` as a subprocess and speaks MCP
over its stdin/stdout, exactly like Claude Desktop or MCP Inspector would.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

import mcp_server
from talktable import tools
from talktable.adapters import MongoAdapter, PostgresAdapter, SQLiteAdapter
from talktable.config import PROJECT_ROOT
from talktable.guardrails import ALLOWED_STAGES
from talktable.pipeline import TalkTable
from data.seed.seed_mongo import INDEXES as MONGO_COLLECTIONS
from data.seed.seed_postgres import metadata as postgres_metadata
from tests.fakes import HashEmbedder, scripted, sql_reply

pytestmark = pytest.mark.anyio

TOOL_NAMES = {"list_databases", "get_schema", "ask", "run_sql", "run_mongo"}
TOP_5_SQL = "SELECT title, price FROM books ORDER BY price DESC LIMIT 5"


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, sqlite_adapter: SQLiteAdapter, postgres_adapter: PostgresAdapter,
           mongo_adapter: MongoAdapter) -> Iterator[TalkTable]:
    """The engine behind the tools: real databases, a fake LLM that knows one query."""
    llm = scripted(router='{"databases": ["sqlite"]}', generator=[sql_reply(TOP_5_SQL)])
    engine = TalkTable(adapters={"sqlite": sqlite_adapter, "postgres": postgres_adapter, "mongo": mongo_adapter},
                     llm=llm, embedder=HashEmbedder())
    monkeypatch.setattr(tools, "get_talktable", lambda: engine)
    yield engine


@pytest.fixture
async def session(engine: TalkTable) -> AsyncIterator[ClientSession]:
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        yield session


async def call(session: ClientSession, tool: str, **arguments: Any) -> Any:
    """Call a tool that should succeed; return its structured result."""
    result = await session.call_tool(tool, arguments)
    assert not result.isError, result.content
    data = result.structuredContent
    assert data is not None
    return data.get("result", data)  # FastMCP wraps list results as {"result": [...]}


async def call_error(session: ClientSession, tool: str, **arguments: Any) -> str:
    """Call a tool that should fail; return the error text the client sees."""
    result: CallToolResult = await session.call_tool(tool, arguments)
    assert result.isError
    return " ".join(getattr(block, "text", "") for block in result.content)


# =============================================================================
# Tool listing
# =============================================================================


async def test_lists_the_five_read_only_tools(session: ClientSession) -> None:
    listed = (await session.list_tools()).tools
    assert {t.name for t in listed} == TOOL_NAMES
    for tool in listed:
        assert tool.description and len(tool.description) > 80, tool.name
        assert tool.annotations is not None and tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.outputSchema is not None, tool.name
    schemas = {t.name: t.inputSchema for t in listed}
    assert schemas["list_databases"].get("required", []) == []
    assert schemas["get_schema"]["required"] == ["database"]
    assert schemas["ask"]["required"] == ["question"]  # the Context parameter isn't exposed
    assert schemas["run_sql"]["required"] == ["database", "query"]
    assert schemas["run_mongo"]["required"] == ["collection", "pipeline"]
    pipeline_types = {option["type"] for option in schemas["run_mongo"]["properties"]["pipeline"]["anyOf"]}
    assert pipeline_types == {"string", "array"}  # JSON text, or the array itself
    for name, schema in schemas.items():
        for prop in schema.get("properties", {}).values():
            assert prop.get("description"), name


async def test_run_mongo_description_lists_every_allowed_stage(session: ClientSession) -> None:
    description = next(t.description for t in (await session.list_tools()).tools if t.name == "run_mongo")
    assert all(stage in description for stage in ALLOWED_STAGES)


def test_instructions_describe_every_database() -> None:
    text = mcp_server._instructions()
    for line in ("- sqlite: Book catalog (authors, awards, book_awards, book_series, book_tags, books, categories,",
                 "- postgres: Sales (carriers, coupons, customers, inventory, loyalty_accounts, order_coupons,",
                 "- mongo: Customer engagement (clickstream, email_campaigns, recommendations, reviews, search_logs,"):
        assert line in text


# =============================================================================
# The tools
# =============================================================================


async def test_list_databases(session: ClientSession) -> None:
    databases = await call(session, "list_databases")
    assert [d["name"] for d in databases] == ["sqlite", "postgres", "mongo"]
    assert all(d["online"] for d in databases)
    by_name = {d["name"]: d for d in databases}
    assert by_name["mongo"]["kind"] == "mongo" and set(by_name["mongo"]["tables"]) == set(MONGO_COLLECTIONS)
    assert sum(len(d["tables"]) for d in databases) == 30
    assert "price" in by_name["sqlite"]["description"]


async def test_get_schema(session: ClientSession) -> None:
    tables = {t["name"]: t for t in await call(session, "get_schema", database="PostgreSQL")}  # alias accepted
    assert set(tables) == set(postgres_metadata.tables)
    orders = tables["orders"]
    assert orders["row_count"] > 0 and orders["primary_key"] == ["order_id"]
    assert "customer_id -> customers.customer_id" in orders["foreign_keys"]
    assert any(c["name"] == "status" and c["description"] for c in orders["columns"])
    fields = {c["name"] for c in (await call(session, "get_schema", database="mongo"))[0]["columns"]}
    assert fields & {"events.type", "rating"}


async def test_get_schema_of_an_unknown_database(session: ClientSession) -> None:
    message = await call_error(session, "get_schema", database="oracle")
    assert "Unknown database 'oracle'" in message and "sqlite, postgres, mongo" in message


async def test_run_sql(session: ClientSession) -> None:
    result = await call(session, "run_sql", database="sqlite", query=TOP_5_SQL)
    assert result["columns"] == ["title", "price"] and result["rows"][0]["price"] == 110.95
    assert result["row_count"] == 5 and result["truncated"] is False
    result = await call(session, "run_sql", database="postgres", query="SELECT order_id FROM orders")
    assert result["row_count"] == 200 and result["truncated"] is True
    assert result["query"].endswith("LIMIT 200")  # what actually ran


@pytest.mark.parametrize(
    ("database", "query", "expected"),
    [
        ("sqlite", "DELETE FROM books", "Blocked by the read-only guardrails"),
        ("postgres", "SELECT 1; DROP TABLE orders", "Blocked by the read-only guardrails"),
        ("postgres", "UPDATE orders SET status = 'x' -- hi", "Blocked by the read-only guardrails"),
        ("sqlite", "SELECT nope FROM books", "no such column: nope"),
        ("mongo", "SELECT 1", "use run_mongo"),
    ],
)
async def test_run_sql_rejections(session: ClientSession, database: str, query: str, expected: str) -> None:
    assert expected in await call_error(session, "run_sql", database=database, query=query)


async def test_run_sql_never_changes_data(session: ClientSession, sqlite_adapter: SQLiteAdapter) -> None:
    for query in ("DELETE FROM books", "DROP TABLE books", "INSERT INTO books (book_id) VALUES (0)"):
        await call_error(session, "run_sql", database="sqlite", query=query)
    assert sqlite_adapter.run("SELECT COUNT(*) AS n FROM books").rows[0]["n"] == 1000


async def test_run_mongo(session: ClientSession) -> None:
    pipeline = json.dumps([
        {"$match": {"rating": 1, "created_at": {"$gte": {"$date": "2025-01-01T00:00:00Z"}}}},
        {"$count": "n"},
    ])
    result = await call(session, "run_mongo", collection="reviews", pipeline=pipeline)
    assert result["columns"] == ["n"] and result["rows"][0]["n"] > 0
    assert '"$limit": 200' in result["query"]
    as_array = await call(session, "run_mongo", collection="reviews", pipeline=json.loads(pipeline))
    assert as_array["rows"] == result["rows"]


@pytest.mark.parametrize(
    ("collection", "pipeline", "expected"),
    [
        ("reviews", '[{"$out": "copy"}]', "$out"),
        ("reviews", [{"$out": "copy"}], "$out"),  # sent as an array rather than JSON text
        ("reviews", [{"$match": {"$where": "sleep(1000)"}}], "$where"),
        ("reviews", '[{"$match": {"$where": "sleep(1000)"}}]', "$where"),
        ("reviews", '[{"$merge": {"into": "reviews"}}]', "$merge"),
        ("reviews", "[{not json", "not valid"),
        ("reviews", '{"$match": {}}', "JSON array"),
        ("system.users", "[]", "can't be queried"),
        ("orders", "[]", "Unknown collection 'orders'"),
    ],
)
async def test_run_mongo_rejections(session: ClientSession, collection: str, pipeline: Any, expected: str) -> None:
    message = await call_error(session, "run_mongo", collection=collection, pipeline=pipeline)
    assert expected in message, message


async def test_ask(session: ClientSession) -> None:
    progress: list[tuple[float, str | None]] = []

    async def on_progress(value: float, total: float | None, message: str | None) -> None:
        progress.append((value, message))

    result = await session.call_tool("ask", {"question": "What are the 5 most expensive books?"},
                                     progress_callback=on_progress)
    assert not result.isError
    answer = result.structuredContent
    assert answer is not None and answer["error"] is None
    assert answer["databases_used"] == ["sqlite"] and answer["queries"] == [TOP_5_SQL]
    assert answer["rows"][0]["price"] == 110.95 and "110.95" in answer["answer"]
    messages = [m for _, m in progress]
    assert messages[0] == "Starting" and "Choosing a database" in messages
    assert any(m and m.startswith("Writing the query") for m in messages)
    values = [v for v, _ in progress]
    assert all(a < b for a, b in zip(values, values[1:]))  # strictly increasing, as MCP requires


async def test_ask_sends_heartbeats_while_waiting(monkeypatch: pytest.MonkeyPatch, engine: TalkTable) -> None:
    monkeypatch.setattr(mcp_server, "HEARTBEAT_SECONDS", 0.05)
    slow = engine.generator._llm.handler  # type: ignore[attr-defined]

    def handler(messages: list, json_mode: bool) -> str:
        time.sleep(0.3)  # a slow LLM
        return slow(messages, json_mode)

    engine.generator._llm.handler = handler  # type: ignore[attr-defined]
    progress: list[str | None] = []

    async def on_progress(value: float, total: float | None, message: str | None) -> None:
        progress.append(message)

    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        await session.call_tool("ask", {"question": "Most expensive books?"}, progress_callback=on_progress)
    assert progress.count("Choosing a database") >= 3  # repeated while the router call is running


async def test_ask_problems_come_back_in_the_result(session: ClientSession) -> None:
    answer = await call(session, "ask", question="   ")
    assert answer["error"] == "Please type a question." and answer["rows"] == []


async def test_unexpected_errors_are_not_leaked(session: ClientSession, engine: TalkTable,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(database: str) -> None:
        raise RuntimeError("password=hunter2 at postgresql://talktable:hunter2@db")

    monkeypatch.setattr(engine, "schema", broken)
    message = await call_error(session, "get_schema", database="sqlite")
    assert "Something went wrong inside TalkTable" in message and "hunter2" not in message


# =============================================================================
# The real thing: mcp_server.py over stdio
# =============================================================================


async def test_stdio_server_end_to_end(sqlite_adapter: SQLiteAdapter) -> None:
    """Start `python mcp_server.py` like an MCP client does, and call tools over stdio."""
    params = StdioServerParameters(command=sys.executable, args=[str(PROJECT_ROOT / "mcp_server.py")],
                                   cwd="/", env={"LOG_LEVEL": "WARNING"})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        assert init.serverInfo.name == "TalkTable" and "read-only" in (init.instructions or "")
        assert {t.name for t in (await session.list_tools()).tools} == TOOL_NAMES
        result = await call(session, "run_sql", database="sqlite", query=TOP_5_SQL)
        assert result["rows"][0]["price"] == 110.95
        assert "Blocked" in await call_error(session, "run_sql", database="sqlite", query="DROP TABLE books")
        assert any(d["name"] == "sqlite" and d["online"] for d in await call(session, "list_databases"))
