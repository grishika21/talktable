"""The five TalkTable tools, shared by the stdio MCP server (mcp_server.py) and the
Gradio app (app.py).

Each function is a thin wrapper around the shared engine. Hand-written queries go
through exactly the same guardrails as generated ones (`adapter.run()` always
validates), and every failure is raised as a `ToolError` whose message is safe to show
to the caller: never a stack trace, never a connection string.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from typing import Any, ParamSpec, TypeVar

from talktable.adapters import DatabaseAdapter, QueryError
from talktable.conversation import Turn
from talktable.guardrails import GuardrailError
from talktable.models import AskResult, DatabaseStatus, QueryResult, TableInfo
from talktable.pipeline import TalkTable, ProgressFn, get_talktable

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_MAX_LOGGED_CHARS = 300


class ToolError(Exception):
    """A tool call failed. The message explains why and is safe to show the caller."""


def _safe(fn: Callable[P, R]) -> Callable[P, R]:
    """Log the call, and turn unexpected exceptions into a generic ToolError."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        shown = ", ".join([repr(a) for a in args]
                          + [f"{k}={v!r}" for k, v in kwargs.items() if k not in ("on_step", "history")])
        logger.info("tool %s(%s)", fn.__name__, shown[:_MAX_LOGGED_CHARS])
        try:
            return fn(*args, **kwargs)
        except ToolError as exc:
            logger.info("tool %s failed: %s", fn.__name__, exc)
            raise
        except Exception:
            logger.exception("tool %s crashed", fn.__name__)
            raise ToolError("Something went wrong inside TalkTable. The server log has the details.") from None

    return wrapper


def _adapter(engine: TalkTable, database: str) -> DatabaseAdapter:
    try:
        return engine.adapters[engine.resolve_database(database)]
    except ValueError as exc:
        raise ToolError(str(exc)) from None


def _run(adapter: DatabaseAdapter, query: object) -> QueryResult:
    try:
        return adapter.run(query)
    except GuardrailError as exc:
        raise ToolError(f"Blocked by the read-only guardrails: {exc}") from None
    except QueryError as exc:  # includes timeouts and "database unavailable"
        raise ToolError(str(exc)) from None


@_safe
def list_databases() -> list[DatabaseStatus]:
    """Every database with its status, description, and tables or collections."""
    engine = get_talktable()
    if any(not s.online and s.name in engine.adapters for s in engine.status()):
        engine.refresh()  # a database that was down at start-up may be back by now
    return engine.status()


@_safe
def get_schema(database: str) -> list[TableInfo]:
    """The tables (or collections) of one database, with columns, keys and descriptions."""
    try:
        return get_talktable().schema(database)
    except ValueError as exc:
        raise ToolError(str(exc)) from None


@_safe
def ask(question: str, *, database: str | None = None, on_step: ProgressFn | None = None,
        history: Sequence[Turn] = (), use_cache: bool = True) -> AskResult:
    """Answer a plain-English question. Problems come back in `AskResult.error`.
    `history` (earlier chat turns) lets follow-up questions work."""
    return get_talktable().ask(question, database=database, on_step=on_step, history=history, use_cache=use_cache)


@_safe
def run_sql(database: str, query: str) -> QueryResult:
    """Run a hand-written SELECT on a SQL database, after the guardrails."""
    engine = get_talktable()
    adapter = _adapter(engine, database)
    if adapter.kind != "sql":
        raise ToolError(f"{adapter.name} is a MongoDB database: use run_mongo with an aggregation pipeline.")
    return _run(adapter, query)


@_safe
def run_mongo(collection: str, pipeline: str | list[dict[str, Any]]) -> QueryResult:
    """Run a hand-written aggregation pipeline (a list of stages, or that list as JSON
    text) on a Mongo collection."""
    adapter = _adapter(get_talktable(), "mongo")
    # The guardrails parse `pipeline` (Extended JSON), check the collection, every stage
    # and operator, and cap the output.
    return _run(adapter, {"collection": collection, "pipeline": pipeline})
