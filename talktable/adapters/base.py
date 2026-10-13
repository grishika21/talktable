"""The common, read-only interface every database adapter implements.

`DatabaseAdapter.run()` is the only way to execute a query, and it always validates
the query with the guardrails first. Subclasses implement `_execute()`, which is only
ever handed already-validated queries.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time as dt_time, timedelta
from decimal import Decimal
from typing import Any, ClassVar, Generic, TypeVar

from bson import Decimal128, ObjectId

from talktable.guardrails import DEFAULT_MAX_ROWS, SqlDialect, validate_sql
from talktable.models import DatabaseKind, QueryResult, TableInfo

logger = logging.getLogger(__name__)

SafeQueryT = TypeVar("SafeQueryT")


class QueryError(Exception):
    """The database could not run a (validated) query. The message is safe to show and
    is useful feedback for the LLM, e.g. 'column "foo" does not exist'."""


class QueryTimeoutError(QueryError):
    """The query ran longer than the configured timeout and was cancelled."""


class DatabaseUnavailableError(QueryError):
    """The database could not be reached (not running, wrong URL, missing file, ...)."""


class DatabaseAdapter(ABC, Generic[SafeQueryT]):
    """A read-only connection to one database.

    `SafeQueryT` is the validated query type: SQL text for SQL adapters, a `MongoQuery`
    for MongoDB. Creating an adapter never connects. Connections are opened lazily, so
    an unreachable database doesn't stop the app from starting (check `ping()`).
    """

    name: ClassVar[str]  # "sqlite" | "postgres" | "mongo"
    kind: ClassVar[DatabaseKind]  # "sql" | "mongo"

    def __init__(self, *, max_rows: int = DEFAULT_MAX_ROWS, timeout_s: float = 5.0) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    # --- public interface (spec 7.1) ----------------------------------------

    @abstractmethod
    def get_schema(self) -> list[TableInfo]:
        """Describe every table/collection: columns/fields, types, row counts, keys."""

    @abstractmethod
    def ping(self) -> bool:
        """True if the database is reachable right now. Never raises."""

    def run(self, query: Any, limit: int | None = None) -> QueryResult:
        """Validate `query` with the guardrails, execute it, and return JSON-safe rows.

        `limit` caps the rows returned. It is clamped to 1..max_rows (default max_rows).

        Raises:
            GuardrailError: the query was rejected and never sent to the database.
            QueryError: the database failed to run it (QueryTimeoutError and
                DatabaseUnavailableError are the special cases).
        """
        cap = self.row_cap(limit)
        safe = self.validate(query, cap)
        started = time.perf_counter()
        columns, rows = self._execute(safe, cap)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        rows = rows[:cap]
        logger.info("%s: %d row(s) in %d ms", self.name, len(rows), elapsed_ms)
        return QueryResult(
            query=self.query_text(safe),
            columns=columns,
            rows=[{key: to_jsonable(value) for key, value in row.items()} for row in rows],
            row_count=len(rows),
            truncated=len(rows) >= cap,
            elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Release connections. The adapter reconnects if used again."""

    # --- hooks for subclasses -----------------------------------------------

    @abstractmethod
    def validate(self, query: Any, limit: int | None = None) -> SafeQueryT:
        """Run the guardrails and return the query in the exact form that will execute."""

    @abstractmethod
    def _execute(self, safe_query: SafeQueryT, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
        """Execute an already-validated query; return (column names, up to `limit` rows)."""

    @abstractmethod
    def query_text(self, safe_query: SafeQueryT) -> str:
        """The validated query as text, for display and logging."""

    def row_cap(self, limit: int | None) -> int:
        return self.max_rows if limit is None else max(1, min(limit, self.max_rows))


class SQLAdapter(DatabaseAdapter[str]):
    """Shared behaviour of the SQL adapters: SQL guardrails plus a table allowlist."""

    kind: ClassVar[DatabaseKind] = "sql"
    dialect: ClassVar[SqlDialect]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._table_names: frozenset[str] | None = None

    def table_names(self) -> frozenset[str]:
        """Names of the queryable tables (read from the schema once, then cached)."""
        if self._table_names is None:
            self._table_names = frozenset(t.name for t in self.get_schema())
        return self._table_names

    def validate(self, query: Any, limit: int | None = None) -> str:
        return validate_sql(query, self.dialect, max_rows=self.row_cap(limit),
                            allowed_tables=self.table_names())

    def query_text(self, safe_query: str) -> str:
        return safe_query


# --- helpers -----------------------------------------------------------------


def unique_column_names(names: Iterable[str]) -> list[str]:
    """Make result column names unique: SELECT a.name, b.name -> ['name', 'name_2']."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f"{name}_{count}")
    return unique


def to_jsonable(value: Any) -> Any:
    """Convert database values (Decimal, dates, ObjectId, bytes, ...) to plain JSON types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, (ObjectId, uuid.UUID)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return str(value)
