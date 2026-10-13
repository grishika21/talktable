"""SQLite adapter (the book catalog).

Read-only in four independent layers:
  1. the guardrails (in `run()`);
  2. the file is opened with `mode=ro`, so SQLite itself refuses to write it;
  3. `PRAGMA query_only = ON`;
  4. an authorizer callback that allows only reading tables and calling safe functions,
     enforced by SQLite's own parser on exactly what it executes.
A progress handler cancels queries that run longer than the timeout.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, ClassVar

from talktable.adapters.base import (
    DatabaseUnavailableError,
    QueryError,
    QueryTimeoutError,
    SQLAdapter,
    unique_column_names,
)
from talktable.guardrails import SqlDialect, is_denied_function
from talktable.models import ColumnInfo, TableInfo

logger = logging.getLogger(__name__)

# Authorizer actions a read-only SELECT needs. Everything else (INSERT, PRAGMA, ATTACH,
# CREATE, ...) is denied. SQLITE_RECURSIVE (33) is missing from older Python builds.
_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_READ_ACTIONS = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, _SQLITE_RECURSIVE})

# How many SQLite VM instructions run between timeout checks.
_PROGRESS_INTERVAL = 10_000


def _authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, source: str | None) -> int:
    if action in _READ_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:  # arg2 is the function name
        return sqlite3.SQLITE_DENY if arg2 and is_denied_function(arg2) else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLiteAdapter(SQLAdapter):
    name: ClassVar[str] = "sqlite"
    dialect: ClassVar[SqlDialect] = "sqlite"

    def __init__(self, path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = path

    def _connect(self, *, restricted: bool = True) -> sqlite3.Connection:
        """Open a fresh read-only connection. (Cheap for SQLite, and thread-safe.)

        `restricted=False` skips the authorizer. Only our own schema-reading code uses
        it, because PRAGMA table_info would otherwise be denied.
        """
        if not self.path.is_file():
            raise DatabaseUnavailableError(f"SQLite database file not found: {self.path.name}")
        uri = f"{self.path.resolve().as_uri()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_s, check_same_thread=False)
        except sqlite3.Error as exc:
            raise DatabaseUnavailableError(f"Could not open SQLite database: {exc}") from None
        conn.execute("PRAGMA query_only = ON")
        if restricted:
            conn.set_authorizer(_authorizer)
        return conn

    def _execute(self, safe_query: str, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
        with closing(self._connect()) as conn:
            deadline = time.monotonic() + self.timeout_s
            # A non-zero return value makes SQLite abort the query with "interrupted".
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), _PROGRESS_INTERVAL)
            try:
                cursor = conn.execute(safe_query)
                columns = unique_column_names(d[0] for d in cursor.description or ())
                rows = cursor.fetchmany(limit)
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc):
                    raise QueryTimeoutError(f"The query took longer than {self.timeout_s:g}s and was cancelled.") from None
                raise QueryError(str(exc)) from None
            except (sqlite3.DatabaseError, sqlite3.Warning) as exc:
                raise QueryError(str(exc)) from None
        return columns, [dict(zip(columns, row)) for row in rows]

    def get_schema(self) -> list[TableInfo]:
        with closing(self._connect(restricted=False)) as conn:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            tables: list[TableInfo] = []
            for name in names:
                quoted = '"' + name.replace('"', '""') + '"'
                # PRAGMA table_info rows: (cid, name, type, notnull, default, pk)
                info = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
                # PRAGMA foreign_key_list rows: (id, seq, table, from, to, ...)
                fks = conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
                tables.append(TableInfo(
                    name=name,
                    columns=[ColumnInfo(name=col[1], type=col[2] or "ANY") for col in info],
                    primary_key=[col[1] for col in sorted(info, key=lambda c: c[5]) if col[5]],
                    foreign_keys=[f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks],
                    row_count=conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0],
                ))
        return tables

    def ping(self) -> bool:
        try:
            with closing(self._connect()) as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except (QueryError, sqlite3.Error) as exc:
            logger.warning("sqlite is unavailable: %s", exc)
            return False
