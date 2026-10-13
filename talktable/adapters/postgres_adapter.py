"""PostgreSQL adapter (the sales database: Neon in production, Docker locally).

Read-only in four independent layers:
  1. the guardrails (in `run()`);
  2. every session: SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY + statement_timeout;
  3. every query runs in its own READ ONLY transaction with SET LOCAL statement_timeout,
     and is rolled back afterwards. This also holds behind transaction-pooling proxies
     (PgBouncer, Neon's pooler) where session settings may not stick;
  4. the database user itself should be read-only (see README / docker/postgres-init.sh).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import psycopg
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from talktable.adapters.base import (
    DatabaseUnavailableError,
    QueryError,
    QueryTimeoutError,
    SQLAdapter,
    unique_column_names,
)
from talktable.guardrails import SqlDialect
from talktable.models import ColumnInfo, TableInfo

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 600


class PostgresAdapter(SQLAdapter):
    name: ClassVar[str] = "postgres"
    dialect: ClassVar[SqlDialect] = "postgres"

    def __init__(self, url: str, *, connect_timeout_s: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._url = url  # contains the password: never log it
        self._connect_timeout_s = connect_timeout_s
        self._engine: Engine | None = None

    @property
    def _timeout_ms(self) -> int:
        return max(1, round(self.timeout_s * 1000))

    def _get_engine(self) -> Engine:
        if self._engine is None:
            engine = create_engine(
                self._url,
                pool_size=2,
                max_overflow=3,
                pool_pre_ping=True,  # Neon suspends idle computes; drop dead connections
                pool_recycle=300,
                connect_args={"connect_timeout": self._connect_timeout_s, "application_name": "talktable"},
            )
            event.listen(engine, "connect", self._on_connect)
            self._engine = engine
        return self._engine

    def _on_connect(self, dbapi_conn: psycopg.Connection, _record: Any) -> None:
        """Session defaults for every new connection (spec 7.1)."""
        with dbapi_conn.cursor() as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cur.execute(f"SET statement_timeout = {self._timeout_ms}")
        dbapi_conn.commit()

    def _execute(self, safe_query: str, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
        try:
            with self._get_engine().connect() as conn:
                # These must be the first statements of the transaction SQLAlchemy opens.
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {self._timeout_ms}")
                # no_parameters: pass the SQL to the driver verbatim, so '%' (LIKE) and
                # ':name' inside string literals aren't mistaken for bind parameters.
                result = conn.execution_options(no_parameters=True).exec_driver_sql(safe_query)
                columns = unique_column_names(result.keys())
                rows = result.fetchmany(limit)
                conn.rollback()
        except DBAPIError as exc:
            raise self._translate(exc) from None
        except SQLAlchemyError as exc:
            raise QueryError(str(exc).splitlines()[0]) from None
        return columns, [dict(zip(columns, row)) for row in rows]

    def _translate(self, exc: DBAPIError) -> QueryError:
        """Turn a driver error into a clean, LLM-friendly message."""
        orig = exc.orig
        if isinstance(orig, psycopg.errors.QueryCanceled):
            return QueryTimeoutError(f"The query took longer than {self.timeout_s:g}s and was cancelled.")
        if exc.connection_invalidated or isinstance(orig, psycopg.OperationalError):
            return DatabaseUnavailableError("Could not connect to PostgreSQL.")
        return QueryError(str(orig).strip()[:_MAX_ERROR_CHARS])

    def get_schema(self) -> list[TableInfo]:
        try:
            engine = self._get_engine()
            inspector = inspect(engine)
            tables: list[TableInfo] = []
            with engine.connect() as conn:
                for name in sorted(inspector.get_table_names(schema="public")):
                    columns = inspector.get_columns(name, schema="public")
                    pk = inspector.get_pk_constraint(name, schema="public")
                    fks = inspector.get_foreign_keys(name, schema="public")
                    quoted = '"' + name.replace('"', '""') + '"'
                    row_count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM public.{quoted}").scalar_one()
                    tables.append(TableInfo(
                        name=name,
                        columns=[ColumnInfo(name=c["name"], type=str(c["type"])) for c in columns],
                        primary_key=list(pk.get("constrained_columns") or []),
                        foreign_keys=[
                            f"{', '.join(fk['constrained_columns'])} -> "
                            f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                            for fk in fks
                        ],
                        row_count=row_count,
                    ))
                conn.rollback()
            return tables
        except DBAPIError as exc:
            raise self._translate(exc) from None
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(f"Could not read the PostgreSQL schema: {str(exc).splitlines()[0]}") from None

    def ping(self) -> bool:
        try:
            with self._get_engine().connect() as conn:
                conn.exec_driver_sql("SELECT 1").scalar_one()
            return True
        except SQLAlchemyError as exc:
            logger.warning("postgres is unavailable: %s", type(exc.__cause__ or exc).__name__)
            return False

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
