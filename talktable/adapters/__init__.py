"""Database adapters: one per backend, all behind the same read-only interface."""

from __future__ import annotations

from talktable.adapters.base import (
    DatabaseAdapter,
    DatabaseUnavailableError,
    QueryError,
    QueryTimeoutError,
    SQLAdapter,
)
from talktable.adapters.mongo_adapter import MongoAdapter
from talktable.adapters.postgres_adapter import PostgresAdapter
from talktable.adapters.sqlite_adapter import SQLiteAdapter
from talktable.config import Settings

ALL_DATABASES = ("sqlite", "postgres", "mongo")

__all__ = [
    "ALL_DATABASES",
    "DatabaseAdapter",
    "DatabaseUnavailableError",
    "MongoAdapter",
    "PostgresAdapter",
    "QueryError",
    "QueryTimeoutError",
    "SQLAdapter",
    "SQLiteAdapter",
    "create_adapters",
]


def create_adapters(settings: Settings) -> dict[str, DatabaseAdapter]:
    """Build an adapter for every configured database, keyed by name.

    A database whose URL isn't set is simply left out. Nothing connects here, so an
    unreachable database never prevents start-up: use `ping()` to see what's online.
    """
    limits = {"max_rows": settings.max_rows, "timeout_s": settings.query_timeout_seconds}
    adapters: dict[str, DatabaseAdapter] = {"sqlite": SQLiteAdapter(settings.sqlite_file, **limits)}
    if settings.postgres_url is not None:
        adapters["postgres"] = PostgresAdapter(settings.postgres_url.get_secret_value(), **limits)
    if settings.mongo_url is not None:
        adapters["mongo"] = MongoAdapter(settings.mongo_url.get_secret_value(), settings.mongo_db, **limits)
    return adapters
