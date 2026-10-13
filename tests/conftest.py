"""Shared fixtures. Postgres/Mongo tests are skipped (with a reason) when unreachable."""

from __future__ import annotations

import os
from collections.abc import Iterator

# Tests never read or write the real answer cache (data/answer_cache.sqlite). Cache tests
# build their own, in a temporary folder. This runs before any Settings are created.
os.environ["CACHE_ENABLED"] = "false"
# ... and keep schema vectors in memory, whatever .env says about Qdrant.
os.environ["VECTOR_STORE"] = "numpy"

import pytest

from talktable.adapters import MongoAdapter, PostgresAdapter, SQLiteAdapter
from talktable.config import Settings, get_settings
from talktable.schema_index import SentenceEmbedder


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """`live` tests call the real LLM (slow): run them with TALKTABLE_LIVE_TESTS=1."""
    if os.environ.get("TALKTABLE_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="calls the real LLM: set TALKTABLE_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def embedder(settings: Settings) -> SentenceEmbedder:
    """The real embedding model, loaded once per test session."""
    return SentenceEmbedder(settings.embedding_model)


@pytest.fixture(scope="session")
def sqlite_adapter(settings: Settings) -> SQLiteAdapter:
    adapter = SQLiteAdapter(settings.sqlite_file)
    if not adapter.ping():
        pytest.skip("data/bookstore.db is missing: run python -m data.seed.seed_sqlite")
    return adapter


@pytest.fixture(scope="session")
def postgres_adapter(settings: Settings) -> Iterator[PostgresAdapter]:
    if settings.postgres_url is None:
        pytest.skip("POSTGRES_URL is not set")
    adapter = PostgresAdapter(settings.postgres_url.get_secret_value(), connect_timeout_s=3)
    if not adapter.ping():
        pytest.skip("Postgres is not reachable (start it with: docker/dev-db.sh up)")
    yield adapter
    adapter.close()


@pytest.fixture(scope="session")
def mongo_adapter(settings: Settings) -> Iterator[MongoAdapter]:
    if settings.mongo_url is None:
        pytest.skip("MONGO_URL is not set")
    adapter = MongoAdapter(settings.mongo_url.get_secret_value(), settings.mongo_db,
                           server_selection_timeout_ms=2000)
    if not adapter.ping():
        pytest.skip("MongoDB is not reachable (start it with: docker/dev-db.sh up)")
    yield adapter
    adapter.close()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
