"""Adapter tests against the real demo databases.

SQLite always runs (the file is in the repo). Postgres and MongoDB tests need the local
Docker databases (docker/dev-db.sh up) and are skipped otherwise.

Several tests deliberately bypass the guardrails by calling `_execute()` directly, to
prove the database-level layers (read-only connections, users, timeouts, the SQLite
authorizer) hold on their own.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from talktable.adapters import (
    DatabaseUnavailableError,
    MongoAdapter,
    PostgresAdapter,
    QueryError,
    QueryTimeoutError,
    SQLiteAdapter,
    create_adapters,
)
from talktable.adapters.base import to_jsonable, unique_column_names
from talktable.adapters.mongo_adapter import infer_fields
from talktable.config import Settings
from talktable.guardrails import GuardrailError
from data.seed.seed_mongo import INDEXES as MONGO_COLLECTIONS
from data.seed.seed_postgres import metadata as postgres_metadata
from data.seed.seed_sqlite import TABLES as SQLITE_TABLES

# =============================================================================
# Helpers
# =============================================================================


def test_unique_column_names() -> None:
    assert unique_column_names(["name", "name", "id", "name"]) == ["name", "name_2", "id", "name_3"]


def test_to_jsonable() -> None:
    oid = ObjectId()
    assert to_jsonable(Decimal("14.99")) == 14.99
    assert to_jsonable(date(2025, 1, 31)) == "2025-01-31"
    assert to_jsonable(datetime(2025, 1, 31, 8, 30)) == "2025-01-31T08:30:00"
    assert to_jsonable(oid) == str(oid)
    assert to_jsonable({"a": [Decimal("1.5"), {"b": date(2020, 1, 1)}]}) == {"a": [1.5, {"b": "2020-01-01"}]}
    assert to_jsonable(b"\x00\x01") == "AAE="


def test_infer_fields_handles_nested_arrays() -> None:
    docs = [{"_id": ObjectId(), "a": 1, "events": [{"type": "view", "ts": datetime(2025, 1, 1)}]},
            {"_id": ObjectId(), "a": None, "events": []}]
    assert infer_fields(docs) == {"a": "int|null", "events": "array", "events.type": "string", "events.ts": "date"}


def test_create_adapters_skips_unconfigured_databases(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("POSTGRES_URL", "MONGO_URL"):
        monkeypatch.delenv(var, raising=False)
    assert set(create_adapters(Settings(_env_file=None))) == {"sqlite"}


def test_create_adapters_never_connects() -> None:
    started = time.perf_counter()
    adapters = create_adapters(Settings(_env_file=None, postgres_url="postgresql://u:p@203.0.113.1:5432/x",
                                        mongo_url="mongodb://203.0.113.1:27017"))
    assert set(adapters) == {"sqlite", "postgres", "mongo"}
    assert time.perf_counter() - started < 1.0  # 203.0.113.1 is unroutable: any connect would hang


# =============================================================================
# SQLite
# =============================================================================


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sqlite_schema(sqlite_adapter: SQLiteAdapter) -> None:
    tables = {t.name: t for t in sqlite_adapter.get_schema()}
    assert set(tables) == set(SQLITE_TABLES)
    books = tables["books"]
    assert [c.name for c in books.columns] == [
        "book_id", "title", "author_id", "category_id", "price", "published_year", "pages"]
    assert books.primary_key == ["book_id"]
    assert "author_id -> authors.author_id" in books.foreign_keys
    assert books.row_count == 1000
    assert sorted(tables["book_tags"].primary_key) == ["book_id", "tag_id"]  # composite key
    assert set(tables["editions"].foreign_keys) == {"book_id -> books.book_id",
                                                    "publisher_id -> publishers.publisher_id"}


def test_sqlite_run_returns_json_rows(sqlite_adapter: SQLiteAdapter) -> None:
    result = sqlite_adapter.run("SELECT title, price FROM books ORDER BY price DESC LIMIT 5")
    assert result.columns == ["title", "price"]
    assert result.row_count == 5 and not result.truncated
    assert result.rows[0]["price"] == max(r["price"] for r in result.rows)
    assert result.query == "SELECT title, price FROM books ORDER BY price DESC LIMIT 5"


def test_sqlite_row_cap(sqlite_adapter: SQLiteAdapter) -> None:
    capped = sqlite_adapter.run("SELECT * FROM books")
    assert capped.row_count == 200 and capped.truncated and capped.query.endswith("LIMIT 200")
    small = sqlite_adapter.run("SELECT * FROM books LIMIT 100", limit=7)
    assert small.row_count == 7 and small.query.endswith("LIMIT 7")


def test_sqlite_duplicate_column_names(sqlite_adapter: SQLiteAdapter) -> None:
    result = sqlite_adapter.run("SELECT a.name, c.name FROM authors a, categories c LIMIT 1")
    assert result.columns == ["name", "name_2"]


def test_sqlite_guardrails_run_before_anything_executes(sqlite_adapter: SQLiteAdapter, settings: Settings) -> None:
    before = _sha256(settings.sqlite_file)
    for sql in ("DROP TABLE books", "SELECT 1; DELETE FROM books", "SELECT * FROM sqlite_master"):
        with pytest.raises(GuardrailError):
            sqlite_adapter.run(sql)
    assert _sha256(settings.sqlite_file) == before


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM books",
        "UPDATE books SET price = 0",
        "INSERT INTO categories VALUES (99, 'Hacked')",
        "CREATE TABLE hacked (x INTEGER)",
        "DROP TABLE books",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA query_only = OFF",
        "PRAGMA writable_schema = ON",
        "SELECT load_extension('evil')",
        "SELECT randomblob(10)",
    ],
)
def test_sqlite_blocks_writes_even_without_guardrails(sqlite_adapter: SQLiteAdapter, settings: Settings,
                                                      sql: str) -> None:
    """Bypass the guardrails: the authorizer + read-only connection must still refuse."""
    before = _sha256(settings.sqlite_file)
    with pytest.raises(QueryError):
        sqlite_adapter._execute(sql, 10)
    assert _sha256(settings.sqlite_file) == before


def test_sqlite_authorizer_alone_blocks_writes(settings: Settings, tmp_path: Path) -> None:
    """On a *writable* copy (no mode=ro, no query_only), the authorizer still says no."""
    from talktable.adapters.sqlite_adapter import _authorizer

    copy = tmp_path / "writable.db"
    copy.write_bytes(settings.sqlite_file.read_bytes())
    with closing(sqlite3.connect(copy)) as conn:
        conn.set_authorizer(_authorizer)
        assert conn.execute("SELECT count(*) FROM books").fetchone() == (1000,)
        for sql in ("DELETE FROM books", "CREATE TABLE t (x)", "ATTACH DATABASE ':memory:' AS m",
                    "PRAGMA journal_mode = DELETE", "SELECT load_extension('x')"):
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                conn.execute(sql)


def test_sqlite_timeout(settings: Settings) -> None:
    adapter = SQLiteAdapter(settings.sqlite_file, timeout_s=0.3)
    endless = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    started = time.perf_counter()
    with pytest.raises(QueryTimeoutError):
        adapter.run(endless)  # passes the guardrails (it's a read), so the timeout must stop it
    assert time.perf_counter() - started < 3


def test_sqlite_missing_file(tmp_path: Path) -> None:
    adapter = SQLiteAdapter(tmp_path / "missing.db")
    assert adapter.ping() is False
    with pytest.raises(DatabaseUnavailableError):
        adapter.run("SELECT 1")
    assert not (tmp_path / "missing.db").exists()  # mode=ro never creates the file


SQLITE_REALISTIC = [
    "SELECT title, price FROM books ORDER BY price DESC, book_id LIMIT 5",
    "SELECT c.name, COUNT(*) AS n, ROUND(AVG(b.price), 2) AS avg_price FROM books b "
    "JOIN categories c ON c.category_id = b.category_id GROUP BY c.name ORDER BY n DESC, c.name",
    "SELECT a.name, COUNT(b.book_id) AS books FROM authors a LEFT JOIN books b ON a.author_id = b.author_id "
    "GROUP BY a.author_id ORDER BY books DESC, a.name LIMIT 10",
    "SELECT published_year / 10 * 10 AS decade, COUNT(*) FROM books GROUP BY decade ORDER BY decade",
    "SELECT substr(title, 1, 12) AS short, length(title), upper(substr(title, 1, 1)) FROM books "
    "WHERE title LIKE '%Collector%' ORDER BY book_id LIMIT 5",
    "SELECT country, COUNT(*) FROM authors WHERE birth_year BETWEEN 1950 AND 1980 GROUP BY country ORDER BY 2 DESC, 1",
    "SELECT title, CASE WHEN pages > 500 THEN 'long' ELSE 'short' END AS size, ifnull(NULL, 'x') "
    "FROM books ORDER BY book_id LIMIT 3 OFFSET 10",
    "SELECT strftime('%Y', date('2025-03-04')) AS y, CAST(price AS INTEGER) FROM books ORDER BY book_id LIMIT 2",
]


@pytest.mark.parametrize("sql", SQLITE_REALISTIC)
def test_sqlite_rewritten_sql_gives_same_results(sqlite_adapter: SQLiteAdapter, settings: Settings, sql: str) -> None:
    """The guardrails regenerate SQL from the syntax tree; results must not change."""
    with closing(sqlite3.connect(f"file:{settings.sqlite_file}?mode=ro", uri=True)) as raw:
        expected = [list(row) for row in raw.execute(sql).fetchall()]
    result = sqlite_adapter.run(sql)
    assert [list(row.values()) for row in result.rows] == expected


# =============================================================================
# PostgreSQL
# =============================================================================


def test_postgres_schema(postgres_adapter: PostgresAdapter) -> None:
    tables = {t.name: t for t in postgres_adapter.get_schema()}
    assert set(tables) == set(postgres_metadata.tables)
    assert tables["customers"].row_count == 2000 and tables["orders"].row_count == 8000
    assert set(tables["shipments"].foreign_keys) == {"order_id -> orders.order_id",
                                                     "carrier_id -> carriers.carrier_id"}
    assert "customer_id -> customers.customer_id" in tables["orders"].foreign_keys
    assert tables["orders"].primary_key == ["order_id"]


def test_postgres_run_converts_types(postgres_adapter: PostgresAdapter) -> None:
    result = postgres_adapter.run(
        "SELECT date_trunc('month', order_date)::date AS month, count(*) AS orders, sum(total_amount) AS revenue "
        "FROM orders WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01' GROUP BY 1 ORDER BY 1"
    )
    assert result.row_count == 12
    first = result.rows[0]
    assert first["month"] == "2025-01-01"
    assert isinstance(first["orders"], int) and isinstance(first["revenue"], float)


def test_postgres_literals_are_not_bind_parameters(postgres_adapter: PostgresAdapter) -> None:
    result = postgres_adapter.run("SELECT '100%' AS pct, ':not_a_param' AS colon, 'a::b' AS casts, "
                                  "count(*) AS n FROM customers WHERE email LIKE '%@example.%'")
    row = result.rows[0]
    assert (row["pct"], row["colon"], row["casts"]) == ("100%", ":not_a_param", "a::b")
    assert row["n"] > 0


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET status = 'cancelled'",
        "INSERT INTO customers (customer_id, name, email, city, signup_date) "
        "VALUES (99999, 'x', 'x@example.com', 'x', '2025-01-01')",
        "CREATE TABLE hacked (x int)",
        "DROP TABLE order_items",
        "TRUNCATE orders",
    ],
)
def test_postgres_blocks_writes_even_without_guardrails(postgres_adapter: PostgresAdapter, sql: str) -> None:
    """Bypass the guardrails: the READ ONLY transaction must refuse the write."""
    with pytest.raises(QueryError, match="read-only transaction"):
        postgres_adapter._execute(sql, 10)
    assert postgres_adapter.run("SELECT count(*) AS n FROM orders").rows[0]["n"] == 8000


def test_postgres_statement_timeout(settings: Settings) -> None:
    assert settings.postgres_url is not None
    adapter = PostgresAdapter(settings.postgres_url.get_secret_value(), timeout_s=1, connect_timeout_s=3)
    if not adapter.ping():
        pytest.skip("Postgres is not reachable")
    try:
        started = time.perf_counter()
        with pytest.raises(QueryTimeoutError):
            adapter._execute("SELECT pg_sleep(5)", 1)  # bypasses the guardrails on purpose
        assert time.perf_counter() - started < 3
    finally:
        adapter.close()


def test_postgres_guardrails_block_pg_sleep_before_execution(postgres_adapter: PostgresAdapter) -> None:
    started = time.perf_counter()
    with pytest.raises(GuardrailError):
        postgres_adapter.run("SELECT pg_sleep(30)")
    assert time.perf_counter() - started < 1


def test_postgres_errors_are_clean(postgres_adapter: PostgresAdapter) -> None:
    with pytest.raises(QueryError) as caught:
        postgres_adapter.run("SELECT no_such_column FROM orders")
    message = str(caught.value)
    assert 'column "no_such_column" does not exist' in message
    assert "sqlalche.me" not in message and "Traceback" not in message


def test_postgres_unreachable() -> None:
    adapter = PostgresAdapter("postgresql://nobody:secret@127.0.0.1:1/nothing", connect_timeout_s=2)
    assert adapter.ping() is False
    with pytest.raises(DatabaseUnavailableError):
        adapter.run("SELECT 1")


def test_postgres_app_user_is_read_only(settings: Settings, postgres_adapter: PostgresAdapter) -> None:
    """POSTGRES_URL must point at a read-only user: even an explicit READ WRITE
    transaction (no TalkTable settings at all) must fail to write."""
    assert settings.postgres_url is not None
    with psycopg.connect(settings.postgres_url.get_secret_value(), connect_timeout=3) as conn:
        for sql in ("CREATE TABLE talktable_probe (x int)", "DELETE FROM orders WHERE order_id = 1"):
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ WRITE")
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(sql)
            conn.rollback()


POSTGRES_REALISTIC = [
    "SELECT date_trunc('month', order_date)::date AS month, count(*) FROM orders "
    "WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01' GROUP BY 1 ORDER BY 1",
    "SELECT EXTRACT(MONTH FROM order_date) AS m, COUNT(*) FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2025 "
    "GROUP BY m ORDER BY m",
    "SELECT to_char(order_date, 'YYYY-MM') AS ym, sum(total_amount) FROM orders "
    "WHERE status NOT IN ('cancelled', 'returned') GROUP BY ym ORDER BY ym DESC LIMIT 6",
    "SELECT c.city, count(*) FILTER (WHERE o.status = 'delivered') AS delivered FROM customers c "
    "JOIN orders o USING (customer_id) GROUP BY c.city ORDER BY delivered DESC, c.city LIMIT 5",
    "WITH spend AS (SELECT customer_id, SUM(total_amount) AS s FROM orders GROUP BY customer_id) "
    "SELECT c.name, spend.s, RANK() OVER (ORDER BY spend.s DESC) AS r FROM spend "
    "JOIN customers c ON c.customer_id = spend.customer_id ORDER BY r, c.name LIMIT 10",
    "SELECT book_id, SUM(quantity * unit_price)::numeric(10, 2) AS revenue FROM order_items "
    "GROUP BY book_id ORDER BY revenue DESC, book_id LIMIT 5",
    "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY total_amount) AS median FROM orders",
    "SELECT DISTINCT ON (customer_id) customer_id, order_date FROM orders ORDER BY customer_id, order_date DESC LIMIT 5",
    "SELECT name FROM customers WHERE name ILIKE '%son%' ORDER BY customer_id LIMIT 5",
    "SELECT signup_date + INTERVAL '30 days' AS trial_end FROM customers ORDER BY customer_id LIMIT 3",
    "SELECT order_date FROM orders WHERE order_date BETWEEN DATE '2025-03-01' AND DATE '2025-03-31' "
    "ORDER BY total_amount DESC, order_id FETCH FIRST 3 ROWS ONLY",
]


@pytest.mark.parametrize("sql", POSTGRES_REALISTIC)
def test_postgres_rewritten_sql_gives_same_results(postgres_adapter: PostgresAdapter, settings: Settings,
                                                   sql: str) -> None:
    """The guardrails regenerate SQL from the syntax tree; results must not change."""
    assert settings.postgres_url is not None
    with psycopg.connect(settings.postgres_url.get_secret_value(), connect_timeout=3) as conn:
        expected = [[to_jsonable(v) for v in row] for row in conn.execute(sql).fetchall()]
    result = postgres_adapter.run(sql)
    assert [list(row.values()) for row in result.rows] == expected


# =============================================================================
# MongoDB
# =============================================================================


def test_mongo_schema(mongo_adapter: MongoAdapter) -> None:
    tables = {t.name: t for t in mongo_adapter.get_schema()}
    assert set(tables) == set(MONGO_COLLECTIONS)
    assert tables["reviews"].row_count == 6000 and tables["clickstream"].row_count == 5000
    fields = {c.name: c.type for c in tables["clickstream"].columns}
    assert fields["events"] == "array" and fields["events.ts"] == "date" and fields["events.type"] == "string"
    assert {c.name for c in tables["reviews"].columns} == {
        "book_id", "customer_id", "rating", "text", "created_at", "helpful_votes"}


def test_mongo_run(mongo_adapter: MongoAdapter) -> None:
    result = mongo_adapter.run({"collection": "reviews", "pipeline": [
        {"$match": {"rating": 1}},
        {"$group": {"_id": "$book_id", "one_star": {"$sum": 1}}},
        {"$sort": {"one_star": -1, "_id": 1}},
        {"$limit": 5},
    ]})
    assert result.columns == ["_id", "one_star"] and result.row_count == 5
    counts = [row["one_star"] for row in result.rows]
    assert counts == sorted(counts, reverse=True)
    assert '"$limit": 5' in result.query


def test_mongo_extended_json_dates(mongo_adapter: MongoAdapter, settings: Settings) -> None:
    query = ('{"collection": "reviews", "pipeline": [{"$match": {"created_at": '
             '{"$gte": {"$date": "2025-01-01T00:00:00Z"}, "$lt": {"$date": "2026-01-01T00:00:00Z"}}}}, '
             '{"$count": "n"}]}')
    result = mongo_adapter.run(query)
    assert settings.mongo_url is not None
    with MongoClient(settings.mongo_url.get_secret_value(), serverSelectionTimeoutMS=2000) as client:
        expected = client[settings.mongo_db].reviews.count_documents(
            {"created_at": {"$gte": datetime(2025, 1, 1), "$lt": datetime(2026, 1, 1)}})
    assert result.rows == [{"n": expected}] and expected > 0


def test_mongo_output_is_capped(mongo_adapter: MongoAdapter) -> None:
    result = mongo_adapter.run({"collection": "reviews", "pipeline": []})
    assert result.row_count == 200 and result.truncated
    assert isinstance(result.rows[0]["_id"], str)  # ObjectId -> str
    assert isinstance(result.rows[0]["created_at"], str)  # datetime -> ISO string


def test_mongo_guardrails_run_before_execution(mongo_adapter: MongoAdapter, settings: Settings) -> None:
    for pipeline in ([{"$out": "stolen"}], [{"$merge": {"into": "stolen"}}]):
        with pytest.raises(GuardrailError):
            mongo_adapter.run({"collection": "reviews", "pipeline": pipeline})
    assert settings.mongo_url is not None
    with MongoClient(settings.mongo_url.get_secret_value(), serverSelectionTimeoutMS=2000) as client:
        assert "stolen" not in client[settings.mongo_db].list_collection_names()


def test_mongo_errors_are_clean(mongo_adapter: MongoAdapter) -> None:
    with pytest.raises(QueryError) as caught:
        mongo_adapter.run({"collection": "reviews", "pipeline": [{"$group": {"total": {"$sum": 1}}}]})
    assert "_id" in str(caught.value)  # "a group specification must include an _id"


def test_mongo_unreachable() -> None:
    adapter = MongoAdapter("mongodb://127.0.0.1:1", "nothing", server_selection_timeout_ms=500)
    assert adapter.ping() is False
    with pytest.raises(DatabaseUnavailableError):
        adapter.run({"collection": "reviews", "pipeline": []})


def test_mongo_app_user_is_read_only(settings: Settings, mongo_adapter: MongoAdapter) -> None:
    """MONGO_URL must point at a user with only the `read` role."""
    assert settings.mongo_url is not None
    with MongoClient(settings.mongo_url.get_secret_value(), serverSelectionTimeoutMS=2000) as client:
        with pytest.raises(OperationFailure, match="not authorized"):
            client[settings.mongo_db].reviews.insert_one({"hacked": True})
