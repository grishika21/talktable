"""Phase 1 checks for the demo data: no database server needed.

* descriptions.yaml describes exactly the tables/columns that really exist
* the generator is deterministic and matches the sizes in the spec
* shared keys (book_id, customer_id) are consistent across the three databases
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
import yaml

from talktable.config import DESCRIPTIONS_PATH, get_settings
from data.seed import generate as gen
from data.seed.extended import generate_engagement_extras
from data.seed.seed_postgres import metadata as postgres_metadata


@pytest.fixture(scope="module")
def descriptions() -> dict[str, Any]:
    return yaml.safe_load(DESCRIPTIONS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reviews() -> list[dict[str, Any]]:
    return gen.generate_reviews()


@pytest.fixture(scope="module")
def clickstream() -> list[dict[str, Any]]:
    return gen.generate_clickstream()


def described(descriptions: dict[str, Any], db: str) -> dict[str, set[str]]:
    """{table_or_collection: {column_or_field, ...}} as written in descriptions.yaml."""
    spec = descriptions["databases"][db]
    objects = spec.get("tables") or spec.get("collections")
    key = "columns" if "tables" in spec else "fields"
    return {name: set(obj[key]) for name, obj in objects.items()}


# --- descriptions.yaml -------------------------------------------------------


def test_descriptions_have_routing_info(descriptions: dict[str, Any]) -> None:
    assert set(descriptions["databases"]) == {"sqlite", "postgres", "mongo"}
    for db, spec in descriptions["databases"].items():
        assert spec["kind"] in {"sql", "mongo"}, db
        assert len(spec["summary"]) > 50, db
        assert spec["examples"], db
        objects = spec.get("tables") or spec.get("collections")
        for name, obj in objects.items():
            assert obj["description"].strip(), f"{db}.{name}"
            fields = obj.get("columns") or obj.get("fields")
            assert all(str(text).strip() for text in fields.values()), f"{db}.{name}"


def test_descriptions_match_sqlite_schema(descriptions: dict[str, Any]) -> None:
    path = get_settings().sqlite_file
    if not path.exists():
        pytest.skip("data/bookstore.db not built yet: run python -m data.seed.seed_sqlite")
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        actual = {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")} for t in tables}
    assert actual == described(descriptions, "sqlite")


def test_descriptions_match_postgres_schema(descriptions: dict[str, Any]) -> None:
    actual = {t.name: {c.name for c in t.columns} for t in postgres_metadata.sorted_tables}
    assert actual == described(descriptions, "postgres")


def test_descriptions_match_mongo_fields(
    descriptions: dict[str, Any], reviews: list[dict[str, Any]], clickstream: list[dict[str, Any]]
) -> None:
    def field_paths(docs: list[dict[str, Any]]) -> set[str]:
        paths: set[str] = set()
        for doc in docs:
            for key, value in doc.items():
                paths.add(key)
                if isinstance(value, list):
                    paths.update(f"{key}.{sub}" for item in value if isinstance(item, dict) for sub in item)
        return paths

    collections = {"reviews": reviews, "clickstream": clickstream, **generate_engagement_extras()}
    assert {name: field_paths(docs) for name, docs in collections.items()} == described(descriptions, "mongo")


# --- generator ---------------------------------------------------------------


def test_sizes_match_spec(reviews: list[dict[str, Any]], clickstream: list[dict[str, Any]]) -> None:
    catalog, sales = gen.generate_catalog(), gen.generate_sales()
    assert (len(catalog.authors), len(catalog.categories), len(catalog.books)) == (150, 15, 1000)
    assert (len(sales.customers), len(sales.orders)) == (2000, 8000)
    assert 18_000 <= len(sales.order_items) <= 22_000
    assert (len(reviews), len(clickstream)) == (6000, 5000)


def test_generation_is_deterministic() -> None:
    # __wrapped__ bypasses the lru_cache so the data is really rebuilt.
    assert gen.generate_catalog.__wrapped__() == gen.generate_catalog()
    assert gen.generate_sales.__wrapped__() == gen.generate_sales()
    assert gen.generate_reviews() == gen.generate_reviews()
    assert gen.generate_clickstream() == gen.generate_clickstream()


def test_unique_titles_names_and_emails() -> None:
    catalog, sales = gen.generate_catalog(), gen.generate_sales()
    assert len({b.title for b in catalog.books}) == len(catalog.books)
    assert len({a.name for a in catalog.authors}) == len(catalog.authors)
    assert len({c.email for c in sales.customers}) == len(sales.customers)
    domains = {c.email.split("@")[1] for c in sales.customers}
    assert all(d.endswith(("example.com", "example.org", "example.net")) for d in domains)


def test_catalog_is_consistent() -> None:
    catalog = gen.generate_catalog()
    authors = {a.author_id: a for a in catalog.authors}
    category_ids = {c.category_id for c in catalog.categories}
    for book in catalog.books:
        assert book.category_id in category_ids
        assert book.price > 0 and book.pages > 0
        # Published during the author's adult life, and never in the future.
        assert authors[book.author_id].birth_year + 22 <= book.published_year <= gen.DATA_END.year


def test_sales_are_consistent() -> None:
    catalog, sales = gen.generate_catalog(), gen.generate_sales()
    books = {b.book_id: b for b in catalog.books}
    customers = {c.customer_id: c for c in sales.customers}
    items_by_order: dict[int, list[gen.OrderItem]] = defaultdict(list)
    for item in sales.order_items:
        assert item.book_id in books  # shared key -> SQLite
        assert item.quantity >= 1 and item.unit_price > 0
        items_by_order[item.order_id].append(item)

    for order in sales.orders:
        customer = customers[order.customer_id]
        assert customer.signup_date <= order.order_date <= gen.DATA_END
        assert order.status in gen.ORDER_STATUSES
        items = items_by_order[order.order_id]
        assert items, f"order {order.order_id} has no items"
        assert order.total_amount == sum((i.unit_price * i.quantity for i in items), Decimal("0"))
        # A book can't be bought before it was published.
        assert all(books[i.book_id].published_year <= order.order_date.year for i in items)

    # The spec's Postgres demo question needs orders in every month of 2025.
    assert {o.order_date.month for o in sales.orders if o.order_date.year == 2025} == set(range(1, 13))


def test_reviews_are_verified_purchases(reviews: list[dict[str, Any]]) -> None:
    sales = gen.generate_sales()
    orders = {o.order_id: o for o in sales.orders}
    first_purchase: dict[tuple[int, int], datetime] = {}
    for item in sales.order_items:
        order = orders[item.order_id]
        if order.status in ("delivered", "returned"):
            first_purchase.setdefault((order.customer_id, item.book_id),
                                      datetime.combine(order.order_date, datetime.min.time()))

    seen: set[tuple[int, int]] = set()
    for review in reviews:
        key = (review["customer_id"], review["book_id"])
        assert key in first_purchase, f"review {key} has no matching purchase"
        assert key not in seen, f"duplicate review {key}"
        seen.add(key)
        assert review["created_at"] > first_purchase[key]
        assert review["rating"] in {1, 2, 3, 4, 5}
        assert review["helpful_votes"] >= 0 and review["text"]

    # The spec's Mongo demo question needs books with several 1-star reviews.
    one_star: dict[int, int] = defaultdict(int)
    for review in reviews:
        if review["rating"] == 1:
            one_star[review["book_id"]] += 1
    assert sorted(one_star.values(), reverse=True)[4] >= 3


def test_clickstream_is_consistent(clickstream: list[dict[str, Any]]) -> None:
    catalog, sales = gen.generate_catalog(), gen.generate_sales()
    book_ids = {b.book_id for b in catalog.books}
    customers = {c.customer_id: c for c in sales.customers}
    assert len({s["session_id"] for s in clickstream}) == len(clickstream)
    for session in clickstream:
        assert session["device"] in gen.DEVICES
        events = session["events"]
        assert events and events[0]["type"] == "view"
        timestamps = [e["ts"] for e in events]
        assert timestamps == sorted(timestamps)
        assert timestamps[0].date() >= customers[session["customer_id"]].signup_date
        for event in events:
            assert event["type"] in gen.EVENT_TYPES
            assert event["book_id"] in book_ids
