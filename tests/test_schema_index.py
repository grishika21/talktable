"""Schema retrieval: multi-vector scoring, join expansion, token budget, vector stores.

Most tests use HashEmbedder (fast, deterministic). The last ones run the held-out
benchmark questions (eval/retrieval_questions.yaml) with the real model as a guard
against retrieval regressions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from talktable.adapters import MongoAdapter, PostgresAdapter, SQLiteAdapter
from talktable.config import PROJECT_ROOT, Settings, get_settings
from talktable.descriptions import DatabaseDescription, FewShotExample, describe_tables, load_descriptions
from talktable.generator import estimate_tokens, format_schema
from talktable.models import ColumnInfo, TableInfo
from talktable.pipeline import TalkTable
from talktable.schema_index import SchemaIndex, SentenceEmbedder, table_text
from talktable.vector_store import QdrantVectorStore
from tests.fakes import HashEmbedder


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts += texts
        return super().encode(texts)


@pytest.fixture(scope="module")
def catalog(sqlite_adapter: SQLiteAdapter) -> tuple[list[TableInfo], DatabaseDescription]:
    description = load_descriptions().databases["sqlite"]
    return describe_tables(sqlite_adapter.get_schema(), description), description


def index_of(catalog: tuple[list[TableInfo], DatabaseDescription], embedder=None, store=None) -> SchemaIndex:  # type: ignore[no-untyped-def]
    index = SchemaIndex(embedder or HashEmbedder(), store)
    index.add_database("sqlite", *catalog)
    return index


# --- indexing ------------------------------------------------------------------


def test_each_table_gets_a_vector_per_text(catalog) -> None:  # type: ignore[no-untyped-def]
    tables, description = catalog
    embedder = CountingEmbedder()
    index_of(catalog, embedder)
    questions = sum(len(t.questions) for t in tables)
    assert len(embedder.texts) == len(tables) + questions + len(description.few_shot)
    assert "Example questions" not in table_text(tables[0], "Book catalog")  # embedded on their own


def test_only_changed_texts_are_embedded_again(catalog) -> None:  # type: ignore[no-untyped-def]
    tables, description = catalog
    embedder = CountingEmbedder()
    index = index_of(catalog, embedder)
    assert index.add_database("sqlite", tables, description) == 0  # nothing changed
    changed = [t.model_copy(update={"description": "Changed."}) if t.name == "tags" else t for t in tables]
    fewer = [t for t in changed if t.name != "series"]
    assert index.add_database("sqlite", fewer, description) == 1  # just the tags description
    keys = index.store.hashes("sqlite")
    assert "sqlite/tags/table" in keys and not any(key.startswith("sqlite/series/") for key in keys)


def test_a_sample_question_finds_its_table(catalog) -> None:  # type: ignore[no-untyped-def]
    index = index_of(catalog)
    selection = index.retrieve("Which books are tagged 'staff pick'?", "sqlite", k=1)
    assert selection.names[0] == "book_tags" and selection.scores["book_tags"] == pytest.approx(1.0, abs=1e-5)


def test_bridges_lookups_and_joins(catalog) -> None:  # type: ignore[no-untyped-def]
    index = index_of(catalog)
    selection = index.retrieve("Which publisher publishes the most books? Which books won the Starfall Award?",
                               "sqlite", k=2)
    assert {"publishers", "book_awards"} <= set(selection.names)
    assert selection.roles["editions"] == "bridge" and selection.roles["books"] == "bridge"
    assert "editions.publisher_id = publishers.publisher_id" in selection.joins
    assert "book_awards.book_id = books.book_id" in selection.joins
    assert selection.compact == {name for name, role in selection.roles.items() if role != "match"}


def test_weak_matches_are_left_out(catalog) -> None:  # type: ignore[no-untyped-def]
    selection = index_of(catalog).retrieve("When was the Golden Quill Prize first awarded?", "sqlite", k=5)
    assert selection.names[0] == "awards"
    assert all(score >= selection.scores["awards"] - 0.25 for score in selection.scores.values())
    assert len(selection.scores) < 5


def test_similar_examples_are_chosen_from_the_library(catalog) -> None:  # type: ignore[no-untyped-def]
    tables, description = catalog
    library = [FewShotExample(question=f"How many {thing} are there?", query=f"SELECT COUNT(*) FROM {thing}")
               for thing in ("authors", "books", "publishers", "awards", "tags")]
    index = SchemaIndex(HashEmbedder())
    index.add_database("sqlite", tables, description.model_copy(update={"few_shot": library}))
    picked = index.retrieve("How many awards are there?", "sqlite", examples=2).examples
    assert len(picked) == 2 and library[3] in picked
    assert picked == [e for e in library if e in picked]  # library order: a stable prompt
    assert index.retrieve("How many awards are there?", "sqlite", examples=0).examples == []


def test_wide_tables_show_their_keys_and_the_relevant_columns() -> None:
    columns = [ColumnInfo(name="id", type="INTEGER"), ColumnInfo(name="customer_id", type="INTEGER")] + [
        ColumnInfo(name=f"metric_{i}", type="REAL", description=f"Daily {word} figure")
        for i, word in enumerate(["revenue", "refunds", "visits", "signups", "returns", "clicks", "carts",
                                  "orders", "reviews", "searches", "emails", "opens", "bounces", "tickets",
                                  "points", "coupons", "shipments", "stock", "views", "ratings"])]
    wide = TableInfo(name="daily_stats", columns=columns, primary_key=["id"],
                     foreign_keys=["customer_id -> customers.customer_id"], description="Daily figures.")
    index = SchemaIndex(HashEmbedder())
    index.add_database("ops", [wide, TableInfo(name="customers", columns=columns[:1], primary_key=["id"])], None)
    shown = next(t for t in index.retrieve("Daily refunds figure", "ops", k=1).tables if t.name == "daily_stats")
    names = [c.name for c in shown.columns]
    assert len(names) == 14 and names[:2] == ["id", "customer_id"] and "metric_1" in names
    assert "Showing 14 of its 22 columns" in (shown.description or "")


def test_the_schema_is_kept_under_the_token_budget(catalog) -> None:  # type: ignore[no-untyped-def]
    index = index_of(catalog)
    question = "Which publisher publishes the most books? Which books won the Starfall Award?"
    roomy = index.retrieve(question, "sqlite", k=4)
    tight = index.retrieve(question, "sqlite", k=4, token_budget=350)
    best = roomy.names[0]
    assert tight.names[0] == best and best not in tight.compact  # the best match stays, in full
    assert len(tight.names) < len(roomy.names) or tight.compact > roomy.compact
    size = estimate_tokens(format_schema(tight.tables, "sql", compact=tight.compact, joins=tight.joins))
    assert size <= 350 or tight.names == [best]
    assert set(tight.joins) <= set(roomy.joins)


def test_rank_databases_uses_the_best_table(catalog, postgres_adapter: PostgresAdapter) -> None:  # type: ignore[no-untyped-def]
    descriptions = load_descriptions()
    index = index_of(catalog)
    index.add_database("postgres", describe_tables(postgres_adapter.get_schema(), descriptions.databases["postgres"]),
                       descriptions.databases["postgres"])
    ranked = index.rank_databases("Which books are out of stock?", ["sqlite", "postgres"])
    assert ranked[0][0] == "postgres" and ranked[0][1] == pytest.approx(1.0, abs=1e-5)  # inventory's sample
    assert index.rank_databases("anything", ["mongo"]) == []  # not indexed


# --- vector stores ----------------------------------------------------------------


def example_scores(index: SchemaIndex, question: str) -> list[float]:
    return [h.score for h in index.store.search(index.embed(question), databases=["sqlite"], kinds=["example"],
                                                limit=3)]


def test_numpy_and_qdrant_give_the_same_selection(catalog) -> None:  # type: ignore[no-untyped-def]
    numpy_index = index_of(catalog)
    qdrant_index = index_of(catalog, store=QdrantVectorStore("schema"))
    try:
        for question in ("Which books won the Starfall Award?", "Average price of ebooks per publisher",
                         "Which tags are most common?"):
            a, b = numpy_index.retrieve(question, "sqlite"), qdrant_index.retrieve(question, "sqlite")
            assert (a.names, a.joins, a.compact, len(a.examples)) == (b.names, b.joins, b.compact, len(b.examples))
            assert a.scores == pytest.approx(b.scores, abs=1e-4)
            # HashEmbedder ties examples often, and the stores may break ties differently
            assert example_scores(numpy_index, question) == pytest.approx(example_scores(qdrant_index, question))
    finally:
        qdrant_index.store.close()


def test_qdrant_keeps_vectors_between_runs(catalog, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    embedder = CountingEmbedder()
    first = index_of(catalog, embedder, QdrantVectorStore("schema", path=tmp_path / "q"))
    first.store.close()
    embedded = len(embedder.texts)
    second = SchemaIndex(embedder, QdrantVectorStore("schema", path=tmp_path / "q"))
    try:
        assert second.add_database("sqlite", *catalog) == 0 and len(embedder.texts) == embedded
        assert second.retrieve("Which tags are most common?", "sqlite").names[0] == "tags"
    finally:
        second.store.close()


def test_auto_picks_qdrant_for_large_schemas() -> None:
    settings = Settings(_env_file=None, vector_store="auto", vector_store_auto_tables=300)
    assert settings.vector_backend(30) == "numpy" and settings.vector_backend(300) == "qdrant"
    assert Settings(_env_file=None, vector_store="auto", qdrant_url="http://qdrant:6333").vector_backend(30) == "qdrant"
    assert Settings(_env_file=None, vector_store="numpy").vector_backend(10_000) == "numpy"


def test_the_engine_uses_the_configured_store(sqlite_adapter: SQLiteAdapter, tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"vector_store": "qdrant", "qdrant_path": tmp_path / "q"})
    engine = TalkTable(settings, adapters={"sqlite": sqlite_adapter}, llm=None, embedder=HashEmbedder(), cache=None)
    try:
        assert engine.index.store.backend == "qdrant" and engine.index.store.count() > 0
        engine.refresh()  # nothing changed: no new vectors, same store
        assert engine.index.store.backend == "qdrant"
    finally:
        engine.close()


# --- prompt rendering ---------------------------------------------------------------


def test_brief_tables_and_join_conditions_in_the_prompt(catalog, postgres_adapter: PostgresAdapter) -> None:  # type: ignore[no-untyped-def]
    tables, _ = catalog
    by_name = {t.name: t for t in tables}
    text = format_schema([by_name["books"], by_name["authors"]], "sql", compact=frozenset({"authors"}),
                         joins=["books.author_id = authors.author_id"])
    assert "Table books, 1,000 rows:" in text and "  - title TEXT" in text
    assert "Related tables (shown briefly):\n- authors, 150 rows: " in text
    assert "author_id INTEGER (primary key), name TEXT" in text and "Join conditions:\n- books.author_id" in text
    sales = {t.name: t for t in describe_tables(postgres_adapter.get_schema(),
                                                load_descriptions().databases["postgres"])}
    brief = format_schema([sales["order_items"]], "sql", compact=frozenset({"order_items"}))
    assert "book_id INTEGER (The book that was bought (the catalog's books.book_id).)" in brief  # cross-db note kept
    brief = format_schema([sales["warehouses"]], "sql", compact=frozenset({"warehouses"}))
    assert "'United States'" in brief  # quoted values are kept: filters need the exact spelling
    assert "Primary key of the warehouse" not in brief  # other notes are dropped


# --- retrieval quality with the real model ---------------------------------------------


@pytest.fixture(scope="module")
def real_index(embedder: SentenceEmbedder, sqlite_adapter: SQLiteAdapter, postgres_adapter: PostgresAdapter,
               mongo_adapter: MongoAdapter) -> SchemaIndex:
    descriptions = load_descriptions()
    index = SchemaIndex(embedder)
    for name, adapter in (("sqlite", sqlite_adapter), ("postgres", postgres_adapter), ("mongo", mongo_adapter)):
        description = descriptions.databases[name]
        index.add_database(name, describe_tables(adapter.get_schema(), description), description)
    return index


def test_held_out_questions_find_their_tables(real_index: SchemaIndex) -> None:
    """The benchmark (python -m eval.benchmark_retrieval) as a regression guard."""
    spec = yaml.safe_load((PROJECT_ROOT / "eval" / "retrieval_questions.yaml").read_text(encoding="utf-8"))
    questions = spec["questions"]
    routed = sum(real_index.rank_databases(q["q"], real_index.databases)[0][0] == q["db"] for q in questions)
    found = sum(set(q["tables"]) <= set(real_index.retrieve(q["q"], q["db"]).names) for q in questions)
    assert routed / len(questions) >= 0.90 and found / len(questions) >= 0.92


def test_out_of_stock_goes_to_inventory(real_index: SchemaIndex) -> None:
    """The case that motivated one vector per sample question: the inventory table's
    description talks about warehouses, so as a single vector it lost to books."""
    assert real_index.rank_databases("Which books are out of stock?", real_index.databases)[0][0] == "postgres"
    assert real_index.retrieve("Which books are out of stock?", "postgres").names[0] == "inventory"
