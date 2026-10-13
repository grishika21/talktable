"""Answer cache: matching rules (with the real embedding model), persistence, and how the
pipeline and UI use it. The real cache file is never touched (see conftest.py)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from talktable import tools, webui
from talktable.adapters import SQLiteAdapter
from talktable.cache import AnswerCache, anchors, content_words, normalize, rows_fingerprint
from talktable.models import AskResult
from talktable.pipeline import TalkTable
from talktable.schema_index import SentenceEmbedder
from tests.fakes import HashEmbedder, scripted, sql_reply

TOP_5_SQL = "SELECT title, price FROM books ORDER BY price DESC LIMIT 5"
Q = "What are the 5 most expensive books?"


@pytest.fixture(scope="module")
def embed(embedder: SentenceEmbedder) -> Callable[[str], np.ndarray]:
    return lambda text: embedder.encode([text])[0]


def cache_with(embed: Callable[[str], np.ndarray], path: Path | None = None, **kwargs) -> AnswerCache:  # type: ignore[no-untyped-def]
    return AnswerCache(path, embed, version="v1", **kwargs)


# =============================================================================
# Matching rules
# =============================================================================


def test_normalize() -> None:
    assert normalize("What are the FIVE most expensive books?") == "what are the 5 most expensive books"
    assert normalize("Which five books have the most one-star reviews?") == \
        "which 5 books have the most 1 star reviews"
    assert normalize("Orders over $1,000.50 in 2024.") == "orders over $1000.50 in 2024"


def test_anchors() -> None:
    assert anchors("Which customers in London spent the most in 2024?") == ("#2024", "@london", "~max")
    assert anchors("each month") == anchors("per month")  # harmless rewording
    assert anchors("cheapest books") != anchors("most expensive books")
    assert anchors('books in the "Science Fiction" category') == ('"science fiction"', "@fiction", "@science")


def test_content_words() -> None:
    assert content_words("Show me the 5 most expensive books") == content_words("What are the five most expensive books?")
    assert content_words("How many fantasy books are there?") != content_words("How many fiction books are there?")


# The same calibration pairs as in the docs: rewordings must hit, near-misses must not.
SHOULD_HIT = [
    ("What are the 5 most expensive books?", "what are the five most expensive books", "same"),
    ("Which 5 books have the most 1-star reviews?", "Which five books have the most one-star reviews?", "same"),
    ("What are the 5 most expensive books?", "Show me the 5 most expensive books", "reworded"),
    ("How many orders were placed each month in 2025?", "How many orders were placed per month in 2025?", "reworded"),
    ("Who are the top 5 customers by total spend?", "Show the top 5 customers by total spend", "reworded"),
]
MUST_MISS = [
    ("Which 5 books sold the most copies in 2024?", "Which 5 books sold the most copies in 2025?"),
    ("What are the 5 most expensive books?", "What are the 10 most expensive books?"),
    ("What are the 5 most expensive books?", "What are the 5 cheapest books?"),
    ("How many orders were placed each month in 2025?", "How many orders were cancelled each month in 2025?"),
    ("Which customers in London spent the most?", "Which customers in Berlin spent the most?"),
    ("How many fantasy books are there?", "How many fiction books are there?"),
    ("What is the average price of books?", "What is the total price of books?"),
    ("Which books have the most reviews?", "Which books have the most 5-star reviews?"),
    ("How many orders were placed in 2024?", "How many orders were placed before 2024?"),
    ("Which books have the most pages?", "Which books have the fewest pages?"),
    ("How many customers signed up each year?", "How many customers signed up each month?"),
    ("Which authors are from Japan?", "Which authors are from Germany?"),
    ("How many sessions came from mobile devices?", "How many sessions came from tablet devices?"),
    ("What is the average rating of fantasy books?", "What is the average rating of romance books?"),
]


@pytest.mark.parametrize(("stored", "asked", "match"), SHOULD_HIT)
def test_rewordings_hit(embed, stored: str, asked: str, match: str) -> None:  # type: ignore[no-untyped-def]
    cache = cache_with(embed)
    cache.store(stored, database="sqlite", query="SELECT 1", rows_hash="h", answer="a")
    hit = cache.lookup(asked)
    assert hit is not None and hit.match == match and hit.question == stored


@pytest.mark.parametrize(("stored", "asked"), MUST_MISS)
def test_near_misses_never_hit(embed, stored: str, asked: str) -> None:  # type: ignore[no-untyped-def]
    cache = cache_with(embed)
    cache.store(stored, database="sqlite", query="SELECT 1", rows_hash="h", answer="a")
    assert cache.lookup(asked) is None


# =============================================================================
# Storage
# =============================================================================


def test_survives_a_restart(embed, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "cache.sqlite"
    cache_with(embed, path).store(Q, database="sqlite", query=TOP_5_SQL, rows_hash="h", answer="a")
    reopened = cache_with(embed, path)
    hit = reopened.lookup("Show me the 5 most expensive books")  # the stored vector is used
    assert hit is not None and hit.query == TOP_5_SQL and len(reopened) == 1


def test_a_new_version_retires_old_entries(embed, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "cache.sqlite"
    cache_with(embed, path).store(Q, database="sqlite", query=TOP_5_SQL, rows_hash="h", answer="a")
    newer = AnswerCache(path, embed, version="v2")
    assert newer.lookup(Q) is None
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 0


def test_entries_expire(embed) -> None:  # type: ignore[no-untyped-def]
    cache = cache_with(embed, ttl_hours=0.5 / 3600)  # half a second
    cache.store(Q, database="sqlite", query=TOP_5_SQL, rows_hash="h", answer="a")
    assert cache.lookup(Q) is not None
    time.sleep(0.6)
    assert cache.lookup(Q) is None and len(cache) == 0


def test_scopes_are_separate(embed) -> None:  # type: ignore[no-untyped-def]
    cache = cache_with(embed)
    cache.store(Q, database="sqlite", query=TOP_5_SQL, rows_hash="h", answer="a", scope="sqlite")
    assert cache.lookup(Q, scope="auto") is None and cache.lookup(Q, scope="sqlite") is not None


def test_forget_and_size_limit(embed, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "cache.sqlite"
    cache = cache_with(embed, path, max_entries=2)
    for year in (2022, 2023, 2024):
        cache.store(f"How many orders were placed in {year}?", database="postgres", query="SELECT 1",
                    rows_hash="h", answer="a")
    assert len(cache) == 2 and cache.lookup("How many orders were placed in 2022?") is None  # oldest dropped
    cache.forget("How many orders were placed in 2024?")
    assert len(cache) == 1 and len(cache_with(embed, path)) == 1  # also gone from the file


def test_a_broken_cache_file_is_not_fatal(embed, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "cache.sqlite"
    path.write_bytes(b"this is not a sqlite database")
    cache = cache_with(embed, path)  # starts empty instead of crashing
    cache.store(Q, database="sqlite", query=TOP_5_SQL, rows_hash="h", answer="a")  # write fails quietly
    assert cache.lookup(Q) is not None  # still works in memory


# =============================================================================
# In the pipeline
# =============================================================================


@pytest.fixture
def cached_engine(sqlite_adapter: SQLiteAdapter):  # type: ignore[no-untyped-def]
    def build(llm):  # type: ignore[no-untyped-def]
        index_embedder = HashEmbedder()
        engine = TalkTable(adapters={"sqlite": sqlite_adapter}, llm=llm, embedder=index_embedder, cache=None)
        engine.cache = AnswerCache(None, engine.index.embed, version="test")
        return engine
    return build


def test_a_repeated_question_skips_the_llm(cached_engine) -> None:  # type: ignore[no-untyped-def]
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])
    engine = cached_engine(llm)
    first = engine.ask(Q)
    second = engine.ask("what are the five most expensive books")
    assert not first.cached and second.cached
    assert second.rows == first.rows and second.answer == first.answer
    assert llm.count("generator") == 1 and llm.count("explainer") == 1  # nothing new on the second ask


def test_changed_data_gets_a_new_explanation(cached_engine) -> None:  # type: ignore[no-untyped-def]
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])
    engine = cached_engine(llm)
    engine.cache.store(Q, database="sqlite", query=TOP_5_SQL, rows_hash="rows-from-last-week", answer="stale")
    result = engine.ask(Q)
    assert result.cached and result.answer != "stale" and llm.count("explainer") == 1
    assert llm.count("generator") == 0  # the saved query was reused
    assert engine.cache.lookup(Q).rows_hash == rows_fingerprint(result.columns, result.rows)  # refreshed


def test_use_cache_false_answers_afresh(cached_engine) -> None:  # type: ignore[no-untyped-def]
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])
    engine = cached_engine(llm)
    engine.ask(Q)
    result = engine.ask(Q, use_cache=False)
    assert not result.cached and llm.count("generator") == 2


def test_failed_answers_are_not_cached(cached_engine) -> None:  # type: ignore[no-untyped-def]
    llm = scripted(generator=[sql_reply("SELECT nope FROM books")])
    engine = cached_engine(llm)
    assert engine.ask(Q).error is not None
    assert len(engine.cache) == 0


def test_a_saved_query_that_stopped_working_is_replaced(cached_engine) -> None:  # type: ignore[no-untyped-def]
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])
    engine = cached_engine(llm)
    engine.cache.store(Q, database="sqlite", query="DROP TABLE books", rows_hash="h", answer="a")
    result = engine.ask(Q)
    assert not result.cached and result.error is None and len(result.rows) == 5  # answered afresh
    assert engine.cache.lookup(Q).query == TOP_5_SQL


def test_small_talk_is_never_cached(cached_engine) -> None:  # type: ignore[no-untyped-def]
    engine = cached_engine(scripted())
    engine.ask("hi")
    assert len(engine.cache) == 0


def test_the_ui_marks_cached_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    result = AskResult(question=Q, databases_used=["sqlite"], queries=[TOP_5_SQL], rows=[{"title": "A"}],
                       columns=["title"], answer="A.", retries=0, latency_ms=420, cached=True)
    text = webui.render_answer(result, {"sqlite": "Book catalog"})
    assert '<span class="tt-badge tt-cached">⚡ Cached</span>' in text
    assert "saved query, re-run on current data · 0.4 s" in text

    seen: dict = {}

    def fake_ask(question: str, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs, question=question)
        return result

    monkeypatch.setattr(tools, "ask", fake_ask)
    monkeypatch.setattr(tools, "get_talktable", lambda: type("E", (), {"status": lambda self: []})())
    turns = [{"question": Q, "databases": ["sqlite"], "query": TOP_5_SQL, "answer": "A.", "rows_preview": ""}]
    list(webui.ask_again_fresh([], turns))
    assert seen["question"] == Q and seen["use_cache"] is False and seen["history"] == []
