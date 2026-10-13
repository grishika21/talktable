"""Router tests: the LLM path with a scripted fake, and the embedding fallback with the real model."""

from __future__ import annotations

import json

import pytest

from talktable.adapters import MongoAdapter, PostgresAdapter, SQLiteAdapter
from talktable.descriptions import describe_tables, load_descriptions
from talktable.llm import LLMError
from talktable.router import Router, is_small_talk
from talktable.schema_index import SchemaIndex, SentenceEmbedder
from tests.fakes import FakeLLM

ALL = ["sqlite", "postgres", "mongo"]


@pytest.fixture(scope="module")
def index(embedder: SentenceEmbedder, sqlite_adapter: SQLiteAdapter,
          postgres_adapter: PostgresAdapter, mongo_adapter: MongoAdapter) -> SchemaIndex:
    """The real schema index over all three databases (real embedding model)."""
    descriptions = load_descriptions()
    index = SchemaIndex(embedder)
    for adapter in (sqlite_adapter, postgres_adapter, mongo_adapter):
        description = descriptions.databases[adapter.name]
        index.add_database(adapter.name, describe_tables(adapter.get_schema(), description), description)
    return index


def router_with(reply: str | Exception, index: SchemaIndex) -> tuple[Router, FakeLLM]:
    def handler(messages: list, json_mode: bool) -> str:
        if isinstance(reply, Exception):
            raise reply
        return reply

    llm = FakeLLM(handler)
    return Router(llm, index, load_descriptions()), llm


def test_single_database_skips_the_llm(index: SchemaIndex) -> None:
    router, llm = router_with('{"databases": ["postgres"]}', index)
    decision = router.route("How many reviews have 5 stars?", ["mongo"])
    assert decision.databases == ["mongo"] and decision.method == "single"
    assert llm.calls == []


@pytest.mark.parametrize("message", ["hi", "Hello there!", "thanks", "Thank you so much.", "ok", "good morning"])
def test_small_talk_skips_the_llm(index: SchemaIndex, message: str) -> None:
    router, llm = router_with('{"databases": ["postgres"]}', index)
    decision = router.route(message, ALL)
    assert decision.databases == [] and decision.method == "small_talk" and llm.calls == []


@pytest.mark.parametrize("message", ["hi, which books are the most expensive?", "thanks, now only for 2024",
                                     "okay so which city has the most customers", "hello world sales"])
def test_questions_that_start_like_small_talk_are_questions(message: str) -> None:
    assert not is_small_talk(message)


@pytest.mark.parametrize("message", ["Who won the football world cup in 2022?", "Tell me a joke", "What's the weather?"])
def test_llm_says_no_database_and_embeddings_agree(index: SchemaIndex, message: str) -> None:
    router, _ = router_with('{"databases": [], "reason": "not about the data"}', index)
    decision = router.route(message, ALL)
    assert decision.databases == [] and decision.method == "off_topic"


def test_llm_says_no_database_but_the_question_is_about_data(index: SchemaIndex) -> None:
    """A wrong "no database" from the LLM is overruled when the question clearly matches one."""
    router, _ = router_with('{"databases": []}', index)
    decision = router.route("What is the average star rating of all reviews?", ALL)
    assert decision.databases == ["mongo"] and decision.method == "embedding"


def test_off_topic_is_detected_when_the_llm_is_down(index: SchemaIndex) -> None:
    router, _ = router_with(LLMError("offline"), index)
    assert router.route("Tell me a joke", ALL).method == "off_topic"


def test_off_topic_with_a_single_database(index: SchemaIndex) -> None:
    router, llm = router_with('{"databases": ["mongo"]}', index)
    decision = router.route("Tell me a joke", ["mongo"])
    assert decision.databases == [] and decision.method == "off_topic" and llm.calls == []


def test_llm_choice_is_used(index: SchemaIndex) -> None:
    router, llm = router_with('{"databases": ["postgres"], "reason": "orders live in sales"}', index)
    decision = router.route("How many orders last year?", ALL)
    assert decision.databases == ["postgres"] and decision.method == "llm"
    assert decision.reason == "orders live in sales"
    assert llm.calls[0][1][1]["content"] == "How many orders last year?"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"databases": ["PostgreSQL"]}', ["postgres"]),
        ('{"databases": ["MongoDB", "sqlite3"]}', ["mongo", "sqlite"]),
        ('{"databases": "mongo"}', ["mongo"]),  # a bare string instead of a list
        ('{"databases": ["sqlite", "sqlite", "mongo", "postgres"]}', ["sqlite", "mongo"]),  # dedupe, max 2
        ('```json\n{"databases": ["mongo"]}\n```', ["mongo"]),  # code fence
    ],
)
def test_llm_reply_is_normalised(index: SchemaIndex, reply: str, expected: list[str]) -> None:
    router, _ = router_with(reply, index)
    assert router.route("q", ALL).databases == expected


@pytest.mark.parametrize(
    "reply",
    ['{"databases": ["oracle"]}', '{"databases": []}', '{"reason": "no idea"}', "I think sqlite", "[]",
     LLMError("Ollama is down")],
)
def test_bad_llm_replies_fall_back_to_embeddings(index: SchemaIndex, reply: str | Exception) -> None:
    router, _ = router_with(reply, index)
    decision = router.route("Which authors were born before 1950?", ALL)
    assert decision.method == "embedding" and decision.databases == ["sqlite"]


def test_offline_databases_are_never_chosen(index: SchemaIndex) -> None:
    router, llm = router_with('{"databases": ["postgres"]}', index)
    decision = router.route("How many orders were placed in 2024?", ["sqlite", "mongo"])
    assert "postgres" not in decision.databases
    prompt = llm.calls[0][1][0]["content"]
    assert "- sqlite (" in prompt and "- mongo (" in prompt and "- postgres (" not in prompt


def test_prompt_includes_descriptions_and_routing_notes(index: SchemaIndex) -> None:
    router, llm = router_with('{"databases": ["sqlite"]}', index)
    router.route("q", ALL)
    prompt = llm.calls[0][1][0]["content"]
    assert "Book catalog" in prompt and "Tables: authors, awards, book_awards, book_series, book_tags, books, categories," in prompt
    for note in load_descriptions().routing_notes:
        assert " ".join(note.split()) in prompt
    assert json.loads('{"databases": ["<database name>"], "reason": "<one short sentence>"}')  # format shown


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # The three demo questions from the spec (not used as examples anywhere).
        ("What are the 5 most expensive books?", "sqlite"),
        ("How many orders were placed each month in 2025?", "postgres"),
        ("Which 5 books have the most 1-star reviews?", "mongo"),
        # Others
        ("Which authors are from Japan?", "sqlite"),
        ("How many fantasy books have more than 500 pages?", "sqlite"),
        ("What was the total revenue in 2024?", "postgres"),
        ("Which city has the most customers?", "postgres"),
        ("What is the average star rating of all reviews?", "mongo"),
        ("How many browsing sessions happened on tablets?", "mongo"),
    ],
)
def test_embedding_fallback_picks_the_right_database(index: SchemaIndex, question: str, expected: str) -> None:
    """With the LLM out of the picture, description similarity alone routes sensibly."""
    router, _ = router_with(LLMError("offline"), index)
    assert router.route(question, ALL).databases == [expected]
