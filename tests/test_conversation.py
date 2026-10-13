"""Follow-up questions: detection, the rewrite, and how the pipeline and chat UI use it."""

from __future__ import annotations

import pytest

from talktable import tools, webui
from talktable.adapters import MongoAdapter, SQLiteAdapter
from talktable.conversation import Condenser, Turn, build_messages, looks_like_follow_up, preview_rows
from talktable.llm import LLMError
from talktable.models import AskResult
from talktable.pipeline import TalkTable
from tests.fakes import FakeLLM, HashEmbedder, scripted, sql_reply

TOP_5_SQL = "SELECT title, price FROM books ORDER BY price DESC LIMIT 5"
TOP_10_SQL = "SELECT title, price FROM books ORDER BY price DESC LIMIT 10"

FIRST_TURN = Turn(question="What are the 5 most expensive books?", databases=["sqlite"], query=TOP_5_SQL,
                  answer="The most expensive is The Hidden Life of Black Holes at $110.95.",
                  rows_preview="title=The Hidden Life of Black Holes, price=110.95")


def engine_with(llm: FakeLLM, sqlite_adapter: SQLiteAdapter, **adapters) -> TalkTable:  # type: ignore[no-untyped-def]
    return TalkTable(adapters={"sqlite": sqlite_adapter, **adapters}, llm=llm, embedder=HashEmbedder())


# =============================================================================
# Detection and the rewrite itself
# =============================================================================


@pytest.mark.parametrize("message", [
    "and in 2024?", "now only for London", "What about the top 10 instead?", "what are their titles?",
    "Show the same for 2023", "only fiction", "how about mobile users", "And which of those were cancelled?",
])
def test_follow_ups_are_detected(message: str) -> None:
    assert looks_like_follow_up(message)


@pytest.mark.parametrize("message", [
    "How many orders were placed each month in 2025?",
    "Which 5 books have the most 1-star reviews?",
    "Who are the top 5 customers by total spend in London?",
])
def test_standalone_questions_skip_the_rewrite(message: str) -> None:
    assert not looks_like_follow_up(message)


def test_the_rewrite_sees_the_earlier_query_and_rows() -> None:
    prompt = build_messages("and the top 10?", [FIRST_TURN])[1]["content"]
    assert "[1] Question: What are the 5 most expensive books?" in prompt
    assert f"Query: {TOP_5_SQL}" in prompt and "First rows: title=The Hidden Life" in prompt
    assert prompt.endswith("Latest message: and the top 10?")


@pytest.mark.parametrize("reply", ['{"question": ""}', '{"q": "x"}', "not json", LLMError("down")])
def test_a_failed_rewrite_keeps_the_message(reply: str | Exception) -> None:
    def handler(messages: list, json_mode: bool) -> str:
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert Condenser(FakeLLM(handler)).standalone("and the top 10?", [FIRST_TURN]) == "and the top 10?"


def test_no_history_means_no_rewrite() -> None:
    llm = scripted(condenser="should not be used")
    assert Condenser(llm).standalone("and the top 10?", []) == "and the top 10?" and llm.calls == []


def test_turn_from_result_keeps_what_the_rewrite_needs() -> None:
    rows = [{"_id": n, "one_star": 20 - n} for n in range(15)]
    result = AskResult(question="and the top 15?", interpreted_as="Which 15 books have the most 1-star reviews?",
                       databases_used=["mongo"], queries=['{"collection": "reviews"}'], rows=rows,
                       columns=["_id", "one_star"], answer="Book 0 has 20 one-star reviews. " * 30, retries=0,
                       latency_ms=1)
    turn = Turn.from_result(result)
    assert turn.question == "Which 15 books have the most 1-star reviews?"  # the standalone version
    assert turn.rows_preview.startswith("_id=0, one_star=20; _id=1, one_star=19")
    assert "(+5 more rows)" in turn.rows_preview and len(turn.answer) <= 300


def test_preview_is_capped() -> None:
    rows = [{"text": "x" * 500}] * 3
    assert len(preview_rows(["text"], rows)) <= 600


# =============================================================================
# In the pipeline
# =============================================================================


def test_a_follow_up_is_answered_as_a_standalone_question(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(condenser="What are the 10 most expensive books?", generator=[sql_reply(TOP_10_SQL)])
    result = engine_with(llm, sqlite_adapter).ask("and the top 10?", history=[FIRST_TURN])
    assert result.error is None and len(result.rows) == 10
    assert result.question == "and the top 10?"
    assert result.interpreted_as == "What are the 10 most expensive books?"
    generator_prompt = [msgs for step, msgs in llm.calls if step == "generator"][0]
    assert generator_prompt[-1]["content"] == "What are the 10 most expensive books?"
    assert [step for step, _ in llm.calls][0] == "condenser"


def test_routing_uses_the_rewritten_question(sqlite_adapter: SQLiteAdapter, mongo_adapter: MongoAdapter) -> None:
    llm = scripted(condenser="What are the titles of the books with book_id 158 and 399?",
                   router='{"databases": ["sqlite"]}',
                   generator=[sql_reply("SELECT book_id, title FROM books WHERE book_id IN (158, 399) LIMIT 2")])
    mongo_turn = Turn(question="Which 2 books have the most 1-star reviews?", databases=["mongo"],
                      query='{"collection": "reviews", "pipeline": []}', answer="Books 158 and 399.",
                      rows_preview="_id=158, n=17; _id=399, n=16")
    result = engine_with(llm, sqlite_adapter, mongo=mongo_adapter).ask("what are their titles?",
                                                                      history=[mongo_turn])
    router_prompt = [msgs for step, msgs in llm.calls if step == "router"][0]
    assert router_prompt[-1]["content"] == "What are the titles of the books with book_id 158 and 399?"
    assert result.databases_used == ["sqlite"] and {row["book_id"] for row in result.rows} == {158, 399}


def test_standalone_questions_in_a_chat_skip_the_rewrite(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(condenser="should not be used", generator=[sql_reply(TOP_5_SQL)])
    result = engine_with(llm, sqlite_adapter).ask("Which books have the highest price in the catalog?",
                                                  history=[FIRST_TURN])
    assert llm.count("condenser") == 0 and result.interpreted_as is None


def test_small_talk_in_a_chat_skips_the_rewrite(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(condenser="should not be used")
    result = engine_with(llm, sqlite_adapter).ask("thanks!", history=[FIRST_TURN])
    assert result.answer.startswith("You're welcome!") and llm.calls == []


def test_a_failed_rewrite_still_answers(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])  # no condenser reply: the rewrite fails
    result = engine_with(llm, sqlite_adapter).ask("and the most expensive ones?", history=[FIRST_TURN])
    assert result.error is None and result.interpreted_as is None and len(result.rows) == 5


# =============================================================================
# In the chat UI
# =============================================================================


def test_the_chat_remembers_turns_for_follow_ups(monkeypatch: pytest.MonkeyPatch,
                                                 sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(condenser="What are the 10 most expensive books?",
                   generator=[sql_reply(TOP_5_SQL), sql_reply(TOP_10_SQL)])
    engine = engine_with(llm, sqlite_adapter)
    monkeypatch.setattr(tools, "get_talktable", lambda: engine)

    chat, _, _, turns = list(webui.answer_stream("What are the 5 most expensive books?", [], []))[-1]
    assert [t["question"] for t in turns] == ["What are the 5 most expensive books?"]

    chat, _, table, turns = list(webui.answer_stream("and the top 10?", chat, turns))[-1]
    assert "Interpreted as: <b>What are the 10 most expensive books?</b>" in chat[-1]["content"]
    assert table.label == "Results · 10 rows"
    assert [t["question"] for t in turns] == ["What are the 5 most expensive books?",
                                              "What are the 10 most expensive books?"]
