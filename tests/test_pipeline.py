"""Pipeline tests.

Most use a scripted fake LLM against the real SQLite catalog, so they are fast and
deterministic. The `live` tests at the bottom run the spec's three demo questions
through the real LLM (Ollama by default): TALKTABLE_LIVE_TESTS=1 pytest -m live
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest

from talktable.adapters import MongoAdapter, PostgresAdapter, SQLiteAdapter
from talktable.config import get_settings
from talktable.descriptions import describe_tables, load_descriptions
from talktable.explainer import NO_ROWS_ANSWER, ungrounded_numbers
from talktable.generator import QueryContext, build_messages, parse_reply
from talktable.llm import LLMError, parse_json_object
from talktable.models import QueryResult
from talktable.pipeline import TalkTable
from talktable.schema_index import SchemaIndex
from tests.fakes import HashEmbedder, quote_first_row, scripted, sql_reply

TOP_5_SQL = "SELECT title, price FROM books ORDER BY price DESC LIMIT 5"


def engine_with(llm, sqlite_adapter: SQLiteAdapter, **adapters) -> TalkTable:  # type: ignore[no-untyped-def]
    # HashEmbedder similarities aren't calibrated, so no relevance floor here. The floor
    # is tested with the real model (test_off_topic_question_runs_no_query, test_router).
    settings = get_settings().model_copy(update={"min_relevance": 0.0})
    return TalkTable(settings, adapters={"sqlite": sqlite_adapter, **adapters}, llm=llm, embedder=HashEmbedder())


# =============================================================================
# The pipeline with a scripted LLM
# =============================================================================


def test_happy_path(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])
    result = engine_with(llm, sqlite_adapter).ask("What are the 5 most expensive books?")
    assert result.error is None
    assert result.databases_used == ["sqlite"]
    assert result.queries == [TOP_5_SQL]
    assert len(result.rows) == 5 and result.columns == ["title", "price"]
    assert result.rows[0]["price"] == 110.95
    assert "110.95" in result.answer
    assert result.retries == 0 and result.latency_ms >= 0
    assert llm.count("router") == 0  # only one database online: no routing call


def test_self_correction_after_a_database_error(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply("SELECT titel FROM books LIMIT 3"), sql_reply(TOP_5_SQL)])
    result = engine_with(llm, sqlite_adapter).ask("Most expensive books?")
    assert result.error is None and result.retries == 1 and len(result.rows) == 5
    correction = [msgs for step, msgs in llm.calls if step == "generator"][1]
    assert correction[-2]["role"] == "assistant" and "titel" in correction[-2]["content"]
    assert "no such column: titel" in correction[-1]["content"]


def test_guardrail_rejection_triggers_a_retry(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply("DELETE FROM books"), sql_reply(TOP_5_SQL)])
    result = engine_with(llm, sqlite_adapter).ask("Most expensive books?")
    assert result.retries == 1 and result.error is None
    correction = [msgs for step, msgs in llm.calls if step == "generator"][1]
    assert "Only SELECT queries are allowed" in correction[-1]["content"]
    assert sqlite_adapter.run("SELECT COUNT(*) AS n FROM books").rows[0]["n"] == 1000


def test_gives_up_after_two_retries(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply("SELECT nope FROM books")])
    result = engine_with(llm, sqlite_adapter).ask("Which books have the most pages?")
    assert result.error is not None and "no such column: nope" in result.answer
    assert result.retries == 2 and llm.count("generator") == 3
    assert result.rows == [] and result.queries == ["SELECT nope FROM books"]


def test_retries_are_validated_again(sqlite_adapter: SQLiteAdapter) -> None:
    """A 'correction' that tries something forbidden is caught too, every time."""
    llm = scripted(generator=[sql_reply("SELECT nope FROM books"), sql_reply("DROP TABLE books"),
                              sql_reply("SELECT 1; DELETE FROM books")])
    result = engine_with(llm, sqlite_adapter).ask("Which books are the longest?")
    assert result.error is not None and "Only one SQL statement" in result.answer
    assert sqlite_adapter.run("SELECT COUNT(*) AS n FROM books").rows[0]["n"] == 1000


def test_invented_numbers_are_never_shown(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply(TOP_5_SQL)], explainer="The priciest book costs $999.")
    result = engine_with(llm, sqlite_adapter).ask("Most expensive books?")
    assert "999" not in result.answer and "110.95" in result.answer  # template fallback
    assert llm.count("explainer") == 2  # one retry with feedback, then the fallback
    retry_prompt = [msgs for step, msgs in llm.calls if step == "explainer"][1][-1]["content"]
    assert "999" in retry_prompt


def test_explainer_gets_a_second_chance(sqlite_adapter: SQLiteAdapter) -> None:
    replies = iter(["The top book costs $999.", None])

    def explainer(messages: list) -> str:
        reply = next(replies)
        return reply if reply is not None else quote_first_row(messages)

    llm = scripted(generator=[sql_reply(TOP_5_SQL)], explainer=explainer)
    result = engine_with(llm, sqlite_adapter).ask("Most expensive books?")
    assert result.answer.startswith("The top result is")


def test_no_rows_answer_skips_the_llm(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[sql_reply("SELECT title FROM books WHERE price > 100000 LIMIT 5")])
    result = engine_with(llm, sqlite_adapter).ask("Books over $100,000?")
    assert result.answer == NO_ROWS_ANSWER and result.error is None
    assert llm.count("explainer") == 0


def test_llm_failure_is_a_friendly_error(sqlite_adapter: SQLiteAdapter) -> None:
    llm = scripted(generator=[])  # raises LLMError
    result = engine_with(llm, sqlite_adapter).ask("Most expensive books?")
    assert result.error and "Traceback" not in result.answer


@pytest.mark.parametrize(
    ("message", "intro"),
    [("hi", "Hi! I'm TalkTable"), ("Thanks!", "You're welcome!"), ("bye", "Goodbye!")],
)
def test_small_talk_gets_a_friendly_reply_without_a_query(sqlite_adapter: SQLiteAdapter, message: str,
                                                          intro: str) -> None:
    llm = scripted(generator=[sql_reply(TOP_5_SQL)])
    result = engine_with(llm, sqlite_adapter).ask(message)
    assert result.answer.startswith(intro) and "**Book catalog**: authors, awards, book_awards," in result.answer
    assert result.error is None and result.queries == [] and result.rows == [] and result.databases_used == []
    assert llm.calls == []  # not even the router


def test_off_topic_question_runs_no_query(sqlite_adapter: SQLiteAdapter, mongo_adapter: MongoAdapter,
                                         embedder) -> None:  # type: ignore[no-untyped-def]
    # The real embedding model: the relevance floor is calibrated for it.
    llm = scripted(router='{"databases": [], "reason": "not about the data"}', generator=[sql_reply(TOP_5_SQL)])
    engine = TalkTable(adapters={"sqlite": sqlite_adapter, "mongo": mongo_adapter}, llm=llm, embedder=embedder)
    result = engine.ask("Who won the football world cup in 2022?")
    assert result.answer.startswith("That doesn't look like a question about the bookstore's data")
    assert result.queries == [] and llm.count("generator") == 0


@pytest.mark.parametrize(("question", "message"), [("   ", "Please type a question"), ("x" * 501, "under 500")])
def test_bad_questions(sqlite_adapter: SQLiteAdapter, question: str, message: str) -> None:
    result = engine_with(scripted(), sqlite_adapter).ask(question)
    assert result.error and message in result.answer


def test_offline_databases_are_reported_and_skipped(sqlite_adapter: SQLiteAdapter) -> None:
    offline = PostgresAdapter("postgresql://nobody:x@127.0.0.1:1/none", connect_timeout_s=1)
    engine = engine_with(scripted(generator=[sql_reply(TOP_5_SQL)]), sqlite_adapter, postgres=offline)
    status = {s.name: s for s in engine.status()}
    assert status["sqlite"].online and not status["postgres"].online
    assert "not configured" in (status["mongo"].detail or "")
    assert engine.online == ["sqlite"]
    assert engine.ask("Most expensive books?").databases_used == ["sqlite"]


def test_forcing_an_offline_database_is_an_error(sqlite_adapter: SQLiteAdapter) -> None:
    result = engine_with(scripted(), sqlite_adapter).ask("q", database="mongo")
    assert result.error and "not available" in result.answer


def test_routing_between_databases(sqlite_adapter: SQLiteAdapter, mongo_adapter: MongoAdapter) -> None:
    pipeline = '{"collection": "reviews", "pipeline": [{"$match": {"rating": 1}}, {"$count": "n"}]}'
    llm = scripted(router='{"databases": ["mongo"], "reason": "reviews"}', generator=[pipeline])
    result = engine_with(llm, sqlite_adapter, mongo=mongo_adapter).ask("How many 1-star reviews are there?")
    assert result.databases_used == ["mongo"] and result.error is None
    assert result.rows[0]["n"] > 0 and '"$limit": 200' in result.queries[0]  # output capped


def test_separate_query_model(sqlite_adapter: SQLiteAdapter, mongo_adapter: MongoAdapter) -> None:
    """With LLM_QUERY_MODEL set, only query writing (and correcting) goes to that model."""
    main = scripted(router='{"databases": ["sqlite"]}')
    coder = scripted(generator=[sql_reply("SELECT titel FROM books LIMIT 3"), sql_reply(TOP_5_SQL)])
    engine = TalkTable(adapters={"sqlite": sqlite_adapter, "mongo": mongo_adapter}, llm=main, query_llm=coder,
                     embedder=HashEmbedder())
    result = engine.ask("Most expensive books?")
    assert result.error is None and result.retries == 1
    assert [step for step, _ in main.calls] == ["router", "explainer"]
    assert [step for step, _ in coder.calls] == ["generator", "generator"]


def test_query_model_setting_builds_a_second_client(settings, sqlite_adapter: SQLiteAdapter) -> None:  # type: ignore[no-untyped-def]
    split = settings.model_copy(update={"llm_provider": "ollama", "llm_model": "big", "llm_query_model": "coder"})
    engine = TalkTable(split, adapters={"sqlite": sqlite_adapter}, embedder=HashEmbedder())
    assert (engine.router._llm.model, engine.generator._llm.model, engine.explainer._llm.model) == \
        ("big", "coder", "big")
    single = settings.model_copy(update={"llm_provider": "ollama", "llm_model": "big", "llm_query_model": None})
    engine = TalkTable(single, adapters={"sqlite": sqlite_adapter}, embedder=HashEmbedder())
    assert engine.generator._llm.model == "big"


# =============================================================================
# Building blocks
# =============================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"sql": "SELECT 1"}', "SELECT 1"),
        ('```json\n{"sql": "SELECT 1"}\n```', "SELECT 1"),
        ('Here you go: {"sql": "SELECT 1"} Hope it helps!', "SELECT 1"),
        ('{"sql": "SELECT a\n  FROM b"}', "SELECT a\n  FROM b"),  # raw newline inside the string
        ('<think>hmm</think>{"query": "SELECT 2"}', "SELECT 2"),
    ],
)
def test_parse_sql_replies(text: str, expected: str) -> None:
    assert parse_reply(text, "sql") == expected


def test_parse_mongo_replies() -> None:
    reply = '{"query": {"collection": "reviews", "pipeline": [{"$count": "n"}]}}'
    assert parse_reply(reply, "mongo") == {"collection": "reviews", "pipeline": [{"$count": "n"}]}


@pytest.mark.parametrize("text", ["", "no json here", '{"sql": ""}', '{"answer": 42}', "[1, 2]"])
def test_unusable_replies_raise(text: str) -> None:
    with pytest.raises(LLMError):
        parse_reply(text, "sql")


def test_parse_json_object_rejects_arrays() -> None:
    with pytest.raises(LLMError):
        parse_json_object("[1, 2, 3]")


def _context(adapter, name: str) -> QueryContext:  # type: ignore[no-untyped-def]
    description = load_descriptions().databases[name]
    return QueryContext(database=name, kind=adapter.kind, title=description.title,
                        tables=describe_tables(adapter.get_schema(), description),
                        few_shot=description.few_shot[:3])  # the pipeline shows the 3 most similar


def test_sql_prompt_contents(sqlite_adapter: SQLiteAdapter) -> None:
    system, user = build_messages("Q?", _context(sqlite_adapter, "sqlite"), today=date(2026, 10, 12))
    prompt = system["content"]
    assert user == {"role": "user", "content": "Q?"}
    assert "SQLite" in prompt and "strftime" in prompt and "SUBSTR" in prompt
    assert "Table books, 1,000 rows" in prompt and "author_id INTEGER (references authors.author_id)" in prompt
    assert "Current list price in US dollars" in prompt  # column descriptions from the YAML
    assert prompt.count("Question: ") == 3 and '"sql": "SELECT c.name AS category' in prompt
    assert "Today's date is 2026-10-12" in prompt and "LIMIT of at most 200" in prompt


def test_prompt_is_stable_across_questions(sqlite_adapter: SQLiteAdapter) -> None:
    """Same system prompt whatever the question or table order, so local models can
    reuse their prompt cache."""
    context = _context(sqlite_adapter, "sqlite")
    shuffled = QueryContext(**{**context.__dict__, "tables": list(reversed(context.tables))})
    first = build_messages("a", context, today=date(2026, 1, 1))[0]
    second = build_messages("b", shuffled, today=date(2026, 1, 1))[0]
    assert first == second


def test_postgres_prompt_explains_date_arithmetic(postgres_adapter: PostgresAdapter) -> None:
    """DATE - DATE is already a number of days in PostgreSQL; EXTRACT on it fails. Without
    this note the query model kept writing EXTRACT(DAY FROM a - b) for delivery times."""
    prompt = build_messages("Q?", _context(postgres_adapter, "postgres"))[0]["content"]
    assert "This is PostgreSQL." in prompt
    assert "Subtracting one DATE from another gives the number of days as an integer" in prompt
    assert "AVG(delivered_date - shipped_date)" in prompt and "Don't put EXTRACT around it" in prompt


def test_mongo_prompt_contents(mongo_adapter: MongoAdapter) -> None:
    prompt = build_messages("Q?", _context(mongo_adapter, "mongo"))[0]["content"]
    assert "Collection reviews, 6,000 documents" in prompt
    assert "$unwind" in prompt and "Never use $out" in prompt and '{"$date": "2025-01-01T00:00:00Z"}' in prompt
    assert '"collection": "clickstream"' in prompt  # few-shot answers


def test_retrieve_adds_foreign_key_targets(sqlite_adapter: SQLiteAdapter) -> None:
    description = load_descriptions().databases["sqlite"]
    index = SchemaIndex(HashEmbedder())
    index.add_database("sqlite", describe_tables(sqlite_adapter.get_schema(), description), description)
    selection = index.retrieve("Which books have more than 500 pages?", "sqlite", k=1)
    assert selection.names[0] == "books" and set(selection.names) == {"books", "authors", "categories"}
    assert selection.roles == {"books": "match", "authors": "lookup", "categories": "lookup"}
    assert selection.compact == {"authors", "categories"}
    assert selection.joins == ["books.author_id = authors.author_id", "books.category_id = categories.category_id"]


@pytest.mark.parametrize(
    ("question", "database", "terms"),
    [
        ("What was the total revenue in 2024?", "postgres", ["revenue"]),
        ("Which books are BESTSELLERS?", "postgres", ["bestseller"]),  # case and plural
        ("Top-selling books last year", "postgres", ["bestseller"]),  # a synonym
        ("Which customers signed up in 2024?", "postgres", ["new customer"]),
        ("What is the conversion rate on mobile?", "mongo", ["conversion"]),
        ("How many negative reviews does each book have?", "mongo", ["negative review"]),
        ("What was the revenue from reviews?", "mongo", []),  # revenue is a sales term only
        ("What are the 5 most expensive books?", "sqlite", []),
        ("Show the bounced sessions and the abandoned carts", "mongo", ["cart abandonment", "bounce"]),
    ],
)
def test_glossary_matching(question: str, database: str, terms: list[str]) -> None:
    found = [entry.term for entry in load_descriptions().glossary_for(question, database)]
    assert sorted(found) == sorted(terms)


def test_glossary_definitions_reach_the_prompt(sqlite_adapter: SQLiteAdapter, postgres_adapter: PostgresAdapter) -> None:
    llm = scripted(router='{"databases": ["postgres"]}',
                   generator=[sql_reply("SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders "
                                        "WHERE status NOT IN ('cancelled', 'returned') LIMIT 1")])
    engine_with(llm, sqlite_adapter, postgres=postgres_adapter).ask("What was the total revenue?")
    generator_messages = [msgs for step, msgs in llm.calls if step == "generator"][0]
    user = generator_messages[-1]["content"]
    assert user.startswith("What was the total revenue?\n\nUse these definitions:\n- revenue: Money from orders")
    assert "Use these definitions" not in generator_messages[0]["content"]  # system prompt unchanged
    router_prompt = [msgs for step, msgs in llm.calls if step == "router"][0][0]["content"]
    assert "Business terms: revenue, bestseller, average order value" in router_prompt


def test_glossary_definitions_use_real_columns() -> None:
    """Every table.column a definition mentions must exist, or the LLM would be misled."""
    descriptions = load_descriptions()
    for entry in descriptions.glossary:
        assert entry.databases, entry.term
        for database in entry.databases:
            objects = descriptions.databases[database].objects
            for table, column in re.findall(r"\b([a-z_]+)\.([a-z_]+)\b", entry.meaning):
                assert table in objects and column in objects[table].fields, (entry.term, f"{table}.{column}")


def test_ungrounded_numbers() -> None:
    result = QueryResult(query="q", columns=["title", "price", "month"], row_count=2, truncated=False,
                         elapsed_ms=1, rows=[{"title": "A", "price": 11981.1, "month": "2025-01-01"},
                                             {"title": "B", "price": 4.27, "month": "2025-02-01"}])
    question = "Top 5 books in 2025?"
    assert ungrounded_numbers("A costs about 11,981 and B 4.3 in 2025.", result, question) == []
    assert ungrounded_numbers("A costs about 12,000 and B about 4.", result, question) == []
    assert ungrounded_numbers("Top 5 shown; 2 rows.", result, question) == []
    # List markers and ordinals are ranks, not data.
    assert ungrounded_numbers("The books are:\n1. A at 11981.1\n2. B at 4.27\n3) none", result, question) == []
    assert ungrounded_numbers("B is 2nd, and the 3rd place is empty.", result, question) == []
    assert ungrounded_numbers("A costs 12,500 and the total is 11985.37.", result, question) == ["12,500", "11985.37"]


def _run_few_shot(adapter, name: str) -> None:  # type: ignore[no-untyped-def]
    for example in load_descriptions().databases[name].few_shot:
        query = json.loads(example.query) if name == "mongo" else example.query
        assert adapter.run(query).rows, f"{name} few-shot returned no rows: {example.question}"


def test_few_shot_examples_run_on_sqlite(sqlite_adapter: SQLiteAdapter) -> None:
    _run_few_shot(sqlite_adapter, "sqlite")


def test_few_shot_examples_run_on_postgres(postgres_adapter: PostgresAdapter) -> None:
    _run_few_shot(postgres_adapter, "postgres")


def test_few_shot_examples_run_on_mongo(mongo_adapter: MongoAdapter) -> None:
    _run_few_shot(mongo_adapter, "mongo")


# =============================================================================
# Live: the spec's Phase 3 "done when" questions, through the real LLM
# =============================================================================


@pytest.fixture(scope="module")
def live_engine(sqlite_adapter: SQLiteAdapter, postgres_adapter: PostgresAdapter, mongo_adapter: MongoAdapter):  # type: ignore[no-untyped-def]
    engine = TalkTable(adapters={"sqlite": sqlite_adapter, "postgres": postgres_adapter, "mongo": mongo_adapter})
    yield engine


def _ints(rows: list[dict]) -> list[int]:
    return [v for row in rows for v in row.values() if isinstance(v, int)]


@pytest.mark.live
def test_live_most_expensive_books(live_engine: TalkTable) -> None:
    result = live_engine.ask("What are the 5 most expensive books?")
    assert result.error is None and result.databases_used == ["sqlite"]
    assert len(result.rows) == 5
    assert 110.95 in result.rows[0].values()
    assert "110.95" in result.answer or "111" in result.answer


@pytest.mark.live
def test_live_orders_per_month_2025(live_engine: TalkTable) -> None:
    result = live_engine.ask("How many orders were placed each month in 2025?")
    assert result.error is None and result.databases_used == ["postgres"]
    assert len(result.rows) == 12
    expected = [170, 137, 165, 192, 198, 186, 220, 260, 206, 252, 334, 397]
    assert [n for n in _ints(result.rows) if n > 12] == expected


@pytest.mark.live
def test_live_most_one_star_reviews(live_engine: TalkTable) -> None:
    result = live_engine.ask("Which 5 books have the most 1-star reviews?")
    assert result.error is None and result.databases_used == ["mongo"]
    pairs = [tuple(_ints([row])) for row in result.rows]
    assert pairs[:2] == [(158, 17), (399, 16)]
    assert set(pairs[2:4]) == {(522, 10), (810, 10)}
    assert pairs[4] in {(180, 6), (648, 6)}  # tie for fifth place


@pytest.mark.live
def test_live_fastest_carrier(live_engine: TalkTable) -> None:
    """Date arithmetic in PostgreSQL (delivered_date - shipped_date), a sidebar example."""
    truth = live_engine.adapters["postgres"].run(
        "SELECT c.name, AVG(s.delivered_date - s.shipped_date)::float AS days FROM shipments s "
        "JOIN carriers c ON c.carrier_id = s.carrier_id WHERE s.delivered_date IS NOT NULL "
        "GROUP BY c.carrier_id, c.name ORDER BY days LIMIT 1").rows[0]
    result = live_engine.ask("Which carrier delivers the fastest?")
    assert result.error is None and result.databases_used == ["postgres"]
    first = result.rows[0]
    assert truth["name"] in first.values()
    days = [float(v) for v in first.values() if isinstance(v, (int, float)) or hasattr(v, "is_finite")]
    assert any(abs(d - truth["days"]) < 0.01 for d in days)
