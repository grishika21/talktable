"""Web UI tests: the Gradio app's public API (= its MCP tools), rendering, and the chat
handler, against the real databases with a scripted fake LLM. Nothing is launched."""

from __future__ import annotations

import json
from collections.abc import Iterator

import gradio as gr
import pytest

from talktable import tools, webui
from talktable.adapters import MongoAdapter, PostgresAdapter, SQLiteAdapter
from talktable.models import AskResult, DatabaseStatus
from talktable.pipeline import TalkTable
from data.seed.seed_sqlite import TABLES as SQLITE_TABLES
from tests.fakes import HashEmbedder, scripted, sql_reply

TOP_5_SQL = "SELECT title, price FROM books ORDER BY price DESC LIMIT 5"
TITLES = {"sqlite": "Book catalog", "postgres": "Sales", "mongo": "Customer engagement"}


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, sqlite_adapter: SQLiteAdapter, postgres_adapter: PostgresAdapter,
           mongo_adapter: MongoAdapter) -> Iterator[TalkTable]:
    llm = scripted(router='{"databases": ["sqlite"]}', generator=[sql_reply(TOP_5_SQL)])
    engine = TalkTable(adapters={"sqlite": sqlite_adapter, "postgres": postgres_adapter, "mongo": mongo_adapter},
                     llm=llm, embedder=HashEmbedder())
    monkeypatch.setattr(tools, "get_talktable", lambda: engine)
    yield engine


def result(**overrides: object) -> AskResult:
    values: dict = dict(question="q", databases_used=["sqlite"], queries=[TOP_5_SQL],
                        rows=[{"title": "A", "price": 1.5}], columns=["title", "price"],
                        answer="The top book costs $1.50.", retries=0, latency_ms=2345)
    values.update(overrides)
    return AskResult(**values)


# =============================================================================
# The app
# =============================================================================


def test_only_the_five_tools_are_public(engine: TalkTable) -> None:
    """Gradio makes every public endpoint an MCP tool: UI events must stay private."""
    info = webui.build_app().get_api_info()
    assert set(info["named_endpoints"]) == {"/list_databases", "/get_schema", "/ask", "/run_sql", "/run_mongo"}
    assert not info.get("unnamed_endpoints")
    params = {name: [p["parameter_name"] for p in ep["parameters"]] for name, ep in info["named_endpoints"].items()}
    assert params["/run_sql"] == ["database", "query"] and params["/run_mongo"] == ["collection", "pipeline"]


def test_mcp_tool_docstrings() -> None:
    """Gradio builds tool and parameter descriptions from these docstrings."""
    for fn in webui.MCP_TOOLS:
        description, parameters, returns = gr.utils.get_function_description(fn)
        assert len(description) > 80 and returns, fn.__name__
        assert all(parameters.values()), fn.__name__  # every parameter is documented
    run_mongo_doc = gr.utils.get_function_description(webui.run_mongo)[0]
    assert all(stage in run_mongo_doc for stage in ("$match", "$group", "$lookup", "$facet"))


def test_theme_survives_gradios_builtin_theme_check() -> None:
    # launch() compares the theme with Gradio's built-in ones; plain-string fonts used to crash it.
    theme = webui.build_theme()
    assert theme.to_dict() != gr.themes.Default().to_dict()
    assert theme.to_dict()["theme"]["button_primary_background_fill_dark"] == webui.ACCENT


# =============================================================================
# MCP tool functions
# =============================================================================


def test_mcp_tools_return_json(engine: TalkTable) -> None:
    databases = json.loads(webui.list_databases())
    assert [d["name"] for d in databases] == ["sqlite", "postgres", "mongo"]
    rows = json.loads(webui.run_sql("sqlite", TOP_5_SQL))["rows"]
    assert rows[0]["price"] == 110.95
    docs = json.loads(webui.run_mongo("reviews", '[{"$match": {"rating": 1}}, {"$count": "n"}]'))
    assert docs["rows"][0]["n"] > 0
    answer = json.loads(webui.ask("What are the 5 most expensive books?"))
    assert answer["error"] is None and answer["rows"][0]["price"] == 110.95
    assert {t["name"] for t in json.loads(webui.get_schema("sqlite"))} == set(SQLITE_TABLES)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: webui.run_sql("sqlite", "DROP TABLE books"), "Blocked by the read-only guardrails"),
        (lambda: webui.run_mongo("reviews", '[{"$out": "x"}]'), "$out is not allowed"),
        (lambda: webui.get_schema("oracle"), "Unknown database 'oracle'"),
    ],
)
def test_mcp_tool_errors_are_friendly(engine: TalkTable, call, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(gr.Error) as caught:
        call()
    assert message in str(caught.value.message)


# =============================================================================
# Rendering
# =============================================================================


def test_render_answer() -> None:
    text = webui.render_answer(result(), TITLES)
    assert '<span class="tt-badge tt-sqlite">SQLite · Book catalog</span>' in text
    assert "The top book costs $1.50." in text
    assert "```sql\nSELECT\n  title,\n  price\nFROM books\nORDER BY\n  price DESC\nLIMIT 5\n```" in text
    assert text.endswith('<div class="tt-foot">1 row · no retries · 2.3 s</div>')


def test_render_answer_with_retries_and_mongo() -> None:
    mongo_query = '{"collection": "reviews", "pipeline": [{"$match": {"rating": 1}}]}'
    text = webui.render_answer(result(databases_used=["mongo"], queries=[mongo_query], retries=2,
                                      rows=[{"n": 1}, {"n": 2}]), TITLES)
    assert "MongoDB · Customer engagement" in text
    assert '```json\n{\n  "collection": "reviews",' in text  # pretty-printed
    assert "2 rows · self-corrected 2 times" in text


def test_render_answer_escapes_html() -> None:
    text = webui.render_answer(result(answer="<img src=x onerror=alert(1)> & co"), TITLES)
    assert "<img" not in text and "&lt;img src=x onerror=alert(1)&gt; &amp; co" in text


def test_render_error() -> None:
    text = webui.render_answer(result(error="no", answer="I couldn't get a working query.", rows=[], columns=[]),
                               TITLES)
    assert text.startswith('<span class="tt-badge tt-error">No answer</span>')
    assert "I couldn't get a working query." in text and "row" not in text.split("tt-foot")[1]


def test_render_small_talk_reply() -> None:
    text = webui.render_answer(result(databases_used=[], queries=[], rows=[], columns=[], answer="Hi! I'm TalkTable."),
                               TITLES)
    assert text == "Hi! I'm TalkTable."  # no badges, query or run details


def test_render_status() -> None:
    html = webui.render_status([
        DatabaseStatus(name="sqlite", title="Book catalog", kind="sql", online=True, tables=["authors", "books"]),
        DatabaseStatus(name="mongo", title="Customer engagement", kind="mongo", online=False,
                       detail="not configured (MONGO_URL is not set)"),
    ])
    assert 'class="tt-db tt-online"' in html and "authors · books" in html
    assert 'class="tt-db tt-offline"' in html and "offline: not configured (MONGO_URL is not set)" in html


def test_results_frame_flattens_nested_values() -> None:
    frame = webui.results_frame(result(rows=[{"_id": 1, "events": [{"type": "view"}], "text": "x" * 500}],
                                       columns=["_id", "events", "text"]))
    assert list(frame.columns) == ["_id", "events", "text"]
    assert frame.iloc[0]["events"] == '[{"type": "view"}]'
    assert len(frame.iloc[0]["text"]) == 300 and frame.iloc[0]["text"].endswith("…")


# =============================================================================
# The chat handler
# =============================================================================


def test_answer_stream(engine: TalkTable) -> None:
    updates = list(webui.answer_stream("What are the 5 most expensive books?", []))
    history, box, table, _ = updates[0]
    assert box == "" and history[0] == {"role": "user", "content": "What are the 5 most expensive books?"}
    assert "Starting" in history[1]["content"]
    history, _, table, turns = updates[-1]
    assert len(turns) == 1 and turns[0]["question"] == "What are the 5 most expensive books?"
    assert len(history) == 2 and "110.95" in history[1]["content"] and "```sql" in history[1]["content"]
    assert isinstance(table, gr.Dataframe) and table.visible is True
    assert table.label == "Results · 5 rows"


def test_answer_stream_keeps_earlier_turns(engine: TalkTable) -> None:
    earlier = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    history = list(webui.answer_stream("Most expensive books?", earlier))[-1][0]
    assert history[:2] == earlier and len(history) == 4


def test_answer_stream_ignores_blank_questions(engine: TalkTable) -> None:
    assert len(list(webui.answer_stream("   ", []))) == 1


def test_answer_stream_shows_problems_in_the_chat(engine: TalkTable) -> None:
    history, _, table, turns = list(webui.answer_stream("x" * 600, []))[-1]
    assert "No answer" in history[-1]["content"] and "under 500 characters" in history[-1]["content"]
    assert isinstance(table, gr.Dataframe) and table.visible is False
    assert not isinstance(turns, list)  # failed answers aren't remembered (gr.skip())


# =============================================================================
# The real thing: the launched app's MCP endpoint over HTTP
# =============================================================================


@pytest.mark.anyio
async def test_mcp_over_http(engine: TalkTable) -> None:
    import socket

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = webui.build_app()
    app.launch(mcp_server=True, server_port=port, prevent_thread_lock=True, quiet=True, ssr_mode=False)
    try:
        url = f"http://127.0.0.1:{port}/gradio_api/mcp/"
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            assert {t.name for t in (await session.list_tools()).tools} == {
                "list_databases", "get_schema", "ask", "run_sql", "run_mongo"}
            ok = await session.call_tool("run_sql", {"database": "sqlite", "query": TOP_5_SQL})
            assert not ok.isError and json.loads(ok.content[0].text)["rows"][0]["price"] == 110.95
            blocked = await session.call_tool("run_sql", {"database": "sqlite", "query": "DELETE FROM books"})
            assert blocked.isError and "Only SELECT queries are allowed" in blocked.content[0].text
            answer = await session.call_tool("ask", {"question": "What are the 5 most expensive books?"})
            assert not answer.isError and "110.95" in json.loads(answer.content[0].text)["answer"]
    finally:
        app.close()
