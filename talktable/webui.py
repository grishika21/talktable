"""The Gradio web app: a chat UI over the TalkTable engine, plus the five MCP tools.

`build_app()` returns the Blocks; `app.py` launches it with `mcp_server=True`, so the
same page is both a chat demo and a remote MCP server (at /gradio_api/mcp).

Every UI event is registered with `api_visibility="private"`. Gradio turns each
public endpoint into an MCP tool, and only the five TalkTable tools should be public.
"""

from __future__ import annotations

import html
import inspect
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import gradio as gr
import pandas as pd

from talktable import tools
from talktable.config import get_settings
from talktable.conversation import MAX_TURNS, Turn
from talktable.guardrails import format_sql
from talktable.models import AskResult, DatabaseStatus

logger = logging.getLogger(__name__)

ACCENT = "#00D4AA"
GITHUB_URL = "https://github.com/grishika21/talktable"
LINKEDIN_URL = "https://www.linkedin.com/in/rishika-guggilam"

ENGINE_NAMES = {"sqlite": "SQLite", "postgres": "PostgreSQL", "mongo": "MongoDB"}

# Sidebar buttons, per database. The first one of each is a Phase 3 demo question.
EXAMPLES: dict[str, list[str]] = {
    "sqlite": [
        "What are the 5 most expensive books?",
        "Which country has the most authors?",
        "Which publisher has the most audiobooks?",
    ],
    "postgres": [
        "How many orders were placed each month in 2025?",
        "Who are the top 5 customers by total spend?",
        "Which carrier delivers the fastest?",
    ],
    "mongo": [
        "Which 5 books have the most 1-star reviews?",
        "How many reviews were posted in 2025?",
        "Which books are on the most wishlists?",
    ],
}

_MAX_CELL_CHARS = 300
_POLL_SECONDS = 1.0

# --- theme ---------------------------------------------------------------------

_TEAL = gr.themes.Color(
    c50="#e6fbf6", c100="#b3f3e3", c200="#80ead0", c300="#4de1bd", c400="#1ad9aa", c500="#00d4aa",
    c600="#00aa88", c700="#008066", c800="#005544", c900="#002b22", c950="#00150f", name="talktable-teal",
)

# GitHub's dark palette with the teal accent. Light and dark values are the same, and
# FORCE_DARK_JS switches Gradio to dark mode, so the app always looks the same.
_PALETTE = {
    "body_background_fill": "#0d1117",
    "background_fill_primary": "#0d1117",
    "background_fill_secondary": "#161b22",
    "block_background_fill": "#161b22",
    "block_border_color": "#30363d",
    "border_color_primary": "#30363d",
    "body_text_color": "#e6edf3",
    "body_text_color_subdued": "#8b949e",
    "block_label_text_color": "#8b949e",
    "block_title_text_color": "#e6edf3",
    "input_background_fill": "#0d1117",
    "input_border_color": "#30363d",
    "input_border_color_focus": ACCENT,
    "color_accent": ACCENT,
    "color_accent_soft": "#00d4aa26",
    "link_text_color": ACCENT,
    "link_text_color_hover": "#4de1bd",
    "button_primary_background_fill": ACCENT,
    "button_primary_background_fill_hover": "#4de1bd",
    "button_primary_text_color": "#0d1117",
    "button_secondary_background_fill": "#21262d",
    "button_secondary_background_fill_hover": "#30363d",
    "button_secondary_text_color": "#e6edf3",
    "button_secondary_border_color": "#30363d",
}


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=_TEAL,
        secondary_hue=_TEAL,
        neutral_hue="slate",
        radius_size="md",
        # System fonts only: nothing is fetched from Google Fonts.
        font=[gr.themes.Font(name) for name in ("Inter", "ui-sans-serif", "system-ui", "sans-serif")],
        font_mono=[gr.themes.Font(name) for name in ("JetBrains Mono", "ui-monospace", "Consolas", "monospace")],
    )
    settable = set(inspect.signature(theme.set).parameters)  # not every variable has a _dark twin
    dark = {f"{key}_dark": value for key, value in _PALETTE.items() if f"{key}_dark" in settable}
    return theme.set(**_PALETTE, **dark)


FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

CSS = """
html, body { scrollbar-color: #30363d #0d1117; }
.tt-brand { display: flex; align-items: center; gap: .6rem; margin-bottom: .2rem; }
.tt-logo { width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
           background: #00d4aa1f; border: 1px solid #00d4aa66; color: #00D4AA; font-weight: 700; }
.tt-brand h1 { font-size: 1.35rem; margin: 0; color: #e6edf3; }
.tt-brand h1 span { color: #00D4AA; }
.tt-pitch { color: #8b949e; font-size: .9rem; margin: 0 0 .4rem; }
.tt-section { color: #8b949e; font-size: .72rem; font-weight: 600; letter-spacing: .08em;
              text-transform: uppercase; margin: .9rem 0 .35rem; }
.tt-db { display: flex; gap: .6rem; padding: .5rem .65rem; margin-bottom: .4rem;
         border: 1px solid #30363d; border-radius: 8px; background: #0d1117; }
.tt-dot { flex: none; width: 9px; height: 9px; margin-top: .4rem; border-radius: 50%;
          background: #3fb950; box-shadow: 0 0 6px #3fb950aa; }
.tt-db.tt-offline .tt-dot { background: #f85149; box-shadow: none; }
.tt-db b { color: #e6edf3; }
.tt-db small { color: #8b949e; }
.tt-engine { color: #8b949e; font-size: .78rem; }
.tt-examples button { justify-content: flex-start !important; text-align: left !important;
                      font-weight: 400 !important; font-size: .85rem !important; }
.tt-badge { display: inline-block; padding: 1px 10px; margin: 0 6px 6px 0; border-radius: 999px;
            border: 1px solid; font-size: .76rem; font-weight: 600; }
.tt-badge.tt-sqlite { color: #79c0ff; border-color: #79c0ff66; background: #79c0ff14; }
.tt-badge.tt-postgres { color: #d2a8ff; border-color: #d2a8ff66; background: #d2a8ff14; }
.tt-badge.tt-mongo { color: #56d364; border-color: #56d36466; background: #56d36414; }
.tt-badge.tt-error { color: #ff7b72; border-color: #ff7b7266; background: #ff7b7214; }
.tt-badge.tt-cached { color: #00D4AA; border-color: #00d4aa66; background: #00d4aa14; }
.tt-foot { color: #8b949e; font-size: .8rem; margin-top: .3rem; }
.tt-progress { color: #8b949e; }
.tt-interp { color: #8b949e; font-size: .82rem; margin: 0 0 .2rem; }
.tt-interp b { color: #c9d1d9; font-weight: 500; }
.tt-progress b { color: #00D4AA; font-weight: 600; }
.tt-small { color: #8b949e; font-size: .8rem; }
.tt-small code { font-size: .75rem; white-space: normal; overflow-wrap: anywhere; }
.tt-group { color: #8b949e; font-size: .8rem; margin-top: .5rem; }
#tt-sidebar .html-container { padding: 0 !important; }
#tt-sidebar .column { gap: 8px !important; }
#tt-sidebar .tt-examples { gap: 6px !important; }
#tt-sidebar, #tt-sidebar * { scrollbar-color: #30363d transparent; scrollbar-width: thin; }
#tt-chat .placeholder { max-width: 620px; margin: 0 auto; text-align: center; color: #8b949e; }
#tt-chat .placeholder h3 { color: #e6edf3; }
"""

# --- rendering -------------------------------------------------------------------


def _text(value: object) -> str:
    """Escape text for the chat's Markdown/HTML (quotes are fine to keep)."""
    return html.escape(str(value), quote=False)


def render_status(statuses: list[DatabaseStatus]) -> str:
    """The sidebar's database list: a green or red dot, the title, and the tables."""
    cards = []
    for s in statuses:
        if s.online:
            state, detail = "tt-online", " · ".join(s.tables)
        else:
            state, detail = "tt-offline", f"offline: {s.detail or 'not reachable'}"
        cards.append(
            f'<div class="tt-db {state}"><span class="tt-dot"></span><div>'
            f'<b>{_text(s.title)}</b> <span class="tt-engine">{ENGINE_NAMES.get(s.name, s.name)}</span><br>'
            f"<small>{_text(detail)}</small></div></div>"
        )
    return "".join(cards)


def render_badges(databases: list[str], titles: dict[str, str]) -> str:
    return "".join(
        f'<span class="tt-badge tt-{_text(db)}">{ENGINE_NAMES.get(db, _text(db))} · {_text(titles.get(db, db))}</span>'
        for db in databases
    )


def _code_block(query: str, database: str | None) -> str:
    """The query in a fenced code block, pretty-printed (SQL or a Mongo pipeline)."""
    language = "sql"
    try:
        parsed = json.loads(query)
        if isinstance(parsed, dict) and "pipeline" in parsed:
            query, language = json.dumps(parsed, indent=2, ensure_ascii=False), "json"
    except ValueError:
        if database in ("sqlite", "postgres"):
            query = format_sql(query, database)
    fence = "~~~~" if "```" in query else "```"
    return f"{fence}{language}\n{query}\n{fence}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def render_answer(result: AskResult, titles: dict[str, str]) -> str:
    """One assistant message: badges, the answer, the query, and the run details."""
    if not result.queries and not result.error:  # a reply to small talk: just the text
        return _text(result.answer)
    parts = []
    if result.error:
        parts.append('<span class="tt-badge tt-error">No answer</span>' + render_badges(result.databases_used, titles))
    else:
        cached = '<span class="tt-badge tt-cached">⚡ Cached</span>' if result.cached else ""
        parts.append(render_badges(result.databases_used, titles) + cached)
    if result.interpreted_as:  # a follow-up: show how it was understood
        parts.append(f'<div class="tt-interp">Interpreted as: <b>{_text(result.interpreted_as)}</b></div>')
    parts.append(_text(result.answer))
    database = result.databases_used[0] if result.databases_used else None
    parts += [_code_block(q, database) for q in result.queries]
    details = [] if result.error else [_plural(len(result.rows), "row")]
    if result.retries or result.queries:
        details.append("self-corrected " + _plural(result.retries, "time") if result.retries else "no retries")
    if result.cached:
        details.append("saved query, re-run on current data")
    details.append(f"{result.latency_ms / 1000:.1f} s")
    parts.append(f'<div class="tt-foot">{" · ".join(details)}</div>')
    return "\n\n".join(p for p in parts if p)


def render_progress(step: str, seconds: float) -> str:
    return f'<div class="tt-progress">⏳ <b>{_text(step)}</b> … {seconds:.0f} s</div>'


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        return value[:_MAX_CELL_CHARS - 1] + "…"
    return value


def results_frame(result: AskResult) -> pd.DataFrame:
    columns = result.columns or (list(result.rows[0]) if result.rows else [])
    return pd.DataFrame([[_cell(row.get(c)) for c in columns] for row in result.rows], columns=columns)


def results_label(result: AskResult) -> str:
    capped = " (capped)" if len(result.rows) >= get_settings().max_rows else ""
    return f"Results · {_plural(len(result.rows), 'row')}{capped}"


# --- the chat handler ------------------------------------------------------------

_DONE = object()


def answer_stream(question: str, history: list[dict[str, Any]], turns: list[dict[str, Any]] | None = None,
                  use_cache: bool = True) -> Iterator[tuple[Any, Any, Any, Any]]:
    """Gradio handler: append the question, show live progress, then the answer.

    `turns` (per-session state) remembers the last answered questions, so follow-ups
    like "and in 2024?" work. Yields (chat, question box, results table, turns) updates.
    """
    question = (question or "").strip()
    if not question:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    earlier = [Turn.model_validate(turn) for turn in (turns or [])]
    history = list(history or []) + [
        {"role": "user", "content": _text(question)},
        {"role": "assistant", "content": render_progress("Starting", 0)},
    ]
    yield history[:], "", gr.skip(), gr.skip()  # copies: later steps replace the last message

    steps: queue.Queue[object] = queue.Queue()
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["result"] = tools.ask(question, on_step=steps.put, history=earlier, use_cache=use_cache)
        except tools.ToolError as exc:
            outcome["error"] = str(exc)
        finally:
            steps.put(_DONE)

    threading.Thread(target=work, name="talktable-ui-ask", daemon=True).start()
    started, step = time.monotonic(), "Starting"
    while True:
        try:
            message = steps.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            message = None
        if message is _DONE:
            break
        if isinstance(message, str):
            step = message
        history[-1] = {"role": "assistant", "content": render_progress(step, time.monotonic() - started)}
        yield history[:], gr.skip(), gr.skip(), gr.skip()

    result: AskResult | None = outcome.get("result")
    if result is None:
        history[-1] = {"role": "assistant",
                       "content": '<span class="tt-badge tt-error">Error</span>\n\n' + _text(outcome.get("error"))}
        yield history, gr.skip(), gr.skip(), gr.skip()
        return
    titles = {s.name: s.title for s in tools.get_talktable().status()}
    history[-1] = {"role": "assistant", "content": render_answer(result, titles)}
    if result.rows:
        table = gr.Dataframe(value=results_frame(result), label=results_label(result), visible=True)
    else:
        table = gr.Dataframe(visible=False)
    if result.queries and not result.error:  # remember answered questions for follow-ups
        turns = [*(turns or []), Turn.from_result(result).model_dump()][-MAX_TURNS:]
        yield history, gr.skip(), table, turns
    else:
        yield history, gr.skip(), table, gr.skip()


def ask_again_fresh(history: list[dict[str, Any]],
                    turns: list[dict[str, Any]] | None) -> Iterator[tuple[Any, Any, Any, Any]]:
    """Gradio handler: answer the last question again from scratch, ignoring the cache
    (the new answer replaces the cached one)."""
    if not turns:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    yield from answer_stream(turns[-1]["question"], history, turns[:-1], use_cache=False)


def refresh_status() -> str:
    try:
        return render_status(tools.list_databases())
    except tools.ToolError as exc:
        return f'<div class="tt-small">{_text(exc)}</div>'


# --- MCP tools -------------------------------------------------------------------
# Gradio exposes these over MCP, named after the functions. Gradio reads the
# description and parameter docs from the docstrings ("Args:" section) and sends the
# return value as text, so each returns JSON.


def _as_json(value: Any) -> str:
    if isinstance(value, list):
        value = [v.model_dump() if hasattr(v, "model_dump") else v for v in value]
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value, ensure_ascii=False, default=str)


def _tool(fn: Callable[[], Any]) -> str:
    try:
        return _as_json(fn())
    except tools.ToolError as exc:
        raise gr.Error(str(exc), print_exception=False) from None


def list_databases() -> str:
    """List TalkTable's databases: name, kind (sql or mongo), whether it is online, what data it holds, and its tables or collections. Call this first to see what is available.

    Returns:
        A JSON array with one object per database.
    """
    return _tool(tools.list_databases)


def get_schema(database: str) -> str:
    """Get the tables (SQL) or collections (MongoDB) of one database: every column or field with its type and meaning, the row count, primary key and foreign keys. Use it before writing a query for run_sql or run_mongo.

    Args:
        database: The database: sqlite, postgres or mongo.

    Returns:
        A JSON array with one object per table or collection.
    """
    return _tool(lambda: tools.get_schema(database))


def ask(question: str) -> str:
    """Answer a question about the online bookstore's data (books, authors, customers, orders, reviews, browsing sessions) in plain English. TalkTable picks the right database, writes the SQL or MongoDB query, checks it against read-only guardrails, runs it (fixing it up to 2 times if it fails), and explains the result.

    Args:
        question: A question in plain English, e.g. "Which 5 books have the most 1-star reviews?"

    Returns:
        JSON with the answer, the databases and queries used, the result rows, retries and latency. If "error" is set, no answer was found and "answer" says why.
    """
    return _tool(lambda: tools.ask(question))


def run_sql(database: str, query: str) -> str:
    """Run your own read-only SQL query on the sqlite (book catalog, SQLite dialect) or postgres (sales, PostgreSQL dialect) database. Only a single SELECT (or WITH ... SELECT) statement is allowed; anything that could change data is rejected with an explanation. At most 200 rows are returned.

    Args:
        database: sqlite or postgres.
        query: One SELECT statement.

    Returns:
        JSON with the query as executed, the columns, the rows, and whether the rows were capped.
    """
    return _tool(lambda: tools.run_sql(database, query))


def run_mongo(collection: str, pipeline: str) -> str:
    """Run your own read-only MongoDB aggregation pipeline on a collection of the mongo database (reviews or clickstream). Allowed stages: $match, $project, $group, $sort, $limit, $skip, $unwind, $count, $lookup, $addFields, $facet. Write dates as {"$date": "2025-01-01T00:00:00Z"}. Stages that write data or run code ($out, $merge, $function, $where) are rejected. At most 200 documents are returned.

    Args:
        collection: The collection, e.g. reviews.
        pipeline: The aggregation pipeline as a JSON array of stages, e.g. [{"$match": {"rating": 1}}, {"$count": "n"}]

    Returns:
        JSON with the pipeline as executed, the columns, the documents, and whether they were capped.
    """
    return _tool(lambda: tools.run_mongo(collection, pipeline))


MCP_TOOLS = (list_databases, get_schema, ask, run_sql, run_mongo)

# --- layout ----------------------------------------------------------------------


def mcp_url() -> str:
    """Where MCP clients connect: the Space's public URL, or the local server."""
    host = os.environ.get("SPACE_HOST", "").split(",")[0].strip()
    return f"https://{host}/gradio_api/mcp" if host else "http://127.0.0.1:7860/gradio_api/mcp"


def _model_label() -> str:
    settings = get_settings()
    label = f"{settings.llm_provider} · {settings.llm_model}"
    if settings.llm_query_model and settings.llm_query_model != settings.llm_model:
        label += f" (queries: {settings.llm_query_model})"
    return label


def build_app() -> gr.Blocks:
    """The Blocks app. Theme and CSS are passed to `launch()` (Gradio 6), see app.py."""
    statuses = tools.list_databases()
    titles = {s.name: s.title for s in statuses}

    with gr.Blocks(title="TalkTable: ask your databases in plain English", fill_height=True) as app:
        with gr.Sidebar(open=True, width=330, elem_id="tt-sidebar"):
            gr.HTML(
                '<div class="tt-brand"><div class="tt-logo">T</div><h1>Talk<span>Table</span></h1></div>'
                '<p class="tt-pitch">Ask questions in plain English across SQLite, PostgreSQL and MongoDB. '
                "Read-only, guarded, and explained.</p>"
                '<div class="tt-section">Databases</div>'
            )
            status_html = gr.HTML(render_status(statuses))
            refresh = gr.Button("Check connections", size="sm", variant="secondary")
            gr.HTML('<div class="tt-section">Try a question</div>')
            example_buttons: list[tuple[gr.Button, str]] = []
            with gr.Column(elem_classes="tt-examples"):
                for db, questions in EXAMPLES.items():
                    gr.HTML(f'<div class="tt-group">{ENGINE_NAMES[db]} · {_text(titles.get(db, db))}</div>')
                    for q in questions:
                        example_buttons.append((gr.Button(q, size="sm", variant="secondary"), q))
            gr.HTML(
                '<div class="tt-section">About</div>'
                f'<div class="tt-small">LLM: <code>{_text(_model_label())}</code><br>'
                f"MCP endpoint: <code>{_text(mcp_url())}</code><br>"
                f'<a href="{GITHUB_URL}" target="_blank">GitHub</a> · '
                f'<a href="{LINKEDIN_URL}" target="_blank">Rishika Guggilam</a></div>'
            )

        with gr.Column():
            chat = gr.Chatbot(
                elem_id="tt-chat",
                show_label=False,
                height=560,
                latex_delimiters=[],  # answers contain prices like $91.99
                buttons=["copy"],
                placeholder=(
                    "### Ask your bookstore data anything\n"
                    "Books and authors live in **SQLite**, sales in **PostgreSQL**, and reviews and "
                    "browsing sessions in **MongoDB**. TalkTable picks the right one, writes the query, "
                    "checks it is read-only, runs it, and explains the result."
                ),
            )
            with gr.Row(equal_height=True):
                question = gr.Textbox(
                    placeholder="e.g. Which 5 books have the most 1-star reviews?",
                    show_label=False, scale=8, max_lines=3, autofocus=True, container=False,
                )
                send = gr.Button("Ask", variant="primary", scale=1, min_width=90)
                new_chat = gr.Button("New chat", variant="secondary", scale=1, min_width=100)
            again = gr.Button("↻ Ask the last question again (skip the cache)", size="sm", variant="secondary")
            table = gr.Dataframe(visible=False, interactive=False, wrap=True, max_height=380,
                                 show_search="search", buttons=["copy", "fullscreen"])

        turns = gr.State([])  # this browser session's answered questions, for follow-ups
        inputs, outputs = [question, chat, turns], [chat, question, table, turns]
        private: dict[str, Any] = {"api_visibility": "private"}
        question.submit(answer_stream, inputs, outputs, concurrency_limit=4, **private)
        send.click(answer_stream, inputs, outputs, concurrency_limit=4, **private)
        for button, q in example_buttons:
            button.click(lambda q=q: q, None, question, queue=False, **private).then(
                answer_stream, inputs, outputs, concurrency_limit=4, **private)
        again.click(ask_again_fresh, [chat, turns], outputs, concurrency_limit=4, **private)
        refresh.click(refresh_status, None, status_html, **private)
        new_chat.click(lambda: ([], gr.Dataframe(visible=False), []), None, [chat, table, turns], queue=False,
                       **private)

        # The five MCP tools. Only ask() is slow, so the rest skip the queue.
        for fn in MCP_TOOLS:
            gr.api(fn, api_name=fn.__name__, queue=fn is ask, concurrency_limit=4 if fn is ask else None)
    return app
