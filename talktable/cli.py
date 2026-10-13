"""Command-line interface.

    python -m talktable.cli "What are the 5 most expensive books?"
    python -m talktable.cli --db postgres "How many orders were placed each month in 2025?"
    python -m talktable.cli --json "Which 5 books have the most 1-star reviews?"
    python -m talktable.cli --status
    python -m talktable.cli --chat          # a conversation: follow-ups like "and in 2024?" work
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from talktable.adapters import ALL_DATABASES
from talktable.config import get_settings
from talktable.conversation import MAX_TURNS, Turn
from talktable.models import AskResult
from talktable.pipeline import TalkTable

_MAX_ROWS_PRINTED = 15
_MAX_CELL = 40


def _table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= _MAX_CELL else text[:_MAX_CELL - 1] + "…"

    shown = rows[:_MAX_ROWS_PRINTED]
    grid = [[cell(c) for c in columns]] + [[cell(row.get(c)) for c in columns] for row in shown]
    widths = [max(len(r[i]) for r in grid) for i in range(len(columns))]
    lines = ["  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in grid]
    lines.insert(1, "  " + "  ".join("-" * w for w in widths))
    if len(rows) > len(shown):
        lines.append(f"  ... {len(rows) - len(shown)} more row(s)")
    return "\n".join(lines)


def _print_result(result: AskResult) -> None:
    print(f"\nQuestion : {result.question}")
    if result.interpreted_as:
        print(f"Meaning  : {result.interpreted_as}")
    print(f"Database : {', '.join(result.databases_used) or '-'}")
    for query in result.queries:
        print(f"Query    : {query}")
    print(f"\nAnswer   : {result.answer}\n")
    if result.rows:
        print(f"Rows ({len(result.rows)}):")
        print(_table(result.columns or list(result.rows[0]), result.rows))
    cached = "   (cached query, re-run on current data)" if result.cached else ""
    print(f"\nRetries: {result.retries}   Latency: {result.latency_ms / 1000:.1f}s{cached}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m talktable.cli", description="Ask TalkTable a question.")
    parser.add_argument("question", nargs="*", help="the question, in plain English")
    parser.add_argument("--db", choices=ALL_DATABASES, help="skip routing and use this database")
    parser.add_argument("--json", action="store_true", help="print the full AskResult as JSON")
    parser.add_argument("--status", action="store_true", help="show which databases are online, then exit")
    parser.add_argument("--chat", action="store_true", help="ask several questions in a row (follow-ups work)")
    parser.add_argument("--no-cache", action="store_true", help="don't reuse a cached answer")
    parser.add_argument("-v", "--verbose", action="store_true", help="log each pipeline step")
    args = parser.parse_args(argv)

    settings = get_settings()
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    engine = TalkTable(settings)
    try:
        if args.status:
            query_model = f" (queries: {settings.llm_query_model})" if settings.llm_query_model else ""
            print(f"LLM      {settings.llm_provider}:{settings.llm_model}{query_model}")
            for status in engine.status():
                state = "online " if status.online else "OFFLINE"
                detail = ", ".join(status.tables) if status.online else status.detail
                print(f"{state}  {status.name:<8}  {status.title:<20}  {detail}")
            return 0
        if args.chat:
            return _chat(engine, args.db, use_cache=not args.no_cache)
        question = " ".join(args.question)
        if not question:
            parser.error("please give a question (or --status / --chat)")
        result = engine.ask(question, database=args.db, use_cache=not args.no_cache)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            _print_result(result)
        return 1 if result.error else 0
    finally:
        engine.close()


def _chat(engine: TalkTable, database: str | None, *, use_cache: bool = True) -> int:
    """A terminal conversation. Earlier answers are remembered, so follow-ups work."""
    print('Ask a question, or a follow-up like "and in 2024?". Empty line or Ctrl+D to quit.')
    history: list[Turn] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in {"exit", "quit"}:
            return 0
        result = engine.ask(question, database=database, history=history, use_cache=use_cache)
        _print_result(result)
        if result.queries and not result.error:
            history = [*history, Turn.from_result(result)][-MAX_TURNS:]


if __name__ == "__main__":
    sys.exit(main())
