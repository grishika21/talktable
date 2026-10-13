"""Explainer: turn query results into a 1-3 sentence plain-English answer.

The model sees the question and the first rows of the result. Its answer is then
checked: every number it mentions must appear in the results (or the question). If
the model invents a number, it gets one more try with that feedback. After that we
fall back to a plain, template-based summary, so made-up numbers never reach the user.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from talktable.llm import ChatModel, LLMError, Message
from talktable.models import QueryResult

logger = logging.getLogger(__name__)

MAX_ROWS_SHOWN = 20
_MAX_CELL_CHARS = 80
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Numbers that are ranks rather than data: list markers ("2. ", "3) ") and ordinals ("3rd").
_ENUMERATION = re.compile(r"(?m)^\s*\d{1,2}[.)](?=\s)|\b\d+(?:st|nd|rd|th)\b")

NO_ROWS_ANSWER = "I couldn't find any matching data for that question."

_SYSTEM = """You explain database query results to a non-technical user of an online bookstore.
Answer the user's question in 1 to 3 short sentences of plain English.

Rules:
- Use ONLY facts and numbers that appear in the results. Never estimate, add up,
  average, extrapolate or invent numbers yourself.
- Report values; don't calculate differences, growth or trends between rows. You may
  name the highest or lowest value.
- You may round a number from the results (e.g. 11981.1 -> about 11,981).
- Mention the most important values, using names when the results include them.
- If the results only contain IDs, refer to them as IDs (e.g. "book 158"); don't guess names.
- If the results were capped, say they show the first rows only.
- Don't mention SQL, queries, tables, collections or JSON."""

_RETRY = """Your answer mentioned {numbers}, which do not appear in the results.
Rewrite the answer using only numbers that appear in the results."""


def format_rows(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """A compact pipe-separated table of the rows, for the prompt."""
    def cell(value: Any) -> str:
        text = "null" if value is None else str(value)
        return text if len(text) <= _MAX_CELL_CHARS else text[:_MAX_CELL_CHARS - 1] + "…"

    lines = [" | ".join(columns)]
    lines += [" | ".join(cell(row.get(c)) for c in columns) for row in rows]
    return "\n".join(lines)


def _numbers_in(text: str) -> list[tuple[str, float]]:
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group().rstrip(",")
        try:
            found.append((raw, float(raw.replace(",", ""))))
        except ValueError:
            continue
    return found


def _rounding_tolerance(raw: str) -> float:
    """How far off a number may be, judging by how precisely it is written:
    '4.3' -> 0.05, '11,981' -> 0.5, '12,000' -> 500 (rounded to the thousand)."""
    if "." in raw:
        return 0.5 * 10 ** -len(raw.split(".", 1)[1])
    digits = raw.replace(",", "")
    trailing_zeros = len(digits) - len(digits.rstrip("0"))
    return 0.5 * 10 ** trailing_zeros if trailing_zeros < len(digits) else 0.5


def ungrounded_numbers(answer: str, result: QueryResult, question: str) -> list[str]:
    """Numbers in `answer` that don't match any number in the results or the question.

    Rounding is allowed: '11,981' and 'about 12,000' match 11981.1, and '4.3' matches 4.27,
    but '11,985.37' does not.
    """
    allowed = {value for row in result.rows for v in row.values() for _, value in _numbers_in(str(v))}
    allowed |= {value for _, value in _numbers_in(question)}
    allowed |= {float(result.row_count), 0.0, 1.0}

    def matches(raw: str, value: float) -> bool:
        tolerance = _rounding_tolerance(raw)
        return any(abs(value - a) <= tolerance for a in allowed)

    return [raw for raw, value in _numbers_in(_ENUMERATION.sub(" ", answer)) if not matches(raw, value)]


def fallback_answer(result: QueryResult) -> str:
    """A plain summary built from the data alone (no LLM)."""
    if not result.rows:
        return NO_ROWS_ANSWER
    first = result.rows[0]
    if len(result.rows) == 1 and len(first) == 1:
        return f"The answer is {next(iter(first.values()))}."
    shown = ", ".join(f"{k}: {v}" for k, v in list(first.items())[:4])
    capped = f" (showing the first {result.row_count})" if result.truncated else ""
    return f"Found {result.row_count} result(s){capped}. The first one is {shown}."


class Explainer:
    def __init__(self, llm: ChatModel) -> None:
        self._llm = llm

    def explain(self, question: str, result: QueryResult) -> str:
        if not result.rows:
            return NO_ROWS_ANSWER
        shown = result.rows[:MAX_ROWS_SHOWN]
        note = ""
        if result.truncated:
            note = f" (capped at {result.row_count} rows)"
        elif len(shown) < result.row_count:
            note = f" (showing the first {len(shown)} of {result.row_count})"
        messages: list[Message] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nResults{note}:\n"
                                        f"{format_rows(result.columns, shown)}"},
        ]
        try:
            for attempt in range(2):
                answer = self._llm.complete(messages, max_tokens=250).strip()
                invented = ungrounded_numbers(answer, result, question)
                if not invented:
                    return answer
                logger.info("explainer: attempt %d mentioned numbers not in the results: %s", attempt + 1, invented)
                messages += [{"role": "assistant", "content": answer},
                             {"role": "user", "content": _RETRY.format(numbers=", ".join(invented))}]
        except LLMError as exc:
            logger.warning("explainer: LLM failed (%s); using the plain summary", exc)
        return fallback_answer(result)
