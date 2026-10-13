"""Follow-up questions in a chat: "now only for 2024", "what are their titles?".

The chat UI keeps the last few turns. Before routing, a follow-up is rewritten into a
question that stands on its own ("How many orders were placed each month in 2024?"),
using one LLM call. That question then goes through the normal pipeline, so routing,
schema retrieval, the glossary and the explanation all work exactly as for a fresh
question. Messages that already stand on their own skip the rewrite.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from talktable.llm import ChatModel, LLMError, Message, parse_json_object
from talktable.models import AskResult

logger = logging.getLogger(__name__)

MAX_TURNS = 3  # how many earlier turns the rewrite sees
_MAX_PREVIEW_ROWS = 10
_MAX_PREVIEW_CHARS = 600
_MAX_ANSWER_CHARS = 300
_MAX_STANDALONE_CHARS = 1000

# Words and openings that usually mean "this refers to the conversation".
_REFERENCES = re.compile(
    r"\b(it|its|they|them|their|theirs|those|these|that one|this one|same|previous|above|earlier|instead"
    r"|also|too|as well|respectively|ones|again|the rest|others)\b",
    re.IGNORECASE,
)
_OPENERS = re.compile(
    r"^(and|but|also|now|then|so|only|just|same|instead|ok|okay|what about|how about|what if|show me more"
    r"|more|less|fewer|except|excluding|without|including|plus|vs|versus|compared)\b",
    re.IGNORECASE,
)
_SHORT = 6  # messages of this many words or fewer are usually follow-ups ("and in 2024?")

_SYSTEM = """You rewrite the latest message of a conversation about an online bookstore's data into
ONE question that can be understood without the conversation.

Rules:
- Keep everything the message relies on from earlier turns: what is measured, filters, time
  periods, groupings, limits, names and IDs.
- If the message changes something (a year, a limit, a filter), apply the change and keep the rest.
- If the message refers to earlier results ("their titles", "those books"), write the specific IDs
  or names from those results into the question.
- If the message already stands on its own, or isn't about the data, return it unchanged.
- Don't answer the question, and don't add anything the user didn't ask for.

Examples:
Earlier question: How many orders were placed each month in 2025?
Latest message: and in 2024?
{"question": "How many orders were placed each month in 2024?"}

Earlier question: Which 5 books have the most 1-star reviews? (results: _id = book_id 158, 399, 522, 810, 180)
Latest message: what are their titles?
{"question": "What are the titles of the books with book_id 158, 399, 522, 810 and 180?"}

Reply with JSON only: {"question": "<the standalone question>"}"""


class Turn(BaseModel):
    """One answered question, as the rewrite needs to see it."""

    question: str  # the standalone question that was answered
    databases: list[str]
    query: str | None
    answer: str
    rows_preview: str  # the first result rows, compact

    @classmethod
    def from_result(cls, result: AskResult) -> Turn:
        return cls(
            question=result.interpreted_as or result.question,
            databases=result.databases_used,
            query=result.queries[-1] if result.queries else None,
            answer=_clip(result.answer, _MAX_ANSWER_CHARS),
            rows_preview=preview_rows(result.columns or (list(result.rows[0]) if result.rows else []), result.rows),
        )


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def preview_rows(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """The first rows as 'col=value, col=value; …', short enough for a prompt."""
    lines = ["; ".join(", ".join(f"{c}={row.get(c)}" for c in columns) for row in rows[:_MAX_PREVIEW_ROWS])]
    if len(rows) > _MAX_PREVIEW_ROWS:
        lines.append(f"(+{len(rows) - _MAX_PREVIEW_ROWS} more rows)")
    return _clip(" ".join(lines), _MAX_PREVIEW_CHARS)


def looks_like_follow_up(message: str) -> bool:
    """A cheap guess at whether a message depends on the conversation. A wrong "yes" only
    costs one LLM call, which returns the message unchanged."""
    message = message.strip()
    return (len(message.split()) <= _SHORT or bool(_OPENERS.match(message))
            or bool(_REFERENCES.search(message)))


def _format_history(history: Sequence[Turn]) -> str:
    blocks = []
    for number, turn in enumerate(history, start=1):
        lines = [f"[{number}] Question: {turn.question}"]
        if turn.databases:
            lines.append(f"    Database: {', '.join(turn.databases)}")
        if turn.query:
            lines.append(f"    Query: {_clip(turn.query, 600)}")
        lines.append(f"    Answer: {turn.answer}")
        if turn.rows_preview:
            lines.append(f"    First rows: {turn.rows_preview}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_messages(message: str, history: Sequence[Turn]) -> list[Message]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Conversation so far:\n{_format_history(history)}\n\nLatest message: {message}"},
    ]


class Condenser:
    """Rewrites follow-up messages into standalone questions."""

    def __init__(self, llm: ChatModel) -> None:
        self._llm = llm

    def standalone(self, message: str, history: Sequence[Turn]) -> str:
        """The standalone version of `message`, or `message` itself if the rewrite fails."""
        recent = list(history)[-MAX_TURNS:]
        if not recent:
            return message
        try:
            reply = parse_json_object(self._llm.complete(build_messages(message, recent), json_mode=True,
                                                         max_tokens=200))
        except LLMError as exc:
            logger.warning("conversation: rewrite failed (%s); using the message as it is", exc)
            return message
        question = reply.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > _MAX_STANDALONE_CHARS:
            logger.warning("conversation: unusable rewrite %r; using the message as it is", reply)
            return message
        return " ".join(question.split())
