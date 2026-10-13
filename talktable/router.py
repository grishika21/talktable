"""Router: decide which database(s) can answer a question.

The LLM picks from short descriptions of the online databases and replies with JSON:
`{"databases": ["postgres"], "reason": "..."}`. If that fails (no JSON, unknown names,
LLM unreachable), we fall back to embedding similarity between the question and each
database's description.

Messages that aren't questions about the data ("hi", "thanks", "tell me a joke") get no
database at all, so the pipeline answers them with a short "what I can do" note instead
of running a made-up query.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from talktable.descriptions import Descriptions
from talktable.llm import ChatModel, LLMError, parse_json_object
from talktable.schema_index import SchemaIndex

logger = logging.getLogger(__name__)

RouteMethod = Literal["llm", "embedding", "single", "forced", "small_talk", "off_topic"]

# Names an LLM might use for our databases.
_ALIASES = {
    "postgresql": "postgres", "pg": "postgres", "neon": "postgres", "sales": "postgres",
    "mongodb": "mongo", "atlas": "mongo", "engagement": "mongo",
    "sqlite3": "sqlite", "catalog": "sqlite",
}
MAX_DATABASES = 2

# A message less similar than this to every database description isn't treated as a
# question about the data. Measured with all-MiniLM-L6-v2: greetings and unrelated
# questions score 0.00-0.15, real questions 0.21 and up. Re-measure it (setting
# MIN_RELEVANCE) if you switch the embedding model.
MIN_RELEVANCE = 0.38

# Greetings, thanks and acknowledgements: answered without calling the LLM.
_SMALL_TALK = re.compile(
    r"^(hi|hii+|hello|hey|hiya|howdy|yo|greetings|good (morning|afternoon|evening|day)"
    r"|thanks|thank you|thx|ty|cheers|ok|okay|cool|great|nice|awesome|bye|goodbye|see you)"
    r"(\s+(there|so much|a lot|again|talktable))?[\s!.,?:)]*$",
    re.IGNORECASE,
)


def is_small_talk(message: str) -> bool:
    return bool(_SMALL_TALK.match(message.strip()))

_SYSTEM = """You route questions about an online bookstore to the database that holds the answer.

Databases:
{catalog}

Reply with JSON only, in this format:
{{"databases": ["<database name>"], "reason": "<one short sentence>"}}

How to choose:
- Pick the ONE database whose tables hold the data the question is about.
- List a second database only if the answer truly needs columns from both.
- Use the database names exactly as written above (before the parentheses).
- If the message is not a question about this bookstore's data (a greeting, thanks,
  small talk, or an unrelated topic), reply {{"databases": [], "reason": "not about the data"}}.
{notes}"""


def canonical_name(name: str) -> str:
    """Map a database name as a person or LLM might write it ("PostgreSQL") to ours ("postgres")."""
    name = name.strip().lower()
    return _ALIASES.get(name, name)


class RouteDecision(BaseModel):
    databases: list[str]  # most relevant first; empty when the message isn't about the data
    reason: str
    method: RouteMethod


def _clean_names(raw: Any, available: Sequence[str]) -> list[str]:
    """Normalise whatever the LLM put in "databases" into known, available names."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = canonical_name(item)
        if name in available and name not in names:
            names.append(name)
    return names[:MAX_DATABASES]


class Router:
    def __init__(self, llm: ChatModel, index: SchemaIndex, descriptions: Descriptions,
                 *, min_relevance: float = MIN_RELEVANCE) -> None:
        self._llm = llm
        self._index = index
        self._descriptions = descriptions
        self._min_relevance = min_relevance

    def _catalog(self, available: Sequence[str]) -> str:
        lines = []
        for name in available:
            desc = self._descriptions.databases.get(name)
            tables = ", ".join(t.name for t in self._index.tables(name))
            if desc:
                summary = " ".join(desc.summary.split())
                terms = self._descriptions.glossary_terms(name)
                business = f" Business terms: {', '.join(terms)}." if terms else ""
                lines.append(f"- {name} ({desc.title}): {summary} Tables: {tables}.{business}")
            else:
                lines.append(f"- {name}: tables {tables}.")
        return "\n".join(lines)

    def route(self, question: str, available: Sequence[str]) -> RouteDecision:
        """Choose database(s) for `question` among the `available` (online) ones.

        Returns no databases (method "small_talk" or "off_topic") when the message
        isn't a question about the data.
        """
        if not available:
            raise ValueError("no databases available")
        if is_small_talk(question):
            return RouteDecision(databases=[], reason="small talk", method="small_talk")
        if len(available) == 1:
            score = self._best_match(question, available)[1]
            if score < self._min_relevance:
                return self._off_topic(score)
            return RouteDecision(databases=[available[0]], reason="the only database online", method="single")

        messages = [
            {"role": "system", "content": _SYSTEM.format(
                catalog=self._catalog(available),
                notes="\n".join(f"- {note}" for note in self._descriptions.routing_notes),
            )},
            {"role": "user", "content": question},
        ]
        try:
            reply = parse_json_object(self._llm.complete(messages, json_mode=True, max_tokens=150))
            names = _clean_names(reply.get("databases"), available)
            if names:
                return RouteDecision(databases=names, reason=str(reply.get("reason") or "").strip(), method="llm")
            if reply.get("databases") == []:
                # "Not about the data". Trust it unless the message clearly resembles one of
                # the databases, in which case the LLM probably got it wrong.
                logger.info("router: the LLM says no database is needed; checking with embeddings")
            else:
                logger.warning("router: no usable database in LLM reply %r; using embeddings", reply)
        except LLMError as exc:
            logger.warning("router: LLM failed (%s); using embeddings", exc)

        best, score = self._best_match(question, available)
        if score < self._min_relevance:
            return self._off_topic(score)
        return RouteDecision(databases=[best], reason=f"closest description match (similarity {score:.2f})",
                             method="embedding")

    def _best_match(self, question: str, available: Sequence[str]) -> tuple[str, float]:
        ranked = self._index.rank_databases(question, list(available))
        return ranked[0] if ranked else (available[0], 0.0)

    @staticmethod
    def _off_topic(score: float) -> RouteDecision:
        return RouteDecision(databases=[], reason=f"not about the data (best match {score:.2f})", method="off_topic")
