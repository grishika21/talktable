"""Answer cache: a repeated question comes back in about a second instead of a minute.

For each answered question we keep the database, the validated query, a fingerprint of
the result rows and the answer text. On a hit the query runs again: it's cheap, and the
data may have changed. If the rows are the same, the cached answer is reused; if not,
only the explanation is redone. The slow LLM steps (routing, writing the query) are skipped.

A question hits the cache in one of two ways:

1. **Same question:** identical after normalisation (case, spacing, punctuation,
   "five" -> 5, "1,000" -> 1000).
2. **Reworded question:** the same *content words* ("show me the 5 most expensive books"
   vs "what are the five most expensive books?"), identical "anchors" (every number and
   year, every name or quoted value, and the words that change what is computed:
   most/least, cheapest, before/after, average/total, per month, cancelled, ...), and an
   embedding similarity of at least CACHE_SIMILARITY as a final sanity check.

   Similarity alone isn't safe. With all-MiniLM-L6-v2, "...in 2024?" vs "...in 2025?"
   scores 0.99 and "mobile" vs "tablet" sessions 0.88, while a true paraphrase
   ("five priciest books") scores only 0.74. So the words decide, and a reworded
   question with different words is simply answered afresh: slower, but never wrong.

Entries are stored in SQLite with the vectors as float32 bytes. There's no vector
database: a few hundred cached questions are one small matrix multiplication. Entries
are tied to a version (the schema descriptions plus the LLM models), so changing
either quietly retires old entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], np.ndarray]

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "dozen": 12,
}
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")

# Words that don't change a question's meaning ("show me" = "what are").
_FILLER = frozenset("""
    what which who whom whose how show list give tell display find get return see let
    is are was were be been do does did the a an me us my our please can could would will
    you i we of for in on at to with and or there have has had that this these those all
    any some much it its their them they
""".split())

# Words that change what a question computes, mapped to a canonical form so harmless
# rewordings ("each month" / "per month") still match while real changes don't.
_MEANING_WORDS: dict[str, str] = {}
for canonical, words in {
    "max": "most top highest high max maximum largest biggest greatest best",
    "min": "least bottom lowest low min minimum smallest fewest worst",
    "cheap": "cheapest cheap cheaper inexpensive",
    "expensive": "expensive priciest pricey pricier costliest",
    "more": "more above over greater exceeding",
    "less": "less below under fewer",
    "first": "first earliest oldest",
    "last": "last latest newest recent",
    "longest": "longest",
    "shortest": "shortest",
    "before": "before until prior",
    "after": "after since",
    "not": "not no without except excluding exclude never",
    "only": "only just",
    "average": "average avg mean",
    "total": "total sum",
    "count": "count number many",
    "median": "median",
    "per": "per each every by",
    "day": "day daily days",
    "week": "week weekly weeks",
    "month": "month monthly months",
    "quarter": "quarter quarterly quarters",
    "year": "year yearly annual years",
    "cancelled": "cancelled canceled",
    "returned": "returned returns",
    "delivered": "delivered",
    "shipped": "shipped",
    "processing": "processing pending",
    "ascending": "ascending asc increasing",
    "descending": "descending desc decreasing",
}.items():
    for word in words.split():
        _MEANING_WORDS[word] = canonical


def normalize(question: str) -> str:
    """Lowercase, numbers as digits, punctuation and extra spaces removed."""
    text = question.lower().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,000 -> 1000
    text = _NUMBER_WORD.sub(lambda m: str(_NUMBER_WORDS[m.group(1)]), text)
    text = re.sub(r"[^\w\s.$]", " ", text)  # keep decimal points and $
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # ...but not full stops
    return " ".join(text.split())


def anchors(question: str) -> tuple[str, ...]:
    """What must be identical for two questions to share an answer: numbers and years,
    names (capitalised words after the first) and quoted values, and meaning words."""
    found: set[str] = set()
    normal = normalize(question)
    found |= {f"#{n}" for n in re.findall(r"\d+(?:\.\d+)?", normal)}
    found |= {f'"{q.lower()}"' for q in re.findall(r"""['"]([^'"]{1,60})['"]""", question)}
    words = re.findall(r"[A-Za-z][\w'-]*", question)
    found |= {f"@{w.lower()}" for w in words[1:] if w[0].isupper() and w.lower() not in ("i",)}
    found |= {f"~{_MEANING_WORDS[w]}" for w in normal.split() if w in _MEANING_WORDS}
    return tuple(sorted(found))


def content_words(question: str) -> frozenset[str]:
    """The words that carry the question's subject, without filler, numbers or meaning
    words (those are anchors). Plurals are folded: "books" = "book"."""
    words = set()
    for word in normalize(question).split():
        if word in _FILLER or word in _MEANING_WORDS or re.fullmatch(r"[\d.$]+", word):
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(word)
    return frozenset(words)


def rows_fingerprint(columns: list[str], rows: list[dict[str, Any]]) -> str:
    payload = json.dumps({"columns": columns, "rows": rows}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CachedAnswer:
    question: str  # the question this entry was stored for
    database: str
    query: str  # the validated query, as executed
    rows_hash: str
    answer: str
    match: str  # "same" or "reworded"


@dataclass
class _Entry:
    id: int
    normalized: str
    question: str
    anchors: tuple[str, ...]
    words: frozenset[str]
    vector: np.ndarray
    database: str
    query: str
    rows_hash: str
    answer: str
    created_at: float


class AnswerCache:
    """A persistent, thread-safe answer cache. `embed` turns a question into a unit vector."""

    def __init__(self, path: Path | None, embed: EmbedFn, *, version: str, similarity: float = 0.80,
                 ttl_hours: float = 168.0, max_entries: int = 1000) -> None:
        self._path = path
        self._embed = embed
        self._version = version
        self._similarity = similarity
        self._ttl_s = ttl_hours * 3600
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, list[_Entry]] = {}  # scope -> entries
        self._next_id = 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # --- persistence -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        conn = sqlite3.connect(self._path, timeout=5)
        conn.execute("""CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY, version TEXT NOT NULL, scope TEXT NOT NULL, normalized TEXT NOT NULL,
            question TEXT NOT NULL, anchors TEXT NOT NULL, vector BLOB NOT NULL, database TEXT NOT NULL,
            query TEXT NOT NULL, rows_hash TEXT NOT NULL, answer TEXT NOT NULL, created_at REAL NOT NULL,
            UNIQUE (version, scope, normalized))""")
        return conn

    def _load(self) -> None:
        cutoff = time.time() - self._ttl_s
        try:
            with self._connect() as conn:
                # Retire entries from older versions or past their time-to-live.
                conn.execute("DELETE FROM answers WHERE version != ? OR created_at < ?", (self._version, cutoff))
                rows = conn.execute("SELECT id, scope, normalized, question, anchors, vector, database, query, "
                                    "rows_hash, answer, created_at FROM answers ORDER BY id").fetchall()
        except sqlite3.Error as exc:
            logger.warning("cache: can't read %s (%s); starting empty", self._path, exc)
            return
        for (id_, scope, normalized, question, anchors_json, vector, database, query, rows_hash, answer,
             created_at) in rows:
            self._entries.setdefault(scope, []).append(_Entry(
                id=id_, normalized=normalized, question=question, anchors=tuple(json.loads(anchors_json)),
                words=content_words(question),
                vector=np.frombuffer(vector, dtype=np.float32).copy(), database=database, query=query,
                rows_hash=rows_hash, answer=answer, created_at=created_at))
            self._next_id = max(self._next_id, id_ + 1)
        logger.info("cache: %d answer(s) loaded", sum(len(v) for v in self._entries.values()))

    # --- lookups and updates -----------------------------------------------------

    def lookup(self, question: str, scope: str = "auto") -> CachedAnswer | None:
        """A cached answer for `question`, or None. `scope` is "auto" (routed) or the
        name of a database the caller forced."""
        normalized = normalize(question)
        with self._lock:
            live = [e for e in self._entries.get(scope, []) if not self._expired(e)]
            for entry in live:
                if entry.normalized == normalized:
                    return self._hit(entry, "same")
            if not live:
                return None
            wanted, words = anchors(question), content_words(question)
            candidates = [e for e in live if e.anchors == wanted and e.words == words]
            if not candidates:
                return None
            vector = self._embed(question)
            scores = np.stack([e.vector for e in candidates]) @ vector
            best = int(np.argmax(scores))
            if float(scores[best]) >= self._similarity:
                logger.info("cache: reworded question %r (%.3f)", candidates[best].question, float(scores[best]))
                return self._hit(candidates[best], "reworded")
        return None

    def store(self, question: str, *, database: str, query: str, rows_hash: str, answer: str,
              scope: str = "auto") -> None:
        normalized = normalize(question)
        entry = _Entry(id=0, normalized=normalized, question=question, anchors=anchors(question),
                       words=content_words(question),
                       vector=np.asarray(self._embed(question), dtype=np.float32), database=database, query=query,
                       rows_hash=rows_hash, answer=answer, created_at=time.time())
        with self._lock:
            entries = [e for e in self._entries.get(scope, []) if e.normalized != normalized]
            entry.id = self._next_id
            self._next_id += 1
            entries.append(entry)
            dropped = entries[:-self._max_entries] if len(entries) > self._max_entries else []
            self._entries[scope] = entries[-self._max_entries:]
            self._write(scope, entry, dropped)

    def forget(self, question: str, scope: str = "auto") -> None:
        """Drop the entry stored for `question` (e.g. its query stopped working)."""
        normalized = normalize(question)
        with self._lock:
            self._entries[scope] = [e for e in self._entries.get(scope, []) if e.normalized != normalized]
            self._delete(scope, [normalized])

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for entries in self._entries.values() for e in entries if not self._expired(e))

    # --- helpers -------------------------------------------------------------------

    def _expired(self, entry: _Entry) -> bool:
        return time.time() - entry.created_at > self._ttl_s

    @staticmethod
    def _hit(entry: _Entry, match: str) -> CachedAnswer:
        return CachedAnswer(question=entry.question, database=entry.database, query=entry.query,
                            rows_hash=entry.rows_hash, answer=entry.answer, match=match)

    def _write(self, scope: str, entry: _Entry, dropped: Iterable[_Entry]) -> None:
        if self._path is None:
            return
        try:
            with self._connect() as conn:
                conn.execute("INSERT OR REPLACE INTO answers (version, scope, normalized, question, anchors, vector, "
                             "database, query, rows_hash, answer, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                             (self._version, scope, entry.normalized, entry.question, json.dumps(entry.anchors),
                              entry.vector.tobytes(), entry.database, entry.query, entry.rows_hash, entry.answer,
                              entry.created_at))
                for old in dropped:
                    conn.execute("DELETE FROM answers WHERE version = ? AND scope = ? AND normalized = ?",
                                 (self._version, scope, old.normalized))
        except sqlite3.Error as exc:  # the cache is an optimisation: never fail a question over it
            logger.warning("cache: can't write %s (%s)", self._path, exc)

    def _delete(self, scope: str, normalized: Iterable[str]) -> None:
        if self._path is None:
            return
        try:
            with self._connect() as conn:
                conn.executemany("DELETE FROM answers WHERE version = ? AND scope = ? AND normalized = ?",
                                 [(self._version, scope, n) for n in normalized])
        except sqlite3.Error as exc:
            logger.warning("cache: can't update %s (%s)", self._path, exc)
