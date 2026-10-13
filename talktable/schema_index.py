"""Schema retrieval: find the tables/collections relevant to a question.

Every table is embedded as several vectors (see vector_store.py): its description and
columns, plus each of its sample questions on its own. A question is matched against
all of them and a table scores as well as its best match, so "Which books are out of
stock?" finds inventory through its sample question even though its description talks
about warehouses and reorder levels. A database scores as well as its best table.

The top tables are then completed with the foreign-key graph (schema_graph.py): bridge
tables that connect them, and the tables they reference (for names and labels). Those
extras are shown to the query writer in a brief form, with the join conditions spelled
out, and the whole schema is kept under a token budget.

Vectors live in NumPy for small schemas and in Qdrant for large ones; only texts that
changed since the last start-up are embedded again.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, NamedTuple, Protocol

import numpy as np

from talktable.descriptions import DatabaseDescription, FewShotExample
from talktable.generator import estimate_tokens, format_schema
from talktable.models import TableInfo
from talktable.schema_graph import JoinGraph
from talktable.vector_store import NumpyVectorStore, Point, VectorStore

logger = logging.getLogger(__name__)

RETRIEVE_K = 5  # best-matching tables per question, before join expansion ...
MATCH_MARGIN = 0.25  # ... keeping those within this similarity of the best one
N_EXAMPLES = 3  # worked examples given to the query writer
WIDE_TABLE = 20  # tables with more columns than this are trimmed ...
KEEP_COLUMNS = 12  # ... to their keys plus this many best-matching columns
SCHEMA_TOKEN_BUDGET = 2000  # estimated tokens for the schema part of the prompt

TableRole = Literal["match", "bridge", "lookup"]
_TABLE_KINDS = ("table", "question")


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """One L2-normalised vector per text, shape (len(texts), dim)."""
        ...


class SentenceEmbedder:
    """sentence-transformers on CPU. The model is loaded once, on first use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import: keep it lazy
            from transformers.utils import logging as transformers_logging

            transformers_logging.disable_progress_bar()  # no "Loading weights" bars in logs
            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False, batch_size=64)
        return np.asarray(vectors, dtype=np.float32)


def table_text(table: TableInfo, db_title: str) -> str:
    """The chunk embedded for a table: what it is, and every column and its meaning.
    (Its sample questions are embedded separately: they match questions best on their own.)"""
    columns = "; ".join(f"{c.name}: {c.description}" if c.description else c.name for c in table.columns)
    return f"{db_title}. Table {table.name}: {table.description or ''} Columns: {columns}."


class _Text(NamedTuple):
    key: str
    text: str
    kind: str
    table: str = ""
    ref: str = ""


@dataclass(frozen=True)
class SchemaSelection:
    """The part of a database's schema shown to the query writer for one question."""

    tables: list[TableInfo]  # matches (best first), then bridges, then lookups
    roles: dict[str, TableRole]
    scores: dict[str, float]  # similarity of each matched table
    joins: list[str]  # e.g. "books.author_id = authors.author_id"
    compact: frozenset[str] = frozenset()  # tables shown briefly (no column descriptions)
    examples: list[FewShotExample] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tables]


class SchemaIndex:
    """Embedding index over the tables of every online database."""

    def __init__(self, embedder: Embedder, store: VectorStore | None = None) -> None:
        self._embedder = embedder
        self._model_id = getattr(embedder, "model_name", type(embedder).__name__)
        self._store: VectorStore = store or NumpyVectorStore()
        self._tables: dict[str, list[TableInfo]] = {}
        self._graphs: dict[str, JoinGraph] = {}
        self._examples: dict[str, list[FewShotExample]] = {}
        self._embed_question = lru_cache(maxsize=256)(self._embed_one)

    def _embed_one(self, text: str) -> np.ndarray:
        return self._embedder.encode([text])[0]

    def embed(self, text: str) -> np.ndarray:
        """The (cached) unit vector for a question."""
        return self._embed_question(text)

    # --- indexing ---------------------------------------------------------------

    @property
    def store(self) -> VectorStore:
        return self._store

    def use_store(self, store: VectorStore) -> None:
        """Switch to another vector store. It starts empty here: add the databases again."""
        if store is self._store:
            return
        self._store.close()
        self._store = store
        for registry in (self._tables, self._graphs, self._examples):
            registry.clear()

    def _texts(self, db: str, tables: list[TableInfo], description: DatabaseDescription | None) -> Iterator[_Text]:
        title = description.title if description else db
        for table in tables:
            yield _Text(f"{db}/{table.name}/table", table_text(table, title), "table", table.name)
            for i, question in enumerate(table.questions):
                yield _Text(f"{db}/{table.name}/question/{i}", question, "question", table.name)
            if len(table.columns) > WIDE_TABLE:
                for column in table.columns:
                    yield _Text(f"{db}/{table.name}/column/{column.name}",
                                f"{table.name}.{column.name}: {column.description or column.name}",
                                "column", table.name, column.name)
        for i, example in enumerate(description.few_shot if description else []):
            yield _Text(f"{db}/-/example/{i}", example.question, "example", ref=str(i))

    def _hash(self, text: str) -> str:
        return hashlib.sha1(f"{self._model_id}\0{text}".encode()).hexdigest()[:16]

    def add_database(self, name: str, tables: list[TableInfo], description: DatabaseDescription | None) -> int:
        """Index (or re-index) one database's tables. `tables` should carry descriptions.

        Returns how many texts had to be embedded: unchanged ones are kept from before
        (which, with Qdrant, includes earlier runs of the app).
        """
        texts = {t.key: t for t in self._texts(name, tables, description)}
        hashes = {key: self._hash(t.text) for key, t in texts.items()}
        known = self._store.hashes(name)
        fresh = [texts[key] for key in texts if known.get(key) != hashes[key]]
        if fresh:
            vectors = self._embedder.encode([t.text for t in fresh])
            self._store.upsert([Point(t.key, vector, name, t.kind, t.table, t.ref, hashes[t.key])
                                for t, vector in zip(fresh, vectors)])
        stale = set(known) - set(texts)
        if stale:
            self._store.delete(name, stale)
        self._tables[name] = list(tables)
        self._graphs[name] = JoinGraph(tables)
        self._examples[name] = list(description.few_shot) if description else []
        logger.info("indexed %s: %d table(s), %d vector(s), %d newly embedded (%s)", name, len(tables),
                    len(texts), len(fresh), self._store.backend)
        return len(fresh)

    def remove_database(self, name: str) -> None:
        """Stop using a database (it went offline). Its vectors stay in a persistent store,
        so they don't need embedding again when it comes back."""
        for registry in (self._tables, self._graphs, self._examples):
            registry.pop(name, None)

    @property
    def databases(self) -> list[str]:
        return list(self._tables)

    def tables(self, db: str) -> list[TableInfo]:
        return list(self._tables.get(db, []))

    def graph(self, db: str) -> JoinGraph:
        return self._graphs[db]

    # --- retrieval --------------------------------------------------------------

    def retrieve(self, question: str, db: str, k: int = RETRIEVE_K, *, examples: int = N_EXAMPLES,
                 token_budget: int = SCHEMA_TOKEN_BUDGET, kind: str = "sql") -> SchemaSelection:
        """The schema of `db` the query writer needs for `question`.

        Up to `k` best-matching tables (those close to the best match), plus the bridge
        tables that join them and the tables they reference, with the join conditions,
        the most similar worked examples, and at most about `token_budget` tokens of schema.
        """
        tables = self._tables.get(db)
        if not tables:
            raise KeyError(f"database {db!r} is not indexed")
        by_name = {t.name: t for t in tables}
        query = self.embed(question)
        hits = self._store.search(query, databases=[db], kinds=_TABLE_KINDS, limit=k, group_by="group")
        hits = [hit for hit in hits if hit.payload["table"] in by_name]
        # A weak match far behind the best one is usually noise, and joining it in would
        # drag its bridge tables along too.
        scores = {hit.payload["table"]: hit.score for hit in hits if hit.score >= hits[0].score - MATCH_MARGIN}
        matches = list(scores) or [tables[0].name]

        graph = self._graphs[db]
        bridges = graph.bridges(matches)
        lookups = [parent for name in matches for parent in graph.parents(name)
                   if parent not in matches and parent not in bridges]
        roles: dict[str, TableRole] = {name: "match" for name in matches}
        roles.update({name: "bridge" for name in bridges})
        roles.update({name: "lookup" for name in dict.fromkeys(lookups)})

        chosen = [self._trim_columns(by_name[name], db, query) for name in roles]
        selection = SchemaSelection(
            tables=chosen, roles=roles, scores=scores, joins=graph.joins(list(roles)),
            compact=frozenset(name for name, role in roles.items() if role != "match"),
            examples=self._similar_examples(query, db, examples),
        )
        return self._fit(selection, db, kind, token_budget)

    def _trim_columns(self, table: TableInfo, db: str, query: np.ndarray) -> TableInfo:
        """A wide table keeps its key columns and the columns most similar to the question."""
        if len(table.columns) <= WIDE_TABLE:
            return table
        hits = self._store.search(query, databases=[db], kinds=["column"], table=table.name, limit=KEEP_COLUMNS)
        keys = set(table.primary_key) | {fk.split("->")[0].strip() for fk in table.foreign_keys}
        keep = keys | {hit.payload["ref"] for hit in hits}
        columns = [c for c in table.columns if c.name in keep]
        note = f" (Showing {len(columns)} of its {len(table.columns)} columns.)"
        return table.model_copy(update={"columns": columns, "description": (table.description or "") + note})

    def _similar_examples(self, query: np.ndarray, db: str, n: int) -> list[FewShotExample]:
        library = self._examples.get(db, [])
        if n <= 0 or not library:
            return []
        hits = self._store.search(query, databases=[db], kinds=["example"], limit=n)
        picked = sorted({int(hit.payload["ref"]) for hit in hits if int(hit.payload["ref"]) < len(library)})
        return [library[i] for i in picked]  # library order, so the prompt reads the same way each time

    def _fit(self, selection: SchemaSelection, db: str, kind: str, budget: int) -> SchemaSelection:
        """Shrink the selection until its schema text fits the token budget: first show
        the less relevant matches briefly, then drop lookups, bridges and weak matches."""
        tables, compact = list(selection.tables), set(selection.compact)
        graph = self._graphs[db]

        def size() -> int:
            names = [t.name for t in tables]
            return estimate_tokens(format_schema(tables, kind, compact=frozenset(compact), joins=graph.joins(names)))

        if size() <= budget:
            return selection
        matches = [name for name, role in selection.roles.items() if role == "match"]
        for name in reversed(matches[1:]):
            compact.add(name)
            if size() <= budget:
                break
        droppable = [[n for n, r in selection.roles.items() if r == role] for role in ("lookup", "bridge")]
        for name in [*reversed(droppable[0]), *reversed(droppable[1]), *reversed(matches[1:])]:
            if size() <= budget:
                break
            tables = [t for t in tables if t.name != name]
        names = [t.name for t in tables]
        logger.info("schema trimmed to the %d-token budget: %s", budget, ", ".join(names))
        return SchemaSelection(tables=tables, roles={n: selection.roles[n] for n in names},
                               scores={n: s for n, s in selection.scores.items() if n in names},
                               joins=graph.joins(names), compact=frozenset(compact & set(names)),
                               examples=selection.examples)

    def rank_databases(self, question: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Candidate databases, best first, each scored by its best-matching table."""
        candidates = [db for db in candidates if db in self._tables]
        if not candidates:
            return []
        hits = self._store.search(self.embed(question), databases=candidates, kinds=_TABLE_KINDS,
                                  limit=len(candidates), group_by="database")
        return [(hit.payload["database"], hit.score) for hit in hits]
