"""Where the schema vectors live: in memory with NumPy, or in Qdrant.

Each point is one embedded text with a payload saying what it describes:

    kind      what was embedded
    table     a table's description and columns
    question  one sample question the table answers (from descriptions.yaml)
    column    one column of a wide table (used to trim its column list)
    example   the question of one worked example (few_shot) for the query writer

Both stores answer the same search: the best points for a vector, filtered by
database/kind/table, optionally keeping only the best point per group (per table, or
per database), which is what "a table scores as well as its best-matching text" needs.

- `NumpyVectorStore`: exact cosine similarity over in-memory arrays. Rebuilt at every
  start-up, which is quick for a few hundred tables.
- `QdrantVectorStore`: a Qdrant collection, embedded on disk (`path`) or on a server
  (`url`). Vectors persist, so a restart only re-embeds tables whose text changed, and a
  server adds HNSW search and payload indexes for large schemas.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np

logger = logging.getLogger(__name__)

PointKind = str  # "table" | "question" | "column" | "example"
_ID_NAMESPACE = uuid.UUID("5b0d7f1e-3c1a-4f7e-9a59-7a1c2f5e7d10")  # Qdrant point ids must be UUIDs


@dataclass(frozen=True)
class Point:
    key: str  # unique and stable, e.g. "sqlite/books/question/1"
    vector: np.ndarray
    database: str
    kind: PointKind
    table: str = ""  # empty for points that belong to no table (examples)
    ref: str = ""  # which example/column, for kinds that need it
    hash: str = ""  # of the embedded text (and model): unchanged hash = no need to re-embed

    def payload(self) -> dict[str, Any]:
        return {"key": self.key, "database": self.database, "kind": self.kind, "table": self.table,
                "group": f"{self.database}.{self.table}", "ref": self.ref, "hash": self.hash}


@dataclass(frozen=True)
class Hit:
    key: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    backend: str

    def upsert(self, points: Sequence[Point]) -> None: ...

    def delete(self, database: str, keys: Iterable[str] | None = None) -> None:
        """Remove the given points of `database`, or all of them."""
        ...

    def hashes(self, database: str) -> dict[str, str]:
        """key -> text hash for every point of `database` (to skip re-embedding)."""
        ...

    def search(self, vector: np.ndarray, *, databases: Sequence[str], kinds: Sequence[str], limit: int,
               group_by: str | None = None, table: str | None = None) -> list[Hit]:
        """The `limit` best points, most similar first. With `group_by` ("group" for
        tables, "database" for databases) only the best point of each group counts."""
        ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class _Block(NamedTuple):
    """One database's points as arrays, for vectorised filtering and scoring."""

    points: list[Point]
    matrix: np.ndarray  # (n, dim)
    kinds: np.ndarray  # (n,) str
    tables: np.ndarray  # (n,) str


class NumpyVectorStore:
    """Exact search over in-memory arrays, one block per database."""

    backend = "numpy"

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Point]] = {}  # database -> key -> point
        self._blocks: dict[str, _Block] = {}  # built lazily, dropped on every change
        self._lock = threading.Lock()

    def upsert(self, points: Sequence[Point]) -> None:
        with self._lock:
            for point in points:
                self._points.setdefault(point.database, {})[point.key] = point
                self._blocks.pop(point.database, None)

    def delete(self, database: str, keys: Iterable[str] | None = None) -> None:
        with self._lock:
            if keys is None:
                self._points.pop(database, None)
            else:
                block = self._points.get(database, {})
                for key in keys:
                    block.pop(key, None)
            self._blocks.pop(database, None)

    def hashes(self, database: str) -> dict[str, str]:
        with self._lock:
            return {key: point.hash for key, point in self._points.get(database, {}).items()}

    def _block(self, database: str) -> _Block | None:
        if database not in self._blocks:
            points = list(self._points.get(database, {}).values())
            if not points:
                return None
            self._blocks[database] = _Block(points, np.stack([p.vector for p in points]),
                                            np.array([p.kind for p in points]), np.array([p.table for p in points]))
        return self._blocks[database]

    def search(self, vector: np.ndarray, *, databases: Sequence[str], kinds: Sequence[str], limit: int,
               group_by: str | None = None, table: str | None = None) -> list[Hit]:
        blocks: list[_Block] = []
        parts: list[np.ndarray] = []
        with self._lock:
            for database in databases:
                block = self._block(database)
                if block is None:
                    continue
                mask = np.isin(block.kinds, list(kinds))
                if table is not None:
                    mask &= block.tables == table
                blocks.append(block)
                parts.append(np.where(mask, block.matrix @ vector, -np.inf))  # filtered out: never ranked
        if not parts:
            return []
        scores = np.concatenate(parts)
        offsets = np.cumsum([0] + [len(block.points) for block in blocks])
        hits: list[Hit] = []
        seen: set[str] = set()
        for flat in np.argsort(-scores, kind="stable"):
            if scores[flat] == -np.inf:
                break
            which = int(np.searchsorted(offsets, flat, side="right")) - 1
            point = blocks[which].points[flat - offsets[which]]
            payload = point.payload()
            if group_by is not None:
                if payload[group_by] in seen:
                    continue
                seen.add(payload[group_by])
            hits.append(Hit(point.key, float(scores[flat]), payload))
            if len(hits) == limit:
                break
        return hits

    def count(self) -> int:
        with self._lock:
            return sum(len(block) for block in self._points.values())

    def close(self) -> None:
        pass


class QdrantVectorStore:
    """A Qdrant collection: embedded on disk when `url` is None, else a Qdrant server.

    Raises (from the constructor):
        RuntimeError: Qdrant can't be used (server unreachable, or the local folder is
            open in another process; embedded mode allows one process per folder).
    """

    backend = "qdrant"

    def __init__(self, collection: str, *, url: str | None = None, api_key: str | None = None,
                 path: Path | None = None, timeout: float = 10.0) -> None:
        from qdrant_client import QdrantClient  # optional dependency: import only when used

        self._collection = collection
        self._server = url is not None
        try:
            if url is not None:
                self._client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout))
                self.location = url
            elif path is not None:
                path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path))
                self.location = str(path)
            else:
                self._client = QdrantClient(location=":memory:")
                self.location = ":memory:"
            self._client.get_collections()  # fail now, not on the first question
        except Exception as exc:  # noqa: BLE001 - connection, lock and version errors vary by mode
            raise RuntimeError(f"can't use Qdrant at {url or path or ':memory:'}: {exc}") from exc
        self._dim: int | None = None

    # --- collection -----------------------------------------------------------

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client import models

        if self._dim == dim:
            return
        if self._client.collection_exists(self._collection):
            existing = self._client.get_collection(self._collection).config.params.vectors
            size = existing.size if isinstance(existing, models.VectorParams) else None
            if size == dim:
                self._dim = dim
                return
            logger.warning("qdrant: collection %s has %s-dim vectors, not %d; recreating it",
                           self._collection, size, dim)
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        if self._server:  # payload indexes make filtered search fast; embedded mode doesn't use them
            for name in ("database", "kind", "table"):
                self._client.create_payload_index(self._collection, name, models.PayloadSchemaType.KEYWORD)
        self._dim = dim

    def _exists(self) -> bool:
        return self._dim is not None or self._client.collection_exists(self._collection)

    @staticmethod
    def _filter(databases: Sequence[str], kinds: Sequence[str] | None = None, table: str | None = None) -> Any:
        from qdrant_client import models

        must: list[Any] = [models.FieldCondition(key="database", match=models.MatchAny(any=list(databases)))]
        if kinds is not None:
            must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
        if table is not None:
            must.append(models.FieldCondition(key="table", match=models.MatchValue(value=table)))
        return models.Filter(must=must)

    # --- VectorStore ------------------------------------------------------------

    def upsert(self, points: Sequence[Point]) -> None:
        from qdrant_client import models

        if not points:
            return
        self._ensure_collection(len(points[0].vector))
        structs = [models.PointStruct(id=str(uuid.uuid5(_ID_NAMESPACE, p.key)), vector=p.vector.tolist(),
                                      payload=p.payload()) for p in points]
        for start in range(0, len(structs), 512):
            self._client.upsert(self._collection, points=structs[start:start + 512], wait=True)

    def delete(self, database: str, keys: Iterable[str] | None = None) -> None:
        from qdrant_client import models

        if not self._exists():
            return
        if keys is None:
            selector: Any = models.FilterSelector(filter=self._filter([database]))
        else:
            ids = [str(uuid.uuid5(_ID_NAMESPACE, key)) for key in keys]
            if not ids:
                return
            selector = models.PointIdsList(points=ids)
        self._client.delete(self._collection, points_selector=selector, wait=True)

    def hashes(self, database: str) -> dict[str, str]:
        if not self._exists():
            return {}
        found: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(self._collection, scroll_filter=self._filter([database]),
                                                 limit=1024, offset=offset, with_payload=["key", "hash"],
                                                 with_vectors=False)
            found.update({p.payload["key"]: p.payload.get("hash", "") for p in points if p.payload})
            if offset is None:
                return found

    def search(self, vector: np.ndarray, *, databases: Sequence[str], kinds: Sequence[str], limit: int,
               group_by: str | None = None, table: str | None = None) -> list[Hit]:
        if not databases or not self._exists():
            return []
        query_filter = self._filter(databases, kinds, table)
        if group_by is None:
            response = self._client.query_points(self._collection, query=vector.tolist(), limit=limit,
                                                 query_filter=query_filter, with_payload=True)
            return [Hit(p.payload["key"], float(p.score), dict(p.payload)) for p in response.points if p.payload]
        groups = self._client.query_points_groups(self._collection, query=vector.tolist(), group_by=group_by,
                                                  limit=limit, group_size=1, query_filter=query_filter,
                                                  with_payload=True).groups
        hits = [Hit(g.hits[0].payload["key"], float(g.hits[0].score), dict(g.hits[0].payload))
                for g in groups if g.hits and g.hits[0].payload]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def count(self) -> int:
        return self._client.count(self._collection).count if self._exists() else 0

    def close(self) -> None:
        self._client.close()


def create_vector_store(backend: str, *, collection: str = "talktable_schema", url: str | None = None,
                        api_key: str | None = None, path: Path | None = None) -> VectorStore:
    """A store for `backend` ("numpy" or "qdrant"). If Qdrant can't be used, says why in
    the log and falls back to NumPy, so the app still starts."""
    if backend == "numpy":
        return NumpyVectorStore()
    if backend != "qdrant":
        raise ValueError(f"unknown vector store {backend!r}; use numpy or qdrant")
    try:
        store = QdrantVectorStore(collection, url=url, api_key=api_key, path=path)
        logger.info("vector store: qdrant (%s, collection %s)", store.location, collection)
        return store
    except (ImportError, RuntimeError) as exc:
        logger.warning("vector store: %s; using the in-memory NumPy store instead", exc)
        return NumpyVectorStore()
