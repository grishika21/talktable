"""The two vector stores must agree: same hits, filters, grouping and bookkeeping."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from talktable.vector_store import NumpyVectorStore, Point, QdrantVectorStore, VectorStore, create_vector_store


def unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@pytest.fixture(scope="module")
def points() -> list[Point]:
    rng = np.random.default_rng(7)
    made = []
    for db in ("sqlite", "postgres"):
        for t in range(6):
            made.append(Point(f"{db}/t{t}/table", unit(rng.normal(size=16)), db, "table", f"t{t}", hash=f"h{t}"))
            made += [Point(f"{db}/t{t}/question/{i}", unit(rng.normal(size=16)), db, "question", f"t{t}",
                           hash=f"q{t}{i}") for i in range(3)]
        made += [Point(f"{db}/-/example/{i}", unit(rng.normal(size=16)), db, "example", ref=str(i)) for i in range(4)]
    return made


@pytest.fixture(params=["numpy", "qdrant-memory", "qdrant-disk"])
def store(request: pytest.FixtureRequest, points: list[Point], tmp_path: Path) -> Iterator[VectorStore]:
    if request.param == "numpy":
        made: VectorStore = NumpyVectorStore()
    elif request.param == "qdrant-memory":
        made = QdrantVectorStore("test")
    else:
        made = QdrantVectorStore("test", path=tmp_path / "qdrant")
    made.upsert(points)
    yield made
    made.close()


def brute_force(points: list[Point], query: np.ndarray, **match: str) -> list[tuple[str, float]]:
    scored = [(p.key, float(p.vector @ query)) for p in points
              if all(getattr(p, field) == value for field, value in match.items())]
    return sorted(scored, key=lambda item: -item[1])


QUERY = unit(np.random.default_rng(1).normal(size=16))


def test_search_matches_brute_force(store: VectorStore, points: list[Point]) -> None:
    hits = store.search(QUERY, databases=["sqlite"], kinds=["question"], limit=5)
    expected = brute_force(points, QUERY, database="sqlite", kind="question")[:5]
    assert [h.key for h in hits] == [key for key, _ in expected]
    assert [round(h.score, 4) for h in hits] == [round(score, 4) for _, score in expected]


def test_group_by_keeps_the_best_point_of_each_table(store: VectorStore, points: list[Point]) -> None:
    hits = store.search(QUERY, databases=["sqlite"], kinds=["table", "question"], limit=3, group_by="group")
    best: dict[str, float] = {}
    for key, score in brute_force(points, QUERY, database="sqlite"):
        table = key.split("/")[1]
        if table != "-":
            best.setdefault(table, score)
    expected = sorted(best.items(), key=lambda item: -item[1])[:3]
    assert [h.payload["table"] for h in hits] == [table for table, _ in expected]
    assert len({h.payload["table"] for h in hits}) == 3


def test_group_by_database_ranks_databases(store: VectorStore, points: list[Point]) -> None:
    hits = store.search(QUERY, databases=["sqlite", "postgres"], kinds=["table", "question"], limit=2,
                        group_by="database")
    tops = {db: brute_force([p for p in points if p.kind != "example"], QUERY, database=db)[0][1]
            for db in ("sqlite", "postgres")}
    assert [h.payload["database"] for h in hits] == sorted(tops, key=lambda db: -tops[db])
    assert hits[0].score == pytest.approx(max(tops.values()), abs=1e-4)


def test_filters_by_table_and_kind(store: VectorStore) -> None:
    hits = store.search(QUERY, databases=["postgres"], kinds=["question"], limit=10, table="t2")
    assert {h.key for h in hits} == {f"postgres/t2/question/{i}" for i in range(3)}
    examples = store.search(QUERY, databases=["postgres"], kinds=["example"], limit=10)
    assert sorted(h.payload["ref"] for h in examples) == ["0", "1", "2", "3"]


def test_hashes_and_delete(store: VectorStore) -> None:
    assert store.count() == 2 * (6 * 4 + 4)
    hashes = store.hashes("sqlite")
    assert len(hashes) == 28 and hashes["sqlite/t3/table"] == "h3"
    store.delete("sqlite", ["sqlite/t3/table", "sqlite/t3/question/0"])
    assert len(store.hashes("sqlite")) == 26 and "sqlite/t3/table" not in store.hashes("sqlite")
    store.delete("sqlite")
    assert store.hashes("sqlite") == {} and len(store.hashes("postgres")) == 28
    assert store.search(QUERY, databases=["sqlite"], kinds=["table"], limit=3) == []


def test_upsert_replaces_a_point(store: VectorStore) -> None:
    replacement = Point("sqlite/t0/table", QUERY, "sqlite", "table", "t0", hash="new")
    store.upsert([replacement])
    assert store.hashes("sqlite")["sqlite/t0/table"] == "new" and store.count() == 56
    top = store.search(QUERY, databases=["sqlite"], kinds=["table"], limit=1)[0]
    assert top.key == "sqlite/t0/table" and top.score == pytest.approx(1.0, abs=1e-5)


def test_qdrant_on_disk_persists_between_runs(points: list[Point], tmp_path: Path) -> None:
    first = QdrantVectorStore("schema", path=tmp_path / "q")
    first.upsert(points)
    first.close()
    second = QdrantVectorStore("schema", path=tmp_path / "q")
    try:
        assert second.count() == len(points) and second.hashes("postgres")["postgres/t1/table"] == "h1"
    finally:
        second.close()


def test_qdrant_recreates_the_collection_for_another_embedding_size(points: list[Point]) -> None:
    store = QdrantVectorStore("schema")
    try:
        store.upsert(points)
        store.upsert([Point("sqlite/x/table", unit(np.ones(8)), "sqlite", "table", "x")])
        assert store.count() == 1  # the old 16-dim vectors were dropped
    finally:
        store.close()


def test_falls_back_to_numpy_when_qdrant_is_busy(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    holder = QdrantVectorStore("schema", path=tmp_path / "q")  # embedded mode: one process per folder
    try:
        store = create_vector_store("qdrant", path=tmp_path / "q")
        assert store.backend == "numpy" and "using the in-memory NumPy store" in caplog.text
    finally:
        holder.close()
    assert create_vector_store("qdrant", url="http://127.0.0.1:1").backend == "numpy"  # nothing listening
    with pytest.raises(ValueError, match="unknown vector store"):
        create_vector_store("chroma")
