"""How schema retrieval holds up as the schema grows.

Synthetic tables (HR, finance, logistics, store operations, ...; with foreign keys and
sample questions, like real ones) are added to the 30 real tables, spread over the three
databases. For each size, and for each vector store:

  - start-up:  embedding + indexing on a first start, and on a restart
  - retrieval: time per question (the question's own embedding excluded)
  - quality:   do the held-out questions still get every table they need?
  - prompt:    schema tokens shown to the query writer, vs. sending every table

    python -m eval.benchmark_scale [--sizes 300 1000 3000] [--qdrant-url http://localhost:6333]

Needs the three databases (for the real schemas) but no LLM. Qdrant runs embedded in a
temporary folder; with --qdrant-url, a Qdrant server is measured too (its collections
are deleted afterwards).
"""

from __future__ import annotations

import argparse
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from talktable.adapters import create_adapters
from talktable.config import get_settings
from talktable.descriptions import DatabaseDescription, describe_tables, load_descriptions
from talktable.generator import estimate_tokens, format_schema
from talktable.models import ColumnInfo, TableInfo
from talktable.schema_index import SchemaIndex, SentenceEmbedder
from talktable.vector_store import NumpyVectorStore, QdrantVectorStore, VectorStore

QUESTIONS = Path(__file__).with_name("retrieval_questions.yaml")

DOMAINS = [
    ("hr", "human resources", ["employees", "departments", "payroll runs", "leave requests", "job openings",
                               "candidates", "interviews", "training sessions"]),
    ("finance", "finance", ["invoices", "ledger entries", "budgets", "expense claims", "vendors", "tax filings",
                            "bank transfers", "purchase orders"]),
    ("marketing", "marketing", ["ad campaigns", "ad spend", "landing pages", "promotions", "influencers",
                                "press releases", "social posts", "event sponsorships"]),
    ("logistics", "logistics", ["trucks", "drivers", "routes", "fuel logs", "depots", "pallets", "customs forms",
                                "freight quotes"]),
    ("it", "IT operations", ["servers", "incidents", "deployments", "software licenses", "laptops", "backups",
                             "alerts", "access requests"]),
    ("facilities", "facilities", ["buildings", "meeting rooms", "room bookings", "maintenance jobs",
                                  "energy readings", "parking permits", "visitor badges", "cleaning rounds"]),
    ("legal", "legal", ["contracts", "disputes", "trademarks", "compliance checks", "policies", "audits",
                        "nda signatures", "insurance claims"]),
    ("product", "product", ["features", "experiments", "releases", "bug reports", "roadmap items",
                            "user interviews", "survey responses", "sprints"]),
    # Close to the real data on purpose: hard negatives for bookstore questions.
    ("store_ops", "store operations", ["pop-up events", "author signings", "book fairs", "shelf audits",
                                       "price changes", "display plans", "supplier visits", "staff shifts"]),
    ("analytics", "analytics", ["daily kpis", "cohort snapshots", "funnel metrics", "forecast runs",
                                "dashboard views", "data quality checks", "etl jobs", "model scores"]),
]
REGIONS = ["", "emea", "apac", "americas"]
YEARS = ["", "2021", "2022", "2023", "2024", "2025"]
ENVIRONMENTS = ["", "staging", "backup"]
COLUMNS = {
    "status": "Current status of the {one}: open, approved, closed or cancelled.",
    "created_at": "When the {one} was created.",
    "amount": "Amount in US dollars.",
    "owner": "Employee responsible for the {one}.",
    "region": "Region the {one} belongs to.",
    "priority": "Priority from 1 (highest) to 5.",
    "notes": "Free-text notes.",
}


def synthetic_tables(n: int, databases: list[str], seed: int = 42) -> dict[str, list[TableInfo]]:
    """`n` made-up tables, dealt round-robin to `databases`."""
    rng = random.Random(seed)
    variants = [(r, y, e) for e in ENVIRONMENTS for y in YEARS for r in REGIONS]
    combos = [(d, v) for v in variants for d in range(len(DOMAINS))]
    if n > len(combos) * 8:
        raise ValueError(f"at most {len(combos) * 8} synthetic tables")
    rng.shuffle(combos)
    out: dict[str, list[TableInfo]] = {db: [] for db in databases}
    count = 0
    for d, (region, year, env) in combos:
        prefix, title, entities = DOMAINS[d]
        suffix = "".join(f"_{part}" for part in (region, year, env) if part)
        where = "".join([f" in {region.upper() if region != 'americas' else 'the Americas'}" if region else "",
                         f" for {year}" if year else "", " (staging copy)" if env == "staging" else "",
                         " (backup copy)" if env == "backup" else ""])
        parent = f"{prefix}_{entities[0].replace(' ', '_')}{suffix}"
        parent_key = f"{entities[0].rstrip('s').replace(' ', '_')}_id"
        for i, entity in enumerate(entities):
            if count == n:
                return out
            one = entity[:-1] if entity.endswith("s") else entity
            key = f"{one.replace(' ', '_')}_id"
            picked = rng.sample(list(COLUMNS), rng.randint(3, 6))
            columns = [ColumnInfo(name=key, type="INTEGER", description=f"Primary key of the {one}.")]
            fks: list[str] = []
            if i:  # every table of a group points at the group's first table
                columns.append(ColumnInfo(name=parent_key, type="INTEGER", description=f"The {entities[0][:-1]}."))
                fks.append(f"{parent_key} -> {parent}.{parent_key}")
            columns += [ColumnInfo(name=c, type="TEXT", description=COLUMNS[c].format(one=one)) for c in picked]
            out[databases[count % len(databases)]].append(TableInfo(
                name=f"{prefix}_{entity.replace(' ', '_')}{suffix}", columns=columns, primary_key=[key],
                foreign_keys=fks, row_count=rng.randint(100, 100_000),
                description=f"{entity.capitalize()} recorded by the {title} team{where}, one row per {one}.",
                questions=[f"How many {entity} were created in {year or '2024'}{where}?",
                           f"What is the total amount of {entity} by status{where}?"],
            ))
            count += 1
    return out


class CachingEmbedder:
    """Embeds each text once (so every store gets identical vectors) and keeps count."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._inner = SentenceEmbedder(model_name)
        self._cache: dict[str, np.ndarray] = {}
        self.requested = 0  # texts the index asked to embed
        self.seconds = 0.0  # time spent embedding texts never seen before

    def encode(self, texts: list[str]) -> np.ndarray:
        self.requested += len(texts)
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            started = time.perf_counter()
            self._cache.update(zip(new, self._inner.encode(new)))
            self.seconds += time.perf_counter() - started
        return np.stack([self._cache[t] for t in texts])


def build(embedder: CachingEmbedder, store: VectorStore, schemas: dict[str, list[TableInfo]],
          descriptions: dict[str, DatabaseDescription]) -> tuple[SchemaIndex, float, int]:
    """An index over `schemas`; returns it, the seconds it took and the texts embedded."""
    before = embedder.requested
    started = time.perf_counter()
    index = SchemaIndex(embedder, store)
    for db, tables in schemas.items():
        index.add_database(db, tables, descriptions.get(db))
    return index, time.perf_counter() - started, embedder.requested - before


def evaluate(index: SchemaIndex, questions: list[dict[str, Any]], kinds: dict[str, str]) -> dict[str, float]:
    for item in questions:
        index.embed(item["q"])  # cached: time the store, not the embedding model
    found, millis, tokens = 0, [], []
    for item in questions:
        started = time.perf_counter()
        selection = index.retrieve(item["q"], item["db"], kind=kinds[item["db"]])
        millis.append((time.perf_counter() - started) * 1000)
        found += set(item["tables"]) <= set(selection.names)
        tokens.append(estimate_tokens(format_schema(selection.tables, kinds[item["db"]],
                                                    compact=selection.compact, joins=selection.joins)))
    return {"recall": found / len(questions), "ms": statistics.median(millis), "tokens": statistics.mean(tokens)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 3000],
                        help="synthetic tables to add (default: 300 1000 3000)")
    parser.add_argument("--qdrant-url", help="also measure a Qdrant server, e.g. http://localhost:6333")
    args = parser.parse_args()

    settings = get_settings()
    descriptions = load_descriptions().databases
    adapters = create_adapters(settings)
    real = {db: describe_tables(adapter.get_schema(), descriptions.get(db)) for db, adapter in adapters.items()}
    kinds = {db: adapter.kind for db, adapter in adapters.items()}
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    embedder = CachingEmbedder(settings.embedding_model)
    embedder.encode(["warm up"])  # load the model outside the timings

    rows = []
    for size in [0, *args.sizes]:
        extra = synthetic_tables(size, list(real))
        schemas = {db: real[db] + extra[db] for db in real}
        tables = sum(len(t) for t in schemas.values())
        every_table = statistics.mean(estimate_tokens(format_schema(schemas[q["db"]], kinds[q["db"]]))
                                      for q in questions)
        embedder.seconds = 0.0
        index, numpy_s, _ = build(embedder, NumpyVectorStore(), schemas, descriptions)
        embed_s, vectors = embedder.seconds, index.store.count()
        result = {"tables": tables, "vectors": vectors, "embed_s": embed_s, "every_table_tokens": every_table,
                  "numpy_index_s": numpy_s - embed_s, "numpy": evaluate(index, questions, kinds)}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qdrant"
            index, first_s, _ = build(embedder, QdrantVectorStore("bench", path=path), schemas, descriptions)
            index.store.close()
            index, restart_s, re_embedded = build(embedder, QdrantVectorStore("bench", path=path), schemas,
                                                  descriptions)
            result.update(qdrant_first_s=first_s, qdrant_restart_s=restart_s, qdrant_re_embedded=re_embedded,
                          qdrant=evaluate(index, questions, kinds))
            index.store.close()

        if args.qdrant_url:
            collection = f"talktable_bench_{size}"
            store = QdrantVectorStore(collection, url=args.qdrant_url)
            index, server_first_s, _ = build(embedder, store, schemas, descriptions)
            index, server_restart_s, _ = build(embedder, QdrantVectorStore(collection, url=args.qdrant_url),
                                               schemas, descriptions)
            result.update(server_first_s=server_first_s, server_restart_s=server_restart_s,
                          server=evaluate(index, questions, kinds))
            store._client.delete_collection(collection)  # benchmark data: don't leave it behind
            store.close()
            index.store.close()
        rows.append(result)
        print(_line(result), flush=True)

    print("\n" + _markdown(rows, bool(args.qdrant_url)))


def _line(r: dict[str, Any]) -> str:
    server = f", server {r['server']['ms']:.1f} ms" if "server" in r else ""
    return (f"{r['tables']:>5} tables / {r['vectors']:>6} vectors: embed {r['embed_s']:.1f} s | "
            f"recall {r['numpy']['recall']:.0%} (qdrant {r['qdrant']['recall']:.0%}) | "
            f"retrieve numpy {r['numpy']['ms']:.1f} ms, qdrant {r['qdrant']['ms']:.1f} ms{server} | "
            f"qdrant restart {r['qdrant_restart_s']:.1f} s ({r['qdrant_re_embedded']} re-embedded) | "
            f"schema ~{r['numpy']['tokens']:.0f} tokens vs ~{r['every_table_tokens']:.0f}")


def _markdown(rows: list[dict[str, Any]], server: bool) -> str:
    """Two tables: what the query writer gets, and what each vector store costs."""
    quality = ["| Tables | All needed tables found | Schema tokens in the prompt | Every table would be |",
               "|---:|---:|---:|---:|"]
    stores = ["NumPy", "Qdrant embedded"] + (["Qdrant server"] if server else [])
    head = ["Tables", "Vectors", "Start-up: NumPy (every start)",
            *[f"Start-up: {name} (first / restart)" for name in stores[1:]], *[f"Search: {name}" for name in stores]]
    cost = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
    for r in rows:
        quality.append(f"| {r['tables']:,} | {r['numpy']['recall']:.0%} | ~{r['numpy']['tokens']:,.0f} "
                       f"| ~{r['every_table_tokens']:,.0f} |")
        cells = [f"{r['tables']:,}", f"{r['vectors']:,}", f"{r['embed_s'] + r['numpy_index_s']:.1f} s",
                 f"{r['embed_s'] + r['qdrant_first_s']:.1f} s / {r['qdrant_restart_s']:.1f} s"]
        if server:
            cells.append(f"{r['embed_s'] + r['server_first_s']:.1f} s / {r['server_restart_s']:.1f} s")
        cells += [f"{r['numpy']['ms']:.1f} ms", f"{r['qdrant']['ms']:.1f} ms"]
        if server:
            cells.append(f"{r['server']['ms']:.1f} ms")
        cost.append("| " + " | ".join(cells) + " |")
    return "\n".join(quality) + "\n\n" + "\n".join(cost)


if __name__ == "__main__":
    main()
