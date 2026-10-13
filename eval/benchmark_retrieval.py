"""Benchmark schema retrieval on held-out questions (eval/retrieval_questions.yaml).

For every question it checks what the query writer would be shown:
  - routing:  does the embedding fallback pick the right database?
  - recall:   are all the tables the query needs in the selected schema?
  - prompt:   how big is the query writer's system prompt (vs. sending every table)?
and whether messages that aren't about the data stay below the relevance floor.

    python -m eval.benchmark_retrieval [--k 4] [--show-misses] [--json results.json]

Needs the three databases (it reads their real schemas) but no LLM. Token counts are
estimated as characters / 4.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from talktable.adapters import create_adapters
from talktable.config import get_settings
from talktable.descriptions import describe_tables, load_descriptions
from talktable.generator import QueryContext, build_messages, format_schema
from talktable.schema_index import RETRIEVE_K, SchemaIndex, SentenceEmbedder

QUESTIONS = Path(__file__).with_name("retrieval_questions.yaml")


def _tokens(text: str) -> int:
    return round(len(text) / 4)


def run(k: int) -> dict[str, Any]:
    settings = get_settings()
    descriptions = load_descriptions()
    adapters = create_adapters(settings)
    index = SchemaIndex(SentenceEmbedder(settings.embedding_model))
    started = time.perf_counter()
    for name, adapter in adapters.items():
        index.add_database(name, describe_tables(adapter.get_schema(), descriptions.databases.get(name)),
                           descriptions.databases.get(name))
    build_s = time.perf_counter() - started
    databases = index.databases

    spec = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in spec["questions"]:
        question, db, needed = item["q"], item["db"], set(item["tables"])
        description = descriptions.databases[db]
        ranked = index.rank_databases(question, databases)
        t0 = time.perf_counter()
        selection = index.retrieve(question, db, k=k, kind=adapters[db].kind,
                                   token_budget=settings.schema_token_budget)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        context = QueryContext(database=db, kind=adapters[db].kind, title=description.title,
                               tables=selection.tables, joins=selection.joins, compact=selection.compact,
                               few_shot=selection.examples)
        full = QueryContext(database=db, kind=adapters[db].kind, title=description.title,
                            tables=index.tables(db), few_shot=description.few_shot)
        selected = {t.name for t in context.tables}
        rows.append({
            "question": question, "db": db, "routed": ranked[0][0], "score": ranked[0][1],
            "needed": sorted(needed), "selected": [t.name for t in context.tables],
            "missing": sorted(needed - selected), "recall": len(needed & selected) / len(needed),
            "prompt_tokens": _tokens(build_messages(question, context)[0]["content"]),
            "full_prompt_tokens": _tokens(build_messages(question, full)[0]["content"]),
            "schema_tokens": _tokens(format_schema(context.tables, context.kind, compact=context.compact,
                                                   joins=context.joins)),
            "full_schema_tokens": _tokens(format_schema(full.tables, full.kind)),
            "joins": len(selection.joins), "bridges": sum(r == "bridge" for r in selection.roles.values()),
            "retrieve_ms": retrieve_ms,
        })
    off_topic = [(message, index.rank_databases(message, databases)[0][1]) for message in spec["off_topic"]]

    return {
        "k": k,
        "tables": sum(len(index.tables(db)) for db in databases),
        "index_build_s": build_s,
        "questions": len(rows),
        "routing_accuracy": sum(r["routed"] == r["db"] for r in rows) / len(rows),
        "full_recall": sum(not r["missing"] for r in rows) / len(rows),
        "table_recall": statistics.mean(r["recall"] for r in rows),
        "multi_table_full_recall": _share(r for r in rows if len(r["needed"]) > 1),
        "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in rows),
        "full_schema_prompt_tokens_mean": statistics.mean(r["full_prompt_tokens"] for r in rows),
        "schema_tokens_mean": statistics.mean(r["schema_tokens"] for r in rows),
        "full_schema_tokens_mean": statistics.mean(r["full_schema_tokens"] for r in rows),
        "tables_shown_mean": statistics.mean(len(r["selected"]) for r in rows),
        "questions_with_bridges": sum(r["bridges"] > 0 for r in rows),
        "questions_with_joins": sum(r["joins"] > 0 for r in rows),
        "retrieve_ms_mean": statistics.mean(r["retrieve_ms"] for r in rows),
        "lowest_question_score": min(r["score"] for r in rows),
        "highest_off_topic_score": max(score for _, score in off_topic),
        "rows": rows,
        "off_topic": off_topic,
    }


def _share(rows: Any) -> float:
    rows = list(rows)
    return sum(not r["missing"] for r in rows) / len(rows) if rows else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=RETRIEVE_K,
                        help=f"tables retrieved before join expansion (default {RETRIEVE_K})")
    parser.add_argument("--show-misses", action="store_true", help="list questions with missing tables")
    parser.add_argument("--json", type=Path, help="also write the full results here")
    args = parser.parse_args()

    results = run(args.k)
    print(f"{results['questions']} questions over {results['tables']} tables (k={results['k']}, "
          f"index built in {results['index_build_s']:.1f} s)")
    print(f"  routing (embedding fallback): {results['routing_accuracy']:.0%}")
    print(f"  all needed tables shown:      {results['full_recall']:.0%} "
          f"(multi-table questions: {results['multi_table_full_recall']:.0%})")
    print(f"  needed tables shown:          {results['table_recall']:.0%}")
    print(f"  tables shown per question:    {results['tables_shown_mean']:.1f}")
    print(f"  schema in the prompt:         ~{results['schema_tokens_mean']:.0f} tokens "
          f"(every table: ~{results['full_schema_tokens_mean']:.0f})")
    print(f"  whole query-writer prompt:    ~{results['prompt_tokens_mean']:.0f} tokens "
          f"(every table: ~{results['full_schema_prompt_tokens_mean']:.0f})")
    print(f"  join conditions given:        {results['questions_with_joins']} questions "
          f"({results['questions_with_bridges']} with bridge tables)")
    print(f"  retrieval time:               {results['retrieve_ms_mean']:.1f} ms")
    print(f"  relevance: lowest question {results['lowest_question_score']:.3f}, "
          f"highest off-topic {results['highest_off_topic_score']:.3f}")
    if args.show_misses:
        for row in results["rows"]:
            if row["missing"] or row["routed"] != row["db"]:
                print(f"  - {row['question']}  [{row['db']} -> routed {row['routed']} {row['score']:.2f}] "
                      f"missing {row['missing']} shown {row['selected']}")
        for message, score in sorted(results["off_topic"], key=lambda item: -item[1])[:3]:
            print(f"  ~ off-topic {score:.3f}: {message}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
