"""Generator: turn a question plus the relevant schema into a SQL query or Mongo pipeline.

Prompts are dialect-aware (SQLite vs PostgreSQL vs MongoDB) and include the worked
examples (`few_shot`) from descriptions.yaml. The LLM replies with JSON:
  SQL:   {"sql": "SELECT ..."}
  Mongo: {"collection": "...", "pipeline": [...]}

Nothing here is trusted: every generated query still goes through the guardrails.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from talktable.descriptions import FewShotExample, GlossaryEntry
from talktable.guardrails import ALLOWED_STAGES
from talktable.llm import ChatModel, LLMError, Message, parse_json_object
from talktable.models import ColumnInfo, DatabaseKind, TableInfo

GeneratedQuery = str | dict[str, Any]  # SQL text, or {"collection": ..., "pipeline": [...]}

_MAX_TOKENS = 700

_DIALECT_NOTES: dict[str, list[str]] = {
    "sqlite": [
        f"This is SQLite {sqlite3.sqlite_version}. Use only functions this version supports.",
        "Dates and years: strftime('%Y', col), strftime('%m', col), date(col). There is no YEAR() or DATE_TRUNC().",
        "Use SUBSTR (not SUBSTRING), || to join strings, and LIKE for text matching (it ignores case for ASCII).",
        "RIGHT JOIN and FULL JOIN are not available: rewrite them as LEFT JOINs.",
        "ROUND(x, 2) rounds a number to 2 decimals.",
    ],
    "postgres": [
        "This is PostgreSQL.",
        "Dates: date_trunc('month', col)::date, EXTRACT(YEAR FROM col)::int, to_char(col, 'YYYY-MM'). "
        "(date_trunc returns a timestamp and EXTRACT a decimal, so cast them as shown.) "
        "Filter ranges with col >= DATE '2025-01-01' AND col < DATE '2026-01-01'.",
        "Subtracting one DATE from another gives the number of days as an integer, e.g. "
        "AVG(delivered_date - shipped_date). Don't put EXTRACT around it: EXTRACT needs a timestamp or interval.",
        "Use ILIKE for case-insensitive matching. Cast with ::numeric or ::int when needed; "
        "ROUND(x, 2) needs a numeric argument, e.g. ROUND(AVG(total_amount)::numeric, 2).",
        "Every non-aggregated column in SELECT must appear in GROUP BY.",
    ],
}

_SQL_SYSTEM = """You are an expert data analyst who writes {dialect_name} queries for an online bookstore.
Write ONE read-only SQL query that answers the user's question.

Rules:
- Use only the tables and columns in the schema below. Never invent columns.
- Write a single SELECT statement (WITH ... SELECT is fine). Never modify data.
- Always end with a LIMIT of at most 200. Use a smaller LIMIT when the question asks for "top N".
- Give computed columns short, readable aliases.
- Add only the filters the question asks for. Don't exclude rows (e.g. by status) on your own.
- When a table holds IDs that refer to another database (marked "the catalog's" / "sales"),
  return the ID: that other database can't be joined here.
- When the schema lists join conditions, use them to connect the tables.
{dialect_notes}
- Today's date is {today}.

Schema:
{schema}

Examples:
{examples}

Reply with JSON only: {{"sql": "<your query>"}}"""

_MONGO_SYSTEM = """You are an expert data analyst who writes MongoDB aggregation pipelines for an online bookstore.
Write ONE aggregation pipeline that answers the user's question.

Rules:
- Use only the collections and fields in the schema below. Field names are case-sensitive.
- Allowed stages: {stages}. Never use $out, $merge, $where, $function or $accumulator.
- Always end the pipeline with {{"$limit": N}}, N at most 200. Use a smaller N for "top N" questions.
- To work with elements of an array field (such as events), $unwind it first.
- Dates are BSON dates. Compare them using Extended JSON, e.g.
  {{"created_at": {{"$gte": {{"$date": "2025-01-01T00:00:00Z"}}}}}}.
- $group puts the grouping key in "_id".
- Add only the filters the question asks for.
- Today's date is {today}.

Schema:
{schema}

Examples:
{examples}

Reply with JSON only: {{"collection": "<collection name>", "pipeline": [<stages>]}}"""

_CORRECTION = """That query failed with this error:
{error}

Fix the query so that it answers the original question, following all the rules.
Reply with JSON only, in the same format as before."""


@dataclass(frozen=True)
class QueryContext:
    """Everything the generator needs to know about the target database."""

    database: str  # adapter name: sqlite | postgres | mongo
    kind: DatabaseKind
    title: str
    tables: list[TableInfo]  # the retrieved, described tables/collections
    few_shot: list[FewShotExample] = field(default_factory=list)
    glossary: list[GlossaryEntry] = field(default_factory=list)  # business terms the question uses
    joins: list[str] = field(default_factory=list)  # join conditions between the tables
    compact: frozenset[str] = frozenset()  # tables to show briefly (bridges and lookups)


# Column notes kept even in the brief form: those saying an ID belongs to another
# database, and those quoting values (filters need the exact spelling: 'United States').
_KEEP_NOTE = re.compile(r"the catalog's|different database|not joinable|'[^']+'", re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English and code)."""
    return (len(text) + 3) // 4


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    head, dot, _ = text.partition(". ")
    return head + "." if dot else text


def format_schema(tables: list[TableInfo], kind: DatabaseKind, *, compact: frozenset[str] = frozenset(),
                  joins: list[str] | tuple[str, ...] = ()) -> str:
    """Schema block for the prompt. Tables are sorted by name, so the same tables always
    give the same text (local models can reuse their prompt cache). Tables in `compact`
    get one line each, after the others; `joins` are listed at the end."""
    noun, unit = ("Collection", "documents") if kind == "mongo" else ("Table", "rows")
    blocks: list[str] = []
    brief: list[str] = []
    for table in sorted(tables, key=lambda t: t.name):
        count = f", {table.row_count:,} {unit}" if table.row_count is not None else ""
        fks = {fk.split("->")[0].strip(): fk.split("->")[1].strip() for fk in table.foreign_keys}

        def keys(column: ColumnInfo) -> list[str]:
            return (["primary key"] if column.name in table.primary_key else []) + (
                [f"references {fks[column.name]}"] if column.name in fks else [])

        if table.name in compact:  # one line: name, first sentence, columns with types and keys
            columns = []
            for column in table.columns:
                notes = keys(column)
                if column.description and _KEEP_NOTE.search(column.description):
                    notes.append(" ".join(column.description.split()))
                columns.append(f"{column.name} {column.type}" + (f" ({'; '.join(notes)})" if notes else ""))
            summary = " ".join(filter(None, [_first_sentence(table.description or ""), "Columns:"]))
            brief.append(f"- {table.name}{count}: {summary} {', '.join(columns)}")
            continue
        lines = [f"{noun} {table.name}{count}: {' '.join((table.description or '').split())}"]
        for column in table.columns:
            extra = f" ({', '.join(keys(column))})" if keys(column) else ""
            description = " ".join((column.description or "").split())
            lines.append(f"  - {column.name} {column.type}{extra}: {description}".rstrip(": "))
        blocks.append("\n".join(lines))
    if brief:
        blocks.append(f"Related {noun.lower()}s (shown briefly):\n" + "\n".join(brief))
    if joins:
        blocks.append("Join conditions:\n" + "\n".join(f"- {condition}" for condition in joins))
    return "\n\n".join(blocks)


def _format_examples(examples: list[FewShotExample], kind: DatabaseKind) -> str:
    rendered = []
    for example in examples:
        if kind == "mongo":
            answer = json.dumps(json.loads(example.query), separators=(", ", ": "))
        else:
            answer = json.dumps({"sql": " ".join(example.query.split())})
        rendered.append(f"Question: {example.question}\nAnswer: {answer}")
    return "\n\n".join(rendered) or "(none)"


def build_messages(question: str, context: QueryContext, today: date | None = None) -> list[Message]:
    """The system prompt (rules, schema, examples) and the user question."""
    today_text = (today or date.today()).isoformat()
    schema = format_schema(context.tables, context.kind, compact=context.compact, joins=context.joins)
    examples = _format_examples(context.few_shot, context.kind)
    if context.kind == "mongo":
        system = _MONGO_SYSTEM.format(stages=", ".join(sorted(ALLOWED_STAGES)), today=today_text,
                                      schema=schema, examples=examples)
    else:
        notes = _DIALECT_NOTES.get(context.database, [])
        system = _SQL_SYSTEM.format(
            dialect_name="SQLite" if context.database == "sqlite" else "PostgreSQL",
            dialect_notes="\n".join(f"- {note}" for note in notes),
            today=today_text, schema=schema, examples=examples,
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": _with_definitions(question, context)}]


def _with_definitions(question: str, context: QueryContext) -> str:
    """The question, plus the meaning of any business terms it uses. They go in the user
    message, after the long system prompt, so that prompt stays identical (and cached)."""
    if not context.glossary:
        return question
    definitions = "\n".join(f"- {entry.term}: {' '.join(entry.meaning.split())}" for entry in context.glossary)
    return f"{question}\n\nUse these definitions:\n{definitions}"


def query_to_text(query: GeneratedQuery) -> str:
    """A generated query as text (for logs, retries, and display)."""
    return query if isinstance(query, str) else json.dumps(query, default=str)


def _as_reply_json(query: GeneratedQuery, kind: DatabaseKind) -> str:
    """The query in the JSON format the model was asked for (used to replay its answer)."""
    if kind == "mongo":
        return query_to_text(query)
    return json.dumps({"sql": query})


def parse_reply(text: str, kind: DatabaseKind) -> GeneratedQuery:
    """Extract the query from the model's JSON reply.

    Raises:
        LLMError: the reply doesn't contain a usable query.
    """
    data = parse_json_object(text)
    if kind == "mongo":
        inner = data.get("query")
        if isinstance(inner, dict) and "pipeline" in inner:  # {"query": {...}} wrapper
            data = inner
        if "pipeline" not in data or "collection" not in data:
            raise LLMError("The language model's answer had no collection/pipeline.")
        return {"collection": data["collection"], "pipeline": data["pipeline"]}
    sql = data.get("sql") or data.get("query")
    if not isinstance(sql, str) or not sql.strip():
        raise LLMError("The language model's answer had no SQL query.")
    return sql.strip()


class Generator:
    def __init__(self, llm: ChatModel, today: date | None = None) -> None:
        self._llm = llm
        self._today = today

    def generate(self, question: str, context: QueryContext) -> GeneratedQuery:
        """First attempt at a query for `question`."""
        messages = build_messages(question, context, self._today)
        return parse_reply(self._llm.complete(messages, json_mode=True, max_tokens=_MAX_TOKENS), context.kind)

    def correct(self, question: str, context: QueryContext, failed: GeneratedQuery, error: str) -> GeneratedQuery:
        """A corrected query, given the one that failed and the error it produced."""
        messages = build_messages(question, context, self._today)
        messages += [
            {"role": "assistant", "content": _as_reply_json(failed, context.kind)},
            {"role": "user", "content": _CORRECTION.format(error=error)},
        ]
        return parse_reply(self._llm.complete(messages, json_mode=True, max_tokens=_MAX_TOKENS), context.kind)
