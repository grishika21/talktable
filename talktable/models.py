"""Pydantic models shared across TalkTable: schemas, queries and results."""

from __future__ import annotations

from typing import Any, Literal

from bson import json_util
from pydantic import BaseModel, Field

DatabaseKind = Literal["sql", "mongo"]


class ColumnInfo(BaseModel):
    """A column of a SQL table, or a (possibly nested, dotted) field of a Mongo collection."""

    name: str
    type: str
    description: str | None = None


class TableInfo(BaseModel):
    """A SQL table or a Mongo collection."""

    name: str
    columns: list[ColumnInfo]
    row_count: int | None = None
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[str] = Field(default_factory=list)  # e.g. "author_id -> authors.author_id"
    description: str | None = None
    questions: list[str] = Field(default_factory=list)  # sample questions (from descriptions.yaml)


class MongoQuery(BaseModel):
    """A validated MongoDB aggregation: which collection, and the pipeline to run on it."""

    collection: str
    pipeline: list[dict[str, Any]]

    def to_json(self) -> str:
        """Extended JSON, so values such as dates survive the round trip."""
        return json_util.dumps({"collection": self.collection, "pipeline": self.pipeline})


class QueryResult(BaseModel):
    """Rows from a read-only query. `query` is what actually ran, after the guardrails;
    `truncated` means the rows were capped."""

    query: str  # the query that actually ran, after guardrails (SQL, or Mongo extended JSON)
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool  # True when the row cap was reached, so more rows may exist
    elapsed_ms: int


class DatabaseStatus(BaseModel):
    """A database, what it holds, and whether it is online (`detail` says why not)."""

    name: str
    title: str
    kind: DatabaseKind
    online: bool
    description: str = ""  # what's in it (from descriptions.yaml)
    tables: list[str] = Field(default_factory=list)
    detail: str | None = None  # why it is offline


class AskResult(BaseModel):
    """The answer to a plain-English question: the explanation, the database(s) and
    query used, and the result rows (in `columns` order). `error` is set, and `answer`
    says why, when no answer could be found.
    """

    question: str
    databases_used: list[str]
    queries: list[str]  # SQL strings or JSON pipelines
    rows: list[dict[str, Any]]
    answer: str
    retries: int
    latency_ms: int
    columns: list[str] = Field(default_factory=list)
    error: str | None = None
    interpreted_as: str | None = None  # a follow-up rewritten as a standalone question
    cached: bool = False  # the query came from the answer cache (and was re-run on current data)
