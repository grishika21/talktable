"""MongoDB adapter (the engagement database: Atlas in production, Docker locally).

Read-only in three layers:
  1. the guardrails (in `run()`): an allowlist of aggregation stages, no $out/$merge,
     no server-side JavaScript, output capped with $limit;
  2. the adapter only ever calls `aggregate()` (and `find()` to sample the schema),
     always with maxTimeMS;
  3. the database user itself should only have the `read` role (see README).
"""

from __future__ import annotations

import logging
from datetime import datetime
from itertools import islice
from typing import Any, ClassVar

from bson import Decimal128, ObjectId
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ExecutionTimeout, OperationFailure, PyMongoError

from talktable.adapters.base import DatabaseAdapter, DatabaseUnavailableError, QueryError, QueryTimeoutError
from talktable.guardrails import validate_mongo
from talktable.models import ColumnInfo, DatabaseKind, MongoQuery, TableInfo

logger = logging.getLogger(__name__)

_SCHEMA_SAMPLE_SIZE = 200


class MongoAdapter(DatabaseAdapter[MongoQuery]):
    name: ClassVar[str] = "mongo"
    kind: ClassVar[DatabaseKind] = "mongo"

    def __init__(self, url: str, db_name: str, *, server_selection_timeout_ms: int = 5000, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._url = url  # contains the password: never log it
        self.db_name = db_name
        self._server_selection_timeout_ms = server_selection_timeout_ms
        self._client: MongoClient[dict[str, Any]] | None = None
        self._collections: frozenset[str] | None = None

    def _db(self) -> Database[dict[str, Any]]:
        if self._client is None:
            self._client = MongoClient(
                self._url,
                serverSelectionTimeoutMS=self._server_selection_timeout_ms,
                connectTimeoutMS=self._server_selection_timeout_ms,
                appname="talktable",
            )
        return self._client[self.db_name]

    def collection_names(self) -> frozenset[str]:
        """Queryable collections (system.* excluded), read once and cached."""
        if self._collections is None:
            try:
                names = self._db().list_collection_names()
            except PyMongoError as exc:
                raise self._translate(exc) from None
            self._collections = frozenset(n for n in names if not n.startswith("system."))
        return self._collections

    def validate(self, query: Any, limit: int | None = None) -> MongoQuery:
        return validate_mongo(query, allowed_collections=self.collection_names(), max_rows=self.row_cap(limit))

    def query_text(self, safe_query: MongoQuery) -> str:
        return safe_query.to_json()

    def _execute(self, safe_query: MongoQuery, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
        try:
            cursor = self._db()[safe_query.collection].aggregate(
                safe_query.pipeline,
                maxTimeMS=round(self.timeout_s * 1000),
                allowDiskUse=False,
            )
            with cursor:
                docs = list(islice(cursor, limit))
        except PyMongoError as exc:
            raise self._translate(exc) from None
        columns: dict[str, None] = {}  # ordered set of keys, in order of first appearance
        for doc in docs:
            columns.update(dict.fromkeys(doc))
        return list(columns), docs

    def _translate(self, exc: PyMongoError) -> QueryError:
        if isinstance(exc, ExecutionTimeout):
            return QueryTimeoutError(f"The query took longer than {self.timeout_s:g}s and was cancelled.")
        if isinstance(exc, ConnectionFailure):  # includes ServerSelectionTimeoutError
            return DatabaseUnavailableError("Could not connect to MongoDB.")
        if isinstance(exc, OperationFailure):
            details = exc.details or {}
            return QueryError(str(details.get("errmsg") or exc))
        return QueryError(str(exc).splitlines()[0])

    def get_schema(self) -> list[TableInfo]:
        tables: list[TableInfo] = []
        try:
            db = self._db()
            for name in sorted(self.collection_names()):
                collection = db[name]
                sample = list(collection.find({}, limit=_SCHEMA_SAMPLE_SIZE, max_time_ms=round(self.timeout_s * 1000)))
                tables.append(TableInfo(
                    name=name,
                    columns=[ColumnInfo(name=path, type=kind) for path, kind in infer_fields(sample).items()],
                    row_count=collection.estimated_document_count(),
                ))
        except PyMongoError as exc:
            raise self._translate(exc) from None
        return tables

    def ping(self) -> bool:
        try:
            self._db().command("ping")
            return True
        except PyMongoError as exc:
            logger.warning("mongo is unavailable: %s", type(exc).__name__)
            return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _bson_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, datetime):
        return "date"
    if isinstance(value, ObjectId):
        return "objectId"
    if isinstance(value, Decimal128):
        return "decimal"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def infer_fields(docs: list[dict[str, Any]]) -> dict[str, str]:
    """Field paths and their BSON types, e.g. {'events': 'array', 'events.ts': 'date'}.

    Fields of documents nested in arrays or sub-documents get dotted paths, which is
    also how they are addressed in queries. `_id` is left out.
    """
    types: dict[str, set[str]] = {}

    def visit(doc: dict[str, Any], prefix: str) -> None:
        for key, value in doc.items():
            if not prefix and key == "_id":
                continue
            path = f"{prefix}{key}"
            types.setdefault(path, set()).add(_bson_type(value))
            if isinstance(value, dict):
                visit(value, f"{path}.")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        visit(item, f"{path}.")

    for doc in docs:
        visit(doc, "")
    return {path: "|".join(sorted(kinds)) for path, kinds in types.items()}
