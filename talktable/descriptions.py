"""Load data/descriptions.yaml: plain-English descriptions of every database, table and column.

The YAML uses `tables`/`columns` for SQL databases and `collections`/`fields` for
MongoDB. This module normalises both into the same `objects`/`fields` shape.
"""

from __future__ import annotations

import re

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from talktable.config import DESCRIPTIONS_PATH
from talktable.models import ColumnInfo, DatabaseKind, TableInfo


class FewShotExample(BaseModel):
    question: str
    query: str  # SQL text, or a Mongo query as (Extended) JSON text


class ObjectDescription(BaseModel):
    """A table or collection."""

    description: str
    fields: dict[str, str]  # column/field name -> description
    questions: list[str] = Field(default_factory=list)  # sample questions it answers (helps retrieval)


class DatabaseDescription(BaseModel):
    title: str
    kind: DatabaseKind
    summary: str
    examples: list[str] = Field(default_factory=list)
    few_shot: list[FewShotExample] = Field(default_factory=list)
    objects: dict[str, ObjectDescription]  # table/collection name -> description


class GlossaryEntry(BaseModel):
    """A business term ("revenue") and what it means in this data."""

    term: str
    also: list[str] = Field(default_factory=list)  # other spellings and synonyms
    databases: list[str] = Field(default_factory=list)  # where the definition applies; empty = everywhere
    meaning: str

    def mentioned_in(self, text: str) -> bool:
        """Whole-word, case-insensitive match of the term or a synonym (plurals count too)."""
        return any(re.search(rf"\b{re.escape(word)}(?:s|es)?\b", text, re.IGNORECASE)
                   for word in (self.term, *self.also))


class Descriptions(BaseModel):
    shared_keys: dict[str, str] = Field(default_factory=dict)
    routing_notes: list[str] = Field(default_factory=list)  # extra hints for the router prompt
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    databases: dict[str, DatabaseDescription]

    def glossary_for(self, question: str, database: str) -> list[GlossaryEntry]:
        """The glossary entries a question uses that apply to `database`."""
        return [entry for entry in self.glossary
                if (not entry.databases or database in entry.databases) and entry.mentioned_in(question)]

    def glossary_terms(self, database: str) -> list[str]:
        return [entry.term for entry in self.glossary if not entry.databases or database in entry.databases]


def _normalise(raw_db: dict[str, Any]) -> dict[str, Any]:
    objects = raw_db.pop("tables", None) or raw_db.pop("collections", None) or {}
    raw_db["objects"] = {
        name: {"description": obj["description"], "fields": obj.get("columns") or obj.get("fields") or {},
               "questions": obj.get("questions") or []}
        for name, obj in objects.items()
    }
    return raw_db


@lru_cache(maxsize=4)
def load_descriptions(path: Path = DESCRIPTIONS_PATH) -> Descriptions:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    databases = {name: _normalise(dict(spec)) for name, spec in raw["databases"].items()}
    return Descriptions(shared_keys=raw.get("shared_keys") or {}, routing_notes=raw.get("routing_notes") or [],
                        glossary=raw.get("glossary") or [], databases=databases)


def describe_tables(tables: list[TableInfo], description: DatabaseDescription | None) -> list[TableInfo]:
    """Copies of `tables` with descriptions filled in from the YAML (where available)."""
    if description is None:
        return [t.model_copy(deep=True) for t in tables]
    described: list[TableInfo] = []
    for table in tables:
        obj = description.objects.get(table.name)
        field_text = obj.fields if obj else {}
        described.append(table.model_copy(update={
            "description": obj.description.strip() if obj else None,
            "questions": list(obj.questions) if obj else [],
            "columns": [
                ColumnInfo(name=c.name, type=c.type, description=field_text.get(c.name, "").strip() or None)
                for c in table.columns
            ],
        }))
    return described
