"""Read-only guardrails: every query is validated (and if needed rewritten) before it runs.

Nothing reaches a database without passing through `validate_sql` or `validate_mongo`.
Both raise `GuardrailError` with a short reason that is safe to show to users and is
fed back to the LLM so it can fix its query.

SQL (parsed with sqlglot in the target dialect):
  1. exactly one statement;
  2. the root must be a SELECT, or a set operation such as UNION (a WITH ... SELECT
     parses as a SELECT);
  3. no write/DDL/admin node anywhere in the tree. This catches data-modifying CTEs,
     SELECT ... INTO, FOR UPDATE, etc.;
  4. no dangerous functions (pg_sleep, pg_read_file, lo_import, load_extension, ...);
  5. optionally, only known tables (plus the query's own CTEs);
  6. the outer LIMIT is at most `max_rows` (injected or lowered).
The SQL that runs is regenerated from the validated syntax tree with comments removed,
so what executes is exactly what was checked. A final token scan of that text
double-checks the function denylist.

Mongo (aggregation pipelines):
  1. only the stages listed in ALLOWED_STAGES (checked recursively in $facet/$lookup);
  2. no $out, $merge, $function, $where, $accumulator, or JavaScript code anywhere;
  3. the collection (and every $lookup source) must exist;
  4. the final output is capped with $limit <= `max_rows`.

These checks are the first layer. The adapters add more: read-only connections and
users, statement timeouts, and SQLite's authorizer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Iterator, Mapping
from typing import Any, Literal

import sqlglot
from bson import json_util
from bson.code import Code
from sqlglot import exp
from sqlglot.dialects.dialect import rename_func
from sqlglot.dialects.sqlite import SQLite
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

from talktable.models import MongoQuery

# sqlglot logs a warning for every statement it can't fully parse ("Falling back to
# parsing as a 'Command'"). We reject those statements anyway, so silence the noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

SqlDialect = Literal["sqlite", "postgres"]

DEFAULT_MAX_ROWS = 200
MAX_QUERY_CHARS = 20_000
MAX_PIPELINE_STAGES = 50
MAX_NESTING_DEPTH = 32


class GuardrailError(ValueError):
    """A query was rejected. The message says why, in plain words."""


# --- SQL ---------------------------------------------------------------------


class _SQLite(SQLite):
    """sqlglot's SQLite dialect, but emitting SUBSTR(), which every SQLite version has.

    sqlglot writes SUBSTRING(), which only exists in SQLite >= 3.34. Some Python
    builds (e.g. on Ubuntu 20.04) still bundle 3.31.
    """

    class Generator(SQLite.Generator):
        TRANSFORMS = {**SQLite.Generator.TRANSFORMS, exp.Substring: rename_func("SUBSTR")}


_GENERATE_DIALECT: dict[str, Any] = {"sqlite": _SQLite, "postgres": "postgres"}

# Schemas a table may be qualified with. Anything else (pg_catalog, information_schema,
# an ATTACHed database, ...) is rejected when a table allowlist is in use.
_ALLOWED_SCHEMAS: dict[str, frozenset[str]] = {
    "sqlite": frozenset({"main"}),
    "postgres": frozenset({"public"}),
}

# Syntax-tree nodes that must never appear anywhere in a query.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,          # writes (also inside CTEs)
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,     # DDL
    exp.Grant, exp.Revoke,                                   # permissions
    exp.Copy, exp.Attach, exp.Detach, exp.Pragma,            # file / database access
    exp.Set, exp.Use, exp.Transaction, exp.Commit, exp.Rollback, exp.Analyze,
    exp.Command,                                             # anything sqlglot couldn't parse
    exp.Into,                                                # SELECT ... INTO new_table
    exp.Lock,                                                # SELECT ... FOR UPDATE / SHARE
)

# Functions that can sleep, touch files, change settings or sequences, run dynamic SQL,
# or allocate huge amounts of memory. Lower-case. Applied to every dialect.
_DENIED_FUNCTIONS = frozenset({
    # PostgreSQL
    "set_config", "nextval", "setval",
    "query_to_xml", "query_to_xmlschema", "query_to_xml_and_xmlschema",
    "cursor_to_xml", "cursor_to_xmlschema",
    "table_to_xml", "table_to_xmlschema", "table_to_xml_and_xmlschema",
    "schema_to_xml", "schema_to_xmlschema", "schema_to_xml_and_xmlschema",
    "database_to_xml", "database_to_xmlschema", "database_to_xml_and_xmlschema",
    "txid_current",
    # SQLite
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "randomblob", "zeroblob",
})
# Whole families: pg_* (sleep, file access, backend control, advisory locks, ...),
# lo_* (large objects, which read/write server files), dblink* (connect elsewhere),
# pragma_* (SQLite PRAGMA table-valued functions).
_DENIED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "pragma_")
_ALLOWED_PG_FUNCTIONS = frozenset({"pg_typeof"})


def is_denied_function(name: str) -> bool:
    """True if calling a function with this name is never allowed."""
    lowered = name.lower()
    if lowered in _ALLOWED_PG_FUNCTIONS:
        return False
    return lowered in _DENIED_FUNCTIONS or lowered.startswith(_DENIED_FUNCTION_PREFIXES)


def validate_sql(
    sql: str,
    dialect: SqlDialect,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    allowed_tables: Collection[str] | None = None,
) -> str:
    """Check that `sql` is a single read-only SELECT and return the SQL to execute.

    The returned SQL is regenerated from the parsed query: comments removed and the
    outer LIMIT capped at `max_rows`. If `allowed_tables` is given, the query may only
    read those tables (and its own CTEs).

    Raises:
        GuardrailError: the query is not allowed. The message explains why.
    """
    if dialect not in _GENERATE_DIALECT:
        raise ValueError(f"unsupported SQL dialect: {dialect!r}")
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if not isinstance(sql, str) or not sql.strip():
        raise GuardrailError("The query is empty.")
    if len(sql) > MAX_QUERY_CHARS:
        raise GuardrailError(f"The query is too long (over {MAX_QUERY_CHARS:,} characters).")

    root = _parse_single_statement(sql, dialect)
    _check_root_is_select(root)
    _check_forbidden_nodes(root)
    _check_functions(root)
    if allowed_tables is not None:
        _check_tables(root, dialect, allowed_tables)
    _enforce_limit(root, max_rows)

    safe_sql = root.sql(dialect=_GENERATE_DIALECT[dialect], comments=False)
    _verify_output(safe_sql, dialect)
    return safe_sql


def format_sql(sql: str, dialect: SqlDialect) -> str:
    """Pretty-print SQL for display (one clause per line). Display only: what runs is
    always validate_sql()'s output. Returns `sql` unchanged if it doesn't parse."""
    try:
        return sqlglot.parse_one(sql, read=dialect).sql(dialect=_GENERATE_DIALECT[dialect], pretty=True)
    except (sqlglot.errors.SqlglotError, KeyError):
        return sql


def _parse_single_statement(sql: str, dialect: SqlDialect) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:
        raise GuardrailError(f"The SQL could not be parsed: {_parse_error_summary(exc)}") from None
    if not statements:
        raise GuardrailError("The query is empty.")
    if len(statements) > 1:
        raise GuardrailError(
            f"Only one SQL statement is allowed, but found {len(statements)} "
            "(separated by ';')."
        )
    root = statements[0]
    # "(SELECT ...)" parses as a Subquery wrapper: look inside it.
    while isinstance(root, exp.Subquery) and not root.alias:
        root = root.this
    return root


def _parse_error_summary(exc: SqlglotError) -> str:
    """'Invalid expression / Unexpected token (line 1, column 11)', without sqlglot internals."""
    text = (str(exc).strip().splitlines() or ["syntax error"])[0]
    text = re.sub(r"<Token token_type: TokenType\.SENTINEL[^>]*>", "the end of the query", text)
    text = re.sub(r"<Token token_type: [^,]+, text: ([^,]*),[^>]*>", r"'\1'", text)
    text = re.sub(r"<class 'sqlglot\.[\w.]*\.(\w+)'>", lambda m: m[1].upper(), text)
    location = re.search(r"\.?\s*Line (\d+), Col: (\d+)\.?", text)
    if location:
        text = f"{text[:location.start()]} (line {location[1]}, column {location[2]})"
    return text


def _describe(node: exp.Expression) -> str:
    """A short human label for a statement or node, e.g. 'DELETE' or 'EXPLAIN'."""
    if isinstance(node, exp.Command):
        return str(node.this).upper()
    if isinstance(node, exp.Into):
        return "SELECT ... INTO"
    if isinstance(node, exp.Lock):
        return "FOR UPDATE/SHARE locking"
    if isinstance(node, exp.TruncateTable):
        return "TRUNCATE"
    return type(node).__name__.upper()


def _check_root_is_select(root: exp.Expression) -> None:
    if isinstance(root, (exp.Select, exp.SetOperation)):
        return
    if isinstance(root, (*_FORBIDDEN_NODES, exp.Values)):
        raise GuardrailError(f"Only SELECT queries are allowed, not {_describe(root)}.")
    # Parser quirks: e.g. "LISTEN x" parses as an alias, "REINDEX" as a column.
    raise GuardrailError("Only SELECT queries are allowed.")


def _check_forbidden_nodes(root: exp.Expression) -> None:
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise GuardrailError(f"{_describe(node)} is not allowed: TalkTable is read-only.")


def _function_name(node: exp.Func) -> str:
    return node.name if isinstance(node, exp.Anonymous) else node.sql_name()


def _check_functions(root: exp.Expression) -> None:
    for node in root.find_all(exp.Func):
        name = _function_name(node)
        if is_denied_function(name):
            raise GuardrailError(f"The function {name.lower()}() is not allowed.")


def _check_tables(root: exp.Expression, dialect: SqlDialect, allowed_tables: Collection[str]) -> None:
    allowed = {t.lower() for t in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue  # a table-valued function such as generate_series(): checked by _check_functions
        name = table.name.lower()
        schema = table.db.lower()
        if table.catalog or (schema and schema not in _ALLOWED_SCHEMAS[dialect]):
            qualified = ".".join(p for p in (table.catalog, table.db, table.name) if p)
            raise GuardrailError(f"Table {qualified} is outside this database's tables and can't be queried.")
        if name not in allowed and name not in cte_names:
            raise GuardrailError(
                f"Unknown table '{table.name}'. Available tables: {', '.join(sorted(allowed))}."
            )


def _literal_int(node: exp.Expression | None) -> int | None:
    """The value of a plain integer literal, else None (expressions, ALL, -1, ...)."""
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(node.this)
        except ValueError:
            return None
    return None


def _enforce_limit(root: exp.Expression, max_rows: int) -> None:
    """Make sure the outermost query returns at most `max_rows` rows."""
    limit = root.args.get("limit")
    if isinstance(limit, exp.Limit):
        current = _literal_int(limit.expression)
    elif isinstance(limit, exp.Fetch):  # FETCH FIRST n ROWS ONLY
        options = limit.args.get("limit_options")
        is_plain = options is None or not (options.args.get("percent") or options.args.get("with_ties"))
        current = _literal_int(limit.args.get("count")) if is_plain else None
    else:
        current = None
    if current is not None and 0 <= current <= max_rows:
        return
    root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))


def _verify_output(safe_sql: str, dialect: SqlDialect) -> None:
    """Belt and braces: re-check the exact text that will be executed."""
    statements = [s for s in sqlglot.parse(safe_sql, read=dialect) if s is not None]
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
        raise GuardrailError("The query could not be safely rewritten.")
    # A token scan catches a denied function even if the parser had mapped it to a
    # differently named expression type.
    tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(safe_sql)
    for current, following in zip(tokens, tokens[1:]):
        if (
            following.token_type == TokenType.L_PAREN
            and current.token_type in (TokenType.VAR, TokenType.IDENTIFIER)
            and is_denied_function(current.text)
        ):
            raise GuardrailError(f"The function {current.text.lower()}() is not allowed.")


# --- MongoDB -----------------------------------------------------------------

ALLOWED_STAGES = frozenset({
    "$match", "$project", "$group", "$sort", "$limit", "$skip",
    "$unwind", "$count", "$lookup", "$addFields", "$facet",
})
# Never allowed anywhere: they write data or run server-side JavaScript.
FORBIDDEN_OPERATORS = frozenset({"$out", "$merge", "$function", "$where", "$accumulator"})
_WHY_FORBIDDEN = {
    "$out": "it writes data",
    "$merge": "it writes data",
    "$function": "it runs JavaScript on the server",
    "$where": "it runs JavaScript on the server",
    "$accumulator": "it runs JavaScript on the server",
    "$code": "it is JavaScript code",
}


def validate_mongo(
    query: str | Mapping[str, Any],
    *,
    allowed_collections: Collection[str],
    max_rows: int = DEFAULT_MAX_ROWS,
) -> MongoQuery:
    """Check a Mongo aggregation and return a safe copy, with its output capped.

    `query` is `{"collection": "...", "pipeline": [...]}`, as a dict or a JSON string.
    MongoDB Extended JSON is understood, e.g. `{"$date": "2025-01-01T00:00:00Z"}`
    becomes a real date. The `pipeline` value may itself be a JSON string.

    Raises:
        GuardrailError: the query is not allowed. The message explains why.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    doc = _load_mongo_query(query)

    unknown_keys = set(doc) - {"collection", "pipeline"}
    if unknown_keys:
        raise GuardrailError(f"Unexpected keys in the Mongo query: {', '.join(sorted(unknown_keys))}.")
    collection = doc.get("collection")
    pipeline = doc.get("pipeline")
    if isinstance(pipeline, str):
        pipeline = _parse_extended_json(pipeline, what="pipeline")

    allowed = set(allowed_collections)
    _check_collection(collection, allowed, role="collection")
    if not isinstance(pipeline, list):
        raise GuardrailError('"pipeline" must be a JSON array of stages.')

    _scan_for_forbidden(pipeline)
    _check_pipeline(pipeline, allowed, max_rows, path="pipeline")
    _cap_output(pipeline, max_rows)
    return MongoQuery(collection=collection, pipeline=pipeline)


def _parse_extended_json(text: str, *, what: str) -> Any:
    if len(text) > MAX_QUERY_CHARS * 2:
        raise GuardrailError(f"The {what} is too long.")
    try:
        return json_util.loads(text)
    except (ValueError, TypeError) as exc:
        raise GuardrailError(f"The {what} is not valid JSON: {exc}") from None


def _load_mongo_query(query: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse JSON text, or deep-copy a dict, resolving Extended JSON ($date, $oid, ...)."""
    if isinstance(query, str):
        loaded = _parse_extended_json(query, what="Mongo query")
    elif isinstance(query, Mapping):
        # A dict built by json.loads() still has {"$date": ...} as a plain dict, which
        # MongoDB would reject. Round-trip it so those become real BSON values.
        # (This also gives us a deep copy, so the caller's object is never modified.)
        try:
            loaded = json_util.loads(json_util.dumps(query))
        except (ValueError, TypeError) as exc:
            raise GuardrailError(f"The Mongo query is not valid: {exc}") from None
    else:
        raise GuardrailError("The Mongo query must be a JSON object.")
    if not isinstance(loaded, dict):
        raise GuardrailError('The Mongo query must be a JSON object like {"collection": ..., "pipeline": [...]}.')
    return loaded


def _check_collection(name: Any, allowed: set[str], *, role: str) -> None:
    if not isinstance(name, str) or not name:
        raise GuardrailError(f'The {role} must be a collection name (a non-empty string).')
    if name.startswith("system.") or "$" in name:
        raise GuardrailError(f"The {role} '{name}' can't be queried.")
    if name not in allowed:
        raise GuardrailError(f"Unknown {role} '{name}'. Available collections: {', '.join(sorted(allowed))}.")


def _walk(value: Any, depth: int = 0) -> Iterator[tuple[str | None, Any]]:
    """Yield (key, value) for every dict entry and list item, at any depth."""
    if depth > MAX_NESTING_DEPTH:
        raise GuardrailError("The pipeline is nested too deeply.")
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk(item, depth + 1)


def _scan_for_forbidden(pipeline: list[Any]) -> None:
    """Reject forbidden operators and JavaScript anywhere in the pipeline, however deep."""
    for key, value in _walk(pipeline):
        if key in FORBIDDEN_OPERATORS or key == "$code":
            raise GuardrailError(f"{key} is not allowed: {_WHY_FORBIDDEN[key]}.")
        if isinstance(value, Code):
            raise GuardrailError(f"JavaScript code is not allowed: {_WHY_FORBIDDEN['$code']}.")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)  # bool is an int subclass


def _check_pipeline(pipeline: list[Any], allowed: set[str], max_rows: int, *, path: str) -> None:
    if len(pipeline) > MAX_PIPELINE_STAGES:
        raise GuardrailError(f"The pipeline has too many stages (max {MAX_PIPELINE_STAGES}).")
    for index, stage in enumerate(pipeline):
        where = f"{path}[{index}]"
        if not isinstance(stage, dict) or len(stage) != 1:
            raise GuardrailError(f"{where} must be an object with exactly one stage, like {{'$match': {{...}}}}.")
        name, spec = next(iter(stage.items()))
        if name not in ALLOWED_STAGES:
            raise GuardrailError(
                f"Stage {name} is not allowed. Allowed stages: {', '.join(sorted(ALLOWED_STAGES))}."
            )
        if name == "$limit" and not (_is_int(spec) and spec > 0):
            raise GuardrailError(f"{where}: $limit must be a positive integer.")
        if name == "$skip" and not (_is_int(spec) and spec >= 0):
            raise GuardrailError(f"{where}: $skip must be a non-negative integer.")
        if name == "$lookup":
            _check_lookup(spec, allowed, max_rows, where=where)
        if name == "$facet":
            if not isinstance(spec, dict) or not spec:
                raise GuardrailError(f"{where}: $facet must map output names to pipelines.")
            for facet_name, sub_pipeline in spec.items():
                if not isinstance(sub_pipeline, list):
                    raise GuardrailError(f"{where}: $facet.{facet_name} must be a pipeline (an array).")
                _check_pipeline(sub_pipeline, allowed, max_rows, path=f"{where}.$facet.{facet_name}")
                # Each facet becomes an array inside a single document: cap it too.
                _cap_output(sub_pipeline, max_rows)


def _check_lookup(spec: Any, allowed: set[str], max_rows: int, *, where: str) -> None:
    if not isinstance(spec, dict):
        raise GuardrailError(f"{where}: $lookup must be an object.")
    _check_collection(spec.get("from"), allowed, role="$lookup source")
    sub_pipeline = spec.get("pipeline")
    if sub_pipeline is not None:
        if not isinstance(sub_pipeline, list):
            raise GuardrailError(f"{where}: $lookup.pipeline must be an array.")
        _check_pipeline(sub_pipeline, allowed, max_rows, path=f"{where}.$lookup.pipeline")


def _cap_output(pipeline: list[dict[str, Any]], max_rows: int) -> None:
    """Ensure the pipeline's final stage limits output to at most `max_rows` documents.

    Earlier $limit stages are left as they are: they change what gets aggregated, not
    how many documents come back.
    """
    if pipeline and "$limit" in pipeline[-1]:
        pipeline[-1]["$limit"] = min(pipeline[-1]["$limit"], max_rows)
    else:
        pipeline.append({"$limit": max_rows})
