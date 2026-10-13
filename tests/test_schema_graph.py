"""Join paths over the foreign-key graph (NetworkX)."""

from __future__ import annotations

import pytest

from talktable.adapters import PostgresAdapter, SQLiteAdapter
from talktable.models import ColumnInfo, TableInfo
from talktable.schema_graph import JoinGraph, parse_foreign_key


def table(name: str, *fks: str) -> TableInfo:
    return TableInfo(name=name, columns=[ColumnInfo(name="id", type="INTEGER")], foreign_keys=list(fks))


@pytest.fixture(scope="module")
def sales(postgres_adapter: PostgresAdapter) -> JoinGraph:
    return JoinGraph(postgres_adapter.get_schema())


@pytest.fixture(scope="module")
def catalog(sqlite_adapter: SQLiteAdapter) -> JoinGraph:
    return JoinGraph(sqlite_adapter.get_schema())


def test_parse_foreign_key() -> None:
    assert parse_foreign_key("author_id -> authors.author_id") == ("author_id", "authors", "author_id")
    assert parse_foreign_key("not a foreign key") is None
    assert parse_foreign_key("x -> nowhere") is None


def test_real_schemas_become_graphs(catalog: JoinGraph, sales: JoinGraph) -> None:
    assert set(catalog.parents("books")) == {"authors", "categories"}
    assert set(catalog.parents("editions")) == {"books", "publishers"}
    assert catalog.edge_count == 10 and sales.edge_count == 12  # one edge per foreign key here


def test_shortest_join_path(catalog: JoinGraph, sales: JoinGraph) -> None:
    assert catalog.path("tags", "authors") == ["tags", "book_tags", "books", "authors"]
    path = sales.path("returns", "carriers")
    assert path == ["returns", "order_items", "orders", "shipments", "carriers"]


def test_bridges_connect_matched_tables(catalog: JoinGraph, sales: JoinGraph) -> None:
    # "Which publisher has the most award-winning books?"
    assert catalog.bridges(["publishers", "book_awards"]) == ["editions", "books"]
    # "Average satisfaction of gold members": both tables point at customers
    assert sales.bridges(["support_tickets", "loyalty_accounts"]) == ["customers"]
    # Matched tables are never reported as bridges, even when the path runs through them
    assert sales.bridges(["shipments", "orders", "carriers"]) == []


def test_far_fetched_paths_are_not_bridged(sales: JoinGraph) -> None:
    # returns -> carriers needs 3 bridge tables: more than MAX_HOPS allows
    assert sales.bridges(["returns", "carriers"]) == []
    assert sales.bridges(["returns", "carriers"], max_hops=4) == ["order_items", "orders", "shipments"]
    assert sales.bridges(["returns", "carriers"], max_hops=4, max_bridges=2) == []


def test_disconnected_and_unknown_tables() -> None:
    graph = JoinGraph([table("a", "b_id -> b.id"), table("b"), table("island")])
    assert graph.bridges(["a", "island"]) == [] and graph.path("a", "island") is None
    assert graph.bridges(["nope", "a"]) == [] and graph.path("nope", "a") is None
    assert graph.joins(["a", "b", "island"]) == ["a.b_id = b.id"]


def test_joins_lists_every_condition_between_the_tables() -> None:
    graph = JoinGraph([
        table("matches", "home_team_id -> teams.team_id", "away_team_id -> teams.team_id"),
        table("teams"),
        table("cities"),
    ])
    assert graph.joins(["matches", "teams"]) == ["matches.away_team_id = teams.team_id",
                                                 "matches.home_team_id = teams.team_id"]
    assert graph.joins(["matches"]) == [] and graph.parents("matches") == ["teams"]


def test_mongo_collections_have_no_edges() -> None:
    graph = JoinGraph([table("reviews"), table("clickstream")])
    assert graph.edge_count == 0 and graph.bridges(["reviews", "clickstream"]) == []
