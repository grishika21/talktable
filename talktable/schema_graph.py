"""Join paths: the foreign keys of a database as a NetworkX graph.

Tables are nodes and every foreign key is an edge, so "how do I get from returns to
carriers?" is a shortest-path question. Retrieval uses it to

  - add the bridge tables that connect the tables a question matched (returns and
    carriers only meet through order_items, orders and shipments), and
  - spell out the join conditions for the query writer instead of leaving it to guess.

MongoDB collections have no foreign keys, so their graph has no edges.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from talktable.models import TableInfo

MAX_HOPS = 3  # don't connect two matched tables through more than 2 bridge tables
MAX_BRIDGES = 3  # bridge tables added per question, at most


def parse_foreign_key(fk: str) -> tuple[str, str, str] | None:
    """'author_id -> authors.author_id' -> ('author_id', 'authors', 'author_id')."""
    if "->" not in fk:
        return None
    column, target = (part.strip() for part in fk.split("->", 1))
    table, _, target_column = target.partition(".")
    return (column, table, target_column) if table and target_column else None


class JoinGraph:
    """Undirected FK graph of one database. Each edge keeps its join condition(s)."""

    def __init__(self, tables: Sequence[TableInfo]) -> None:
        self._graph = nx.Graph()
        self._parents: dict[str, list[str]] = {t.name: [] for t in tables}
        self._graph.add_nodes_from(t.name for t in tables)
        for table in tables:
            for fk in table.foreign_keys:
                parsed = parse_foreign_key(fk)
                if parsed is None or parsed[1] not in self._parents:
                    continue
                column, target, target_column = parsed
                condition = f"{table.name}.{column} = {target}.{target_column}"
                if self._graph.has_edge(table.name, target):
                    self._graph.edges[table.name, target]["on"].append(condition)
                else:
                    self._graph.add_edge(table.name, target, on=[condition])
                if target not in self._parents[table.name]:
                    self._parents[table.name].append(target)

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def parents(self, table: str) -> list[str]:
        """The tables `table` references through its foreign keys."""
        return list(self._parents.get(table, []))

    def path(self, source: str, target: str) -> list[str] | None:
        """The shortest join path between two tables, or None if they don't connect."""
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def bridges(self, tables: Sequence[str], *, max_hops: int = MAX_HOPS,
                max_bridges: int = MAX_BRIDGES) -> list[str]:
        """Tables needed to connect `tables` (most relevant first), not counting them.

        A greedy Steiner tree: each table, in order, joins the tree built so far by its
        shortest path to any table already in it. Paths longer than `max_hops` (or that
        would exceed `max_bridges`) are skipped, and so are tables with no path: they are
        still shown, just not linked, because a far-fetched join is usually a sign the
        table was only a weak match.
        """
        wanted = [t for t in tables if t in self._graph]
        if not wanted:
            return []
        tree, added = {wanted[0]}, []
        for table in wanted[1:]:
            if table in tree:
                continue
            try:
                hops, path = nx.multi_source_dijkstra(self._graph, tree, target=table)
            except nx.NetworkXNoPath:
                tree.add(table)
                continue
            new = [node for node in path[1:-1] if node not in tree and node not in wanted]
            if hops <= max_hops and len(added) + len(new) <= max_bridges:
                added += new
                tree.update(path)
            else:
                tree.add(table)
        return added

    def joins(self, tables: Sequence[str]) -> list[str]:
        """Join conditions between any two of `tables`, e.g. 'books.author_id = authors.author_id'."""
        present = [t for t in tables if t in self._graph]
        conditions: list[str] = []
        for u, v, data in self._graph.subgraph(present).edges(data=True):
            conditions += data["on"]
        return sorted(conditions)
