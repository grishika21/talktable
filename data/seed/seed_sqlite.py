"""Build the SQLite catalog database from scratch: authors, categories and books, plus
publishers, editions, series, awards and tags.

Usage (from the project root):
    python -m data.seed.seed_sqlite [--path data/bookstore.db]

The file is written to a temporary path and then swapped in, so a failed run never
leaves a half-written database behind.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import asdict
from pathlib import Path

from talktable.config import get_settings
from data.seed.common import print_counts
from data.seed.extended import generate_catalog_extras
from data.seed.generate import generate_catalog

TABLES = ("authors", "categories", "books", "publishers", "editions", "series", "book_series", "awards",
          "book_awards", "tags", "book_tags")

SCHEMA = """
CREATE TABLE authors (
    author_id   INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    country     TEXT    NOT NULL,
    birth_year  INTEGER NOT NULL
);

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id        INTEGER PRIMARY KEY,
    title          TEXT    NOT NULL,
    author_id      INTEGER NOT NULL REFERENCES authors (author_id),
    category_id    INTEGER NOT NULL REFERENCES categories (category_id),
    price          REAL    NOT NULL CHECK (price > 0),
    published_year INTEGER NOT NULL,
    pages          INTEGER NOT NULL CHECK (pages > 0)
);

CREATE INDEX idx_books_author_id   ON books (author_id);
CREATE INDEX idx_books_category_id ON books (category_id);

CREATE TABLE publishers (
    publisher_id INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL UNIQUE,
    country      TEXT    NOT NULL,
    founded_year INTEGER NOT NULL
);

CREATE TABLE editions (
    edition_id   INTEGER PRIMARY KEY,
    book_id      INTEGER NOT NULL REFERENCES books (book_id),
    publisher_id INTEGER NOT NULL REFERENCES publishers (publisher_id),
    format       TEXT    NOT NULL CHECK (format IN ('paperback', 'hardcover', 'ebook', 'audiobook')),
    isbn13       TEXT    NOT NULL UNIQUE,
    release_year INTEGER NOT NULL,
    price        REAL    NOT NULL CHECK (price > 0)
);

CREATE TABLE series (
    series_id INTEGER PRIMARY KEY,
    name      TEXT    NOT NULL
);

CREATE TABLE book_series (
    book_id   INTEGER PRIMARY KEY REFERENCES books (book_id),
    series_id INTEGER NOT NULL REFERENCES series (series_id),
    position  INTEGER NOT NULL
);

CREATE TABLE awards (
    award_id     INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL UNIQUE,
    organization TEXT    NOT NULL,
    country      TEXT    NOT NULL,
    first_year   INTEGER NOT NULL
);

CREATE TABLE book_awards (
    book_award_id INTEGER PRIMARY KEY,
    book_id       INTEGER NOT NULL REFERENCES books (book_id),
    award_id      INTEGER NOT NULL REFERENCES awards (award_id),
    year          INTEGER NOT NULL,
    result        TEXT    NOT NULL CHECK (result IN ('winner', 'shortlisted'))
);

CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY,
    name   TEXT    NOT NULL UNIQUE
);

CREATE TABLE book_tags (
    book_id INTEGER NOT NULL REFERENCES books (book_id),
    tag_id  INTEGER NOT NULL REFERENCES tags (tag_id),
    PRIMARY KEY (book_id, tag_id)
);

CREATE INDEX idx_editions_book_id      ON editions (book_id);
CREATE INDEX idx_editions_publisher_id ON editions (publisher_id);
CREATE INDEX idx_book_series_series_id ON book_series (series_id);
CREATE INDEX idx_book_awards_book_id   ON book_awards (book_id);
CREATE INDEX idx_book_awards_award_id  ON book_awards (award_id);
CREATE INDEX idx_book_tags_tag_id      ON book_tags (tag_id);
"""


def _insert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if rows:
        columns = list(rows[0])
        conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES "
                         f"({', '.join(':' + c for c in columns)})", rows)


def seed(path: Path) -> dict[str, int]:
    """(Re)create the catalog database at `path` and return row counts per table."""
    catalog = generate_catalog()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO authors VALUES (:author_id, :name, :country, :birth_year)",
            [asdict(a) for a in catalog.authors],
        )
        conn.executemany(
            "INSERT INTO categories VALUES (:category_id, :name)",
            [asdict(c) for c in catalog.categories],
        )
        conn.executemany(
            "INSERT INTO books VALUES (:book_id, :title, :author_id, :category_id, "
            ":price, :published_year, :pages)",
            [asdict(b) for b in catalog.books],
        )
        for table, rows in generate_catalog_extras().items():
            _insert(conn, table, rows)
        conn.commit()
        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"foreign key violations: {problems[:5]}")
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
        conn.execute("VACUUM")
    finally:
        conn.close()

    tmp_path.replace(path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, help="SQLite file to create (default: SQLITE_PATH)")
    args = parser.parse_args()

    path = args.path or get_settings().sqlite_file
    print(f"[sqlite] seeding {path}")
    print_counts("sqlite", seed(path))


if __name__ == "__main__":
    main()
