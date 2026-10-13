"""Seed all three demo databases and print a summary of row counts.

Usage (from the project root):
    python -m data.seed.seed_all                 # all three
    python -m data.seed.seed_all postgres mongo  # just these

Each database is seeded independently, so one failing (e.g. Mongo not running) doesn't
stop the others. Exits with status 1 if any database failed.
"""

from __future__ import annotations

import sys
import time

from talktable.config import Settings, get_settings, mask_url
from data.seed import seed_mongo, seed_postgres, seed_sqlite
from data.seed.common import print_counts, resolve_url

DATABASES = ("sqlite", "postgres", "mongo")


def seed_one(label: str, settings: Settings) -> dict[str, int]:
    """Seed a single database using the URLs/paths from settings."""
    if label == "sqlite":
        print(f"[sqlite] seeding {settings.sqlite_file}")
        return seed_sqlite.seed(settings.sqlite_file)
    if label == "postgres":
        url = resolve_url(None, settings.postgres_seed_url, settings.postgres_url, "postgres")
        print(f"[postgres] seeding {mask_url(url)}")
        return seed_postgres.seed(url)
    url = resolve_url(None, settings.mongo_seed_url, settings.mongo_url, "mongo")
    print(f"[mongo] seeding {mask_url(url)} (database: {settings.mongo_db})")
    return seed_mongo.seed(url, settings.mongo_db)


def main(argv: list[str] | None = None) -> int:
    targets = list(argv if argv is not None else sys.argv[1:]) or list(DATABASES)
    unknown = [t for t in targets if t not in DATABASES]
    if unknown:
        print(f"Unknown database(s): {', '.join(unknown)}. Choose from: {', '.join(DATABASES)}.", file=sys.stderr)
        return 2
    settings = get_settings()
    failures: list[str] = []
    started = time.perf_counter()

    for label in targets:
        try:
            print_counts(label, seed_one(label, settings))
        except SystemExit as exc:  # raised by resolve_url when no URL is configured
            print(exc, file=sys.stderr)
            failures.append(label)
        except Exception as exc:  # report it and carry on with the next database
            print(f"[{label}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(label)
        print()

    elapsed = time.perf_counter() - started
    if failures:
        print(f"Finished in {elapsed:.1f}s. Failed: {', '.join(failures)}")
        return 1
    print(f"Finished in {elapsed:.1f}s. Seeded: {', '.join(targets)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
