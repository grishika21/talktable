"""Load the MongoDB engagement database: reviews and clickstream, plus wishlists, search
logs, recommendations and email campaigns.

Usage (from the project root):
    python -m data.seed.seed_mongo [--url mongodb://...] [--db bookstore]

The URL defaults to MONGO_SEED_URL (a user that can write). Only the collections below
are dropped and recreated.
"""

from __future__ import annotations

import argparse
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel, MongoClient

from talktable.config import get_settings, mask_url
from data.seed.common import print_counts, resolve_url
from data.seed.extended import generate_engagement_extras
from data.seed.generate import generate_clickstream, generate_reviews

INDEXES: dict[str, list[IndexModel]] = {
    "reviews": [
        IndexModel([("book_id", ASCENDING)]),
        IndexModel([("customer_id", ASCENDING)]),
        IndexModel([("rating", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
    ],
    "clickstream": [
        IndexModel([("session_id", ASCENDING)], unique=True),
        IndexModel([("customer_id", ASCENDING)]),
        IndexModel([("events.book_id", ASCENDING)]),
    ],
    "wishlists": [
        IndexModel([("customer_id", ASCENDING)], unique=True),
        IndexModel([("items.book_id", ASCENDING)]),
    ],
    "search_logs": [
        IndexModel([("ts", DESCENDING)]),
        IndexModel([("customer_id", ASCENDING)]),
        IndexModel([("clicked_book_id", ASCENDING)]),
    ],
    "recommendations": [
        IndexModel([("customer_id", ASCENDING)]),
        IndexModel([("items.book_id", ASCENDING)]),
    ],
    "email_campaigns": [
        IndexModel([("campaign_id", ASCENDING)], unique=True),
        IndexModel([("sent_at", DESCENDING)]),
    ],
}


def seed(url: str, db_name: str) -> dict[str, int]:
    """Drop, recreate and fill the engagement collections. Returns document counts."""
    collections: dict[str, list[dict[str, Any]]] = {
        "reviews": generate_reviews(),
        "clickstream": generate_clickstream(),
        # insert_many adds an _id to each document, so pass copies of the cached lists
        **{name: [dict(doc) for doc in docs] for name, docs in generate_engagement_extras().items()},
    }
    # A generous server-selection timeout also rides out the brief restart a fresh
    # Docker container does right after its init scripts finish.
    client: MongoClient[dict[str, Any]] = MongoClient(url, serverSelectionTimeoutMS=20_000)
    try:
        db = client[db_name]
        for name, docs in collections.items():
            db.drop_collection(name)
            db[name].insert_many(docs)
            db[name].create_indexes(INDEXES[name])
        return {name: db[name].count_documents({}) for name in collections}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="MongoDB URL (default: MONGO_SEED_URL)")
    parser.add_argument("--db", help="database name (default: MONGO_DB)")
    args = parser.parse_args()

    settings = get_settings()
    url = resolve_url(args.url, settings.mongo_seed_url, settings.mongo_url, "mongo")
    db_name = args.db or settings.mongo_db
    print(f"[mongo] seeding {mask_url(url)} (database: {db_name})")
    print_counts("mongo", seed(url, db_name))


if __name__ == "__main__":
    main()
