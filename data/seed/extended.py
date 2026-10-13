"""22 more tables for the demo data, for a total of 30.

They hang off the original data, so they join to it:

* catalog (SQLite): publishers, editions, series, book_series, awards, book_awards, tags, book_tags
* sales (Postgres): payments, carriers, shipments, returns, coupons, order_coupons,
  warehouses, inventory, loyalty_accounts, support_tickets
* engagement (MongoDB): wishlists, search_logs, recommendations, email_campaigns

Every generator here has its own fixed seed (SEED + 10 and up) and never touches the
random state used by generate.py, so the original eight tables stay byte-for-byte
identical, and so do the known answers the tests check.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from typing import Any

from data.seed.generate import (
    DATA_END,
    DEVICES,
    SEED,
    Catalog,
    Sales,
    generate_catalog,
    generate_sales,
)

Row = dict[str, Any]

# --- vocabulary ------------------------------------------------------------------

PUBLISHERS: list[tuple[str, str, int]] = [  # name, country, founded
    ("Harbor Lane Press", "United States", 1921), ("Northwind Books", "United States", 1948),
    ("Quill & Lantern", "United Kingdom", 1902), ("Bluefern Publishing", "Canada", 1975),
    ("Saffron House", "India", 1989), ("Kestrel Editions", "United Kingdom", 1966),
    ("Meridian Publishing", "United States", 1937), ("Copperleaf Press", "Australia", 1994),
    ("Atlas & Ink", "United States", 2003), ("Silverbirch Books", "Canada", 1958),
    ("Monsoon Press", "India", 2007), ("Ravenstone Publishers", "United Kingdom", 1884),
    ("Brightwater Media", "United States", 2011), ("Lighthouse Kids", "United States", 1999),
    ("Orchard Lane Poetry", "Ireland", 1972), ("Granite Peak Books", "United States", 1982),
    ("Maple & Stone", "Canada", 2015), ("Sundial Publishing", "Germany", 1951),
    ("Ironwood Academic", "United Kingdom", 1930), ("Tidewater Press", "Australia", 1968),
]
AUDIO_PUBLISHERS: list[tuple[str, str, int]] = [
    ("SoundPage Audio", "United States", 1998), ("Narrator's Circle", "United Kingdom", 2009),
    ("ClearVoice Audiobooks", "Canada", 2014),
]
FORMATS = ("paperback", "hardcover", "ebook", "audiobook")
_FORMAT_FACTOR = {"paperback": 1.0, "hardcover": 1.6, "ebook": 0.6, "audiobook": 1.25}

SERIES_WORDS = ["Chronicles", "Saga", "Cycle", "Trilogy", "Files", "Mysteries", "Quartet", "Legacy", "Diaries"]
_STOPWORDS = {"the", "and", "with", "from", "into", "about", "what", "when", "where", "your", "that", "this"}
SERIES_GENRES = {"Fantasy", "Science Fiction", "Mystery", "Thriller", "Romance", "Children's", "Historical Fiction"}

AWARDS: list[tuple[str, str, str, int, set[str]]] = [  # name, organisation, country, first year, categories
    ("Golden Quill Prize", "Golden Quill Foundation", "United States", 1995,
     {"Fiction", "Historical Fiction", "Poetry"}),
    ("Starfall Award", "Speculative Writers Guild", "United States", 1990, {"Science Fiction", "Fantasy"}),
    ("Iron Lantern Dagger", "Crime Writers Circle", "United Kingdom", 1988, {"Mystery", "Thriller"}),
    ("Heartstrings Medal", "Romance Readers Society", "Canada", 2001, {"Romance"}),
    ("Midnight Ink Award", "Horror Writers Collective", "United States", 2004, {"Horror"}),
    ("True Story Prize", "Nonfiction Book Council", "United Kingdom", 1993, {"Biography", "History"}),
    ("Lumen Science Book Prize", "Society for Public Science", "Germany", 2000, {"Science"}),
    ("Open Door Award", "Wellbeing Readers Trust", "Australia", 2010, {"Self-Help"}),
    ("Ledger Business Book Award", "Business Readers Forum", "United States", 2005, {"Business"}),
    ("Little Lantern Medal", "Children's Library Association", "Ireland", 1985, {"Children's"}),
    ("Readers' Choice Award", "TalkTable Readers", "United States", 2012, set()),  # any category
]

TAGS = ["bestseller", "staff pick", "award winner", "debut", "classic", "new release", "short read", "long read",
        "series", "page-turner", "cozy", "dark", "funny", "heartwarming", "thought-provoking", "twisty", "epic",
        "romantic", "inspiring", "book club pick", "illustrated", "translated", "gift idea", "beach read",
        "slow burn", "true story", "practical", "bedtime story", "lyrical", "hidden gem"]
_MOOD_TAGS: dict[str, list[str]] = {
    "Fiction": ["thought-provoking", "lyrical", "book club pick", "heartwarming"],
    "Mystery": ["twisty", "cozy", "page-turner"],
    "Thriller": ["page-turner", "dark", "twisty"],
    "Science Fiction": ["epic", "thought-provoking"],
    "Fantasy": ["epic", "cozy", "dark"],
    "Romance": ["romantic", "slow burn", "beach read", "heartwarming"],
    "Historical Fiction": ["epic", "book club pick", "lyrical"],
    "Horror": ["dark", "page-turner"],
    "Biography": ["true story", "inspiring"],
    "History": ["true story", "thought-provoking"],
    "Science": ["thought-provoking", "illustrated"],
    "Self-Help": ["practical", "inspiring"],
    "Business": ["practical", "thought-provoking"],
    "Children's": ["bedtime story", "illustrated", "funny"],
    "Poetry": ["lyrical", "gift idea"],
}

CARRIERS = ["SwiftShip", "ParcelPro", "GlobalPost", "NorthStar Freight", "QuickDrop", "OceanAir Logistics"]
_CARRIER_WEIGHTS = [30, 25, 18, 12, 10, 5]
PAYMENT_METHODS = ("card", "paypal", "gift_card", "bank_transfer")
RETURN_REASONS = ("damaged", "wrong item", "not as described", "changed mind", "arrived late")
WAREHOUSES = [  # name, city, country
    ("Newark Distribution Center", "Newark", "United States"), ("Reno Fulfillment Center", "Reno", "United States"),
    ("Milton Keynes Hub", "Milton Keynes", "United Kingdom"), ("Toronto East DC", "Toronto", "Canada"),
    ("Pune Fulfillment Center", "Pune", "India"),
]
TICKET_CATEGORIES = ("delivery", "payment", "damaged item", "refund", "account", "other")
CAMPAIGN_THEMES = ["New Releases", "Summer Reading", "Holiday Gift Guide", "Award Winners", "Staff Picks",
                   "Back to School", "Mystery Month", "Cozy Reads", "Big Ideas", "Kids' Corner"]
_SEARCH_EXTRAS = ["gift ideas", "bestsellers", "new releases", "audiobooks", "books like harry potter",
                  "cheap books", "signed copies", "book club", "short stories", "poetry for beginners"]


# --- helpers ---------------------------------------------------------------------


def _money(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _price_99(value: float) -> float:
    """A shop price like 12.99."""
    return max(2.99, round(value) - 0.01)


def _isbn13(rng: random.Random) -> str:
    digits = [9, 7, 8] + [rng.randrange(10) for _ in range(9)]
    check = (10 - sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits)) % 10) % 10
    return "".join(map(str, digits + [check]))


def _clamp(day: date) -> date:
    return min(day, DATA_END)


def _at(rng: random.Random, day: date) -> datetime:
    return datetime.combine(day, time(rng.randrange(7, 23), rng.randrange(60), rng.randrange(60)))


def _random_day(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randrange(max(1, (end - start).days + 1)))


# --- catalog (SQLite) --------------------------------------------------------------


@lru_cache(maxsize=1)
def generate_catalog_extras() -> dict[str, list[Row]]:
    catalog = generate_catalog()
    rng = random.Random(SEED + 10)
    category = {c.category_id: c.name for c in catalog.categories}

    publishers = [{"publisher_id": i, "name": n, "country": c, "founded_year": y}
                  for i, (n, c, y) in enumerate(PUBLISHERS + AUDIO_PUBLISHERS, start=1)]
    print_ids = list(range(1, len(PUBLISHERS) + 1))
    audio_ids = list(range(len(PUBLISHERS) + 1, len(publishers) + 1))
    kids_id, poetry_id = 14, 15  # Lighthouse Kids, Orchard Lane Poetry

    editions: list[Row] = []
    for book in catalog.books:
        genre = category[book.category_id]
        main = kids_id if genre == "Children's" and rng.random() < 0.6 else (
            poetry_id if genre == "Poetry" and rng.random() < 0.6 else rng.choice(print_ids))
        formats = ["paperback"]
        if rng.random() < 0.55:
            formats.append("hardcover")
        if rng.random() < (0.75 if book.published_year >= 2005 else 0.35):
            formats.append("ebook")
        if genre not in ("Poetry", "Children's") and rng.random() < (0.4 if book.published_year >= 2010 else 0.12):
            formats.append("audiobook")
        for fmt in formats:
            year = book.published_year
            if fmt == "ebook":
                year = max(year, 2008 + rng.randrange(3))
            elif fmt == "audiobook":
                year = max(year + rng.randrange(3), 2010 + rng.randrange(4))
            editions.append({
                "edition_id": len(editions) + 1, "book_id": book.book_id,
                "publisher_id": rng.choice(audio_ids) if fmt == "audiobook" and rng.random() < 0.6 else main,
                "format": fmt, "isbn13": _isbn13(rng), "release_year": min(year, 2026),
                "price": _price_99(book.price * _FORMAT_FACTOR[fmt] * rng.uniform(0.95, 1.05)),
            })

    # Series: some authors write 2-5 books in the same series, in publication order.
    by_author: dict[int, list[Any]] = defaultdict(list)
    for book in catalog.books:
        if category[book.category_id] in SERIES_GENRES:
            by_author[book.author_id].append(book)
    series: list[Row] = []
    book_series: list[Row] = []
    used_names: set[str] = set()
    for author_id in sorted(by_author):
        books = sorted(by_author[author_id], key=lambda b: (b.published_year, b.book_id))
        if len(books) < 2 or rng.random() > 0.45:
            continue
        members = books[: rng.randint(2, min(5, len(books)))]
        title = members[0].title.split(" (")[0].split(":")[0]
        words = [w for w in re.findall(r"[A-Za-z']+", title) if len(w) > 3 and w.lower() not in _STOPWORDS]
        name = f"The {max(words, key=len) if words else 'Untold'} {rng.choice(SERIES_WORDS)}"
        while name in used_names:
            name = f"{name} II"
        used_names.add(name)
        series_id = len(series) + 1
        series.append({"series_id": series_id, "name": name})
        book_series += [{"book_id": b.book_id, "series_id": series_id, "position": i}
                        for i, b in enumerate(members, start=1)]

    # Awards: each year, a short list of eligible books published the year before; the
    # best-rated of them tends to win.
    awards = [{"award_id": i, "name": n, "organization": o, "country": c, "first_year": y}
              for i, (n, o, c, y, _) in enumerate(AWARDS, start=1)]
    by_year_genre: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for book in catalog.books:
        by_year_genre[(book.published_year, category[book.category_id])].append(book)
    book_awards: list[Row] = []
    for award_id, (_, _, _, first_year, genres) in enumerate(AWARDS, start=1):
        for year in range(max(first_year, 1990), 2026):
            pool = [b for (y, g), bs in by_year_genre.items() if y == year - 1 and (not genres or g in genres)
                    for b in bs]
            if not pool or rng.random() < 0.15:
                continue
            shortlist = rng.sample(pool, min(len(pool), rng.randint(3, 5)))
            winner = max(shortlist, key=lambda b: catalog.quality[b.book_id] + rng.uniform(-0.4, 0.4))
            for book in sorted(shortlist, key=lambda b: b.book_id):
                book_awards.append({"book_award_id": len(book_awards) + 1, "book_id": book.book_id,
                                    "award_id": award_id, "year": year,
                                    "result": "winner" if book is winner else "shortlisted"})

    # Tags: some follow from the data, plus a mood tag or two per book.
    tags = [{"tag_id": i, "name": n} for i, n in enumerate(TAGS, start=1)]
    tag_id = {n: i for i, n in enumerate(TAGS, start=1)}
    top_popular = set(sorted(catalog.popularity, key=catalog.popularity.get, reverse=True)[:50])
    top_quality = set(sorted(catalog.quality, key=catalog.quality.get, reverse=True)[:80])
    winners = {r["book_id"] for r in book_awards if r["result"] == "winner"}
    in_series = {r["book_id"] for r in book_series}
    first_book = {}
    for book in sorted(catalog.books, key=lambda b: (b.published_year, b.book_id)):
        first_book.setdefault(book.author_id, book.book_id)
    book_tags: list[Row] = []
    for book in catalog.books:
        names = set()
        names |= {"bestseller"} if book.book_id in top_popular else set()
        names |= {"staff pick"} if book.book_id in top_quality and rng.random() < 0.5 else set()
        names |= {"award winner"} if book.book_id in winners else set()
        names |= {"debut"} if first_book[book.author_id] == book.book_id else set()
        names |= {"classic"} if book.published_year < 1990 else set()
        names |= {"new release"} if book.published_year >= 2025 else set()
        names |= {"short read"} if book.pages < 200 else set()
        names |= {"long read"} if book.pages > 600 else set()
        names |= {"series"} if book.book_id in in_series else set()
        moods = _MOOD_TAGS[category[book.category_id]]
        names |= set(rng.sample(moods, rng.randint(1, min(2, len(moods)))))
        if rng.random() < 0.05:
            names.add("hidden gem")
        if rng.random() < 0.06:
            names.add("translated")
        book_tags += [{"book_id": book.book_id, "tag_id": tag_id[n]} for n in sorted(names, key=tag_id.get)]

    return {"publishers": publishers, "editions": editions, "series": series, "book_series": book_series,
            "awards": awards, "book_awards": book_awards, "tags": tags, "book_tags": book_tags}


# --- sales (Postgres) --------------------------------------------------------------


@lru_cache(maxsize=1)
def generate_sales_extras() -> dict[str, list[Row]]:
    catalog, sales = generate_catalog(), generate_sales()
    rng = random.Random(SEED + 11)
    items_by_order: dict[int, list[Any]] = defaultdict(list)
    for item in sales.order_items:
        items_by_order[item.order_id].append(item)

    carriers = [{"carrier_id": i, "name": n} for i, n in enumerate(CARRIERS, start=1)]

    payments: list[Row] = []
    shipments: list[Row] = []
    returns: list[Row] = []
    for order in sales.orders:
        method = rng.choices(PAYMENT_METHODS, weights=(62, 20, 8, 10))[0]
        status = {"cancelled": "refunded" if rng.random() < 0.7 else "failed",
                  "returned": "refunded"}.get(order.status, "paid")
        payments.append({"payment_id": len(payments) + 1, "order_id": order.order_id, "method": method,
                         "amount": order.total_amount, "paid_at": _at(rng, order.order_date), "status": status})
        if order.status not in ("shipped", "delivered", "returned"):
            continue
        shipped = _clamp(order.order_date + timedelta(days=rng.randint(1, 3)))
        delivered = None if order.status == "shipped" else _clamp(shipped + timedelta(days=rng.randint(2, 8)))
        shipments.append({"shipment_id": len(shipments) + 1, "order_id": order.order_id,
                          "carrier_id": rng.choices(range(1, len(CARRIERS) + 1), weights=_CARRIER_WEIGHTS)[0],
                          "shipped_date": shipped, "delivered_date": delivered,
                          "status": "in_transit" if delivered is None else "delivered"})
        if order.status == "returned":
            items = items_by_order[order.order_id]
            chosen = [i for i in items if rng.random() < 0.75] or [items[0]]
            for item in chosen:
                returns.append({"return_id": len(returns) + 1, "order_item_id": item.order_item_id,
                                "reason": rng.choices(RETURN_REASONS, weights=(30, 15, 20, 25, 10))[0],
                                "refund_amount": _money(item.unit_price * item.quantity),
                                "return_date": _clamp(delivered + timedelta(days=rng.randint(2, 25)))})

    coupons: list[Row] = [{"coupon_id": 1, "code": "WELCOME10", "discount_pct": 10,
                           "valid_from": date(2022, 1, 1), "valid_to": DATA_END}]
    for year in range(2022, 2027):
        for season, month in (("SPRING", 3), ("SUMMER", 6), ("FALL", 9), ("HOLIDAY", 11)):
            start = date(year, month, rng.randint(1, 10))
            if start > DATA_END:
                continue
            coupons.append({"coupon_id": len(coupons) + 1, "code": f"{season}{str(year)[2:]}",
                            "discount_pct": rng.choice([10, 15, 20, 25]), "valid_from": start,
                            "valid_to": _clamp(start + timedelta(weeks=rng.randint(2, 6)))})
    order_coupons: list[Row] = []
    for order in sales.orders:
        valid = [c for c in coupons[1:] if c["valid_from"] <= order.order_date <= c["valid_to"]]
        if valid and rng.random() < 0.3:
            order_coupons.append({"order_id": order.order_id, "coupon_id": rng.choice(valid)["coupon_id"]})
        elif rng.random() < 0.03:
            order_coupons.append({"order_id": order.order_id, "coupon_id": 1})

    warehouses = [{"warehouse_id": i, "name": n, "city": c, "country": k}
                  for i, (n, c, k) in enumerate(WAREHOUSES, start=1)]
    max_pop = max(catalog.popularity.values())
    inventory: list[Row] = []
    for book in catalog.books:
        demand = catalog.popularity[book.book_id] / max_pop
        for warehouse_id in sorted(rng.sample(range(1, len(WAREHOUSES) + 1), rng.randint(1, 3))):
            on_hand = 0 if rng.random() < 0.06 else int(max(1, rng.gauss(20 + 300 * demand, 15)))
            inventory.append({"warehouse_id": warehouse_id, "book_id": book.book_id, "quantity_on_hand": on_hand,
                              "reorder_level": rng.choice([10, 15, 20, 30, 40, 50]),
                              "last_restocked": _random_day(rng, date(2026, 1, 1), DATA_END)})

    spend: dict[int, Decimal] = defaultdict(Decimal)
    for order in sales.orders:
        if order.status != "cancelled":
            spend[order.customer_id] += order.total_amount
    loyalty_accounts: list[Row] = []
    for customer in sales.customers:
        if rng.random() > min(0.85, 0.3 + 0.25 * sales.activity[customer.customer_id]):
            continue
        points = max(0, int(spend[customer.customer_id]) + rng.randint(-150, 300))
        tier = ("platinum" if points >= 3000 else "gold" if points >= 1500 else "silver" if points >= 500
                else "bronze")
        loyalty_accounts.append({"customer_id": customer.customer_id, "tier": tier, "points": points,
                                 "joined_date": _clamp(customer.signup_date + timedelta(days=rng.randint(0, 180)))})

    support_tickets: list[Row] = []
    order_weights = [{"returned": 8, "cancelled": 5, "shipped": 4}.get(o.status, 1) for o in sales.orders]
    for order in rng.choices(sales.orders, weights=order_weights, k=1200):
        if order.status == "returned":
            category = rng.choice(["refund", "damaged item", "refund"])
        elif order.status == "cancelled":
            category = rng.choice(["payment", "refund"])
        else:
            category = rng.choices(TICKET_CATEGORIES, weights=(40, 10, 10, 5, 5, 10))[0]
        support_tickets.append(_ticket(rng, len(support_tickets) + 1, order.customer_id, order.order_id,
                                       order.order_date, category))
    for customer in rng.sample(sales.customers, 250):
        support_tickets.append(_ticket(rng, len(support_tickets) + 1, customer.customer_id, None,
                                       customer.signup_date, rng.choice(["account", "other", "payment"])))
    support_tickets.sort(key=lambda t: t["opened_at"])
    for number, ticket in enumerate(support_tickets, start=1):
        ticket["ticket_id"] = number

    return {"carriers": carriers, "payments": payments, "shipments": shipments, "returns": returns,
            "coupons": coupons, "order_coupons": order_coupons, "warehouses": warehouses, "inventory": inventory,
            "loyalty_accounts": loyalty_accounts, "support_tickets": support_tickets}


def _ticket(rng: random.Random, ticket_id: int, customer_id: int, order_id: int | None, after: date,
            category: str) -> Row:
    opened = _at(rng, _clamp(after + timedelta(days=rng.randint(1, 30))))
    still_open = opened.date() > DATA_END - timedelta(days=10) and rng.random() < 0.7
    hours = rng.choice([2, 5, 12, 24, 48, 72, 120, 240])
    resolved = None if still_open else opened + timedelta(hours=hours + rng.randint(0, 3))
    satisfaction = None
    if resolved is not None and rng.random() < 0.8:
        base = 5 - (hours >= 72) - (hours >= 120) - (category in ("damaged item", "delivery"))
        satisfaction = max(1, min(5, base + rng.choice([-1, 0, 0, 0, 1])))
    return {"ticket_id": ticket_id, "customer_id": customer_id, "order_id": order_id, "category": category,
            "opened_at": opened, "resolved_at": resolved, "satisfaction": satisfaction}


# --- engagement (MongoDB) ----------------------------------------------------------


@lru_cache(maxsize=1)
def generate_engagement_extras() -> dict[str, list[Row]]:
    catalog, sales = generate_catalog(), generate_sales()
    rng = random.Random(SEED + 12)
    books = catalog.books
    book_weights = [catalog.popularity[b.book_id] for b in books]
    customers = sales.customers
    customer_weights = [sales.activity[c.customer_id] for c in customers]
    category = {c.category_id: c.name for c in catalog.categories}
    author = {a.author_id: a.name for a in catalog.authors}

    wishlists: list[Row] = []
    for customer in sorted({c.customer_id: c for c in rng.choices(customers, weights=customer_weights,
                                                                    k=1100)}.values(),
                           key=lambda c: c.customer_id)[:900]:
        start = max(customer.signup_date, date(2024, 1, 1))
        picked = {b.book_id: b for b in rng.choices(books, weights=book_weights, k=rng.randint(1, 12))}
        items = sorted(({"book_id": book_id, "added_at": _at(rng, _random_day(rng, start, DATA_END))}
                        for book_id in picked), key=lambda i: i["added_at"])
        wishlists.append({"customer_id": customer.customer_id, "items": items, "updated_at": items[-1]["added_at"]})

    search_logs: list[Row] = []
    for _ in range(6000):
        day = _random_day(rng, date(2025, 1, 1), DATA_END)
        book = rng.choices(books, weights=book_weights)[0]
        kind = rng.random()
        if kind < 0.45:
            words = book.title.split(" (")[0].split(":")[0].lower().split()
            size = rng.randint(1, min(3, len(words)))
            skip = 1 if len(words) > 1 and words[0] in ("the", "a", "an") else 0  # nobody searches "the"
            query, target = " ".join(words[skip: skip + size]), book.book_id
        elif kind < 0.7:
            query, target = author[book.author_id].split()[-1].lower(), book.book_id
        elif kind < 0.9:
            query, target = f"{category[book.category_id].lower()} books", None
        else:
            query, target = rng.choice(_SEARCH_EXTRAS), None
        results = 0 if rng.random() < 0.04 else rng.randint(1, 60)
        clicked = target if results and target and rng.random() < 0.6 else (
            rng.choices(books, weights=book_weights)[0].book_id if results and rng.random() < 0.25 else None)
        search_logs.append({
            "customer_id": rng.choices(customers, weights=customer_weights)[0].customer_id
            if rng.random() < 0.7 else None,
            "query": query, "results_count": results, "clicked_book_id": clicked,
            "device": rng.choices(DEVICES, weights=(55, 35, 10))[0], "ts": _at(rng, day),
        })
    search_logs.sort(key=lambda s: s["ts"])

    recommendations: list[Row] = []
    for customer in rng.sample(customers, 1500):
        picked = list({b.book_id: b for b in rng.choices(books, weights=book_weights, k=12)})[:5]
        score = rng.uniform(0.85, 0.99)
        items = []
        for rank, book_id in enumerate(picked, start=1):
            items.append({"book_id": book_id, "rank": rank, "score": round(score, 3)})
            score -= rng.uniform(0.02, 0.12)
        recommendations.append({
            "customer_id": customer.customer_id,
            "model": rng.choices(("collaborative", "content_based", "trending"), weights=(50, 30, 20))[0],
            "generated_at": _at(rng, _random_day(rng, date(2025, 6, 1), DATA_END)), "items": items,
        })
    recommendations.sort(key=lambda r: (r["generated_at"], r["customer_id"]))

    email_campaigns: list[Row] = []
    month = date(2023, 6, 1)
    while month <= DATA_END and len(email_campaigns) < 40:
        theme = CAMPAIGN_THEMES[len(email_campaigns) % len(CAMPAIGN_THEMES)]
        audience = rng.randint(800, 2000)
        opened = int(audience * rng.uniform(0.18, 0.42))
        email_campaigns.append({
            "campaign_id": len(email_campaigns) + 1, "name": f"{theme} {month:%B %Y}", "theme": theme,
            "sent_at": _at(rng, month + timedelta(days=rng.randint(0, 20))), "audience_size": audience,
            "opened": opened, "clicked": int(opened * rng.uniform(0.08, 0.3)),
            "featured_book_ids": sorted({b.book_id for b in rng.choices(books, weights=book_weights,
                                                                        k=rng.randint(3, 6))}),
        })
        month = (month + timedelta(days=32)).replace(day=1) if rng.random() < 0.85 else \
            (month + timedelta(days=62)).replace(day=1)

    return {"wishlists": wishlists, "search_logs": search_logs, "recommendations": recommendations,
            "email_campaigns": email_campaigns}
