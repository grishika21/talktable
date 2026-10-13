"""Deterministic generator for the "Online Bookstore" demo data.

All three seed scripts build their data through this module. Every generator uses its
own fixed seed, so:

* re-running a seed script always produces identical data, and
* the three databases stay consistent: the same `book_id` / `customer_id` means the same
  book / customer everywhere, whichever seed script runs first.

The data is shaped so that analytical questions have interesting answers: book sales
follow a long tail, orders peak in November/December, the customer base grows over time,
reviews only come from customers who actually received the book, and a few over-hyped
books sell well but collect lots of 1-star reviews.

Nothing here touches a database. The `seed_*.py` scripts do the writing.
"""

from __future__ import annotations

import math
import random
import unicodedata
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from itertools import accumulate
from string import Formatter
from typing import Any, Generic, TypeVar

from faker import Faker

from data.seed import vocab

SEED = 42

N_AUTHORS = 150
N_BOOKS = 1_000
N_CUSTOMERS = 2_000
N_ORDERS = 8_000
N_REVIEWS = 6_000
N_SESSIONS = 5_000

DATA_START = date(2022, 1, 1)  # first customer sign-up
DATA_END = date(2026, 9, 30)  # last day with any activity
CLICKSTREAM_START = date(2025, 1, 1)  # the clickstream keeps ~21 months of sessions

ORDER_STATUSES = ("processing", "shipped", "delivered", "cancelled", "returned")
DEVICES = ("mobile", "desktop", "tablet")
EVENT_TYPES = ("view", "wishlist", "add_to_cart", "remove_from_cart", "purchase")

# Order seasonality: holiday peak in Nov/Dec, a small back-to-school bump in August.
_MONTH_WEIGHT = {1: 0.85, 2: 0.8, 3: 0.9, 4: 0.9, 5: 0.95, 6: 1.0,
                 7: 1.05, 8: 1.1, 9: 0.95, 10: 1.0, 11: 1.3, 12: 1.6}
# Browsing sessions by hour of day: quiet at night, busiest in the evening.
_HOUR_WEIGHT = [1.0, 0.6, 0.4, 0.3, 0.3, 0.4, 0.8, 1.5, 2.0, 2.2, 2.2, 2.3,
                2.6, 2.4, 2.2, 2.2, 2.4, 2.8, 3.2, 3.6, 3.8, 3.4, 2.6, 1.6]

T = TypeVar("T")


# --- Records -----------------------------------------------------------------
# Field names match the database columns exactly, so `dataclasses.asdict()` gives
# ready-to-insert rows.


@dataclass(frozen=True)
class Author:
    author_id: int
    name: str
    country: str
    birth_year: int


@dataclass(frozen=True)
class Category:
    category_id: int
    name: str


@dataclass(frozen=True)
class Book:
    book_id: int
    title: str
    author_id: int
    category_id: int
    price: float  # USD
    published_year: int
    pages: int


@dataclass(frozen=True)
class Customer:
    customer_id: int
    name: str
    email: str
    city: str
    signup_date: date


@dataclass(frozen=True)
class Order:
    order_id: int
    customer_id: int
    order_date: date
    status: str
    total_amount: Decimal  # USD, sum of quantity * unit_price over the order's items


@dataclass(frozen=True)
class OrderItem:
    order_item_id: int
    order_id: int
    book_id: int
    quantity: int
    unit_price: Decimal  # USD, list price minus any discount


@dataclass(frozen=True)
class Catalog:
    authors: list[Author]
    categories: list[Category]
    books: list[Book]
    # Hidden traits (not stored in any database) that drive sales and review ratings.
    popularity: dict[int, float]  # book_id -> relative sales weight
    quality: dict[int, float]  # book_id -> mean review rating before noise


@dataclass(frozen=True)
class Sales:
    customers: list[Customer]
    orders: list[Order]
    order_items: list[OrderItem]
    activity: dict[int, float]  # customer_id -> how often they shop/browse (hidden)


# --- Helpers -----------------------------------------------------------------


class _WeightedPicker(Generic[T]):
    """Repeated weighted random choice from a fixed population (O(log n) per pick)."""

    def __init__(self, items: Sequence[T], weights: Sequence[float]) -> None:
        self.items = list(items)
        self.cum_weights = list(accumulate(weights))

    def pick(self, rng: random.Random) -> T:
        return rng.choices(self.items, cum_weights=self.cum_weights, k=1)[0]

    def pick_distinct(self, rng: random.Random, k: int) -> list[T]:
        k = min(k, len(self.items))
        chosen: list[T] = []
        while len(chosen) < k:
            item = self.pick(rng)
            if item not in chosen:
                chosen.append(item)
        return chosen


def _fakers(locales: Sequence[str], seed: int) -> dict[str, Faker]:
    """One independently seeded Faker per locale, so outputs never depend on call order elsewhere."""
    fakers: dict[str, Faker] = {}
    for offset, locale in enumerate(sorted(set(locales))):
        fake = Faker(locale)
        fake.seed_instance(seed + offset)
        fakers[locale] = fake
    return fakers


def _person_name(fake: Faker, locale: str) -> tuple[str, str]:
    """(first, last) name. Uses romanized names for Japanese so every name is Latin-script."""
    if locale == "ja_JP":
        return fake.first_romanized_name(), fake.last_romanized_name()
    return fake.first_name(), fake.last_name()


def _ascii_slug(text: str) -> str:
    """'José Núñez' -> 'josenunez' (used for email addresses)."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(ch for ch in folded.lower() if ch.isalnum())


def _pluralize(word: str) -> str:
    irregular = {"Mouse": "Mice", "Wolf": "Wolves"}
    if word in irregular:
        return irregular[word]
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def _money(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


# --- Catalog (SQLite) --------------------------------------------------------

# How to fill each title-pattern placeholder (see vocab.TITLE_PATTERNS).
_TITLE_SLOTS: dict[str, Callable[[random.Random, Faker], str]] = {
    "adj": lambda r, f: r.choice(vocab.ADJECTIVES),
    "noun": lambda r, f: r.choice(vocab.NOUNS),
    "nouns": lambda r, f: _pluralize(r.choice(vocab.NOUNS)),
    "place": lambda r, f: r.choice(vocab.PLACES),
    "verb": lambda r, f: r.choice(vocab.VERBS),
    "occupation": lambda r, f: r.choice(vocab.OCCUPATIONS),
    "relative": lambda r, f: r.choice(vocab.RELATIVES),
    "season": lambda r, f: r.choice(vocab.SEASONS),
    "star": lambda r, f: r.choice(vocab.STARS),
    "year": lambda r, f: str(r.randint(1650, 1985)),
    "first": lambda r, f: f.first_name(),
    "full": lambda r, f: f"{f.first_name()} {f.last_name()}",
    "h_topic": lambda r, f: r.choice(vocab.HISTORY_TOPICS),
    "s_one": lambda r, f: r.choice(vocab.SCIENCE_SINGULAR),
    "s_many": lambda r, f: r.choice(vocab.SCIENCE_PLURAL),
    "n": lambda r, f: str(r.choice((3, 5, 7, 10, 12))),
    "days": lambda r, f: str(r.choice((7, 21, 30, 90))),
    "sh_people": lambda r, f: r.choice(vocab.SELFHELP_PEOPLE),
    "sh_gerund": lambda r, f: r.choice(vocab.SELFHELP_GERUNDS),
    "sh_noun": lambda r, f: r.choice(vocab.SELFHELP_NOUNS),
    "sh_thing": lambda r, f: r.choice(vocab.SELFHELP_THINGS),
    "sh_little": lambda r, f: r.choice(vocab.SELFHELP_LITTLE),
    "b_adj": lambda r, f: r.choice(vocab.BUSINESS_ADJ),
    "b_value": lambda r, f: r.choice(vocab.BUSINESS_VALUES),
    "b_scale": lambda r, f: r.choice(vocab.BUSINESS_SCALE),
    "b_play": lambda r, f: r.choice(vocab.BUSINESS_PLAYBOOKS),
    "b_object": lambda r, f: r.choice(vocab.BUSINESS_OBJECTS),
    "animal": lambda r, f: r.choice(vocab.ANIMALS),
    "animals": lambda r, f: _pluralize(r.choice(vocab.ANIMALS)),
    "c_adj": lambda r, f: r.choice(vocab.CHILDREN_ADJ),
    "c_noun": lambda r, f: r.choice(vocab.CHILDREN_NOUNS),
}


def _make_title(rng: random.Random, fake: Faker, category: str, suffix: str, taken: set[str]) -> str:
    """A unique, genre-appropriate title such as 'The Crimson Lighthouse' or 'Murder at Ashford'."""
    subtitles = vocab.SUBTITLES.get(category, [])
    for attempt in range(100):
        pattern = rng.choice(vocab.TITLE_PATTERNS[category])
        fields = {name for _, name, _, _ in Formatter().parse(pattern) if name}
        title = pattern.format(**{name: _TITLE_SLOTS[name](rng, fake) for name in sorted(fields)})
        # Non-fiction: often add a subtitle, and always once plain titles start colliding.
        if subtitles and (attempt >= 3 or rng.random() < 0.5):
            title = f"{title}: {rng.choice(subtitles)}"
        title += suffix
        if title not in taken:
            taken.add(title)
            return title
    raise RuntimeError(f"could not generate a unique {category} title")


@lru_cache(maxsize=1)
def generate_catalog() -> Catalog:
    """Authors, categories and books for the SQLite catalog database."""
    rng = random.Random(SEED)
    fakers = _fakers([locale for locale, _, _ in vocab.AUTHOR_ORIGINS], SEED)
    current_year = DATA_END.year

    categories = [Category(i, name) for i, (name, *_) in enumerate(vocab.CATEGORIES, start=1)]
    price_range = {name: prices for name, _, prices, _ in vocab.CATEGORIES}
    page_range = {name: pages for name, _, _, pages in vocab.CATEGORIES}
    category_picker = _WeightedPicker(categories, [share for _, share, _, _ in vocab.CATEGORIES])
    origin_picker = _WeightedPicker(vocab.AUTHOR_ORIGINS, [w for *_, w in vocab.AUTHOR_ORIGINS])

    # Authors, each writing mainly in one genre (sometimes two).
    authors: list[Author] = []
    author_genres: dict[int, list[Category]] = {}
    taken_names: set[str] = set()
    for author_id in range(1, N_AUTHORS + 1):
        locale, country, _ = origin_picker.pick(rng)
        while True:
            first, last = _person_name(fakers[locale], locale)
            name = f"{first} {last}"
            if name not in taken_names:
                taken_names.add(name)
                break
        authors.append(Author(author_id, name, country, birth_year=rng.randint(1925, 1998)))
        genres = [category_picker.pick(rng)]
        if rng.random() < 0.3:
            genres.append(category_picker.pick(rng))
        author_genres[author_id] = genres

    # A handful of prolific authors write a big share of the catalog.
    author_picker = _WeightedPicker(authors, [min(rng.paretovariate(1.6), 12.0) for _ in authors])

    books: list[Book] = []
    popularity: dict[int, float] = {}
    quality: dict[int, float] = {}
    taken_titles: set[str] = set()
    for book_id in range(1, N_BOOKS + 1):
        author = author_picker.pick(rng)
        genres = author_genres[author.author_id]
        roll = rng.random()
        category = genres[0] if roll < 0.75 else genres[-1] if roll < 0.95 else rng.choice(categories)

        collector = rng.random() < 0.03  # pricey special editions
        suffix = " (Collector's Edition)" if collector else ""
        title = _make_title(rng, fakers["en_US"], category.name, suffix, taken_titles)

        low, high = price_range[category.name]
        base = rng.triangular(low, high, low + 0.35 * (high - low))
        if collector:
            base *= rng.uniform(2.5, 4.0)
        price = round(max(math.floor(base), 1) + rng.choice((0.99, 0.99, 0.49, 0.95)), 2)

        # Published during the author's working life, skewed toward recent years.
        first_year = author.birth_year + 22
        last_year = min(current_year, author.birth_year + 85)
        published_year = max(first_year, last_year - int(rng.expovariate(1 / 10)))

        p_low, p_high = page_range[category.name]
        pages = int(rng.triangular(p_low, p_high, p_low + 0.4 * (p_high - p_low)))

        books.append(Book(book_id, title, author.author_id, category.category_id, price, published_year, pages))

        trait = rng.random()
        if trait < 0.01:  # over-hyped flop: sells well, reviews badly
            quality[book_id] = rng.uniform(1.8, 2.4)
            popularity[book_id] = rng.uniform(15.0, 30.0)
        elif trait < 0.06:  # quiet dud
            quality[book_id] = rng.uniform(1.6, 2.6)
            popularity[book_id] = min(rng.paretovariate(1.15), 40.0) * 0.6
        else:
            quality[book_id] = min(4.9, max(2.6, rng.gauss(3.95, 0.5)))
            popularity[book_id] = min(rng.paretovariate(1.15), 40.0) * (0.6 + 0.1 * quality[book_id])

    return Catalog(authors, categories, books, popularity, quality)


# --- Sales (PostgreSQL) ------------------------------------------------------


def _book_pickers_by_year(catalog: Catalog, years: range) -> dict[int, _WeightedPicker[Book]]:
    """For each year, a popularity-weighted picker over books already published by then."""
    pickers: dict[int, _WeightedPicker[Book]] = {}
    for year in years:
        eligible = [b for b in catalog.books if b.published_year <= year]
        pickers[year] = _WeightedPicker(eligible, [catalog.popularity[b.book_id] for b in eligible])
    return pickers


def _unique_email(rng: random.Random, first: str, last: str, taken: set[str]) -> str:
    local = ".".join(part for part in (_ascii_slug(first), _ascii_slug(last)) if part) or "reader"
    if rng.random() < 0.35:
        local += str(rng.randint(1, 99))
    domain = rng.choice(vocab.EMAIL_DOMAINS)
    email = f"{local}@{domain}"
    while email in taken:
        email = f"{local}{rng.randint(100, 9999)}@{domain}"
    taken.add(email)
    return email


def _order_status(rng: random.Random, order_date: date) -> str:
    """Recent orders are still in flight; older ones are delivered, cancelled or returned."""
    age_days = (DATA_END - order_date).days
    if age_days <= 3:
        return rng.choices(("processing", "shipped"), weights=(60, 40))[0]
    if age_days <= 10:
        return rng.choices(("shipped", "delivered", "cancelled"), weights=(50, 45, 5))[0]
    return rng.choices(("delivered", "cancelled", "returned"), weights=(88, 6, 6))[0]


@lru_cache(maxsize=1)
def generate_sales() -> Sales:
    """Customers, orders and order items for the PostgreSQL sales database."""
    catalog = generate_catalog()
    rng = random.Random(SEED + 1)
    fakers = _fakers([locale for _, locale, _ in vocab.CUSTOMER_CITIES], SEED + 1)
    city_picker = _WeightedPicker(vocab.CUSTOMER_CITIES, [w for *_, w in vocab.CUSTOMER_CITIES])

    # Customers. Sign-ups accelerate over time (the store is growing), and IDs follow
    # sign-up order like a real auto-increment key would.
    span_days = (DATA_END - DATA_START).days
    drafts: list[tuple[date, str, str, str]] = []
    taken_emails: set[str] = set()
    for _ in range(N_CUSTOMERS):
        city, locale, _ = city_picker.pick(rng)
        first, last = _person_name(fakers[locale], locale)
        email = _unique_email(rng, first, last, taken_emails)
        signup = DATA_START + timedelta(days=int(rng.random() ** 0.75 * (span_days - 1)))
        drafts.append((signup, f"{first} {last}", email, city))
    drafts.sort(key=lambda d: d[0])  # stable: ties keep generation order
    customers = [Customer(i, name, email, city, signup)
                 for i, (signup, name, email, city) in enumerate(drafts, start=1)]
    activity = {c.customer_id: rng.lognormvariate(0, 0.9) for c in customers}

    # Orders: busier customers, and customers who joined earlier, place more of them.
    customer_picker = _WeightedPicker(
        customers, [activity[c.customer_id] * max(1, (DATA_END - c.signup_date).days) for c in customers]
    )
    peak = max(_MONTH_WEIGHT.values())
    order_drafts: list[tuple[date, int]] = []
    for _ in range(N_ORDERS):
        customer = customer_picker.pick(rng)
        for _attempt in range(50):  # rejection sampling for seasonality
            order_date = _random_date(rng, customer.signup_date, DATA_END)
            if rng.random() < _MONTH_WEIGHT[order_date.month] / peak:
                break
        order_drafts.append((order_date, customer.customer_id))
    order_drafts.sort()  # order IDs increase with time

    book_pickers = _book_pickers_by_year(catalog, range(DATA_START.year, DATA_END.year + 1))
    orders: list[Order] = []
    items: list[OrderItem] = []
    for order_id, (order_date, customer_id) in enumerate(order_drafts, start=1):
        n_items = rng.choices((1, 2, 3, 4, 5), weights=(25, 30, 22, 13, 10))[0]
        total = Decimal("0.00")
        for book in book_pickers[order_date.year].pick_distinct(rng, n_items):
            quantity = rng.choices((1, 2, 3), weights=(80, 15, 5))[0]
            unit_price = _money(book.price)
            if rng.random() < 0.15:  # promotional discount
                unit_price = _money(unit_price * (1 - Decimal(rng.choice(("0.10", "0.15", "0.20")))))
            items.append(OrderItem(len(items) + 1, order_id, book.book_id, quantity, unit_price))
            total += unit_price * quantity
        orders.append(Order(order_id, customer_id, order_date, _order_status(rng, order_date), _money(total)))

    return Sales(customers, orders, items, activity)


# --- Engagement (MongoDB) ----------------------------------------------------
# These return fresh lists of dicts on every call (no caching), because pymongo's
# insert_many() adds an `_id` key to each document it inserts.


def _review_text(rng: random.Random, rating: int, category: str, returned: bool) -> str:
    tone = "pos" if rating >= 4 else "mid" if rating == 3 else "neg"
    if category == "Children's":
        group = "children"
    elif category == "Poetry":
        group = "poetry"
    elif category in vocab.NONFICTION:
        group = "nonfiction"
    else:
        group = "fiction"
    details = vocab.REVIEW_DETAILS[(group, tone)]
    parts = [rng.choice(vocab.REVIEW_OPENERS[rating])]
    parts += rng.sample(details, k=min(len(details), rng.choice((1, 1, 2))))
    if returned and tone == "neg":
        parts.append(rng.choice(vocab.RETURN_COMPLAINTS))
    else:
        parts.append(rng.choice(vocab.REVIEW_CLOSERS[tone]))
    return " ".join(part for part in parts if part)


def generate_reviews() -> list[dict[str, Any]]:
    """Book reviews, each written by a customer who received that book (verified purchase)."""
    catalog = generate_catalog()
    sales = generate_sales()
    rng = random.Random(SEED + 2)
    category_names = {c.category_id: c.name for c in catalog.categories}
    category_of = {b.book_id: category_names[b.category_id] for b in catalog.books}
    order_by_id = {o.order_id: o for o in sales.orders}

    # (customer_id, book_id) -> (date of first completed purchase, was that order returned?)
    purchases: dict[tuple[int, int], tuple[date, bool]] = {}
    for item in sales.order_items:  # already in date order, so the first purchase wins
        order = order_by_id[item.order_id]
        if order.status in ("delivered", "returned") and (DATA_END - order.order_date).days >= 5:
            purchases.setdefault((order.customer_id, item.book_id),
                                 (order.order_date, order.status == "returned"))
    if len(purchases) < N_REVIEWS:
        raise RuntimeError(f"only {len(purchases)} completed purchases, need {N_REVIEWS} reviews")

    reviews: list[dict[str, Any]] = []
    for customer_id, book_id in rng.sample(sorted(purchases), N_REVIEWS):
        bought_on, returned = purchases[(customer_id, book_id)]
        mean = catalog.quality[book_id] - (1.5 if returned else 0.0)
        rating = min(5, max(1, round(rng.gauss(mean, 0.9))))
        day = min(DATA_END, bought_on + timedelta(days=rng.randint(4, 75)))
        created_at = datetime.combine(day, time(rng.randint(7, 23), rng.randint(0, 59), rng.randint(0, 59)))
        # Strong opinions and older reviews collect more "helpful" votes.
        votes = rng.expovariate(1 / 2.5) * (1.6 if rating in (1, 5) else 1.0) + (DATA_END - day).days / 150
        reviews.append({
            "book_id": book_id,
            "customer_id": customer_id,
            "rating": rating,
            "text": _review_text(rng, rating, category_of[book_id], returned),
            "created_at": created_at,
            "helpful_votes": int(votes),
        })
    reviews.sort(key=lambda r: (r["created_at"], r["customer_id"], r["book_id"]))
    return reviews


def _session_events(rng: random.Random, start: datetime, picker: _WeightedPicker[Book]) -> list[dict[str, Any]]:
    """A browsing session: views, some wishlists and cart adds, and sometimes a checkout."""
    clock = start

    def event(kind: str, book_id: int) -> dict[str, Any]:
        nonlocal clock
        record = {"type": kind, "book_id": book_id, "ts": clock}
        clock += timedelta(seconds=rng.randint(8, 240))
        return record

    events: list[dict[str, Any]] = []
    cart: list[int] = []
    n_views = 1 + min(int(rng.expovariate(1 / 3)), 14)
    for book in picker.pick_distinct(rng, n_views):
        events.append(event("view", book.book_id))
        if rng.random() < 0.08:
            events.append(event("wishlist", book.book_id))
        if rng.random() < 0.22:
            events.append(event("add_to_cart", book.book_id))
            cart.append(book.book_id)

    kept: list[int] = []
    for book_id in cart:
        if rng.random() < 0.15:
            events.append(event("remove_from_cart", book_id))
        else:
            kept.append(book_id)
    if kept and rng.random() < 0.55:  # checkout
        events.extend(event("purchase", book_id) for book_id in kept)
    return events


def generate_clickstream() -> list[dict[str, Any]]:
    """Website browsing sessions with their ordered list of events."""
    catalog = generate_catalog()
    sales = generate_sales()
    rng = random.Random(SEED + 3)

    def days_in_window(c: Customer) -> int:
        return max(1, (DATA_END - max(c.signup_date, CLICKSTREAM_START)).days)

    customer_picker = _WeightedPicker(
        sales.customers, [sales.activity[c.customer_id] * days_in_window(c) for c in sales.customers]
    )
    preferred_device = {c.customer_id: rng.choices(DEVICES, weights=(55, 35, 10))[0] for c in sales.customers}
    book_pickers = _book_pickers_by_year(catalog, range(CLICKSTREAM_START.year, DATA_END.year + 1))

    sessions: list[dict[str, Any]] = []
    for _ in range(N_SESSIONS):
        customer = customer_picker.pick(rng)
        day = _random_date(rng, max(customer.signup_date, CLICKSTREAM_START), DATA_END)
        hour = rng.choices(range(24), weights=_HOUR_WEIGHT)[0]
        start = datetime.combine(day, time(hour, rng.randint(0, 59), rng.randint(0, 59)))
        device = preferred_device[customer.customer_id] if rng.random() < 0.7 else rng.choice(DEVICES)
        sessions.append({
            "customer_id": customer.customer_id,
            "session_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "events": _session_events(rng, start, book_pickers[day.year]),
            "device": device,
        })
    sessions.sort(key=lambda s: (s["events"][0]["ts"], s["session_id"]))
    return sessions
