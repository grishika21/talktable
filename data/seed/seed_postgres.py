"""Load the PostgreSQL sales database: customers, orders and order_items, plus payments,
shipments, carriers, returns, coupons, warehouses, inventory, loyalty and support tickets.

Usage (from the project root):
    python -m data.seed.seed_postgres [--url postgresql://...]

The URL defaults to POSTGRES_SEED_URL (a user that can create tables). Only the tables
below are dropped and recreated; anything else in the database is left alone.
`book_id` in order_items refers to books in the SQLite catalog. A foreign key can't
span two databases, so that link is enforced by the generator instead.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    create_engine,
    func,
    select,
    text,
)

from talktable.config import get_settings, mask_url
from data.seed.common import print_counts, resolve_url
from data.seed.extended import generate_sales_extras
from data.seed.generate import ORDER_STATUSES, generate_sales

metadata = MetaData()

customers = Table(
    "customers",
    metadata,
    Column("customer_id", Integer, primary_key=True, autoincrement=False),
    Column("name", String(120), nullable=False),
    Column("email", String(255), nullable=False, unique=True),
    Column("city", String(80), nullable=False),
    Column("signup_date", Date, nullable=False),
)

orders = Table(
    "orders",
    metadata,
    Column("order_id", Integer, primary_key=True, autoincrement=False),
    Column("customer_id", Integer, ForeignKey("customers.customer_id"), nullable=False, index=True),
    Column("order_date", Date, nullable=False, index=True),
    Column("status", String(20), nullable=False),
    Column("total_amount", Numeric(10, 2), nullable=False),
    CheckConstraint(
        "status IN (" + ", ".join(f"'{s}'" for s in ORDER_STATUSES) + ")", name="orders_status_check"
    ),
)

order_items = Table(
    "order_items",
    metadata,
    Column("order_item_id", Integer, primary_key=True, autoincrement=False),
    Column("order_id", Integer, ForeignKey("orders.order_id"), nullable=False, index=True),
    Column("book_id", Integer, nullable=False, index=True),  # -> SQLite books.book_id
    Column("quantity", Integer, CheckConstraint("quantity > 0"), nullable=False),
    Column("unit_price", Numeric(10, 2), nullable=False),
)

carriers = Table(
    "carriers", metadata,
    Column("carrier_id", Integer, primary_key=True, autoincrement=False),
    Column("name", String(80), nullable=False, unique=True),
)

payments = Table(
    "payments", metadata,
    Column("payment_id", Integer, primary_key=True, autoincrement=False),
    Column("order_id", Integer, ForeignKey("orders.order_id"), nullable=False, index=True),
    Column("method", String(20), nullable=False),
    Column("amount", Numeric(10, 2), nullable=False),
    Column("paid_at", DateTime, nullable=False),
    Column("status", String(20), nullable=False),
    CheckConstraint("method IN ('card', 'paypal', 'gift_card', 'bank_transfer')", name="payments_method_check"),
    CheckConstraint("status IN ('paid', 'refunded', 'failed')", name="payments_status_check"),
)

shipments = Table(
    "shipments", metadata,
    Column("shipment_id", Integer, primary_key=True, autoincrement=False),
    Column("order_id", Integer, ForeignKey("orders.order_id"), nullable=False, index=True),
    Column("carrier_id", Integer, ForeignKey("carriers.carrier_id"), nullable=False, index=True),
    Column("shipped_date", Date, nullable=False),
    Column("delivered_date", Date),
    Column("status", String(20), nullable=False),
)

returns = Table(
    "returns", metadata,
    Column("return_id", Integer, primary_key=True, autoincrement=False),
    Column("order_item_id", Integer, ForeignKey("order_items.order_item_id"), nullable=False, index=True),
    Column("reason", String(40), nullable=False),
    Column("refund_amount", Numeric(10, 2), nullable=False),
    Column("return_date", Date, nullable=False),
)

coupons = Table(
    "coupons", metadata,
    Column("coupon_id", Integer, primary_key=True, autoincrement=False),
    Column("code", String(20), nullable=False, unique=True),
    Column("discount_pct", Integer, nullable=False),
    Column("valid_from", Date, nullable=False),
    Column("valid_to", Date, nullable=False),
)

order_coupons = Table(
    "order_coupons", metadata,
    Column("order_id", Integer, ForeignKey("orders.order_id"), primary_key=True, autoincrement=False),
    Column("coupon_id", Integer, ForeignKey("coupons.coupon_id"), nullable=False, index=True),
)

warehouses = Table(
    "warehouses", metadata,
    Column("warehouse_id", Integer, primary_key=True, autoincrement=False),
    Column("name", String(80), nullable=False, unique=True),
    Column("city", String(80), nullable=False),
    Column("country", String(80), nullable=False),
)

inventory = Table(
    "inventory", metadata,
    Column("warehouse_id", Integer, ForeignKey("warehouses.warehouse_id"), primary_key=True, autoincrement=False),
    Column("book_id", Integer, primary_key=True, autoincrement=False),  # -> SQLite books.book_id
    Column("quantity_on_hand", Integer, nullable=False),
    Column("reorder_level", Integer, nullable=False),
    Column("last_restocked", Date, nullable=False),
)

loyalty_accounts = Table(
    "loyalty_accounts", metadata,
    Column("customer_id", Integer, ForeignKey("customers.customer_id"), primary_key=True, autoincrement=False),
    Column("tier", String(20), nullable=False),
    Column("points", Integer, nullable=False),
    Column("joined_date", Date, nullable=False),
    CheckConstraint("tier IN ('bronze', 'silver', 'gold', 'platinum')", name="loyalty_tier_check"),
)

support_tickets = Table(
    "support_tickets", metadata,
    Column("ticket_id", Integer, primary_key=True, autoincrement=False),
    Column("customer_id", Integer, ForeignKey("customers.customer_id"), nullable=False, index=True),
    Column("order_id", Integer, ForeignKey("orders.order_id"), index=True),
    Column("category", String(30), nullable=False),
    Column("opened_at", DateTime, nullable=False),
    Column("resolved_at", DateTime),
    Column("satisfaction", Integer),
)

# Insert order: parents before children.
EXTRA_TABLES = (carriers, payments, shipments, returns, coupons, order_coupons, warehouses, inventory,
                loyalty_accounts, support_tickets)


def seed(url: str) -> dict[str, int]:
    """Drop, recreate and fill the sales tables. Returns row counts per table."""
    sales = generate_sales()
    engine = create_engine(url)
    try:
        with engine.begin() as conn:  # one transaction: all or nothing
            metadata.drop_all(conn)
            metadata.create_all(conn)
            # SQLAlchemy batches these into multi-row INSERTs, which matters over a
            # network link to a hosted database such as Neon.
            conn.execute(customers.insert(), [asdict(c) for c in sales.customers])
            conn.execute(orders.insert(), [asdict(o) for o in sales.orders])
            conn.execute(order_items.insert(), [asdict(i) for i in sales.order_items])
            extras = generate_sales_extras()
            for table in EXTRA_TABLES:
                conn.execute(table.insert(), extras[table.name])
        with engine.connect() as conn:
            all_tables = (customers, orders, order_items, *EXTRA_TABLES)
            conn.execute(text("ANALYZE " + ", ".join(t.name for t in all_tables)))
            return {t.name: conn.execute(select(func.count()).select_from(t)).scalar_one() for t in all_tables}
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Postgres URL (default: POSTGRES_SEED_URL)")
    args = parser.parse_args()

    settings = get_settings()
    url = resolve_url(args.url, settings.postgres_seed_url, settings.postgres_url, "postgres")
    print(f"[postgres] seeding {mask_url(url)}")
    print_counts("postgres", seed(url))


if __name__ == "__main__":
    main()
