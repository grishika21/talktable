#!/bin/sh
# Runs once, when the Postgres container is first created.
# Creates the read-only login `talktable_ro` that the TalkTable app connects with.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE ROLE talktable_ro LOGIN PASSWORD '${TALKTABLE_RO_PASSWORD}';

    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO talktable_ro;
    GRANT USAGE ON SCHEMA public TO talktable_ro;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO talktable_ro;

    -- Tables the seed script creates later (owned by $POSTGRES_USER) are readable too.
    ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
        GRANT SELECT ON TABLES TO talktable_ro;

    -- Defence in depth: every session for this role is read-only and time-limited.
    ALTER ROLE talktable_ro SET default_transaction_read_only = on;
    ALTER ROLE talktable_ro SET statement_timeout = '5s';
EOSQL
