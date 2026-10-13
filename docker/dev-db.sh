#!/usr/bin/env bash
# Local development databases for TalkTable, using plain `docker run` (no compose plugin needed).
#
#   docker/dev-db.sh up       start Postgres (127.0.0.1:5433) and Mongo (127.0.0.1:27017)
#   docker/dev-db.sh status   show container health
#   docker/dev-db.sh down     stop and remove the containers AND their data volumes
#
# Passwords are throwaway local-dev defaults (override them with env vars). Production
# uses Neon and MongoDB Atlas instead. Both containers listen on 127.0.0.1 only.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PG=talktable-postgres
MONGO=talktable-mongo
PG_ADMIN_PASSWORD="${POSTGRES_ADMIN_PASSWORD:-talktable_dev}"
MONGO_ROOT_PASSWORD="${MONGO_ROOT_PASSWORD:-root_dev}"
RO_PASSWORD="${TALKTABLE_RO_PASSWORD:-talktable_ro_dev}"

exists() { docker container inspect "$1" >/dev/null 2>&1; }

wait_healthy() {
  local name=$1 status
  for _ in $(seq 1 60); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$name" 2>/dev/null || echo missing)
    if [[ $status == healthy ]]; then echo "  $name: healthy"; return 0; fi
    sleep 2
  done
  echo "  $name did not become healthy (last status: $status). See: docker logs $name" >&2
  return 1
}

up() {
  if exists "$PG"; then
    docker start "$PG" >/dev/null
  else
    # Init scripts run only on first start, against an empty data volume.
    # The health check uses TCP because Postgres listens only on a unix socket while it
    # runs init scripts, so the container isn't reported healthy until init is done.
    docker run -d --name "$PG" \
      -e POSTGRES_DB=bookstore \
      -e POSTGRES_USER=talktable \
      -e POSTGRES_PASSWORD="$PG_ADMIN_PASSWORD" \
      -e TALKTABLE_RO_PASSWORD="$RO_PASSWORD" \
      -p 127.0.0.1:5433:5432 \
      -v talktable-pgdata:/var/lib/postgresql/data \
      -v "$HERE/postgres-init.sh:/docker-entrypoint-initdb.d/10-readonly-user.sh:ro" \
      --health-cmd "pg_isready -h 127.0.0.1 -U talktable -d bookstore" \
      --health-interval 2s --health-timeout 3s --health-retries 30 \
      postgres:16-alpine >/dev/null
  fi

  if exists "$MONGO"; then
    docker start "$MONGO" >/dev/null
  else
    docker run -d --name "$MONGO" \
      -e MONGO_INITDB_ROOT_USERNAME=root \
      -e MONGO_INITDB_ROOT_PASSWORD="$MONGO_ROOT_PASSWORD" \
      -e MONGO_INITDB_DATABASE=bookstore \
      -e TALKTABLE_RO_PASSWORD="$RO_PASSWORD" \
      -p 127.0.0.1:27017:27017 \
      -v talktable-mongodata:/data/db \
      -v "$HERE/mongo-init.js:/docker-entrypoint-initdb.d/10-readonly-user.js:ro" \
      --health-cmd "mongosh --quiet --eval 'db.adminCommand(\"ping\").ok'" \
      --health-interval 2s --health-timeout 5s --health-retries 30 \
      mongo:8.0 >/dev/null
  fi

  echo "Waiting for databases..."
  wait_healthy "$PG"
  wait_healthy "$MONGO"
}

down() {
  docker rm -f "$PG" "$MONGO" >/dev/null 2>&1 || true
  docker volume rm talktable-pgdata talktable-mongodata >/dev/null 2>&1 || true
  echo "Removed containers and volumes."
}

status() {
  for name in "$PG" "$MONGO"; do
    if exists "$name"; then
      echo "$name: $(docker inspect -f '{{.State.Status}} ({{.State.Health.Status}})' "$name")"
    else
      echo "$name: not created"
    fi
  done
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 {up|down|status}" >&2; exit 2 ;;
esac
