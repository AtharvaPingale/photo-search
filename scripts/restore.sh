#!/bin/sh
# One-command restore for the Docker setup: `make restore` (newest backup) or
# `make restore BACKUP=backups/photos-20260101-030000.dump`.
# Stops everything that writes to the database, restores, re-applies migrations,
# restarts, and rebuilds any thumbnails missing from ./data.
set -eu
cd "$(dirname "$0")/.."
# BACKUPS_DIR from the environment or .env, like docker-compose.yml
[ -z "${BACKUPS_DIR:-}" ] && [ -f .env ] && BACKUPS_DIR=$(sed -n 's/^BACKUPS_DIR=//p' .env | tail -n 1)
BACKUPS_DIR="${BACKUPS_DIR:-./backups}"
BACKUP="${1:-$(ls -1t "$BACKUPS_DIR"/*.dump 2>/dev/null | head -n 1)}"
[ -n "$BACKUP" ] && [ -f "$BACKUP" ] || { echo "no backup found (looked in $BACKUPS_DIR)"; exit 1; }
echo "Restoring $BACKUP into the photos database. This replaces its current contents."
printf "Continue? [y/N] "
read -r ok
[ "$ok" = "y" ] || [ "$ok" = "Y" ] || exit 1

docker compose stop api worker ingest watcher
docker compose up -d db
docker compose exec -T db pg_restore --clean --if-exists --no-owner --single-transaction \
  -U photos -d photos "/backups/$(basename "$BACKUP")"
docker compose run --rm migrate
docker compose up -d
docker compose exec -T api photo-search repair
echo "Restored $BACKUP"
