#!/bin/sh
# Nightly pg_dump for the `backup` compose service: one custom-format dump per
# day at $BACKUP_HOUR (container local time), dumps older than $KEEP_DAYS pruned,
# the newest always kept. Restore with `make restore`.
set -eu
BACKUP_HOUR="${BACKUP_HOUR:-3}"
KEEP_DAYS="${KEEP_DAYS:-14}"
mkdir -p /backups

dump() {
  out="/backups/photos-$(date +%Y%m%d-%H%M%S).dump"
  if pg_dump --format=custom --compress=6 --file="$out.partial"; then
    mv "$out.partial" "$out"
    echo "$(date -Iseconds) backup ok: $out ($(du -h "$out" | cut -f1))"
  else
    rm -f "$out.partial"
    echo "$(date -Iseconds) backup FAILED" >&2
  fi
  newest=$(ls -1t /backups/*.dump 2>/dev/null | head -n 1 || true)
  find /backups -name '*.dump' -mtime +"$KEEP_DAYS" ! -path "$newest" -print -delete || true
}

# first run: back up right away if there is no backup from the last day
if [ -z "$(find /backups -name '*.dump' -mtime -1 2>/dev/null | head -n 1)" ]; then
  dump
fi
while true; do
  now=$(date +%s)
  next=$(date -d "today ${BACKUP_HOUR}:00" +%s 2>/dev/null || echo $((now + 86400)))
  [ "$next" -le "$now" ] && next=$((next + 86400))
  sleep $((next - now))
  dump
done
