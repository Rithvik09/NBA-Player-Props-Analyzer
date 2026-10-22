#!/usr/bin/env bash
# PostgreSQL backup using pg_dump.
#
# Why pg_dump and not a filesystem copy?
#   Copying the data directory while the server is running captures
#   half-written pages and an un-checkpointed WAL — an inconsistent
#   snapshot. pg_dump runs inside a single repeatable-read transaction, so
#   the output is always a consistent point-in-time logical backup, taken
#   while readers and writers carry on. It is also the only form that
#   restores onto a managed host like Neon, where there is no data
#   directory to copy.
#
# Output: backups/<dbname>_YYYYMMDD_HHMMSS.sql (gzipped).
# Usage:
#   ./scripts/backup_db.sh                       # backs up $DATABASE_URL
#   ./scripts/backup_db.sh postgresql://...      # backs up explicit DSN
#   ./scripts/backup_db.sh --keep 14             # rotate older than 14 days
#
# Cron example (daily at 03:30):
#   30 3 * * * cd /opt/ai-basketball && ./scripts/backup_db.sh --keep 14 >> logs/backup.log 2>&1
set -euo pipefail

DB_DSN=""
KEEP_DAYS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            KEEP_DAYS="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,21p' "$0"; exit 0 ;;
        *)
            DB_DSN="$1"; shift ;;
    esac
done

if [[ -z "$DB_DSN" ]]; then
    DB_DSN="${DATABASE_URL:-}"
fi

if [[ -z "$DB_DSN" ]]; then
    echo "ERROR: no connection string: pass a DSN or set DATABASE_URL" >&2
    exit 1
fi

BACKUP_DIR="${BACKUP_DIR:-backups}"
mkdir -p "$BACKUP_DIR"

# Name the artifact after the database in the DSN, falling back to "db".
DB_BASENAME="$(basename "${DB_DSN%%\?*}")"
DB_BASENAME="${DB_BASENAME:-db}"
TS="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_PATH="${BACKUP_DIR}/${DB_BASENAME}_${TS}.sql"

if ! pg_dump --dbname="$DB_DSN" --no-owner --no-privileges --file="$BACKUP_PATH"; then
    echo "ERROR: pg_dump failed for $DB_BASENAME" >&2
    rm -f "$BACKUP_PATH"
    exit 1
fi

# pg_dump writes this trailer only after the dump has run to completion,
# so its presence is what distinguishes a finished backup from a truncated
# one. We check the COPY, not the source — same reasoning as before.
if ! tail -n 20 "$BACKUP_PATH" | grep -q "PostgreSQL database dump complete"; then
    echo "ERROR: dump is incomplete (no completion marker): $BACKUP_PATH" >&2
    rm -f "$BACKUP_PATH"
    exit 2
fi

# gzip in-place — a plain-SQL dump compresses very well
gzip -f "$BACKUP_PATH"
FINAL_PATH="${BACKUP_PATH}.gz"
echo "OK $FINAL_PATH ($(du -h "$FINAL_PATH" | cut -f1))"

# Retention: drop backups older than --keep days (matches both .sql and .sql.gz)
if [[ -n "$KEEP_DAYS" ]]; then
    if [[ ! "$KEEP_DAYS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --keep must be a non-negative integer" >&2
        exit 3
    fi
    # find -mtime +N is "strictly older than N+1 days" — close enough for ops
    find "$BACKUP_DIR" -maxdepth 1 -type f \( -name "${DB_BASENAME}_*.sql" -o -name "${DB_BASENAME}_*.sql.gz" \) \
        -mtime +"$KEEP_DAYS" -print -delete
fi
