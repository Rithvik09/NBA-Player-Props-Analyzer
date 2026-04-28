#!/usr/bin/env bash
# SQLite backup using the online .backup API.
#
# Why .backup and not `cp`?
#   `cp bankroll.db bankroll.db.bak` reads the file byte-for-byte. If the
#   process is mid-transaction (BEGIN ... COMMIT), the copy can capture an
#   inconsistent snapshot — half-written pages, an open WAL that hasn't been
#   checkpointed, or torn writes. SQLite's `.backup` command issues a proper
#   online backup using the C API: pages are copied while readers/writers
#   continue, with the engine itself coordinating consistency. The output
#   is always a valid, point-in-time SQLite file.
#
# Output: backups/<dbname>_YYYYMMDD_HHMMSS.db (gzipped).
# Usage:
#   ./scripts/backup_db.sh                       # backs up bankroll.db
#   ./scripts/backup_db.sh path/to/other.db      # backs up explicit file
#   ./scripts/backup_db.sh --keep 14             # rotate older than 14 days
#
# Cron example (daily at 03:30):
#   30 3 * * * cd /opt/ai-basketball && ./scripts/backup_db.sh --keep 14 >> logs/backup.log 2>&1
set -euo pipefail

DB_FILE=""
KEEP_DAYS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            KEEP_DAYS="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,18p' "$0"; exit 0 ;;
        *)
            DB_FILE="$1"; shift ;;
    esac
done

if [[ -z "$DB_FILE" ]]; then
    DB_FILE="${BANKROLL_DB:-bankroll.db}"
fi

if [[ ! -f "$DB_FILE" ]]; then
    echo "ERROR: db file not found: $DB_FILE" >&2
    exit 1
fi

BACKUP_DIR="${BACKUP_DIR:-backups}"
mkdir -p "$BACKUP_DIR"

DB_BASENAME="$(basename "$DB_FILE" .db)"
TS="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_PATH="${BACKUP_DIR}/${DB_BASENAME}_${TS}.db"

# .backup uses the SQLite online backup API; safe even if the process is
# actively writing. We then run an integrity_check on the COPY (not the
# source) so the backup itself is verified before we declare success.
sqlite3 "$DB_FILE" ".backup '$BACKUP_PATH'"
INTEG="$(sqlite3 "$BACKUP_PATH" 'PRAGMA integrity_check;' | head -n 1)"
if [[ "$INTEG" != "ok" ]]; then
    echo "ERROR: integrity_check failed on backup: $INTEG" >&2
    rm -f "$BACKUP_PATH"
    exit 2
fi

# gzip in-place — meaningful compression on SQLite (mostly text + small ints)
gzip -f "$BACKUP_PATH"
FINAL_PATH="${BACKUP_PATH}.gz"
echo "OK $FINAL_PATH ($(du -h "$FINAL_PATH" | cut -f1))"

# Retention: drop backups older than --keep days (matches both .db and .db.gz)
if [[ -n "$KEEP_DAYS" ]]; then
    if [[ ! "$KEEP_DAYS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --keep must be a non-negative integer" >&2
        exit 3
    fi
    # find -mtime +N is "strictly older than N+1 days" — close enough for ops
    find "$BACKUP_DIR" -maxdepth 1 -type f \( -name "${DB_BASENAME}_*.db" -o -name "${DB_BASENAME}_*.db.gz" \) \
        -mtime +"$KEEP_DAYS" -print -delete
fi
