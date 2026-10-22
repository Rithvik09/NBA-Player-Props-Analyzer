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
# The connection string is read from DATABASE_URL and from nowhere else.
# It used to be accepted as a positional argument too, which put the
# password into shell history and into the crontab line; it was then
# handed to pg_dump as --dbname=, which put it in `ps` for the length of
# the dump, and used as the output filename, which put it in directory
# listings and in the cron log. The DSN is split into PG* environment
# variables here instead, and the artifact is named after the database.
#
# Output: backups/<dbname>_YYYYMMDD_HHMMSS.sql.gz, mode 0600.
# Usage:
#   DATABASE_URL=postgresql://... ./scripts/backup_db.sh
#   DATABASE_URL=postgresql://... ./scripts/backup_db.sh --keep 14
#
# Cron example (daily at 03:30):
#   30 3 * * * cd /opt/ai-basketball && ./scripts/backup_db.sh --keep 14 >> logs/backup.log 2>&1
set -euo pipefail

# Before anything is created: the dump carries full bankroll, bets and
# prediction data, and gzip preserves whatever mode it finds. The default
# umask of 022 left it 0644 — world-readable.
umask 077

KEEP_DAYS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            KEEP_DAYS="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,27p' "$0"; exit 0 ;;
        *)
            # Deliberately does not echo the argument: it used to be a DSN.
            echo "ERROR: unexpected argument. The connection string is read" \
                 "from DATABASE_URL — passing one on the command line puts" \
                 "the password in ps, in shell history and in the crontab." >&2
            exit 1 ;;
    esac
done

DB_DSN="${DATABASE_URL:-}"
if [[ -z "$DB_DSN" ]]; then
    echo "ERROR: no connection string: set DATABASE_URL" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to split DATABASE_URL into PG* variables" >&2
    exit 1
fi

# Split the DSN into the PG* environment variables libpq reads, so the
# password reaches pg_dump through its environment (readable only by this
# user) rather than through argv (readable by everyone with `ps`).
# Stdlib only: this has to run under whatever python3 cron happens to have.
PG_ENV="$(BACKUP_DSN="$DB_DSN" python3 -I - <<'PY'
import os
import shlex
import sys
from urllib.parse import unquote

dsn = os.environ["BACKUP_DSN"].strip()

# libpq connection keyword -> environment variable.
ENV_FOR = {
    "host": "PGHOST", "hostaddr": "PGHOSTADDR", "port": "PGPORT",
    "user": "PGUSER", "password": "PGPASSWORD", "dbname": "PGDATABASE",
    "passfile": "PGPASSFILE", "service": "PGSERVICE",
    "options": "PGOPTIONS", "application_name": "PGAPPNAME",
    "connect_timeout": "PGCONNECT_TIMEOUT", "client_encoding": "PGCLIENTENCODING",
    "sslmode": "PGSSLMODE", "sslcert": "PGSSLCERT", "sslkey": "PGSSLKEY",
    "sslrootcert": "PGSSLROOTCERT", "sslcrl": "PGSSLCRL",
    "sslcompression": "PGSSLCOMPRESSION", "requiressl": "PGREQUIRESSL",
    "krbsrvname": "PGKRBSRVNAME", "gsslib": "PGGSSLIB",
    "gssencmode": "PGGSSENCMODE", "channel_binding": "PGCHANNELBINDING",
    "target_session_attrs": "PGTARGETSESSIONATTRS",
}


def split_hostport(hostport):
    """``h:5432``, ``[::1]:5432`` and libpq's comma-separated host lists."""
    hosts, ports = [], []
    for item in hostport.split(","):
        if item.startswith("["):
            host, _, rest = item.partition("]")
            hosts.append(unquote(host[1:]))
            ports.append(unquote(rest[1:]) if rest.startswith(":") else "")
        else:
            host, sep, port = item.partition(":")
            hosts.append(unquote(host))
            ports.append(unquote(port) if sep else "")
    return ",".join(hosts), ",".join(ports)


def parse_uri(dsn):
    rest = dsn.split("://", 1)[1]
    out = {}
    if "?" in rest:
        rest, query = rest.split("?", 1)
        for pair in query.split("&"):
            if pair:
                key, _, value = pair.partition("=")
                out[unquote(key)] = unquote(value)
    authority, _, path = rest.partition("/")
    if "@" in authority:
        userinfo, _, hostport = authority.rpartition("@")
        user, sep, password = userinfo.partition(":")
        if user:
            out["user"] = unquote(user)
        if sep:
            out["password"] = unquote(password)
    else:
        hostport = authority
    host, port = split_hostport(hostport)
    if host.strip(","):
        out["host"] = host
    if port.strip(","):
        out["port"] = port
    if path:
        out["dbname"] = unquote(path)
    return out


def parse_keyword_value(dsn):
    """libpq keyword/value form, including '...' quoting and backslashes."""
    out, i, n = {}, 0, len(dsn)
    while i < n:
        while i < n and dsn[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not dsn[i].isspace() and dsn[i] != "=":
            i += 1
        key = dsn[start:i]
        while i < n and dsn[i].isspace():
            i += 1
        if i >= n or dsn[i] != "=":
            # The key is safe to name; the value is not.
            sys.exit(f"missing '=' after '{key}' in DATABASE_URL")
        i += 1
        while i < n and dsn[i].isspace():
            i += 1
        chars = []
        if i < n and dsn[i] == "'":
            i += 1
            while i < n and dsn[i] != "'":
                if dsn[i] == "\\":
                    i += 1
                chars.append(dsn[i])
                i += 1
            i += 1
        else:
            while i < n and not dsn[i].isspace():
                if dsn[i] == "\\":
                    i += 1
                chars.append(dsn[i])
                i += 1
        out[key] = "".join(chars)
    return out


params = (parse_uri(dsn) if dsn.startswith(("postgresql://", "postgres://"))
          else parse_keyword_value(dsn))

unsupported = []
for key, value in params.items():
    var = ENV_FOR.get(key)
    if var is None:
        unsupported.append(key)
    elif value != "":
        print(f"export {var}={shlex.quote(value)}")

if unsupported:
    # Named, not dropped in silence — one of these can be what makes the
    # connection work. Keys only; a value may be a credential.
    print("WARNING: DATABASE_URL parameter(s) have no PG* equivalent and "
          "were ignored: " + " ".join(sorted(unsupported)), file=sys.stderr)
PY
)" || { echo "ERROR: could not parse DATABASE_URL" >&2; exit 1; }
eval "$PG_ENV"
unset PG_ENV DB_DSN DATABASE_URL

BACKUP_DIR="${BACKUP_DIR:-backups}"
mkdir -p "$BACKUP_DIR"

# Name the artifact after the database and nothing else. Using the DSN put
# "u:PASSWORD@host:5432_<ts>.sql.gz" in directory listings, in the success
# line below and in the cron log.
DB_BASENAME="$(printf '%s' "${PGDATABASE:-}" | tr -cd 'A-Za-z0-9_-')"
DB_BASENAME="${DB_BASENAME:-db}"
TS="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_PATH="${BACKUP_DIR}/${DB_BASENAME}_${TS}.sql"

# No --dbname: the connection comes from the PG* variables exported above.
if ! pg_dump --no-owner --no-privileges --file="$BACKUP_PATH"; then
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
