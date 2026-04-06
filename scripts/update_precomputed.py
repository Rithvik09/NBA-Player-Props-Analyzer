"""
CLI wrapper for the daily precompute job.

Prefer this in production via cron, but the app can also kick background updates when stale.
"""

import argparse
import os
import sys

# Ensure repo root is on sys.path so `import src.*` works when executing this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--season", default=None, help="NBA season like 2025-26 (defaults to computed current season)")
    args = ap.parse_args()
    from src.precompute_jobs import update_precomputed

    summary = update_precomputed(db_path=args.db, season=args.season)
    print("[precompute] done:", summary)


if __name__ == "__main__":
    main()


