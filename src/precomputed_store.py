import sqlite3
import time
from typing import Any


class PrecomputedStore:
    """
    Lightweight reader for daily-precomputed datasets stored in SQLite.

    Data sources:
    - DVP by position (scraped daily): `dvp_by_position`
    - Special defenders by team/pos-group (nba_api, daily): `team_special_defenders`
    """

    def __init__(self, db_name: str):
        self.db_name = db_name
        self._cache: dict[str, Any] = {}
        self._cache_at = 0
        self._cache_ttl_seconds = 60 * 10  # 10 minutes

    def _get_db(self):
        return sqlite3.connect(self.db_name)

    def refresh(self, force: bool = False) -> dict[str, Any]:
        now = int(time.time())
        if not force and self._cache and (now - int(self._cache_at) < self._cache_ttl_seconds):
            return self._cache

        dvp = {}  # (team_id, pos) -> stats dict
        dvp_meta = {'updated_at': None}
        dvp_pos_avgs = {}  # pos -> averages dict

        defenders = {}  # (team_id, pos_group) -> list[dict] ordered by rank
        defenders_meta = {'updated_at': None}

        try:
            conn = self._get_db()
            cur = conn.cursor()

            cur.execute(
                """
                SELECT team_id, position, gp, pts, reb, ast, fg3m, stl, blk, tov, fd_pts, updated_at
                FROM dvp_by_position
                """
            )
            rows = cur.fetchall()
            max_ts = None
            for (team_id, pos, gp, pts, reb, ast, fg3m, stl, blk, tov, fd_pts, updated_at) in rows:
                dvp[(int(team_id), str(pos))] = {
                    'gp': int(gp) if gp is not None else 0,
                    'pts': float(pts) if pts is not None else 0.0,
                    'reb': float(reb) if reb is not None else 0.0,
                    'ast': float(ast) if ast is not None else 0.0,
                    'fg3m': float(fg3m) if fg3m is not None else 0.0,
                    'stl': float(stl) if stl is not None else 0.0,
                    'blk': float(blk) if blk is not None else 0.0,
                    'tov': float(tov) if tov is not None else 0.0,
                    'fd_pts': float(fd_pts) if fd_pts is not None else 0.0,
                    'updated_at': int(updated_at) if updated_at is not None else 0,
                }
                if updated_at is not None:
                    max_ts = int(updated_at) if max_ts is None else max(max_ts, int(updated_at))
            dvp_meta['updated_at'] = max_ts

            # Compute position averages (for deltas / shrinkage)
            by_pos = {}
            for (_, pos), s in dvp.items():
                by_pos.setdefault(pos, []).append(s)
            for pos, ss in by_pos.items():
                if not ss:
                    continue
                dvp_pos_avgs[pos] = {
                    'pts': sum(x['pts'] for x in ss) / len(ss),
                    'reb': sum(x['reb'] for x in ss) / len(ss),
                    'ast': sum(x['ast'] for x in ss) / len(ss),
                    'fg3m': sum(x['fg3m'] for x in ss) / len(ss),
                    'stl': sum(x['stl'] for x in ss) / len(ss),
                    'blk': sum(x['blk'] for x in ss) / len(ss),
                    'tov': sum(x['tov'] for x in ss) / len(ss),
                    'fd_pts': sum(x['fd_pts'] for x in ss) / len(ss),
                }

            cur.execute(
                """
                SELECT team_id, pos_group, rank, player_id, player_name,
                       def_rating, def_ws, pct_stl, pct_blk, min_per_game, score01, updated_at
                FROM team_special_defenders
                ORDER BY team_id, pos_group, rank ASC
                """
            )
            rows = cur.fetchall()
            max_ts = None
            for (team_id, pos_group, rank, player_id, player_name, def_rating, def_ws, pct_stl, pct_blk, mpg, score01, updated_at) in rows:
                defenders.setdefault((int(team_id), str(pos_group)), []).append({
                    'rank': int(rank),
                    'player_id': int(player_id) if player_id is not None else None,
                    'player_name': str(player_name) if player_name is not None else None,
                    'def_rating': float(def_rating) if def_rating is not None else None,
                    'def_ws': float(def_ws) if def_ws is not None else None,
                    'pct_stl': float(pct_stl) if pct_stl is not None else None,
                    'pct_blk': float(pct_blk) if pct_blk is not None else None,
                    'min_per_game': float(mpg) if mpg is not None else None,
                    'score01': float(score01) if score01 is not None else None,
                    'updated_at': int(updated_at) if updated_at is not None else 0,
                })
                if updated_at is not None:
                    max_ts = int(updated_at) if max_ts is None else max(max_ts, int(updated_at))
            defenders_meta['updated_at'] = max_ts

            conn.close()

        except Exception:
            # Tables may not exist yet; return empty structures.
            pass

        payload = {
            'dvp': dvp,
            'dvp_meta': dvp_meta,
            'dvp_pos_avgs': dvp_pos_avgs,
            'defenders': defenders,
            'defenders_meta': defenders_meta,
        }
        self._cache = payload
        self._cache_at = now
        return payload

    def is_fresh(self, max_age_seconds: int = 60 * 60 * 26) -> bool:
        data = self.refresh()
        now = int(time.time())
        dvp_ts = data.get('dvp_meta', {}).get('updated_at') or 0
        def_ts = data.get('defenders_meta', {}).get('updated_at') or 0
        newest = max(int(dvp_ts), int(def_ts))
        return newest > 0 and (now - newest) < max_age_seconds


