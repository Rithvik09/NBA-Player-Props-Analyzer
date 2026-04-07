"""
Reusable precompute jobs used by both:
- CLI: scripts/update_precomputed.py
- App: background refresh when stale

Writes to SQLite:
- dvp_by_position
- team_special_defenders
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from nba_api.stats.endpoints import CommonTeamRoster, leaguedashplayerstats, leaguedashteamstats
from nba_api.stats.static import teams

FANTASYPROS_DVP_URL = "https://www.fantasypros.com/daily-fantasy/nba/fanduel-defense-vs-position.php"


def compute_current_season() -> str:
    now = time.localtime()
    y = now.tm_year
    m = now.tm_mon
    if 1 <= m <= 7:
        return f"{y-1}-{str(y)[2:]}"
    return f"{y}-{str(y+1)[2:]}"


def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dvp_by_position (
            team_id INTEGER NOT NULL,
            position TEXT NOT NULL,
            gp INTEGER,
            pts REAL,
            reb REAL,
            ast REAL,
            fg3m REAL,
            stl REAL,
            blk REAL,
            tov REAL,
            fd_pts REAL,
            source TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (team_id, position, source)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_special_defenders (
            team_id INTEGER NOT NULL,
            pos_group TEXT NOT NULL,
            rank INTEGER NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            def_rating REAL,
            def_ws REAL,
            pct_stl REAL,
            pct_blk REAL,
            min_per_game REAL,
            score01 REAL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (team_id, pos_group, rank)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS referee_stats (
            ref_name TEXT PRIMARY KEY,
            games INTEGER,
            foul_rate REAL,
            home_win_pct REAL,
            pace REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_foul_rates (
            team_id INTEGER PRIMARY KEY,
            foul_rate_season REAL,
            foul_rate_last5 REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dvp_rolling (
            team_id INTEGER NOT NULL,
            window INTEGER NOT NULL,
            pts REAL,
            reb REAL,
            ast REAL,
            fg3m REAL,
            stl REAL,
            blk REAL,
            tov REAL,
            gp INTEGER,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (team_id, window)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_stats (
            team_id INTEGER PRIMARY KEY,
            pts_fb REAL,
            pts_off_tov REAL,
            opp_fga REAL,
            opp_fg_pct REAL,
            opp_fg3a REAL,
            opp_fg3_pct REAL,
            opp_tov REAL,
            opp_stl REAL,
            opp_blk REAL,
            opp_pts_paint REAL,
            opp_pts_fb REAL,
            opp_pts_off_tov REAL,
            opp_def_rating_last5 REAL,
            opp_blk_last5 REAL,
            opp_stl_last5 REAL,
            lg_pts_fb REAL,
            lg_pts_off_tov REAL,
            lg_fga REAL,
            lg_fg_pct REAL,
            lg_fg3a REAL,
            lg_tov REAL,
            lg_stl REAL,
            foul_rate_season REAL,
            foul_rate_last5 REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def _team_id_from_abbrev(abbrev: str) -> int | None:
    alias = {
        "NOR": "NOP",
        "NOH": "NOP",
        "PHO": "PHX",
        "UTH": "UTA",
    }
    abbrev = alias.get(abbrev, abbrev)
    t = teams.find_team_by_abbreviation(abbrev)
    if not t:
        return None
    return int(t["id"])


def scrape_dvp_by_position() -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    def fnum(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def inum(x: str) -> int:
        try:
            return int(float(x))
        except Exception:
            return 0

    for pos in ["PG", "SG", "SF", "PF", "C"]:
        url = f"{FANTASYPROS_DVP_URL}?position={pos}"
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            continue

        seen_team_ids = set()
        for tr in table.find_all("tr")[1:]:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) != 10:
                continue

            team_cell = tds[0]  # e.g. "ATL Atlanta Hawks"
            abbrev = team_cell.split(" ")[0].strip()
            team_id = _team_id_from_abbrev(abbrev)
            if not team_id:
                continue
            if team_id in seen_team_ids:
                continue
            seen_team_ids.add(team_id)

            out[(team_id, pos)] = {
                "gp": inum(tds[1]),
                "pts": fnum(tds[2]),
                "reb": fnum(tds[3]),
                "ast": fnum(tds[4]),
                "fg3m": fnum(tds[5]),
                "stl": fnum(tds[6]),
                "blk": fnum(tds[7]),
                "tov": fnum(tds[8]),
                "fd_pts": fnum(tds[9]),
            }

            if len(seen_team_ids) >= 30:
                break

        time.sleep(0.25)

    return out


def _pos_group(position: str | None) -> str:
    p = (position or "").upper()
    if "C" in p and "G" not in p:
        return "C"
    if "G" in p and "C" not in p and "F" not in p:
        return "G"
    if "F" in p and "C" not in p and "G" not in p:
        return "F"
    if "C" in p:
        return "C"
    if "G" in p:
        return "G"
    return "F"


def compute_special_defenders(season: str) -> list[dict[str, Any]]:
    d = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_detailed_defense="Defense")
    df = d.get_data_frames()[0]
    time.sleep(0.5)

    df["MPG"] = df["MIN"] / df["GP"].replace(0, 1)
    df = df[df["GP"] >= 8]
    df = df[df["MPG"] >= 14]

    def z(s):
        mu = s.mean()
        sd = s.std() or 1.0
        return (s - mu) / sd

    score_raw = (
        0.55 * z(-df["DEF_RATING"]) +
        0.30 * z(df["DEF_WS"]) +
        0.10 * z(df["PCT_STL"]) +
        0.05 * z(df["PCT_BLK"])
    )
    df = df.assign(SCORE=float("nan"))
    df["SCORE"] = score_raw

    mn = float(df["SCORE"].min())
    mx = float(df["SCORE"].max())
    denom = (mx - mn) if mx > mn else 1.0
    df["SCORE01"] = (df["SCORE"] - mn) / denom

    roster_pos_group = {}
    for t in teams.get_teams():
        team_id = int(t["id"])
        try:
            roster = CommonTeamRoster(team_id=team_id, timeout=60).get_data_frames()[0]
            roster_pos_group[team_id] = {int(r["PLAYER_ID"]): _pos_group(r.get("POSITION")) for _, r in roster.iterrows()}
            time.sleep(0.35)
        except Exception:
            roster_pos_group[team_id] = {}
            continue

    rows: list[dict[str, Any]] = []
    for team_id, mapping in roster_pos_group.items():
        team_players = df[df["TEAM_ID"] == team_id]
        if team_players.empty:
            continue

        for pos_group in ["G", "F", "C"]:
            candidates = []
            for _, r in team_players.iterrows():
                pid = int(r["PLAYER_ID"])
                if mapping.get(pid) != pos_group:
                    continue
                candidates.append(r)

            if not candidates:
                continue

            candidates = sorted(candidates, key=lambda x: float(x["SCORE01"]), reverse=True)[:2]
            for idx, r in enumerate(candidates, start=1):
                rows.append({
                    "team_id": team_id,
                    "pos_group": pos_group,
                    "rank": idx,
                    "player_id": int(r["PLAYER_ID"]),
                    "player_name": str(r["PLAYER_NAME"]),
                    "def_rating": float(r["DEF_RATING"]),
                    "def_ws": float(r.get("DEF_WS", 0.0) or 0.0),
                    "pct_stl": float(r.get("PCT_STL", 0.0) or 0.0),
                    "pct_blk": float(r.get("PCT_BLK", 0.0) or 0.0),
                    "min_per_game": float(r.get("MPG", 0.0) or 0.0),
                    "score01": float(r.get("SCORE01", 0.0) or 0.0),
                })

    return rows


def upsert_dvp(conn: sqlite3.Connection, dvp: dict[tuple[int, str], dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for (team_id, pos), s in dvp.items():
        cur.execute(
            """
            INSERT OR REPLACE INTO dvp_by_position
              (team_id, position, gp, pts, reb, ast, fg3m, stl, blk, tov, fd_pts, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(team_id), str(pos),
                int(s.get("gp", 0)),
                float(s.get("pts", 0.0)),
                float(s.get("reb", 0.0)),
                float(s.get("ast", 0.0)),
                float(s.get("fg3m", 0.0)),
                float(s.get("stl", 0.0)),
                float(s.get("blk", 0.0)),
                float(s.get("tov", 0.0)),
                float(s.get("fd_pts", 0.0)),
                "fantasypros",
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_defenders(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_special_defenders
              (team_id, pos_group, rank, player_id, player_name, def_rating, def_ws, pct_stl, pct_blk, min_per_game, score01, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r["team_id"]),
                str(r["pos_group"]),
                int(r["rank"]),
                int(r["player_id"]) if r.get("player_id") is not None else None,
                str(r["player_name"]) if r.get("player_name") is not None else None,
                float(r.get("def_rating")) if r.get("def_rating") is not None else None,
                float(r.get("def_ws")) if r.get("def_ws") is not None else None,
                float(r.get("pct_stl")) if r.get("pct_stl") is not None else None,
                float(r.get("pct_blk")) if r.get("pct_blk") is not None else None,
                float(r.get("min_per_game")) if r.get("min_per_game") is not None else None,
                float(r.get("score01")) if r.get("score01") is not None else None,
                int(updated_at),
            ),
        )
    conn.commit()


def scrape_ref_stats() -> list[dict[str, Any]]:
    """
    Scrape referee statistics from Basketball-Reference.
    Returns list of dicts with ref_name, games, foul_rate, home_win_pct, pace.
    Returns empty list silently on any failure.
    """
    try:
        url = "https://www.basketball-reference.com/referees/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", {"id": "referees"}) or soup.find("table")
        if not table:
            return []

        # Parse header row to find column indices
        header_row = table.find("thead")
        col_names: list[str] = []
        if header_row:
            for th in header_row.find_all("th"):
                col_names.append((th.get("data-stat") or th.get_text(strip=True) or "").lower())

        def _col_idx(*candidates) -> int | None:
            for cand in candidates:
                for idx, name in enumerate(col_names):
                    if cand in name:
                        return idx
            return None

        name_idx     = _col_idx("referee", "ref_name", "name")
        games_idx    = _col_idx("g", "games")
        fouls_idx    = _col_idx("foul", "pf", "personal_foul")
        home_pct_idx = _col_idx("home_w", "hw_pct", "home_win")
        pace_idx     = _col_idx("pace", "poss")

        def _safe_float(tds, idx) -> float:
            if idx is None or idx >= len(tds):
                return 0.0
            try:
                return float(tds[idx].get_text(strip=True).replace(",", "") or 0)
            except Exception:
                return 0.0

        def _safe_int(tds, idx) -> int:
            if idx is None or idx >= len(tds):
                return 0
            try:
                return int(float(tds[idx].get_text(strip=True).replace(",", "") or 0))
            except Exception:
                return 0

        results: list[dict[str, Any]] = []
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            # Get ref name
            if name_idx is not None and name_idx < len(tds):
                raw_name = tds[name_idx].get_text(strip=True)
            else:
                # first td with a link is typically the name
                link_td = tr.find("td", {"data-stat": lambda x: x and "name" in x.lower()})
                if link_td:
                    raw_name = link_td.get_text(strip=True)
                elif tds:
                    raw_name = tds[0].get_text(strip=True)
                else:
                    continue
            if not raw_name or raw_name.lower() in ("", "referee"):
                continue

            results.append({
                "ref_name":     raw_name.strip(),
                "games":        _safe_int(tds, games_idx),
                "foul_rate":    _safe_float(tds, fouls_idx),
                "home_win_pct": _safe_float(tds, home_pct_idx),
                "pace":         _safe_float(tds, pace_idx),
            })
        return results
    except Exception:
        return []


def compute_team_foul_rates(season: str) -> list[dict[str, Any]]:
    """
    Fetch team personal-foul stats from NBA API for full season and last-5 games.
    Returns list of dicts: team_id, foul_rate_season, foul_rate_last5.
    """
    try:
        season_df = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed="Base"
        ).get_data_frames()[0]
        time.sleep(0.5)
    except Exception:
        return []

    try:
        last5_df = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed="Base", last_n_games=5
        ).get_data_frames()[0]
        time.sleep(0.5)
    except Exception:
        last5_df = None

    results: list[dict[str, Any]] = []
    for _, row in season_df.iterrows():
        team_id = int(row["TEAM_ID"])
        gp = float(row.get("GP", 1) or 1)
        if gp <= 0:
            gp = 1.0
        foul_rate_season = float(row.get("PF", 0.0) or 0.0) / gp

        foul_rate_last5 = foul_rate_season  # fallback
        if last5_df is not None:
            last5_row = last5_df[last5_df["TEAM_ID"] == row["TEAM_ID"]]
            if not last5_row.empty:
                gp5 = float(last5_row.iloc[0].get("GP", 1) or 1)
                if gp5 <= 0:
                    gp5 = 1.0
                foul_rate_last5 = float(last5_row.iloc[0].get("PF", 0.0) or 0.0) / gp5

        results.append({
            "team_id": team_id,
            "foul_rate_season": foul_rate_season,
            "foul_rate_last5": foul_rate_last5,
        })
    return results


def compute_team_stats(season: str) -> list[dict[str, Any]]:
    """
    Fetch team scoring + opponent defensive stats from LeagueDashTeamStats.
    Replaces compute_team_foul_rates with a richer dataset.
    Returns list of dicts keyed by team_id.
    """
    import pandas as pd

    results: dict[int, dict[str, Any]] = {}

    def safe_float(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) != 'nan' else default
        except Exception:
            return default

    # 1. Base stats (fouls) — full season
    try:
        df_base = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed='Base'
        ).get_data_frames()[0]
        time.sleep(0.6)
        for _, row in df_base.iterrows():
            tid = int(row['TEAM_ID'])
            gp = max(int(row.get('GP', 1) or 1), 1)
            results[tid] = {
                'foul_rate_season': safe_float(row.get('PF', 0)) / gp,
            }
    except Exception as e:
        print(f"team_stats base fetch failed: {e}")

    # 2. Scoring stats (pts_fb, pts_off_tov)
    try:
        df_scoring = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed='Scoring'
        ).get_data_frames()[0]
        time.sleep(0.6)
        for _, row in df_scoring.iterrows():
            tid = int(row['TEAM_ID'])
            gp = max(int(row.get('GP', 1) or 1), 1)
            if tid not in results:
                results[tid] = {}
            results[tid]['pts_fb']      = safe_float(row.get('PTS_FB', 0)) / gp
            results[tid]['pts_off_tov'] = safe_float(row.get('PTS_OFF_TOV', 0)) / gp
    except Exception as e:
        print(f"team_stats scoring fetch failed: {e}")

    # 3. Opponent stats (full season)
    try:
        df_opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed='Opponent'
        ).get_data_frames()[0]
        time.sleep(0.6)
        for _, row in df_opp.iterrows():
            tid = int(row['TEAM_ID'])
            gp = max(int(row.get('GP', 1) or 1), 1)
            if tid not in results:
                results[tid] = {}
            results[tid].update({
                'opp_fga':         safe_float(row.get('OPP_FGA', 0)) / gp,
                'opp_fg_pct':      safe_float(row.get('OPP_FG_PCT', 0.47)),
                'opp_fg3a':        safe_float(row.get('OPP_FG3A', 0)) / gp,
                'opp_fg3_pct':     safe_float(row.get('OPP_FG3_PCT', 0.36)),
                'opp_tov':         safe_float(row.get('OPP_TOV', 0)) / gp,
                'opp_stl':         safe_float(row.get('OPP_STL', 0)) / gp,
                'opp_blk':         safe_float(row.get('OPP_BLK', 0)) / gp,
                'opp_pts_paint':   safe_float(row.get('OPP_PTS_PAINT', 0)) / gp,
                'opp_pts_fb':      safe_float(row.get('OPP_PTS_FB', 0)) / gp,
                'opp_pts_off_tov': safe_float(row.get('OPP_PTS_OFF_TOV', 0)) / gp,
            })
    except Exception as e:
        print(f"team_stats opponent fetch failed: {e}")

    # 4. Opponent last-5 games (defensive trend)
    try:
        df_opp5 = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed='Opponent', last_n_games=5
        ).get_data_frames()[0]
        time.sleep(0.6)
        for _, row in df_opp5.iterrows():
            tid = int(row['TEAM_ID'])
            gp = max(int(row.get('GP', 1) or 1), 1)
            if tid not in results:
                results[tid] = {}
            pts_allowed = safe_float(row.get('OPP_PTS', 0)) / gp
            pace_proxy  = max(safe_float(row.get('OPP_FGA', 86)) / gp, 1.0)
            results[tid]['opp_def_rating_last5'] = pts_allowed / (pace_proxy / 100.0) if pace_proxy > 0 else 110.0
            results[tid]['opp_blk_last5'] = safe_float(row.get('OPP_BLK', 0)) / gp
            results[tid]['opp_stl_last5'] = safe_float(row.get('OPP_STL', 0)) / gp
    except Exception as e:
        print(f"team_stats opp_last5 fetch failed: {e}")

    # 5. Base last-5 games (foul rate trend)
    try:
        df_base5 = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed='Base', last_n_games=5
        ).get_data_frames()[0]
        time.sleep(0.6)
        for _, row in df_base5.iterrows():
            tid = int(row['TEAM_ID'])
            gp = max(int(row.get('GP', 1) or 1), 1)
            if tid not in results:
                results[tid] = {}
            results[tid]['foul_rate_last5'] = safe_float(row.get('PF', 0)) / gp
    except Exception as e:
        print(f"team_stats base_last5 fetch failed: {e}")

    # 6. League averages (average across all teams)
    lg_avgs: dict[str, float] = {}
    try:
        keys_to_avg = ['pts_fb', 'pts_off_tov', 'opp_fga', 'opp_fg_pct', 'opp_fg3a', 'opp_tov', 'opp_stl']
        for k in keys_to_avg:
            vals = [v[k] for v in results.values() if k in v and v[k] is not None]
            lg_avgs[f'lg_{k}'] = float(sum(vals) / len(vals)) if vals else 0.0
        # Rename keys to match expected feature names
        lg_avgs['lg_fga']      = lg_avgs.pop('lg_opp_fga',   lg_avgs.get('lg_opp_fga',   86.0))
        lg_avgs['lg_fg_pct']   = lg_avgs.pop('lg_opp_fg_pct', lg_avgs.get('lg_opp_fg_pct', 0.47))
        lg_avgs['lg_fg3a']     = lg_avgs.pop('lg_opp_fg3a',  lg_avgs.get('lg_opp_fg3a',  35.0))
        lg_avgs['lg_tov']      = lg_avgs.pop('lg_opp_tov',   lg_avgs.get('lg_opp_tov',   14.0))
        lg_avgs['lg_stl']      = lg_avgs.pop('lg_opp_stl',   lg_avgs.get('lg_opp_stl',    7.0))
    except Exception:
        pass

    # Inject league averages into every team row
    for tid in results:
        results[tid].update(lg_avgs)

    return [{'team_id': tid, **data} for tid, data in results.items()]


def compute_team_foul_rates(season: str) -> list[dict[str, Any]]:
    """
    Backward-compatible alias that delegates to compute_team_stats.
    Returns the subset of fields used by the old team_foul_rates table.
    """
    rows = compute_team_stats(season)
    return [
        {
            'team_id':           r['team_id'],
            'foul_rate_season':  r.get('foul_rate_season', 20.0),
            'foul_rate_last5':   r.get('foul_rate_last5', 20.0),
        }
        for r in rows
    ]


def upsert_team_stats(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_stats (
                team_id, pts_fb, pts_off_tov,
                opp_fga, opp_fg_pct, opp_fg3a, opp_fg3_pct, opp_tov, opp_stl, opp_blk,
                opp_pts_paint, opp_pts_fb, opp_pts_off_tov,
                opp_def_rating_last5, opp_blk_last5, opp_stl_last5,
                lg_pts_fb, lg_pts_off_tov, lg_fga, lg_fg_pct, lg_fg3a, lg_tov, lg_stl,
                foul_rate_season, foul_rate_last5, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                int(r['team_id']),
                float(r.get('pts_fb', 0.0) or 0.0),
                float(r.get('pts_off_tov', 0.0) or 0.0),
                float(r.get('opp_fga', 0.0) or 0.0),
                float(r.get('opp_fg_pct', 0.47) or 0.47),
                float(r.get('opp_fg3a', 0.0) or 0.0),
                float(r.get('opp_fg3_pct', 0.36) or 0.36),
                float(r.get('opp_tov', 0.0) or 0.0),
                float(r.get('opp_stl', 0.0) or 0.0),
                float(r.get('opp_blk', 0.0) or 0.0),
                float(r.get('opp_pts_paint', 0.0) or 0.0),
                float(r.get('opp_pts_fb', 0.0) or 0.0),
                float(r.get('opp_pts_off_tov', 0.0) or 0.0),
                float(r.get('opp_def_rating_last5', 110.0) or 110.0),
                float(r.get('opp_blk_last5', 0.0) or 0.0),
                float(r.get('opp_stl_last5', 0.0) or 0.0),
                float(r.get('lg_pts_fb', 0.0) or 0.0),
                float(r.get('lg_pts_off_tov', 0.0) or 0.0),
                float(r.get('lg_fga', 0.0) or 0.0),
                float(r.get('lg_fg_pct', 0.47) or 0.47),
                float(r.get('lg_fg3a', 0.0) or 0.0),
                float(r.get('lg_tov', 0.0) or 0.0),
                float(r.get('lg_stl', 0.0) or 0.0),
                float(r.get('foul_rate_season', 0.0) or 0.0),
                float(r.get('foul_rate_last5', 0.0) or 0.0),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_rolling_dvp(season: str) -> list[dict[str, Any]]:
    """
    Compute team-level opponent stats for last 5 and last 10 games using
    LeagueDashTeamStats with measure_type='Opponent' and last_n_games filtering.

    Returns list of dicts: team_id, window (5 or 10), pts, reb, ast, fg3m,
    stl, blk, tov, gp.
    """
    results: list[dict[str, Any]] = []

    def _fetch(window: int) -> Any:
        try:
            kwargs: dict[str, Any] = {
                "season": season,
                "measure_type_detailed": "Opponent",
            }
            if window > 0:
                kwargs["last_n_games"] = window
            df = leaguedashteamstats.LeagueDashTeamStats(**kwargs).get_data_frames()[0]
            time.sleep(0.5)
            return df
        except Exception:
            return None

    for window in (5, 10):
        df = _fetch(window)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            team_id = int(row["TEAM_ID"])
            gp = int(row.get("GP", 0) or 0)

            def _f(col: str) -> float:
                v = row.get(col)
                if v is None:
                    return 0.0
                try:
                    return float(v)
                except Exception:
                    return 0.0

            results.append({
                "team_id": team_id,
                "window": window,
                "pts":  _f("OPP_PTS"),
                "reb":  _f("OPP_REB"),
                "ast":  _f("OPP_AST"),
                "fg3m": _f("OPP_FG3M"),
                "stl":  _f("OPP_STL"),
                "blk":  _f("OPP_BLK"),
                "tov":  _f("OPP_TOV"),
                "gp":   gp,
            })
    return results


def upsert_ref_stats(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO referee_stats
              (ref_name, games, foul_rate, home_win_pct, pace, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(r["ref_name"]),
                int(r.get("games", 0)),
                float(r.get("foul_rate", 0.0)),
                float(r.get("home_win_pct", 0.0)),
                float(r.get("pace", 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_team_foul_rates(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_foul_rates
              (team_id, foul_rate_season, foul_rate_last5, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(r["team_id"]),
                float(r.get("foul_rate_season", 0.0)),
                float(r.get("foul_rate_last5", 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_rolling_dvp(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO dvp_rolling
              (team_id, window, pts, reb, ast, fg3m, stl, blk, tov, gp, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r["team_id"]),
                int(r["window"]),
                float(r.get("pts", 0.0)),
                float(r.get("reb", 0.0)),
                float(r.get("ast", 0.0)),
                float(r.get("fg3m", 0.0)),
                float(r.get("stl", 0.0)),
                float(r.get("blk", 0.0)),
                float(r.get("tov", 0.0)),
                int(r.get("gp", 0)),
                int(updated_at),
            ),
        )
    conn.commit()


def update_precomputed(db_path: str, season: str | None = None) -> dict[str, Any]:
    """
    Runs full update and returns summary.
    Raises on hard failures.
    """
    season = season or compute_current_season()
    updated_at = int(time.time())

    conn = sqlite3.connect(db_path)
    ensure_tables(conn)

    dvp = scrape_dvp_by_position()
    upsert_dvp(conn, dvp, updated_at=updated_at)

    defenders = compute_special_defenders(season=season)
    upsert_defenders(conn, defenders, updated_at=updated_at)

    ref_rows = scrape_ref_stats()
    upsert_ref_stats(conn, ref_rows, updated_at=updated_at)

    team_stats_rows = compute_team_stats(season=season)
    upsert_team_stats(conn, team_stats_rows, updated_at=updated_at)
    # Also keep the legacy team_foul_rates table populated for backward compat
    foul_rows = [
        {
            'team_id':          r['team_id'],
            'foul_rate_season': r.get('foul_rate_season', 20.0),
            'foul_rate_last5':  r.get('foul_rate_last5', 20.0),
        }
        for r in team_stats_rows
    ]
    upsert_team_foul_rates(conn, foul_rows, updated_at=updated_at)

    rolling_rows = compute_rolling_dvp(season=season)
    upsert_rolling_dvp(conn, rolling_rows, updated_at=updated_at)

    conn.close()
    return {
        "season": season,
        "updated_at": updated_at,
        "dvp_rows": len(dvp),
        "defender_rows": len(defenders),
        "ref_rows": len(ref_rows),
        "team_stats_rows": len(team_stats_rows),
        "foul_rate_rows": len(foul_rows),
        "rolling_dvp_rows": len(rolling_rows),
    }


