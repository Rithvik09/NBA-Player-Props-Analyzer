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
from nba_api.stats.endpoints import (
    leaguedashplayerbiostats,
    leaguedashplayerclutch,
    leaguehustlestatsplayer,
    leaguedashplayerptshot,
    leaguedashoppptshot,
    synergyplaytypes,
    teamplayeronoffsummary,
    playerdashboardbyshootingsplits,
    playerdashboardbygamesplits,
    commonallplayers,
    leaguedashptstats,
    leaguestandingsv3,
    teamdashboardbygeneralsplits,
    leaguedashlineups,
)
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_advanced_stats (
            player_id INTEGER PRIMARY KEY,
            usg_pct REAL, ts_pct REAL, efg_pct REAL, ast_pct REAL,
            oreb_pct REAL, dreb_pct REAL, reb_pct REAL, pie REAL,
            off_rating REAL, def_rating REAL, pace REAL, net_rating REAL,
            age REAL, height_inches REAL, weight REAL, years_experience REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_clutch_stats (
            player_id INTEGER PRIMARY KEY,
            clutch_pts_pg REAL, clutch_fg_pct REAL, clutch_fg3_pct REAL,
            clutch_fta_pg REAL, clutch_plus_minus REAL, clutch_min_pg REAL,
            clutch_games INTEGER,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_hustle_stats (
            player_id INTEGER PRIMARY KEY,
            contested_shots_pg REAL, deflections_pg REAL,
            charges_drawn_pg REAL, screen_assists_pg REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_shot_profile (
            player_id INTEGER PRIMARY KEY,
            open_shot_fg_pct REAL, open_shot_freq REAL,
            tight_shot_fg_pct REAL, tight_shot_freq REAL,
            catch_shoot_fg_pct REAL, catch_shoot_freq REAL,
            pullup_fg_pct REAL, pullup_freq REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_play_types (
            player_id INTEGER PRIMARY KEY,
            iso_poss_pct REAL, iso_ppp REAL,
            pnr_bh_poss_pct REAL, pnr_bh_ppp REAL,
            pnr_roll_poss_pct REAL, pnr_roll_ppp REAL,
            spotup_poss_pct REAL, spotup_ppp REAL,
            transition_poss_pct REAL, transition_ppp REAL,
            postup_poss_pct REAL, cut_poss_pct REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_on_off (
            player_id INTEGER PRIMARY KEY,
            on_court_net_rating REAL,
            off_court_net_rating REAL,
            on_off_differential REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_shot_zones (
            player_id INTEGER PRIMARY KEY,
            rim_fga_pct REAL, rim_fg_pct REAL,
            paint_fga_pct REAL, paint_fg_pct REAL,
            midrange_fga_pct REAL, midrange_fg_pct REAL,
            corner3_fga_pct REAL, corner3_fg_pct REAL,
            above_break3_fga_pct REAL, above_break3_fg_pct REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_quarter_splits (
            player_id INTEGER PRIMARY KEY,
            q1_avg REAL, q2_avg REAL, q3_avg REAL, q4_avg REAL,
            q4_min_pg REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_opp_shot_zones (
            team_id INTEGER PRIMARY KEY,
            rim_fg_pct_allowed REAL,
            paint_fg_pct_allowed REAL,
            midrange_fg_pct_allowed REAL,
            corner3_fg_pct_allowed REAL,
            above_break3_fg_pct_allowed REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_synergy_defense (
            team_id INTEGER PRIMARY KEY,
            pnr_ppp_allowed REAL,
            iso_ppp_allowed REAL,
            spotup_ppp_allowed REAL,
            transition_ppp_allowed REAL,
            postup_ppp_allowed REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_tracking_stats (
            player_id INTEGER PRIMARY KEY,
            avg_speed REAL, avg_speed_off REAL, avg_speed_def REAL,
            dist_miles REAL, dist_miles_off REAL, dist_miles_def REAL,
            touches_pg REAL, time_of_poss_pg REAL, avg_drib_per_touch REAL,
            paint_touches_pg REAL, elbow_touches_pg REAL,
            passes_made_pg REAL, potential_ast_pg REAL, secondary_ast_pg REAL,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_standings (
            team_id INTEGER PRIMARY KEY,
            win_pct REAL, wins INTEGER, losses INTEGER,
            conf_rank INTEGER, games_back REAL,
            home_win_pct REAL, road_win_pct REAL,
            current_streak INTEGER,
            l10_wins INTEGER,
            pts_pg REAL, opp_pts_pg REAL,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_scoring_breakdown (
            player_id INTEGER PRIMARY KEY,
            pct_pts_3pt REAL, pct_pts_paint REAL, pct_pts_ft REAL,
            pct_pts_midrange REAL, pct_uast_fgm REAL, pct_ast_fgm REAL,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_vs_opponent (
            player_id INTEGER,
            opponent_team_id INTEGER,
            gp INTEGER,
            avg_stat_pts REAL,
            fg_pct REAL,
            ts_pct REAL,
            avg_min REAL,
            PRIMARY KEY (player_id, opponent_team_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_rest_splits (
            team_id INTEGER PRIMARY KEY,
            b2b_def_rating REAL,
            b2b_pace REAL,
            b2b_pts_allowed REAL,
            rested_def_rating REAL,
            rested_pace REAL,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_yoy_stats (
            player_id INTEGER PRIMARY KEY,
            yoy_pts_change REAL,
            yoy_ts_change REAL,
            yoy_usage_change REAL,
            seasons_in_league INTEGER,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_home_away_splits (
            team_id INTEGER PRIMARY KEY,
            home_def_rating REAL,
            away_def_rating REAL,
            home_away_def_split REAL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_lineup_stats (
            team_id INTEGER PRIMARY KEY,
            top_lineup_net_rating REAL,
            bench_net_rating REAL,
            bench_strength REAL,
            lineup_continuity REAL,
            lineups_played_count INTEGER,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_injury_status (
            team_id INTEGER PRIMARY KEY,
            key_players_out INTEGER,
            total_players_out INTEGER,
            total_impact REAL,
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
    time.sleep(1.0)

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
            time.sleep(0.8)
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
            season=season, measure_type_detailed_defense="Base"
        ).get_data_frames()[0]
        time.sleep(1.0)
    except Exception:
        return []

    try:
        last5_df = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Base", last_n_games=5
        ).get_data_frames()[0]
        time.sleep(1.0)
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
            season=season, measure_type_detailed_defense='Base'
        ).get_data_frames()[0]
        time.sleep(1.2)
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
            season=season, measure_type_detailed_defense='Scoring'
        ).get_data_frames()[0]
        time.sleep(1.2)
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
            season=season, measure_type_detailed_defense='Opponent'
        ).get_data_frames()[0]
        time.sleep(1.2)
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
            season=season, measure_type_detailed_defense='Opponent', last_n_games=5
        ).get_data_frames()[0]
        time.sleep(1.2)
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
            season=season, measure_type_detailed_defense='Base', last_n_games=5
        ).get_data_frames()[0]
        time.sleep(1.2)
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
                "measure_type_detailed_defense": "Opponent",
            }
            if window > 0:
                kwargs["last_n_games"] = window
            df = leaguedashteamstats.LeagueDashTeamStats(**kwargs).get_data_frames()[0]
            time.sleep(1.0)
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


def compute_advanced_player_stats(season: str) -> list[dict[str, Any]]:
    """Fetch official NBA advanced stats + bio stats and merge by player_id."""
    current_year = time.localtime().tm_year

    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    adv_map: dict[int, dict[str, Any]] = {}
    try:
        adv_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.0)
        for _, row in adv_df.iterrows():
            pid = int(row['PLAYER_ID'])
            oreb = _sf(row.get('OREB_PCT', 0))
            dreb = _sf(row.get('DREB_PCT', 0))
            reb  = _sf(row.get('REB_PCT', oreb + dreb))
            adv_map[pid] = {
                'usg_pct':    _sf(row.get('USG_PCT', 0.18)),
                'ts_pct':     _sf(row.get('TS_PCT', 0.55)),
                'efg_pct':    _sf(row.get('EFG_PCT', 0.50)),
                'ast_pct':    _sf(row.get('AST_PCT', 0.15)),
                'oreb_pct':   oreb,
                'dreb_pct':   dreb,
                'reb_pct':    reb,
                'pie':        _sf(row.get('PIE', 0.10)),
                'off_rating': _sf(row.get('OFF_RATING', 110.0)),
                'def_rating': _sf(row.get('DEF_RATING', 110.0)),
                'pace':       _sf(row.get('PACE', 100.0)),
                'net_rating': _sf(row.get('NET_RATING', 0.0)),
            }
    except Exception as e:
        print(f"compute_advanced_player_stats: advanced fetch failed: {e}")

    bio_map: dict[int, dict[str, Any]] = {}
    try:
        bio_df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=season,
            per_mode_simple='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.0)
        for _, row in bio_df.iterrows():
            pid = int(row['PLAYER_ID'])
            draft_year = row.get('DRAFT_YEAR', 0)
            try:
                draft_year_int = int(float(draft_year)) if draft_year and str(draft_year) not in ('', 'nan', 'Undrafted') else 0
            except Exception:
                draft_year_int = 0
            years_exp = max(1, current_year - draft_year_int) if draft_year_int > 0 else 1
            height_raw = row.get('PLAYER_HEIGHT_INCHES', 78.0)
            bio_map[pid] = {
                'age':            _sf(row.get('AGE', 26.0)),
                'height_inches':  _sf(height_raw, 78.0),
                'weight':         _sf(row.get('PLAYER_WEIGHT', 220.0)),
                'years_experience': float(years_exp),
            }
    except Exception as e:
        print(f"compute_advanced_player_stats: bio fetch failed: {e}")

    all_pids = set(adv_map.keys()) | set(bio_map.keys())
    results: list[dict[str, Any]] = []
    for pid in all_pids:
        adv = adv_map.get(pid, {})
        bio = bio_map.get(pid, {})
        results.append({
            'player_id':      pid,
            'usg_pct':        adv.get('usg_pct', 0.18),
            'ts_pct':         adv.get('ts_pct', 0.55),
            'efg_pct':        adv.get('efg_pct', 0.50),
            'ast_pct':        adv.get('ast_pct', 0.15),
            'oreb_pct':       adv.get('oreb_pct', 0.05),
            'dreb_pct':       adv.get('dreb_pct', 0.15),
            'reb_pct':        adv.get('reb_pct', 0.10),
            'pie':            adv.get('pie', 0.10),
            'off_rating':     adv.get('off_rating', 110.0),
            'def_rating':     adv.get('def_rating', 110.0),
            'pace':           adv.get('pace', 100.0),
            'net_rating':     adv.get('net_rating', 0.0),
            'age':            bio.get('age', 26.0),
            'height_inches':  bio.get('height_inches', 78.0),
            'weight':         bio.get('weight', 220.0),
            'years_experience': bio.get('years_experience', 5.0),
        })
    return results


def compute_clutch_stats(season: str) -> list[dict[str, Any]]:
    """Fetch clutch stats (last 5 min, within 5 pts)."""
    try:
        df = leaguedashplayerclutch.LeagueDashPlayerClutch(
            season=season,
            per_mode_detailed='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.0)
    except Exception as e:
        print(f"compute_clutch_stats: fetch failed: {e}")
        return []

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        def _sf(col, default=0.0):
            try:
                v = row.get(col)
                return float(v) if v is not None and str(v) not in ('nan', '') else default
            except Exception:
                return default
        results.append({
            'player_id':         int(row['PLAYER_ID']),
            'clutch_pts_pg':     _sf('PTS', 0.0),
            'clutch_fg_pct':     _sf('FG_PCT', 0.45),
            'clutch_fg3_pct':    _sf('FG3_PCT', 0.33),
            'clutch_fta_pg':     _sf('FTA', 0.0),
            'clutch_plus_minus': _sf('PLUS_MINUS', 0.0),
            'clutch_min_pg':     _sf('MIN', 0.0),
            'clutch_games':      int(_sf('GP', 0)),
        })
    return results


def compute_hustle_stats(season: str) -> list[dict[str, Any]]:
    """Fetch league hustle stats (per game)."""
    try:
        df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
            season=season,
            per_mode_time='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.0)
    except Exception as e:
        print(f"compute_hustle_stats: fetch failed: {e}")
        return []

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        def _sf(col, default=0.0):
            try:
                v = row.get(col)
                return float(v) if v is not None and str(v) not in ('nan', '') else default
            except Exception:
                return default
        results.append({
            'player_id':            int(row['PLAYER_ID']),
            'contested_shots_pg':   _sf('CONTESTED_SHOTS', 3.0),
            'deflections_pg':       _sf('DEFLECTIONS', 1.0),
            'charges_drawn_pg':     _sf('CHARGES_DRAWN', 0.1),
            'screen_assists_pg':    _sf('SCREEN_ASSISTS', 0.5),
        })
    return results


def compute_shot_profile(season: str) -> list[dict[str, Any]]:
    """Fetch shot profile by defender distance and shot type."""

    def _fetch_ptshot(**kwargs) -> Any:
        try:
            df = leaguedashplayerptshot.LeagueDashPlayerPtShot(
                per_mode_simple='PerGame',
                **kwargs,
            ).get_data_frames()[0]
            time.sleep(1.0)
            return df
        except Exception as e:
            print(f"compute_shot_profile: fetch failed {kwargs}: {e}")
            return None

    def _extract(df, pid_col='PLAYER_ID', freq_col='FGA_FREQUENCY', pct_col='FG_PCT') -> dict[int, tuple]:
        out = {}
        if df is None or df.empty:
            return out
        for _, row in df.iterrows():
            try:
                pid = int(row[pid_col])
                freq = float(row.get(freq_col, 0.0) or 0.0)
                pct  = float(row.get(pct_col, 0.0) or 0.0)
                out[pid] = (freq, pct)
            except Exception:
                continue
        return out

    open_df   = _fetch_ptshot(season=season, close_def_dist_range_nullable='6+ Feet - Wide Open')
    tight_df  = _fetch_ptshot(season=season, close_def_dist_range_nullable='0-2 Feet - Very Tight')
    cs_df     = _fetch_ptshot(season=season, general_range_nullable='Catch and Shoot')
    pullup_df = _fetch_ptshot(season=season, general_range_nullable='Pullups')

    open_map   = _extract(open_df)
    tight_map  = _extract(tight_df)
    cs_map     = _extract(cs_df)
    pullup_map = _extract(pullup_df)

    all_pids = set(open_map) | set(tight_map) | set(cs_map) | set(pullup_map)
    results: list[dict[str, Any]] = []
    for pid in all_pids:
        o_freq, o_pct   = open_map.get(pid, (0.30, 0.50))
        t_freq, t_pct   = tight_map.get(pid, (0.15, 0.38))
        c_freq, c_pct   = cs_map.get(pid, (0.25, 0.40))
        p_freq, p_pct   = pullup_map.get(pid, (0.20, 0.40))
        results.append({
            'player_id':          pid,
            'open_shot_fg_pct':   o_pct,
            'open_shot_freq':     o_freq,
            'tight_shot_fg_pct':  t_pct,
            'tight_shot_freq':    t_freq,
            'catch_shoot_fg_pct': c_pct,
            'catch_shoot_freq':   c_freq,
            'pullup_fg_pct':      p_pct,
            'pullup_freq':        p_freq,
        })
    return results


def compute_synergy_play_types(season: str) -> list[dict[str, Any]]:
    """Fetch player synergy play type stats for offensive play types."""
    play_type_map = {
        'Isolation':    ('iso_poss_pct', 'iso_ppp'),
        'PRBallHandler': ('pnr_bh_poss_pct', 'pnr_bh_ppp'),
        'PRRollman':    ('pnr_roll_poss_pct', 'pnr_roll_ppp'),
        'Postup':       ('postup_poss_pct', 'postup_ppp'),
        'Spotup':       ('spotup_poss_pct', 'spotup_ppp'),
        'Transition':   ('transition_poss_pct', 'transition_ppp'),
        'Cut':          ('cut_poss_pct', 'cut_ppp'),
    }

    player_data: dict[int, dict[str, float]] = {}

    for pt, (pct_key, ppp_key) in play_type_map.items():
        try:
            df = synergyplaytypes.SynergyPlayTypes(
                season=season,
                play_type_nullable=pt,
                player_or_team_abbreviation='P',
                type_grouping_nullable='offensive',
                per_mode_simple='PerGame',
            ).get_data_frames()[0]
            time.sleep(1.0)
            for _, row in df.iterrows():
                try:
                    pid = int(row.get('PLAYER_ID', 0) or row.get('ENTITY_ID', 0) or 0)
                    if pid == 0:
                        continue
                    if pid not in player_data:
                        player_data[pid] = {}
                    player_data[pid][pct_key] = float(row.get('POSS_PCT', 0.0) or 0.0)
                    player_data[pid][ppp_key]  = float(row.get('PPP', 0.9) or 0.9)
                except Exception:
                    continue
        except Exception as e:
            print(f"compute_synergy_play_types: fetch failed for {pt}: {e}")
            time.sleep(1.0)

    results: list[dict[str, Any]] = []
    for pid, data in player_data.items():
        results.append({
            'player_id':          pid,
            'iso_poss_pct':       data.get('iso_poss_pct', 0.0),
            'iso_ppp':            data.get('iso_ppp', 0.9),
            'pnr_bh_poss_pct':    data.get('pnr_bh_poss_pct', 0.0),
            'pnr_bh_ppp':         data.get('pnr_bh_ppp', 0.9),
            'pnr_roll_poss_pct':  data.get('pnr_roll_poss_pct', 0.0),
            'pnr_roll_ppp':       data.get('pnr_roll_ppp', 0.9),
            'spotup_poss_pct':    data.get('spotup_poss_pct', 0.0),
            'spotup_ppp':         data.get('spotup_ppp', 1.0),
            'transition_poss_pct': data.get('transition_poss_pct', 0.0),
            'transition_ppp':     data.get('transition_ppp', 1.1),
            'postup_poss_pct':    data.get('postup_poss_pct', 0.0),
            'cut_poss_pct':       data.get('cut_poss_pct', 0.0),
        })
    return results


def compute_on_off_ratings(season: str) -> list[dict[str, Any]]:
    """Fetch player on/off court net ratings for all teams."""
    results: list[dict[str, Any]] = []

    for t in teams.get_teams():
        team_id = int(t['id'])
        try:
            frames = teamplayeronoffsummary.TeamPlayerOnOffSummary(
                team_id=team_id,
                season=season,
            ).get_data_frames()
            time.sleep(0.8)

            # Look for the on-court frame and off-court frame
            on_df = None
            off_df = None
            for frame in frames:
                if frame.empty:
                    continue
                cols = [c.upper() for c in frame.columns]
                if 'PLAYER_ID' in cols or 'VS_PLAYER_ID' in cols:
                    if on_df is None:
                        on_df = frame
                    elif off_df is None:
                        off_df = frame
                        break

            if on_df is None:
                continue

            on_map: dict[int, float] = {}
            off_map: dict[int, float] = {}

            def _pid_col(df):
                for c in df.columns:
                    if 'PLAYER_ID' in c.upper():
                        return c
                return None

            def _rating_col(df):
                for c in df.columns:
                    if 'NET_RATING' in c.upper():
                        return c
                return None

            pid_c = _pid_col(on_df)
            rat_c = _rating_col(on_df)
            if pid_c and rat_c:
                for _, row in on_df.iterrows():
                    try:
                        pid = int(row[pid_c])
                        on_map[pid] = float(row.get(rat_c, 0.0) or 0.0)
                    except Exception:
                        continue

            if off_df is not None:
                pid_c2 = _pid_col(off_df)
                rat_c2 = _rating_col(off_df)
                if pid_c2 and rat_c2:
                    for _, row in off_df.iterrows():
                        try:
                            pid = int(row[pid_c2])
                            off_map[pid] = float(row.get(rat_c2, 0.0) or 0.0)
                        except Exception:
                            continue

            for pid in on_map:
                on_r  = on_map[pid]
                off_r = off_map.get(pid, 0.0)
                results.append({
                    'player_id':            pid,
                    'on_court_net_rating':  on_r,
                    'off_court_net_rating': off_r,
                    'on_off_differential':  on_r - off_r,
                })
        except Exception as e:
            print(f"compute_on_off_ratings: team {team_id} failed: {e}")
            time.sleep(0.8)

    return results


def _get_active_player_ids(limit: int = 400) -> list[int]:
    """Return up to `limit` active player IDs."""
    try:
        df = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
        ).get_data_frames()[0]
        time.sleep(1.0)
        pids = [int(r['PERSON_ID']) for _, r in df.iterrows()]
        return pids[:limit]
    except Exception as e:
        print(f"_get_active_player_ids failed: {e}")
        return []


def compute_shot_zone_breakdown(season: str, player_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Fetch shot zone breakdown for each active player."""
    if player_ids is None:
        player_ids = _get_active_player_ids(400)

    results: list[dict[str, Any]] = []

    for pid in player_ids:
        try:
            frames = playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits(
                player_id=pid,
                season=season,
                per_mode_detailed='PerGame',
                timeout=10,
            ).get_data_frames()
            time.sleep(1.2)

            zone_df = None
            for frame in frames:
                if frame.empty:
                    continue
                cols_upper = [c.upper() for c in frame.columns]
                if 'GROUP_VALUE' in cols_upper:
                    zone_df = frame
                    break

            if zone_df is None:
                continue

            # Build zone accumulator
            zone_fga: dict[str, float] = {}
            zone_fgm: dict[str, float] = {}
            zone_fg_pct: dict[str, float] = {}

            def _classify(gv: str) -> str | None:
                gv_up = gv.upper()
                if 'RESTRICTED' in gv_up or ('PAINT' in gv_up and '3' not in gv_up):
                    if 'RESTRICTED' in gv_up:
                        return 'rim'
                    return 'paint'
                if 'MID' in gv_up or 'MIDRANGE' in gv_up or 'MID-RANGE' in gv_up:
                    return 'midrange'
                if 'CORNER' in gv_up and '3' in gv_up:
                    return 'corner3'
                if ('ABOVE' in gv_up and 'BREAK' in gv_up) or ('ABOVE BREAK' in gv_up):
                    return 'above_break3'
                if 'IN THE PAINT' in gv_up:
                    return 'paint'
                return None

            total_fga = 0.0
            for _, row in zone_df.iterrows():
                gv = str(row.get('GROUP_VALUE', '') or '')
                zone = _classify(gv)
                if zone is None:
                    continue
                try:
                    fga = float(row.get('FGA', 0.0) or 0.0)
                    pct = float(row.get('FG_PCT', 0.0) or 0.0)
                    fgm = fga * pct
                    zone_fga[zone] = zone_fga.get(zone, 0.0) + fga
                    zone_fgm[zone] = zone_fgm.get(zone, 0.0) + fgm
                    total_fga += fga
                except Exception:
                    continue

            def _zone_stats(z: str) -> tuple[float, float]:
                fga = zone_fga.get(z, 0.0)
                fgm = zone_fgm.get(z, 0.0)
                freq = (fga / total_fga) if total_fga > 0 else 0.0
                pct  = (fgm / fga) if fga > 0 else 0.0
                return freq, pct

            r_freq, r_pct    = _zone_stats('rim')
            pa_freq, pa_pct  = _zone_stats('paint')
            m_freq, m_pct    = _zone_stats('midrange')
            c3_freq, c3_pct  = _zone_stats('corner3')
            ab_freq, ab_pct  = _zone_stats('above_break3')

            results.append({
                'player_id':          pid,
                'rim_fga_pct':        r_freq,
                'rim_fg_pct':         r_pct,
                'paint_fga_pct':      pa_freq,
                'paint_fg_pct':       pa_pct,
                'midrange_fga_pct':   m_freq,
                'midrange_fg_pct':    m_pct,
                'corner3_fga_pct':    c3_freq,
                'corner3_fg_pct':     c3_pct,
                'above_break3_fga_pct': ab_freq,
                'above_break3_fg_pct':  ab_pct,
            })
        except Exception as e:
            print(f"compute_shot_zone_breakdown: player {pid} failed: {e}")
            time.sleep(1.2)

    return results


def compute_quarter_splits(season: str, player_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Fetch per-quarter scoring averages for each active player."""
    if player_ids is None:
        player_ids = _get_active_player_ids(400)

    results: list[dict[str, Any]] = []

    quarter_map = {
        '1ST QTR': 'q1_avg',
        '2ND QTR': 'q2_avg',
        '3RD QTR': 'q3_avg',
        '4TH QTR': 'q4_avg',
        '1ST': 'q1_avg',
        '2ND': 'q2_avg',
        '3RD': 'q3_avg',
        '4TH': 'q4_avg',
        # Numeric period labels (e.g. "1", "2", "3", "4") from Period split frame
        '1': 'q1_avg',
        '2': 'q2_avg',
        '3': 'q3_avg',
        '4': 'q4_avg',
    }

    for pid in player_ids:
        try:
            frames = playerdashboardbygamesplits.PlayerDashboardByGameSplits(
                player_id=pid,
                season=season,
                per_mode_detailed='PerGame',
                timeout=20,
            ).get_data_frames()
            time.sleep(1.2)

            period_df = None
            for frame in frames:
                if frame.empty:
                    continue
                cols_upper = [c.upper() for c in frame.columns]
                if 'GROUP_VALUE' in cols_upper:
                    gvs = [str(v).strip().upper() for v in frame.get('GROUP_VALUE', [])]
                    # Match period/quarter frames by GROUP_SET or by GROUP_VALUE content
                    group_set = str(frame['GROUP_SET'].iloc[0]).upper() if 'GROUP_SET' in frame.columns and len(frame) > 0 else ''
                    if ('PERIOD' in group_set or
                            any('QTR' in g or 'PERIOD' in g or '1ST' in g for g in gvs) or
                            set(gvs) & {'1', '2', '3', '4'}):
                        period_df = frame
                        break

            if period_df is None:
                continue

            q_avgs = {k: 0.0 for k in ('q1_avg', 'q2_avg', 'q3_avg', 'q4_avg')}
            q4_min = 0.0

            for _, row in period_df.iterrows():
                gv = str(row.get('GROUP_VALUE', '') or '').upper()
                key = None
                for pattern, qkey in quarter_map.items():
                    if pattern in gv:
                        key = qkey
                        break
                if key is None:
                    continue
                try:
                    q_avgs[key] = float(row.get('PTS', 0.0) or 0.0)
                    if key == 'q4_avg':
                        q4_min = float(row.get('MIN', 0.0) or 0.0)
                except Exception:
                    continue

            results.append({
                'player_id': pid,
                'q1_avg':    q_avgs['q1_avg'],
                'q2_avg':    q_avgs['q2_avg'],
                'q3_avg':    q_avgs['q3_avg'],
                'q4_avg':    q_avgs['q4_avg'],
                'q4_min_pg': q4_min,
            })
        except Exception as e:
            print(f"compute_quarter_splits: player {pid} failed: {e}")
            time.sleep(1.2)

    return results


def compute_opp_shot_zones(season: str) -> list[dict[str, Any]]:
    """Fetch opponent shot zone FG% allowed per team."""
    try:
        df = leaguedashoppptshot.LeagueDashOppPtShot(
            season=season,
            per_mode_simple='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.0)
    except Exception as e:
        print(f"compute_opp_shot_zones: fetch failed: {e}")
        return []

    # This endpoint has one row per team with zone columns
    # Look for columns like LESS_THAN_6FT_FG_PCT etc.
    # Zone classification from column names
    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            team_id = int(row.get('TEAM_ID', 0) or 0)
            if team_id == 0:
                continue

            def _find_col(keywords: list[str]) -> float:
                for col in df.columns:
                    col_up = col.upper()
                    if all(k in col_up for k in keywords):
                        try:
                            return float(row.get(col, 0.0) or 0.0)
                        except Exception:
                            return 0.0
                return 0.0

            # Try to find rim/paint/midrange/corner3/above_break3 columns
            rim_pct      = _find_col(['LESS_THAN_6', 'FG_PCT']) or _find_col(['RESTRICTED', 'FG_PCT'])
            paint_pct    = _find_col(['PAINT', 'FG_PCT']) or _find_col(['LESS_THAN_10', 'FG_PCT'])
            mid_pct      = _find_col(['MID', 'FG_PCT'])
            corner3_pct  = _find_col(['CORNER', 'FG3_PCT']) or _find_col(['CORNER', 'FG_PCT'])
            above3_pct   = _find_col(['ABOVE', 'BREAK', 'FG3_PCT']) or _find_col(['ABOVE', 'FG3_PCT'])

            # Fallback to league averages if columns not found
            results.append({
                'team_id':                   team_id,
                'rim_fg_pct_allowed':        rim_pct if rim_pct > 0 else 0.62,
                'paint_fg_pct_allowed':      paint_pct if paint_pct > 0 else 0.55,
                'midrange_fg_pct_allowed':   mid_pct if mid_pct > 0 else 0.42,
                'corner3_fg_pct_allowed':    corner3_pct if corner3_pct > 0 else 0.38,
                'above_break3_fg_pct_allowed': above3_pct if above3_pct > 0 else 0.35,
            })
        except Exception as e:
            print(f"compute_opp_shot_zones: row error: {e}")
            continue

    return results


def compute_synergy_team_defense(season: str) -> list[dict[str, Any]]:
    """Fetch team synergy defense stats (PPP allowed per play type)."""
    play_type_map = {
        'PRBallHandler': 'pnr_ppp_allowed',
        'Isolation':     'iso_ppp_allowed',
        'Spotup':        'spotup_ppp_allowed',
        'Transition':    'transition_ppp_allowed',
        'Postup':        'postup_ppp_allowed',
    }

    team_data: dict[int, dict[str, float]] = {}

    for pt, key in play_type_map.items():
        try:
            df = synergyplaytypes.SynergyPlayTypes(
                season=season,
                play_type_nullable=pt,
                player_or_team_abbreviation='T',
                type_grouping_nullable='defensive',
                per_mode_simple='PerGame',
            ).get_data_frames()[0]
            time.sleep(1.0)
            for _, row in df.iterrows():
                try:
                    tid = int(row.get('TEAM_ID', 0) or 0)
                    if tid == 0:
                        continue
                    if tid not in team_data:
                        team_data[tid] = {}
                    team_data[tid][key] = float(row.get('PPP', 0.9) or 0.9)
                except Exception:
                    continue
        except Exception as e:
            print(f"compute_synergy_team_defense: fetch failed for {pt}: {e}")
            time.sleep(1.0)

    results: list[dict[str, Any]] = []
    for tid, data in team_data.items():
        results.append({
            'team_id':               tid,
            'pnr_ppp_allowed':       data.get('pnr_ppp_allowed', 0.9),
            'iso_ppp_allowed':       data.get('iso_ppp_allowed', 0.9),
            'spotup_ppp_allowed':    data.get('spotup_ppp_allowed', 1.0),
            'transition_ppp_allowed': data.get('transition_ppp_allowed', 1.1),
            'postup_ppp_allowed':    data.get('postup_ppp_allowed', 0.9),
        })
    return results


# ---- Upsert functions for new tables ----

def upsert_advanced_player_stats(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_advanced_stats
              (player_id, usg_pct, ts_pct, efg_pct, ast_pct,
               oreb_pct, dreb_pct, reb_pct, pie,
               off_rating, def_rating, pace, net_rating,
               age, height_inches, weight, years_experience, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('usg_pct', 0.18)), float(r.get('ts_pct', 0.55)),
                float(r.get('efg_pct', 0.50)), float(r.get('ast_pct', 0.15)),
                float(r.get('oreb_pct', 0.05)), float(r.get('dreb_pct', 0.15)),
                float(r.get('reb_pct', 0.10)), float(r.get('pie', 0.10)),
                float(r.get('off_rating', 110.0)), float(r.get('def_rating', 110.0)),
                float(r.get('pace', 100.0)), float(r.get('net_rating', 0.0)),
                float(r.get('age', 26.0)), float(r.get('height_inches', 78.0)),
                float(r.get('weight', 220.0)), float(r.get('years_experience', 5.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_clutch_stats(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_clutch_stats
              (player_id, clutch_pts_pg, clutch_fg_pct, clutch_fg3_pct,
               clutch_fta_pg, clutch_plus_minus, clutch_min_pg, clutch_games, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('clutch_pts_pg', 0.0)), float(r.get('clutch_fg_pct', 0.45)),
                float(r.get('clutch_fg3_pct', 0.33)), float(r.get('clutch_fta_pg', 0.0)),
                float(r.get('clutch_plus_minus', 0.0)), float(r.get('clutch_min_pg', 0.0)),
                int(r.get('clutch_games', 0)), int(updated_at),
            ),
        )
    conn.commit()


def upsert_hustle_stats(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_hustle_stats
              (player_id, contested_shots_pg, deflections_pg,
               charges_drawn_pg, screen_assists_pg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('contested_shots_pg', 3.0)), float(r.get('deflections_pg', 1.0)),
                float(r.get('charges_drawn_pg', 0.1)), float(r.get('screen_assists_pg', 0.5)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_shot_profile(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_shot_profile
              (player_id, open_shot_fg_pct, open_shot_freq,
               tight_shot_fg_pct, tight_shot_freq,
               catch_shoot_fg_pct, catch_shoot_freq,
               pullup_fg_pct, pullup_freq, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('open_shot_fg_pct', 0.50)), float(r.get('open_shot_freq', 0.30)),
                float(r.get('tight_shot_fg_pct', 0.38)), float(r.get('tight_shot_freq', 0.15)),
                float(r.get('catch_shoot_fg_pct', 0.40)), float(r.get('catch_shoot_freq', 0.25)),
                float(r.get('pullup_fg_pct', 0.40)), float(r.get('pullup_freq', 0.20)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_play_types(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_play_types
              (player_id, iso_poss_pct, iso_ppp,
               pnr_bh_poss_pct, pnr_bh_ppp,
               pnr_roll_poss_pct, pnr_roll_ppp,
               spotup_poss_pct, spotup_ppp,
               transition_poss_pct, transition_ppp,
               postup_poss_pct, cut_poss_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('iso_poss_pct', 0.0)), float(r.get('iso_ppp', 0.9)),
                float(r.get('pnr_bh_poss_pct', 0.0)), float(r.get('pnr_bh_ppp', 0.9)),
                float(r.get('pnr_roll_poss_pct', 0.0)), float(r.get('pnr_roll_ppp', 0.9)),
                float(r.get('spotup_poss_pct', 0.0)), float(r.get('spotup_ppp', 1.0)),
                float(r.get('transition_poss_pct', 0.0)), float(r.get('transition_ppp', 1.1)),
                float(r.get('postup_poss_pct', 0.0)), float(r.get('cut_poss_pct', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_on_off(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_on_off
              (player_id, on_court_net_rating, off_court_net_rating,
               on_off_differential, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('on_court_net_rating', 0.0)),
                float(r.get('off_court_net_rating', 0.0)),
                float(r.get('on_off_differential', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_shot_zones(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_shot_zones
              (player_id, rim_fga_pct, rim_fg_pct,
               paint_fga_pct, paint_fg_pct,
               midrange_fga_pct, midrange_fg_pct,
               corner3_fga_pct, corner3_fg_pct,
               above_break3_fga_pct, above_break3_fg_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('rim_fga_pct', 0.25)), float(r.get('rim_fg_pct', 0.62)),
                float(r.get('paint_fga_pct', 0.30)), float(r.get('paint_fg_pct', 0.55)),
                float(r.get('midrange_fga_pct', 0.20)), float(r.get('midrange_fg_pct', 0.42)),
                float(r.get('corner3_fga_pct', 0.10)), float(r.get('corner3_fg_pct', 0.38)),
                float(r.get('above_break3_fga_pct', 0.25)), float(r.get('above_break3_fg_pct', 0.35)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_quarter_splits(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_quarter_splits
              (player_id, q1_avg, q2_avg, q3_avg, q4_avg, q4_min_pg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('q1_avg', 0.0)), float(r.get('q2_avg', 0.0)),
                float(r.get('q3_avg', 0.0)), float(r.get('q4_avg', 0.0)),
                float(r.get('q4_min_pg', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_opp_shot_zones(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_opp_shot_zones
              (team_id, rim_fg_pct_allowed, paint_fg_pct_allowed,
               midrange_fg_pct_allowed, corner3_fg_pct_allowed,
               above_break3_fg_pct_allowed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['team_id']),
                float(r.get('rim_fg_pct_allowed', 0.62)),
                float(r.get('paint_fg_pct_allowed', 0.55)),
                float(r.get('midrange_fg_pct_allowed', 0.42)),
                float(r.get('corner3_fg_pct_allowed', 0.38)),
                float(r.get('above_break3_fg_pct_allowed', 0.35)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_synergy_defense(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_synergy_defense
              (team_id, pnr_ppp_allowed, iso_ppp_allowed,
               spotup_ppp_allowed, transition_ppp_allowed,
               postup_ppp_allowed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['team_id']),
                float(r.get('pnr_ppp_allowed', 0.9)),
                float(r.get('iso_ppp_allowed', 0.9)),
                float(r.get('spotup_ppp_allowed', 1.0)),
                float(r.get('transition_ppp_allowed', 1.1)),
                float(r.get('postup_ppp_allowed', 0.9)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_player_tracking(season: str = '2024-25') -> list[dict[str, Any]]:
    """Fetch player tracking stats: speed/distance, possessions, and passing."""
    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    player_data: dict[int, dict[str, Any]] = {}

    # 1. SpeedDistance
    try:
        df = leaguedashptstats.LeagueDashPtStats(
            player_or_team='Player',
            season=season,
            pt_measure_type='SpeedDistance',
            per_mode_simple='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.2)
        for _, row in df.iterrows():
            pid = int(row['PLAYER_ID'])
            if pid not in player_data:
                player_data[pid] = {}
            player_data[pid].update({
                'avg_speed':     _sf(row.get('AVG_SPEED', 0.0)),
                'avg_speed_off': _sf(row.get('AVG_SPEED_OFF', 0.0)),
                'avg_speed_def': _sf(row.get('AVG_SPEED_DEF', 0.0)),
                'dist_miles':    _sf(row.get('DIST_MILES', 0.0)),
                'dist_miles_off': _sf(row.get('DIST_MILES_OFF', 0.0)),
                'dist_miles_def': _sf(row.get('DIST_MILES_DEF', 0.0)),
            })
    except Exception as e:
        print(f"compute_player_tracking: SpeedDistance fetch failed: {e}")

    time.sleep(1.2)

    # 2. Possessions
    try:
        df = leaguedashptstats.LeagueDashPtStats(
            player_or_team='Player',
            season=season,
            pt_measure_type='Possessions',
            per_mode_simple='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.2)
        for _, row in df.iterrows():
            pid = int(row['PLAYER_ID'])
            if pid not in player_data:
                player_data[pid] = {}
            player_data[pid].update({
                'touches_pg':          _sf(row.get('TOUCHES', 0.0)),
                'time_of_poss_pg':     _sf(row.get('TIME_OF_POSS', 0.0)),
                'avg_drib_per_touch':  _sf(row.get('AVG_DRIB_PER_TOUCH', 0.0)),
                'paint_touches_pg':    _sf(row.get('PAINT_TOUCHES', 0.0)),
                'elbow_touches_pg':    _sf(row.get('ELBOW_TOUCHES', 0.0)),
            })
    except Exception as e:
        print(f"compute_player_tracking: Possessions fetch failed: {e}")

    time.sleep(1.2)

    # 3. Passing
    try:
        df = leaguedashptstats.LeagueDashPtStats(
            player_or_team='Player',
            season=season,
            pt_measure_type='Passing',
            per_mode_simple='PerGame',
        ).get_data_frames()[0]
        time.sleep(1.2)
        for _, row in df.iterrows():
            pid = int(row['PLAYER_ID'])
            if pid not in player_data:
                player_data[pid] = {}
            player_data[pid].update({
                'passes_made_pg':     _sf(row.get('PASSES_MADE', 0.0)),
                'potential_ast_pg':   _sf(row.get('POTENTIAL_AST', 0.0)),
                'secondary_ast_pg':   _sf(row.get('SECONDARY_AST', 0.0)),
            })
    except Exception as e:
        print(f"compute_player_tracking: Passing fetch failed: {e}")

    results: list[dict[str, Any]] = []
    for pid, data in player_data.items():
        results.append({
            'player_id':        pid,
            'avg_speed':        data.get('avg_speed', 0.0),
            'avg_speed_off':    data.get('avg_speed_off', 0.0),
            'avg_speed_def':    data.get('avg_speed_def', 0.0),
            'dist_miles':       data.get('dist_miles', 0.0),
            'dist_miles_off':   data.get('dist_miles_off', 0.0),
            'dist_miles_def':   data.get('dist_miles_def', 0.0),
            'touches_pg':       data.get('touches_pg', 0.0),
            'time_of_poss_pg':  data.get('time_of_poss_pg', 0.0),
            'avg_drib_per_touch': data.get('avg_drib_per_touch', 0.0),
            'paint_touches_pg': data.get('paint_touches_pg', 0.0),
            'elbow_touches_pg': data.get('elbow_touches_pg', 0.0),
            'passes_made_pg':   data.get('passes_made_pg', 0.0),
            'potential_ast_pg': data.get('potential_ast_pg', 0.0),
            'secondary_ast_pg': data.get('secondary_ast_pg', 0.0),
        })
    return results


def upsert_player_tracking(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_tracking_stats
              (player_id, avg_speed, avg_speed_off, avg_speed_def,
               dist_miles, dist_miles_off, dist_miles_def,
               touches_pg, time_of_poss_pg, avg_drib_per_touch,
               paint_touches_pg, elbow_touches_pg,
               passes_made_pg, potential_ast_pg, secondary_ast_pg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('avg_speed', 0.0)), float(r.get('avg_speed_off', 0.0)),
                float(r.get('avg_speed_def', 0.0)), float(r.get('dist_miles', 0.0)),
                float(r.get('dist_miles_off', 0.0)), float(r.get('dist_miles_def', 0.0)),
                float(r.get('touches_pg', 0.0)), float(r.get('time_of_poss_pg', 0.0)),
                float(r.get('avg_drib_per_touch', 0.0)), float(r.get('paint_touches_pg', 0.0)),
                float(r.get('elbow_touches_pg', 0.0)), float(r.get('passes_made_pg', 0.0)),
                float(r.get('potential_ast_pg', 0.0)), float(r.get('secondary_ast_pg', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_team_standings(season: str = '2024-25') -> list[dict[str, Any]]:
    """Fetch team standings with win%, streak, L10, and home/road splits."""
    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    def _parse_record(record_str, idx=0):
        """Parse 'W-L' record string, return wins or losses by idx."""
        try:
            parts = str(record_str).split('-')
            return int(parts[idx])
        except Exception:
            return 0

    try:
        df = leaguestandingsv3.LeagueStandingsV3(season=season).get_data_frames()[0]
        time.sleep(1.2)
    except Exception as e:
        print(f"compute_team_standings: fetch failed: {e}")
        return []

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            team_id = int(row.get('TeamID', 0) or 0)
            if team_id == 0:
                continue

            win_pct = _sf(row.get('WinPCT', 0.5))
            wins = int(_sf(row.get('WINS', 0)))
            losses = int(_sf(row.get('LOSSES', 0)))

            # Conference rank
            conf_rank = int(_sf(row.get('ConferenceRank', row.get('PlayoffRank', 15)), 15))

            # Games back
            games_back = _sf(row.get('GamesBehind', row.get('ConferenceGamesBack', 0.0)))

            # Home/road records
            home_rec = str(row.get('HOME', row.get('HomeRecord', '0-0')) or '0-0')
            road_rec = str(row.get('ROAD', row.get('RoadRecord', '0-0')) or '0-0')
            home_w = _parse_record(home_rec, 0)
            home_l = _parse_record(home_rec, 1)
            road_w = _parse_record(road_rec, 0)
            road_l = _parse_record(road_rec, 1)
            home_win_pct = float(home_w) / max(home_w + home_l, 1)
            road_win_pct = float(road_w) / max(road_w + road_l, 1)

            # Current streak (positive = wins, negative = losses)
            streak_val = row.get('CurrentStreak', row.get('strCurrentStreak', 0))
            try:
                current_streak = int(float(str(streak_val).replace('W', '').replace('L', '-').strip()))
            except Exception:
                current_streak = 0

            # L10 record
            l10_rec = str(row.get('L10', row.get('Last10', '5-5')) or '5-5')
            l10_wins = _parse_record(l10_rec, 0)

            # Pts pg / opp pts pg
            pts_pg = _sf(row.get('PointsPG', row.get('PtsPG', 0.0)))
            opp_pts_pg = _sf(row.get('OppPointsPG', row.get('OppPtsPG', 0.0)))

            results.append({
                'team_id':        team_id,
                'win_pct':        win_pct,
                'wins':           wins,
                'losses':         losses,
                'conf_rank':      conf_rank,
                'games_back':     games_back,
                'home_win_pct':   home_win_pct,
                'road_win_pct':   road_win_pct,
                'current_streak': current_streak,
                'l10_wins':       l10_wins,
                'pts_pg':         pts_pg,
                'opp_pts_pg':     opp_pts_pg,
            })
        except Exception as e:
            print(f"compute_team_standings: row error: {e}")
            continue

    return results


def upsert_team_standings(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_standings
              (team_id, win_pct, wins, losses, conf_rank, games_back,
               home_win_pct, road_win_pct, current_streak, l10_wins,
               pts_pg, opp_pts_pg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['team_id']),
                float(r.get('win_pct', 0.5)), int(r.get('wins', 0)), int(r.get('losses', 0)),
                int(r.get('conf_rank', 15)), float(r.get('games_back', 0.0)),
                float(r.get('home_win_pct', 0.5)), float(r.get('road_win_pct', 0.5)),
                int(r.get('current_streak', 0)), int(r.get('l10_wins', 5)),
                float(r.get('pts_pg', 0.0)), float(r.get('opp_pts_pg', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_player_scoring_breakdown(season: str = '2024-25') -> list[dict[str, Any]]:
    """Fetch player scoring breakdown by method (3PT%, paint%, FT%, etc.)."""
    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense='Scoring',
            per_mode_detailed='PerGame',
            timeout=15,
        ).get_data_frames()[0]
        time.sleep(1.2)
    except Exception as e:
        print(f"compute_player_scoring_breakdown: fetch failed: {e}")
        return []

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            pid = int(row['PLAYER_ID'])
            results.append({
                'player_id':       pid,
                'pct_pts_3pt':     _sf(row.get('PCT_PTS_3PT', 0.0)),
                'pct_pts_paint':   _sf(row.get('PCT_PTS_PAINT', 0.0)),
                'pct_pts_ft':      _sf(row.get('PCT_PTS_FT', 0.0)),
                'pct_pts_midrange': _sf(row.get('PCT_PTS_MID_RANGE', 0.0)),
                'pct_uast_fgm':    _sf(row.get('PCT_UAST_2PM', row.get('PCT_UAST_FGM', 0.0))),
                'pct_ast_fgm':     _sf(row.get('PCT_AST_FGM', 0.0)),
            })
        except Exception as e:
            print(f"compute_player_scoring_breakdown: row error: {e}")
            continue

    return results


def upsert_player_scoring_breakdown(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_scoring_breakdown
              (player_id, pct_pts_3pt, pct_pts_paint, pct_pts_ft,
               pct_pts_midrange, pct_uast_fgm, pct_ast_fgm, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('pct_pts_3pt', 0.0)), float(r.get('pct_pts_paint', 0.0)),
                float(r.get('pct_pts_ft', 0.0)), float(r.get('pct_pts_midrange', 0.0)),
                float(r.get('pct_uast_fgm', 0.0)), float(r.get('pct_ast_fgm', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_player_vs_opponent(season: str = '2024-25', max_players: int = 400) -> list[dict[str, Any]]:
    """Fetch each player's historical splits vs each opponent team via game log aggregation."""
    from nba_api.stats.endpoints import playergamelog

    # Build team abbreviation -> team_id map
    all_nba_teams = teams.get_teams()
    abbrev_to_id = {t['abbreviation']: t['id'] for t in all_nba_teams}

    player_ids = _get_active_player_ids(max_players)
    results: list[dict[str, Any]] = []

    for pid in player_ids:
        try:
            df = playergamelog.PlayerGameLog(player_id=pid, season=season, timeout=10).get_data_frames()[0]
            time.sleep(1.2)
            if df.empty:
                continue

            # MATCHUP format: "TOR vs. BOS" or "TOR @ BOS" — opponent is last token
            df = df.copy()
            df['OPP_ABBREV'] = df['MATCHUP'].apply(lambda m: str(m).split()[-1])

            for opp_abbrev, group in df.groupby('OPP_ABBREV'):
                opp_id = abbrev_to_id.get(opp_abbrev, 0)
                if opp_id == 0:
                    continue
                gp  = len(group)
                pts = float(group['PTS'].mean()) if 'PTS' in group.columns else 0.0
                fg_pct = float(group['FG_PCT'].mean()) if 'FG_PCT' in group.columns else 0.45
                fga = float(group['FGA'].mean()) if 'FGA' in group.columns else 10.0
                fta = float(group['FTA'].mean()) if 'FTA' in group.columns else 3.0
                denom  = 2.0 * (fga + 0.44 * fta)
                ts_pct = pts / denom if denom > 0 else 0.55
                avg_min = float(group['MIN'].mean()) if 'MIN' in group.columns else 30.0
                results.append({
                    'player_id': pid, 'opponent_team_id': opp_id,
                    'gp': gp, 'avg_stat_pts': pts, 'fg_pct': fg_pct,
                    'ts_pct': ts_pct, 'avg_min': avg_min,
                })
        except Exception as e:
            print(f"compute_player_vs_opponent: player {pid} failed: {e}")
            time.sleep(1.2)

    return results


def upsert_player_vs_opponent(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_vs_opponent
              (player_id, opponent_team_id, gp, avg_stat_pts, fg_pct, ts_pct, avg_min)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                int(r['opponent_team_id']),
                int(r.get('gp', 0)),
                float(r.get('avg_stat_pts', 0.0)),
                float(r.get('fg_pct', 0.45)),
                float(r.get('ts_pct', 0.55)),
                float(r.get('avg_min', 30.0)),
            ),
        )
    conn.commit()


def compute_team_rest_splits(season: str = '2024-25') -> list[dict[str, Any]]:
    """Fetch how each team performs on B2B vs rested (for opponent context)."""
    from nba_api.stats.endpoints import teamdashboardbygeneralsplits

    all_teams = teams.get_teams()
    results: list[dict[str, Any]] = []

    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    for t in all_teams:
        tid = int(t['id'])
        try:
            frames = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=tid,
                season=season,
                per_mode_detailed='PerGame',
            ).get_data_frames()
            time.sleep(1.2)

            rest_df = None
            for frame in frames:
                if frame.empty:
                    continue
                if 'GROUP_SET' in frame.columns:
                    gsets = [str(v).upper() for v in frame['GROUP_SET'].unique()]
                    if any('REST' in g for g in gsets):
                        rest_df = frame
                        break
                if 'GROUP_VALUE' in frame.columns:
                    gvals = [str(v).upper() for v in frame['GROUP_VALUE'].unique()]
                    if any('REST' in g for g in gvals):
                        rest_df = frame
                        break

            if rest_df is None or rest_df.empty:
                continue

            # filter for rest-day rows
            gv_col = 'GROUP_VALUE' if 'GROUP_VALUE' in rest_df.columns else rest_df.columns[1]
            b2b_rows = rest_df[rest_df[gv_col].astype(str).str.upper().str.contains('REST DAYS 0|REST DAYS 1|0 REST|1 REST|BACK TO BACK|B2B', na=False)]
            rested_rows = rest_df[rest_df[gv_col].astype(str).str.upper().str.contains(r'REST DAYS 2|REST DAYS 3|2\+ REST|3\+ REST', na=False)]

            def _avg_rows(df_sub, col, default):
                vals = []
                for _, row in df_sub.iterrows():
                    v = _sf(row.get(col), default)
                    if v != default or col in row:
                        vals.append(v)
                return float(sum(vals) / len(vals)) if vals else default

            b2b_def  = _avg_rows(b2b_rows, 'DEF_RATING', 112.0)
            b2b_pace = _avg_rows(b2b_rows, 'PACE', 100.0)
            b2b_pts  = _avg_rows(b2b_rows, 'OPP_PTS', 115.0)
            if b2b_pts == 115.0:
                b2b_pts = _avg_rows(b2b_rows, 'PTS', 115.0)  # fallback to own pts

            rest_def  = _avg_rows(rested_rows, 'DEF_RATING', 110.0)
            rest_pace = _avg_rows(rested_rows, 'PACE', 100.0)

            results.append({
                'team_id': tid,
                'b2b_def_rating': b2b_def,
                'b2b_pace': b2b_pace,
                'b2b_pts_allowed': b2b_pts,
                'rested_def_rating': rest_def,
                'rested_pace': rest_pace,
            })
        except Exception as e:
            print(f"compute_team_rest_splits: team {tid} failed: {e}")
            time.sleep(1.2)

    return results


def upsert_team_rest_splits(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO team_rest_splits
              (team_id, b2b_def_rating, b2b_pace, b2b_pts_allowed,
               rested_def_rating, rested_pace, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['team_id']),
                float(r.get('b2b_def_rating', 112.0)),
                float(r.get('b2b_pace', 100.0)),
                float(r.get('b2b_pts_allowed', 115.0)),
                float(r.get('rested_def_rating', 110.0)),
                float(r.get('rested_pace', 100.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_player_yoy(season: str = '2024-25', max_players: int = 400) -> list[dict[str, Any]]:
    """Compute year-over-year stat changes using PlayerCareerStats (reliable endpoint)."""
    from nba_api.stats.endpoints import playercareerstats

    player_ids = _get_active_player_ids(max_players)
    results: list[dict[str, Any]] = []

    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    def _ts(pts, fga, fta):
        denom = 2.0 * (fga + 0.44 * fta)
        return (pts / denom) if denom > 0 else 0.55

    for pid in player_ids:
        try:
            frames = playercareerstats.PlayerCareerStats(
                player_id=pid,
                per_mode36='PerGame',
                timeout=15,
            ).get_data_frames()
            time.sleep(0.8)

            # Frame 0 is regular season per-game stats by season
            career_df = frames[0] if frames and not frames[0].empty else None
            if career_df is None or career_df.empty:
                continue

            # Sort by SEASON_ID descending
            season_col = next((c for c in career_df.columns if 'SEASON_ID' in c.upper()), None)
            if season_col is None:
                continue

            career_df = career_df.sort_values(season_col, ascending=False).reset_index(drop=True)
            seasons_count = len(career_df)

            if seasons_count < 2:
                yoy_pts_change = 0.0
                yoy_ts_change = 0.0
                yoy_usage_change = 0.0
            else:
                curr = career_df.iloc[0]
                prev = career_df.iloc[1]

                curr_pts = _sf(curr.get('PTS', 0.0))
                prev_pts = _sf(prev.get('PTS', 0.0))
                yoy_pts_change = curr_pts - prev_pts

                curr_ts = _ts(_sf(curr.get('PTS', 0.0)), _sf(curr.get('FGA', 10.0)), _sf(curr.get('FTA', 3.0)))
                prev_ts = _ts(_sf(prev.get('PTS', 0.0)), _sf(prev.get('FGA', 10.0)), _sf(prev.get('FTA', 3.0)))
                yoy_ts_change = curr_ts - prev_ts

                # USG_PCT not in career stats — approximate via FGA/(team FGA proxy)
                curr_fga = _sf(curr.get('FGA', 10.0))
                prev_fga = _sf(prev.get('FGA', 10.0))
                yoy_usage_change = (curr_fga - prev_fga) / max(prev_fga, 1.0)

            results.append({
                'player_id': pid,
                'yoy_pts_change': yoy_pts_change,
                'yoy_ts_change': yoy_ts_change,
                'yoy_usage_change': yoy_usage_change,
                'seasons_in_league': seasons_count,
            })
        except Exception as e:
            print(f"compute_player_yoy: player {pid} failed: {e}")
            time.sleep(0.8)

    return results


def upsert_player_yoy(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO player_yoy_stats
              (player_id, yoy_pts_change, yoy_ts_change, yoy_usage_change,
               seasons_in_league, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(r['player_id']),
                float(r.get('yoy_pts_change', 0.0)),
                float(r.get('yoy_ts_change', 0.0)),
                float(r.get('yoy_usage_change', 0.0)),
                int(r.get('seasons_in_league', 1)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_team_home_away_splits(season: str = '2024-25') -> list[dict[str, Any]]:
    """Fetch home vs away defensive rating split for each team."""
    all_teams = teams.get_teams()
    rows: list[dict[str, Any]] = []
    for team in all_teams:
        tid = int(team['id'])
        try:
            dash = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=tid, season=season, measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame', timeout=60,
            )
            dfs = dash.get_data_frames()
            # Frame index 1 is typically the Location split (Home/Road)
            loc_df = None
            for df in dfs:
                if df is not None and not df.empty and 'GROUP_VALUE' in df.columns:
                    vals = df['GROUP_VALUE'].str.upper().tolist()
                    if 'HOME' in vals or 'ROAD' in vals:
                        loc_df = df
                        break
            if loc_df is None or loc_df.empty:
                time.sleep(0.6)
                continue
            home_row = loc_df[loc_df['GROUP_VALUE'].str.upper() == 'HOME']
            away_row = loc_df[loc_df['GROUP_VALUE'].str.upper() == 'ROAD']
            home_def = float(home_row['DEF_RATING'].iloc[0]) if not home_row.empty and 'DEF_RATING' in home_row.columns else 110.0
            away_def = float(away_row['DEF_RATING'].iloc[0]) if not away_row.empty and 'DEF_RATING' in away_row.columns else 110.0
            rows.append({
                'team_id': tid,
                'home_def_rating': home_def,
                'away_def_rating': away_def,
                'home_away_def_split': home_def - away_def,
            })
            time.sleep(0.6)
        except Exception as _e:
            print(f"  home/away split failed for team {tid}: {_e}")
            time.sleep(0.6)
    return rows


def upsert_team_home_away_splits(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_home_away_splits (team_id, home_def_rating, away_def_rating, home_away_def_split, updated_at)
            VALUES (:team_id, :home_def_rating, :away_def_rating, :home_away_def_split, :updated_at)
            ON CONFLICT(team_id) DO UPDATE SET
                home_def_rating=excluded.home_def_rating,
                away_def_rating=excluded.away_def_rating,
                home_away_def_split=excluded.home_away_def_split,
                updated_at=excluded.updated_at
            """,
            {**r, 'updated_at': updated_at},
        )
    conn.commit()


def compute_lineup_stats(season: str = '2024-25') -> list[dict[str, Any]]:
    """Compute lineup depth/continuity metrics per team using LeagueDashLineups."""
    try:
        dash = leaguedashlineups.LeagueDashLineups(
            season=season, measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame', timeout=90,
        )
        df = dash.get_data_frames()[0]
        time.sleep(1.0)
    except Exception as _e:
        print(f"  lineup stats fetch failed: {_e}")
        return []

    if df is None or df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for tid, grp in df.groupby('TEAM_ID'):
        tid = int(tid)
        grp = grp.copy()
        # Sort by minutes descending
        min_col = 'MIN' if 'MIN' in grp.columns else None
        nr_col = 'NET_RATING' if 'NET_RATING' in grp.columns else None
        if min_col:
            grp = grp.sort_values(min_col, ascending=False)
        total_min = float(grp[min_col].sum()) if min_col else 1.0

        # Top lineup = highest-minutes 5-man unit
        top_nr = float(grp[nr_col].iloc[0]) if nr_col and not grp.empty else 0.0

        # Bench lineups: those with GROUP_VALUE containing 0 starters (heuristic: net_rating of bottom half)
        half = max(1, len(grp) // 2)
        bench_grp = grp.iloc[half:]
        bench_nr = float(bench_grp[nr_col].mean()) if nr_col and not bench_grp.empty else 0.0

        # bench_strength: bench net rating normalised to [-1, 1]
        bench_strength = max(-1.0, min(1.0, bench_nr / 10.0))

        # lineup_continuity: fraction of total minutes in top-3 lineups
        top3_min = float(grp[min_col].head(3).sum()) if min_col else total_min
        lineup_continuity = top3_min / max(total_min, 1.0)

        rows.append({
            'team_id': tid,
            'top_lineup_net_rating': top_nr,
            'bench_net_rating': bench_nr,
            'bench_strength': bench_strength,
            'lineup_continuity': lineup_continuity,
            'lineups_played_count': int(len(grp)),
        })
    return rows


def upsert_lineup_stats(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_lineup_stats
                (team_id, top_lineup_net_rating, bench_net_rating, bench_strength, lineup_continuity, lineups_played_count, updated_at)
            VALUES
                (:team_id, :top_lineup_net_rating, :bench_net_rating, :bench_strength, :lineup_continuity, :lineups_played_count, :updated_at)
            ON CONFLICT(team_id) DO UPDATE SET
                top_lineup_net_rating=excluded.top_lineup_net_rating,
                bench_net_rating=excluded.bench_net_rating,
                bench_strength=excluded.bench_strength,
                lineup_continuity=excluded.lineup_continuity,
                lineups_played_count=excluded.lineups_played_count,
                updated_at=excluded.updated_at
            """,
            {**r, 'updated_at': updated_at},
        )
    conn.commit()


def compute_injury_status() -> list[dict[str, Any]]:
    """Fetch current injury status for all 30 teams via InjuryTracker."""
    try:
        from .injury_tracker import InjuryTracker
    except ImportError:
        try:
            from src.injury_tracker import InjuryTracker
        except ImportError:
            print("  InjuryTracker not available")
            return []

    tracker = InjuryTracker()
    all_teams = teams.get_teams()
    rows: list[dict[str, Any]] = []
    for team in all_teams:
        tid = int(team['id'])
        try:
            info = tracker.get_team_injuries(tid)
            rows.append({
                'team_id': tid,
                'key_players_out': int(info.get('key_players_out', 0)),
                'total_players_out': int(info.get('total_players_out', 0)),
                'total_impact': float(info.get('total_impact', 0.0)),
            })
        except Exception as _e:
            print(f"  injury status failed for team {tid}: {_e}")
            rows.append({'team_id': tid, 'key_players_out': 0, 'total_players_out': 0, 'total_impact': 0.0})
    return rows


def upsert_injury_status(conn: sqlite3.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_injury_status (team_id, key_players_out, total_players_out, total_impact, updated_at)
            VALUES (:team_id, :key_players_out, :total_players_out, :total_impact, :updated_at)
            ON CONFLICT(team_id) DO UPDATE SET
                key_players_out=excluded.key_players_out,
                total_players_out=excluded.total_players_out,
                total_impact=excluded.total_impact,
                updated_at=excluded.updated_at
            """,
            {**r, 'updated_at': updated_at},
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

    # ---- New enriched feature jobs ----
    adv_rows: list[dict[str, Any]] = []
    try:
        print("computing advanced player stats...")
        adv_rows = compute_advanced_player_stats(season=season)
        upsert_advanced_player_stats(conn, adv_rows, updated_at=updated_at)
        print(f"  advanced player stats: {len(adv_rows)} rows")
    except Exception as _e:
        print(f"advanced player stats failed: {_e}")

    clutch_rows: list[dict[str, Any]] = []
    try:
        print("computing clutch stats...")
        clutch_rows = compute_clutch_stats(season=season)
        upsert_clutch_stats(conn, clutch_rows, updated_at=updated_at)
        print(f"  clutch stats: {len(clutch_rows)} rows")
    except Exception as _e:
        print(f"clutch stats failed: {_e}")

    hustle_rows: list[dict[str, Any]] = []
    try:
        print("computing hustle stats...")
        hustle_rows = compute_hustle_stats(season=season)
        upsert_hustle_stats(conn, hustle_rows, updated_at=updated_at)
        print(f"  hustle stats: {len(hustle_rows)} rows")
    except Exception as _e:
        print(f"hustle stats failed: {_e}")

    shot_profile_rows: list[dict[str, Any]] = []
    try:
        print("computing shot profile...")
        shot_profile_rows = compute_shot_profile(season=season)
        upsert_shot_profile(conn, shot_profile_rows, updated_at=updated_at)
        print(f"  shot profile: {len(shot_profile_rows)} rows")
    except Exception as _e:
        print(f"shot profile failed: {_e}")

    play_type_rows: list[dict[str, Any]] = []
    try:
        print("computing synergy play types...")
        play_type_rows = compute_synergy_play_types(season=season)
        upsert_play_types(conn, play_type_rows, updated_at=updated_at)
        print(f"  play types: {len(play_type_rows)} rows")
    except Exception as _e:
        print(f"synergy play types failed: {_e}")

    on_off_rows: list[dict[str, Any]] = []
    try:
        print("computing on/off ratings...")
        on_off_rows = compute_on_off_ratings(season=season)
        upsert_on_off(conn, on_off_rows, updated_at=updated_at)
        print(f"  on/off ratings: {len(on_off_rows)} rows")
    except Exception as _e:
        print(f"on/off ratings failed: {_e}")

    shot_zone_rows: list[dict[str, Any]] = []
    try:
        print("computing shot zone breakdown (up to 400 players)...")
        shot_zone_rows = compute_shot_zone_breakdown(season=season)
        upsert_shot_zones(conn, shot_zone_rows, updated_at=updated_at)
        print(f"  shot zones: {len(shot_zone_rows)} rows")
    except Exception as _e:
        print(f"shot zone breakdown failed: {_e}")

    quarter_split_rows: list[dict[str, Any]] = []
    try:
        print("computing quarter splits (up to 400 players)...")
        quarter_split_rows = compute_quarter_splits(season=season)
        upsert_quarter_splits(conn, quarter_split_rows, updated_at=updated_at)
        print(f"  quarter splits: {len(quarter_split_rows)} rows")
    except Exception as _e:
        print(f"quarter splits failed: {_e}")

    opp_shot_zone_rows: list[dict[str, Any]] = []
    try:
        print("computing opponent shot zones...")
        opp_shot_zone_rows = compute_opp_shot_zones(season=season)
        upsert_opp_shot_zones(conn, opp_shot_zone_rows, updated_at=updated_at)
        print(f"  opp shot zones: {len(opp_shot_zone_rows)} rows")
    except Exception as _e:
        print(f"opp shot zones failed: {_e}")

    synergy_def_rows: list[dict[str, Any]] = []
    try:
        print("computing synergy team defense...")
        synergy_def_rows = compute_synergy_team_defense(season=season)
        upsert_synergy_defense(conn, synergy_def_rows, updated_at=updated_at)
        print(f"  synergy defense: {len(synergy_def_rows)} rows")
    except Exception as _e:
        print(f"synergy team defense failed: {_e}")

    # ---- New feature group jobs ----
    tracking_rows: list[dict[str, Any]] = []
    try:
        print("computing player tracking stats...")
        tracking_rows = compute_player_tracking(season=season)
        upsert_player_tracking(conn, tracking_rows, updated_at=updated_at)
        print(f"  player tracking: {len(tracking_rows)} rows")
    except Exception as _e:
        print(f"player tracking failed: {_e}")

    standings_rows: list[dict[str, Any]] = []
    try:
        print("computing team standings...")
        standings_rows = compute_team_standings(season=season)
        upsert_team_standings(conn, standings_rows, updated_at=updated_at)
        print(f"  team standings: {len(standings_rows)} rows")
    except Exception as _e:
        print(f"team standings failed: {_e}")

    scoring_breakdown_rows: list[dict[str, Any]] = []
    try:
        print("computing player scoring breakdown...")
        scoring_breakdown_rows = compute_player_scoring_breakdown(season=season)
        upsert_player_scoring_breakdown(conn, scoring_breakdown_rows, updated_at=updated_at)
        print(f"  scoring breakdown: {len(scoring_breakdown_rows)} rows")
    except Exception as _e:
        print(f"player scoring breakdown failed: {_e}")

    player_vs_opp_rows: list[dict[str, Any]] = []
    try:
        print("computing player vs opponent splits (up to 400 players)...")
        player_vs_opp_rows = compute_player_vs_opponent(season=season)
        upsert_player_vs_opponent(conn, player_vs_opp_rows)
        print(f"  player vs opponent: {len(player_vs_opp_rows)} rows")
    except Exception as _e:
        print(f"player vs opponent failed: {_e}")

    team_rest_rows: list[dict[str, Any]] = []
    try:
        print("computing team rest splits (30 teams)...")
        team_rest_rows = compute_team_rest_splits(season=season)
        upsert_team_rest_splits(conn, team_rest_rows, updated_at=updated_at)
        print(f"  team rest splits: {len(team_rest_rows)} rows")
    except Exception as _e:
        print(f"team rest splits failed: {_e}")

    player_yoy_rows: list[dict[str, Any]] = []
    try:
        print("computing player year-over-year stats (up to 400 players)...")
        player_yoy_rows = compute_player_yoy(season=season)
        upsert_player_yoy(conn, player_yoy_rows, updated_at=updated_at)
        print(f"  player yoy: {len(player_yoy_rows)} rows")
    except Exception as _e:
        print(f"player yoy stats failed: {_e}")

    # ---- New Tier B jobs ----
    home_away_rows: list[dict[str, Any]] = []
    try:
        print("computing team home/away defensive splits (30 teams)...")
        home_away_rows = compute_team_home_away_splits(season=season)
        upsert_team_home_away_splits(conn, home_away_rows, updated_at=updated_at)
        print(f"  home/away splits: {len(home_away_rows)} rows")
    except Exception as _e:
        print(f"team home/away splits failed: {_e}")

    lineup_rows: list[dict[str, Any]] = []
    try:
        print("computing team lineup stats...")
        lineup_rows = compute_lineup_stats(season=season)
        upsert_lineup_stats(conn, lineup_rows, updated_at=updated_at)
        print(f"  lineup stats: {len(lineup_rows)} rows")
    except Exception as _e:
        print(f"lineup stats failed: {_e}")

    injury_rows: list[dict[str, Any]] = []
    try:
        print("computing team injury status (current)...")
        injury_rows = compute_injury_status()
        upsert_injury_status(conn, injury_rows, updated_at=updated_at)
        print(f"  injury status: {len(injury_rows)} rows")
    except Exception as _e:
        print(f"injury status failed: {_e}")

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
        "adv_rows": len(adv_rows),
        "clutch_rows": len(clutch_rows),
        "hustle_rows": len(hustle_rows),
        "shot_profile_rows": len(shot_profile_rows),
        "play_type_rows": len(play_type_rows),
        "on_off_rows": len(on_off_rows),
        "shot_zone_rows": len(shot_zone_rows),
        "quarter_split_rows": len(quarter_split_rows),
        "opp_shot_zone_rows": len(opp_shot_zone_rows),
        "synergy_def_rows": len(synergy_def_rows),
        "tracking_rows": len(tracking_rows),
        "standings_rows": len(standings_rows),
        "scoring_breakdown_rows": len(scoring_breakdown_rows),
        "player_vs_opp_rows": len(player_vs_opp_rows),
        "team_rest_rows": len(team_rest_rows),
        "player_yoy_rows": len(player_yoy_rows),
        "home_away_rows": len(home_away_rows),
        "lineup_rows": len(lineup_rows),
        "injury_rows": len(injury_rows),
    }


