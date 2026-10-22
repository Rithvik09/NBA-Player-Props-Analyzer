"""
Reusable precompute jobs used by both:
- CLI: scripts/update_precomputed.py
- App: background refresh when stale

Writes to PostgreSQL:
- dvp_by_position
- team_special_defenders
"""

from __future__ import annotations

import time
from typing import Any

import psycopg
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

from src import db

FANTASYPROS_DVP_URL = "https://www.fantasypros.com/daily-fantasy/nba/fanduel-defense-vs-position.php"


def compute_current_season() -> str:
    now = time.localtime()
    y = now.tm_year
    m = now.tm_mon
    if 1 <= m <= 7:
        return f"{y-1}-{str(y)[2:]}"
    return f"{y}-{str(y+1)[2:]}"


def ensure_tables(conn: psycopg.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dvp_by_position (
            team_id BIGINT NOT NULL,
            position TEXT NOT NULL,
            gp INTEGER,
            pts DOUBLE PRECISION,
            reb DOUBLE PRECISION,
            ast DOUBLE PRECISION,
            fg3m DOUBLE PRECISION,
            stl DOUBLE PRECISION,
            blk DOUBLE PRECISION,
            tov DOUBLE PRECISION,
            fd_pts DOUBLE PRECISION,
            source TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (team_id, position, source)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_special_defenders (
            team_id BIGINT NOT NULL,
            pos_group TEXT NOT NULL,
            rank INTEGER NOT NULL,
            player_id BIGINT,
            player_name TEXT,
            def_rating DOUBLE PRECISION,
            def_ws DOUBLE PRECISION,
            pct_stl DOUBLE PRECISION,
            pct_blk DOUBLE PRECISION,
            min_per_game DOUBLE PRECISION,
            score01 DOUBLE PRECISION,
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
            foul_rate DOUBLE PRECISION,
            home_win_pct DOUBLE PRECISION,
            pace DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    # game → assigned officials. Used at training time to look up which
    # refs worked a specific historical game so the row's ref_* features
    # come from the AVG of THIS game's refs (real per-row variance) rather
    # than a league-mean constant. Populated by
    # compute_game_refs_for_season() walking BoxScoreSummaryV3.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS game_officials (
            game_id TEXT NOT NULL,
            ref_name TEXT NOT NULL,
            ref_id BIGINT,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (game_id, ref_name)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_game_officials_game ON game_officials(game_id)"
    )
    # Per-game team stats — used to derive rolling opp_pace / def_rating
    # features at training time. Populated by the same NBA-API box-score
    # walk that fills game_officials, so backfilling both is one pass.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS game_team_stats (
            game_id TEXT NOT NULL,
            team_id BIGINT NOT NULL,
            game_date TEXT,
            pace DOUBLE PRECISION,
            off_rating DOUBLE PRECISION,
            def_rating DOUBLE PRECISION,
            pts DOUBLE PRECISION,
            opp_pts DOUBLE PRECISION,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (game_id, team_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_game_team_stats_team_date ON game_team_stats(team_id, game_date)"
    )

    # Per-game national-TV broadcaster from NBA's daily Scoreboard.
    # Raw value is something like "ABC/ESPN/Disney+" or NULL for
    # non-nationally-televised games. We store the raw string so
    # downstream code can do feature engineering (presence flag,
    # specific-network flag, etc.) without re-scraping.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS game_broadcasts (
            game_id TEXT PRIMARY KEY,
            game_date TEXT,
            natl_tv TEXT,
            home_tv TEXT,
            away_tv TEXT,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_foul_rates (
            team_id BIGINT PRIMARY KEY,
            foul_rate_season DOUBLE PRECISION,
            foul_rate_last5 DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dvp_rolling (
            team_id BIGINT NOT NULL,
            "window" INTEGER NOT NULL,
            pts DOUBLE PRECISION,
            reb DOUBLE PRECISION,
            ast DOUBLE PRECISION,
            fg3m DOUBLE PRECISION,
            stl DOUBLE PRECISION,
            blk DOUBLE PRECISION,
            tov DOUBLE PRECISION,
            gp INTEGER,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (team_id, "window")
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_stats (
            team_id BIGINT PRIMARY KEY,
            pts_fb DOUBLE PRECISION,
            pts_off_tov DOUBLE PRECISION,
            opp_fga DOUBLE PRECISION,
            opp_fg_pct DOUBLE PRECISION,
            opp_fg3a DOUBLE PRECISION,
            opp_fg3_pct DOUBLE PRECISION,
            opp_tov DOUBLE PRECISION,
            opp_stl DOUBLE PRECISION,
            opp_blk DOUBLE PRECISION,
            opp_pts_paint DOUBLE PRECISION,
            opp_pts_fb DOUBLE PRECISION,
            opp_pts_off_tov DOUBLE PRECISION,
            opp_def_rating_last5 DOUBLE PRECISION,
            opp_blk_last5 DOUBLE PRECISION,
            opp_stl_last5 DOUBLE PRECISION,
            lg_pts_fb DOUBLE PRECISION,
            lg_pts_off_tov DOUBLE PRECISION,
            lg_fga DOUBLE PRECISION,
            lg_fg_pct DOUBLE PRECISION,
            lg_fg3a DOUBLE PRECISION,
            lg_tov DOUBLE PRECISION,
            lg_stl DOUBLE PRECISION,
            foul_rate_season DOUBLE PRECISION,
            foul_rate_last5 DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_advanced_stats (
            player_id BIGINT PRIMARY KEY,
            usg_pct DOUBLE PRECISION, ts_pct DOUBLE PRECISION, efg_pct DOUBLE PRECISION, ast_pct DOUBLE PRECISION,
            oreb_pct DOUBLE PRECISION, dreb_pct DOUBLE PRECISION, reb_pct DOUBLE PRECISION, pie DOUBLE PRECISION,
            off_rating DOUBLE PRECISION, def_rating DOUBLE PRECISION, pace DOUBLE PRECISION, net_rating DOUBLE PRECISION,
            age DOUBLE PRECISION, height_inches DOUBLE PRECISION, weight DOUBLE PRECISION, years_experience DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_clutch_stats (
            player_id BIGINT PRIMARY KEY,
            clutch_pts_pg DOUBLE PRECISION, clutch_fg_pct DOUBLE PRECISION, clutch_fg3_pct DOUBLE PRECISION,
            clutch_fta_pg DOUBLE PRECISION, clutch_plus_minus DOUBLE PRECISION, clutch_min_pg DOUBLE PRECISION,
            clutch_games INTEGER,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_hustle_stats (
            player_id BIGINT PRIMARY KEY,
            contested_shots_pg DOUBLE PRECISION, deflections_pg DOUBLE PRECISION,
            charges_drawn_pg DOUBLE PRECISION, screen_assists_pg DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_shot_profile (
            player_id BIGINT PRIMARY KEY,
            open_shot_fg_pct DOUBLE PRECISION, open_shot_freq DOUBLE PRECISION,
            tight_shot_fg_pct DOUBLE PRECISION, tight_shot_freq DOUBLE PRECISION,
            catch_shoot_fg_pct DOUBLE PRECISION, catch_shoot_freq DOUBLE PRECISION,
            pullup_fg_pct DOUBLE PRECISION, pullup_freq DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_play_types (
            player_id BIGINT PRIMARY KEY,
            iso_poss_pct DOUBLE PRECISION, iso_ppp DOUBLE PRECISION,
            pnr_bh_poss_pct DOUBLE PRECISION, pnr_bh_ppp DOUBLE PRECISION,
            pnr_roll_poss_pct DOUBLE PRECISION, pnr_roll_ppp DOUBLE PRECISION,
            spotup_poss_pct DOUBLE PRECISION, spotup_ppp DOUBLE PRECISION,
            transition_poss_pct DOUBLE PRECISION, transition_ppp DOUBLE PRECISION,
            postup_poss_pct DOUBLE PRECISION, cut_poss_pct DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_on_off (
            player_id BIGINT PRIMARY KEY,
            on_court_net_rating DOUBLE PRECISION,
            off_court_net_rating DOUBLE PRECISION,
            on_off_differential DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_shot_zones (
            player_id BIGINT PRIMARY KEY,
            rim_fga_pct DOUBLE PRECISION, rim_fg_pct DOUBLE PRECISION,
            paint_fga_pct DOUBLE PRECISION, paint_fg_pct DOUBLE PRECISION,
            midrange_fga_pct DOUBLE PRECISION, midrange_fg_pct DOUBLE PRECISION,
            corner3_fga_pct DOUBLE PRECISION, corner3_fg_pct DOUBLE PRECISION,
            above_break3_fga_pct DOUBLE PRECISION, above_break3_fg_pct DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_quarter_splits (
            player_id BIGINT PRIMARY KEY,
            q1_avg DOUBLE PRECISION, q2_avg DOUBLE PRECISION, q3_avg DOUBLE PRECISION, q4_avg DOUBLE PRECISION,
            q4_min_pg DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_opp_shot_zones (
            team_id BIGINT PRIMARY KEY,
            rim_fg_pct_allowed DOUBLE PRECISION,
            paint_fg_pct_allowed DOUBLE PRECISION,
            midrange_fg_pct_allowed DOUBLE PRECISION,
            corner3_fg_pct_allowed DOUBLE PRECISION,
            above_break3_fg_pct_allowed DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_synergy_defense (
            team_id BIGINT PRIMARY KEY,
            pnr_ppp_allowed DOUBLE PRECISION,
            iso_ppp_allowed DOUBLE PRECISION,
            spotup_ppp_allowed DOUBLE PRECISION,
            transition_ppp_allowed DOUBLE PRECISION,
            postup_ppp_allowed DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_tracking_stats (
            player_id BIGINT PRIMARY KEY,
            avg_speed DOUBLE PRECISION, avg_speed_off DOUBLE PRECISION, avg_speed_def DOUBLE PRECISION,
            dist_miles DOUBLE PRECISION, dist_miles_off DOUBLE PRECISION, dist_miles_def DOUBLE PRECISION,
            touches_pg DOUBLE PRECISION, time_of_poss_pg DOUBLE PRECISION, avg_drib_per_touch DOUBLE PRECISION,
            paint_touches_pg DOUBLE PRECISION, elbow_touches_pg DOUBLE PRECISION,
            passes_made_pg DOUBLE PRECISION, potential_ast_pg DOUBLE PRECISION, secondary_ast_pg DOUBLE PRECISION,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_standings (
            team_id BIGINT PRIMARY KEY,
            win_pct DOUBLE PRECISION, wins INTEGER, losses INTEGER,
            conf_rank INTEGER, games_back DOUBLE PRECISION,
            home_win_pct DOUBLE PRECISION, road_win_pct DOUBLE PRECISION,
            current_streak INTEGER,
            l10_wins INTEGER,
            pts_pg DOUBLE PRECISION, opp_pts_pg DOUBLE PRECISION,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_scoring_breakdown (
            player_id BIGINT PRIMARY KEY,
            pct_pts_3pt DOUBLE PRECISION, pct_pts_paint DOUBLE PRECISION, pct_pts_ft DOUBLE PRECISION,
            pct_pts_midrange DOUBLE PRECISION, pct_uast_fgm DOUBLE PRECISION, pct_ast_fgm DOUBLE PRECISION,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_vs_opponent (
            player_id BIGINT,
            opponent_team_id BIGINT,
            gp INTEGER,
            avg_stat_pts DOUBLE PRECISION,
            fg_pct DOUBLE PRECISION,
            ts_pct DOUBLE PRECISION,
            avg_min DOUBLE PRECISION,
            PRIMARY KEY (player_id, opponent_team_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_rest_splits (
            team_id BIGINT PRIMARY KEY,
            b2b_def_rating DOUBLE PRECISION,
            b2b_pace DOUBLE PRECISION,
            b2b_pts_allowed DOUBLE PRECISION,
            rested_def_rating DOUBLE PRECISION,
            rested_pace DOUBLE PRECISION,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_yoy_stats (
            player_id BIGINT PRIMARY KEY,
            yoy_pts_change DOUBLE PRECISION,
            yoy_ts_change DOUBLE PRECISION,
            yoy_usage_change DOUBLE PRECISION,
            seasons_in_league INTEGER,
            updated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_home_away_splits (
            team_id BIGINT PRIMARY KEY,
            home_def_rating DOUBLE PRECISION,
            away_def_rating DOUBLE PRECISION,
            home_away_def_split DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_lineup_stats (
            team_id BIGINT PRIMARY KEY,
            top_lineup_net_rating DOUBLE PRECISION,
            bench_net_rating DOUBLE PRECISION,
            bench_strength DOUBLE PRECISION,
            lineup_continuity DOUBLE PRECISION,
            lineups_played_count INTEGER,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_injury_status (
            team_id BIGINT PRIMARY KEY,
            key_players_out INTEGER,
            total_players_out INTEGER,
            total_impact DOUBLE PRECISION,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()

    # Idempotent per-season migration of per-player snapshot tables.
    # First run promotes the schema; subsequent runs are no-ops.
    # Existing rows get tagged with current_season; older seasons are
    # filled in by re-running the per-season compute jobs.
    try:
        cur_season = compute_current_season()
        for tbl in PER_PLAYER_TABLES_TO_MIGRATE:
            try:
                # Commits on success — see the note in the helper.
                _migrate_player_table_to_per_season(conn, tbl, cur_season)
            except Exception as e:
                # Migration is best-effort — if one table can't be
                # migrated (e.g. permission, in-use), log and continue.
                # The table just keeps the per-player schema and won't
                # benefit from per-season variance until manually fixed.
                #
                # The rollback is what makes "continue" possible: a failed
                # statement aborts the whole PostgreSQL transaction, so
                # without it every later table would fail too. Already
                # committed tables are unaffected.
                conn.rollback()
                print(f"[migrate] {tbl} skipped: {e}")
    except Exception as e:
        print(f"[migrate] could not determine current season: {e}")


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


def upsert_dvp(conn: psycopg.Connection, dvp: dict[tuple[int, str], dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for (team_id, pos), s in dvp.items():
        cur.execute(
            """
            INSERT INTO dvp_by_position (team_id, position, gp, pts, reb, ast, fg3m, stl, blk, tov, fd_pts, source, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, position, source) DO UPDATE SET
                gp = excluded.gp,
                pts = excluded.pts,
                reb = excluded.reb,
                ast = excluded.ast,
                fg3m = excluded.fg3m,
                stl = excluded.stl,
                blk = excluded.blk,
                tov = excluded.tov,
                fd_pts = excluded.fd_pts,
                updated_at = excluded.updated_at
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


def upsert_defenders(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_special_defenders (team_id, pos_group, rank, player_id, player_name, def_rating, def_ws, pct_stl, pct_blk, min_per_game, score01, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, pos_group, rank) DO UPDATE SET
                player_id = excluded.player_id,
                player_name = excluded.player_name,
                def_rating = excluded.def_rating,
                def_ws = excluded.def_ws,
                pct_stl = excluded.pct_stl,
                pct_blk = excluded.pct_blk,
                min_per_game = excluded.min_per_game,
                score01 = excluded.score01,
                updated_at = excluded.updated_at
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


def compute_referee_stats_from_history(db_path: str | None = None) -> list[dict[str, Any]]:
    """Recompute measured NBA observations instead of scraping a names directory."""
    from .referee_history import rebuild_aggregates
    with db.connect(db_path) as conn:
        if not db.table_columns(conn, 'referee_game_stats'):
            return []
        conn.commit()
        rebuild_aggregates(conn)
        rows = conn.execute('SELECT ref_name,games,foul_rate,home_win_pct,pace FROM referee_stats').fetchall()
        return [dict(zip(('ref_name','games','foul_rate','home_win_pct','pace'), row)) for row in rows]


def scrape_ref_stats() -> list[dict[str, Any]]:
    """Compatibility alias for the validated NBA history collector."""
    return compute_referee_stats_from_history()


def compute_game_refs_for_season(
    season: str,
    *,
    max_games: int | None = None,
    sleep_between: float = 0.7,
    progress_every: int = 50,
    on_progress=None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Walk every game of ``season`` and return:

      1. ``game_officials_rows``: one row per (game, ref) — the lookup
         table the trainer uses to know who refereed a given game.
      2. ``ref_aggregates``: dict ``ref_name -> {games, foul_rate,
         home_win_pct, pace}``, aggregated over every game we observed.
      3. ``game_team_stats_rows``: one row per (game, team) with pace,
         off_rating, def_rating, pts, opp_pts. Used downstream to
         compute rolling team stats (opp_pace_last5, etc.) per training
         row. Costs nothing extra — same 3 box-score calls per game.

    Replaces the basketball-reference scrape (``scrape_ref_stats``)
    which is data-source-blocked: BBR's /referees/ page is just a
    directory of names with no stats. The NBA-API path is an order of
    magnitude slower (~3 calls × 0.7s × 1230 games ≈ 75 min/season)
    but it's the only source that actually has the data.

    Per game we make 3 calls:
      - BoxScoreSummaryV3 → list of officials + winner
      - BoxScoreTraditionalV3 → team-level foulsPersonal (personal fouls)
      - BoxScoreAdvancedV3 → team-level pace

    For each ref, the season aggregates use simple unweighted means.
    Refs who worked < 3 games in the observed window get filtered out
    of the aggregates dict — too few samples to be useful, and they'd
    pull league averages around. The game_officials rows are kept for
    them anyway so a future re-run with more seasons can re-aggregate.
    """
    from nba_api.stats.endpoints import (
        leaguegamefinder, boxscoreadvancedv3,
    )
    import time as _time

    try:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season',
            league_id_nullable='00',
        ).get_data_frames()[0]
        _time.sleep(sleep_between)
    except Exception as e:
        print(f"compute_game_refs: game finder failed for {season}: {e}")
        return [], {}, []

    # Process oldest-first. Modern V3 endpoints cover games missing from
    # V2; explicit homeTeam identity also avoids guessing frame row order.
    if 'GAME_DATE' in gf.columns:
        gf = gf.sort_values('GAME_DATE', ascending=True)
    distinct_games = list(gf['GAME_ID'].drop_duplicates())
    if max_games:
        distinct_games = distinct_games[:max_games]
    print(f"compute_game_refs: {len(distinct_games)} games for {season}")

    game_rows: list[dict[str, Any]] = []
    # ref_name -> running totals
    accum: dict[str, dict[str, Any]] = {}

    def _get_summary_officials_winner(gid: str):
        from .nba_officials import game_summary, summary_officials
        return summary_officials(game_summary(gid))

    def _get_team_fouls(gid: str) -> float | None:
        from nba_api.stats.endpoints import boxscoretraditionalv3
        frames = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=20).get_data_frames()
        team_totals = frames[1] if len(frames) > 1 else None
        if team_totals is None or team_totals.empty or 'foulsPersonal' not in team_totals.columns:
            return None
        return float(team_totals['foulsPersonal'].sum())

    def _get_pace(gid: str) -> float | None:
        try:
            frames = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid).get_data_frames()
        except Exception:
            return None
        # Frame 1 is the team-level totals row with `pace`
        if len(frames) < 2 or frames[1].empty or "pace" not in frames[1].columns:
            return None
        try:
            return float(frames[1]["pace"].mean())
        except Exception:
            return None

    def _get_team_advanced(gid: str) -> dict[int, dict]:
        """Return {team_id: {pace, off_rating, def_rating}} from frame 1 of
        BoxScoreAdvancedV3 — the team-level totals row(s)."""
        try:
            frames = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid).get_data_frames()
        except Exception:
            return {}
        if len(frames) < 2 or frames[1].empty:
            return {}
        out: dict[int, dict] = {}
        f = frames[1]
        for _, r in f.iterrows():
            tid = r.get("teamId")
            if tid is None:
                continue
            try:
                out[int(tid)] = {
                    "pace": float(r["pace"]) if "pace" in r and pd.notna(r.get("pace")) else None,
                    "off_rating": float(r["offensiveRating"]) if "offensiveRating" in r and pd.notna(r.get("offensiveRating")) else None,
                    "def_rating": float(r["defensiveRating"]) if "defensiveRating" in r and pd.notna(r.get("defensiveRating")) else None,
                }
            except Exception:
                continue
        return out

    now = int(_time.time())
    completed = 0
    game_team_stats_rows: list[dict[str, Any]] = []
    for gid in distinct_games:
        try:
            refs, is_home_winner, game_date, team_ls = _get_summary_officials_winner(gid)
            _time.sleep(sleep_between)
            if not refs:
                continue  # data unavailable for this game
            total_fouls = _get_team_fouls(gid)
            _time.sleep(sleep_between)
            team_adv = _get_team_advanced(gid)
            _time.sleep(sleep_between)

            # Per-team game stats: pace + off/def rating + pts + opp_pts
            # (the latter two from line_score). We have 2 teams per game;
            # opp_pts for team A is team B's pts.
            for tid, pts in team_ls:
                opp_pts = next((p for tid2, p in team_ls if tid2 != tid), None)
                adv = team_adv.get(tid, {})
                game_team_stats_rows.append({
                    "game_id": str(gid),
                    "team_id": tid,
                    "game_date": game_date,
                    "pace":       adv.get("pace"),
                    "off_rating": adv.get("off_rating"),
                    "def_rating": adv.get("def_rating"),
                    "pts":        pts,
                    "opp_pts":    opp_pts,
                    "updated_at": now,
                })

            # League-level pace for the per-ref aggregator (avg of two team paces)
            pace_for_ref = None
            if team_adv:
                paces = [v["pace"] for v in team_adv.values() if v.get("pace") is not None]
                if paces:
                    pace_for_ref = sum(paces) / len(paces)

            for ref in refs:
                game_rows.append({
                    "game_id": str(gid),
                    "ref_name": ref["name"],
                    "ref_id": ref["id"],
                    "updated_at": now,
                })
                bucket = accum.setdefault(ref["name"], {
                    "games": 0,
                    "fouls_sum": 0.0, "fouls_n": 0,
                    "home_wins": 0, "wl_n": 0,
                    "pace_sum": 0.0, "pace_n": 0,
                })
                bucket["games"] += 1
                if total_fouls is not None:
                    bucket["fouls_sum"] += total_fouls
                    bucket["fouls_n"] += 1
                if is_home_winner is not None:
                    bucket["wl_n"] += 1
                    if is_home_winner:
                        bucket["home_wins"] += 1
                if pace_for_ref is not None:
                    bucket["pace_sum"] += pace_for_ref
                    bucket["pace_n"] += 1
        except Exception as e:
            print(f"compute_game_refs: game {gid} failed: {e}")
            _time.sleep(sleep_between)
        finally:
            completed += 1
            if on_progress and completed % progress_every == 0:
                on_progress(completed, len(distinct_games))
            elif completed % progress_every == 0:
                print(f"compute_game_refs: {completed}/{len(distinct_games)} games done")

    # Roll up aggregates. Refs with < 3 games go in game_rows but get
    # filtered out of the per-ref stats — too noisy to be useful.
    ref_aggregates: dict[str, dict[str, Any]] = {}
    for name, b in accum.items():
        if b["games"] < 3:
            continue
        ref_aggregates[name] = {
            "ref_name": name,
            "games": b["games"],
            "foul_rate": float(b["fouls_sum"] / b["fouls_n"]) if b["fouls_n"] else 0.0,
            "home_win_pct": float(b["home_wins"] / b["wl_n"]) if b["wl_n"] else 0.5,
            "pace": float(b["pace_sum"] / b["pace_n"]) if b["pace_n"] else 100.0,
        }

    print(f"compute_game_refs: {len(game_rows)} ref-game rows, "
          f"{len(ref_aggregates)} aggregated refs (≥3 games each), "
          f"{len(game_team_stats_rows)} game-team-stats rows")
    return game_rows, ref_aggregates, game_team_stats_rows


def _migrate_player_table_to_per_season(
    conn: psycopg.Connection,
    table_name: str,
    current_season: str,
) -> None:
    """Promote a per-player snapshot table to per-(player, season).

    Many of our trained features come from tables like
    ``player_advanced_stats`` that store ONE row per player. When we
    train on multi-season data, every season's training rows for the
    same player get the same snapshot value → zero within-player
    variance → zero XGBoost importance.

    This helper adds a ``season`` column and switches the primary key
    to ``(player_id, season)`` by rebuilding via create → copy → drop →
    rename. Idempotent: if ``season`` already exists, no-op.

    Existing rows get tagged with ``current_season`` since that's where
    the data came from. The caller can then re-run the per-season
    compute jobs for older seasons to fill in historical rows.
    """
    cur = conn.cursor()
    info = db.column_info(conn, table_name)
    if not info:
        return  # table doesn't exist yet — nothing to migrate
    cols = [name for name, _typ, _nn, _dflt in info]
    if 'season' in cols:
        return  # already migrated

    # Reconstruct full column DDL so the new table matches the old
    # exactly (preserves types, NOT NULL flags, defaults).
    col_defs = []
    col_names = []
    for name, typ, notnull, dflt in info:
        clause = f'"{name}" {typ}'
        if notnull:
            clause += " NOT NULL"
        if dflt is not None:
            clause += f" DEFAULT {dflt}"
        col_defs.append(clause)
        col_names.append(name)

    # Build the new schema with composite PK
    non_pk_defs = [c for c, n in zip(col_defs, col_names) if n != 'player_id']
    new_schema = f"""
        CREATE TABLE {table_name}_new (
            player_id BIGINT NOT NULL,
            season TEXT NOT NULL,
            {", ".join(non_pk_defs)},
            PRIMARY KEY (player_id, season)
        )
    """
    transfer_cols = ", ".join(f'"{c}"' for c in col_names if c != 'player_id')
    cur.execute(new_schema)
    cur.execute(
        f"INSERT INTO {table_name}_new (player_id, season, {transfer_cols}) "
        f"SELECT player_id, %s, {transfer_cols} FROM {table_name}",
        (current_season,),
    )
    cur.execute(f"DROP TABLE {table_name}")
    cur.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pid ON {table_name}(player_id)"
    )
    # Commit per table so one table's failure doesn't undo the ones already
    # migrated. The caller relies on this: it recovers from a failure with
    # conn.rollback(), which must not reach back past this point.
    conn.commit()
    print(f"[migrate] {table_name}: tagged existing rows with season={current_season}")


# Tables that store one snapshot per player and would benefit from
# per-season storage. Listed here so a single migration call can
# promote all of them in a consistent way.
PER_PLAYER_TABLES_TO_MIGRATE = (
    "player_advanced_stats",
    "player_clutch_stats",
    "player_hustle_stats",
    "player_shot_profile",
    "player_play_types",
    "player_on_off",
    "player_shot_zones",
    "player_quarter_splits",
    "player_tracking_stats",
    "player_scoring_breakdown",
    "player_yoy_stats",
)


def compute_game_broadcasts_for_dates(
    dates: list[str],
    *,
    sleep_between: float = 0.7,
) -> list[dict[str, Any]]:
    """Walk NBA scoreboard day-by-day to collect national-TV broadcaster
    info per game.

    NBA-API's ScoreboardV2 (legacy) and ScoreboardV3 expose
    NATL_TV_BROADCASTER_ABBREVIATION on each game in frame 0. Most days
    return None for that field; nationally-televised games (ESPN, ABC,
    TNT, NBA TV slots) get a populated string.

    Walking by date hits the same endpoint once per day, so a full
    season costs ~180 calls × 0.7s ≈ 2 min per season. Dates that
    return no games (off-days) pass cleanly.

    ``dates`` should be ISO YYYY-MM-DD strings.
    """
    from nba_api.stats.endpoints import scoreboardv2
    import time as _time

    rows: list[dict[str, Any]] = []
    now = int(_time.time())
    for d in dates:
        try:
            frames = scoreboardv2.ScoreboardV2(game_date=d).get_data_frames()
            _time.sleep(sleep_between)
        except Exception as e:
            print(f"compute_game_broadcasts: {d} failed: {e}")
            continue
        if not frames or frames[0].empty:
            continue
        sb = frames[0]

        def _str_or_none(v):
            """Treat pandas NaN, empty string, and the literal 'nan'
            as missing. The default ``str(nan or "") or None`` pattern
            doesn't work because pandas nan is truthy and str(nan) is
            the literal 'nan' string."""
            try:
                import math as _m
                if v is None or (isinstance(v, float) and _m.isnan(v)):
                    return None
            except Exception:
                pass
            s = str(v).strip()
            if not s or s.lower() == 'nan':
                return None
            return s

        for _, r in sb.iterrows():
            gid = r.get("GAME_ID")
            if not gid:
                continue
            rows.append({
                "game_id": str(gid),
                "game_date": d,
                "natl_tv": _str_or_none(r.get("NATL_TV_BROADCASTER_ABBREVIATION")),
                "home_tv": _str_or_none(r.get("HOME_TV_BROADCASTER_ABBREVIATION")),
                "away_tv": _str_or_none(r.get("AWAY_TV_BROADCASTER_ABBREVIATION")),
                "updated_at": now,
            })
    print(f"compute_game_broadcasts: {len(rows)} game-broadcast rows over "
          f"{len(dates)} dates")
    return rows


def upsert_game_team_stats(
    conn: psycopg.Connection,
    rows: list[dict[str, Any]],
) -> None:
    """Persist per-(game, team) pace / off_rating / def_rating / pts.
    The trainer reads from here to derive rolling opp stats per game."""
    if not rows:
        return
    cur = conn.cursor()
    for r in rows:
        try:
            cur.execute(
                """
                INSERT INTO game_team_stats (game_id, team_id, game_date, pace, off_rating, def_rating,
                   pts, opp_pts, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id, team_id) DO UPDATE SET
                    game_date = excluded.game_date,
                    pace = excluded.pace,
                    off_rating = excluded.off_rating,
                    def_rating = excluded.def_rating,
                    pts = excluded.pts,
                    opp_pts = excluded.opp_pts,
                    updated_at = excluded.updated_at
                """,
                (str(r["game_id"]), int(r["team_id"]), r.get("game_date"),
                 _ndf(r.get("pace")), _ndf(r.get("off_rating")),
                 _ndf(r.get("def_rating")),
                 _ndf(r.get("pts")), _ndf(r.get("opp_pts")),
                 int(r.get("updated_at") or 0)),
            )
        except Exception as e:
            print(f"upsert_game_team_stats: row failed: {e}")
    conn.commit()


def _ndf(v):
    """Cast to float, treat NaN/None as NULL via returning None."""
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def upsert_game_broadcasts(
    conn: psycopg.Connection,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    cur = conn.cursor()
    for r in rows:
        try:
            cur.execute(
                """
                INSERT INTO game_broadcasts (game_id, game_date, natl_tv, home_tv, away_tv, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    game_date = excluded.game_date,
                    natl_tv = excluded.natl_tv,
                    home_tv = excluded.home_tv,
                    away_tv = excluded.away_tv,
                    updated_at = excluded.updated_at
                """,
                (str(r["game_id"]), r.get("game_date"),
                 r.get("natl_tv"), r.get("home_tv"), r.get("away_tv"),
                 int(r.get("updated_at") or 0)),
            )
        except Exception as e:
            print(f"upsert_game_broadcasts: row failed: {e}")
    conn.commit()


def upsert_game_officials(
    conn: psycopg.Connection,
    rows: list[dict[str, Any]],
) -> None:
    """Idempotent upsert of (game_id, ref_name, ref_id) rows."""
    if not rows:
        return
    cur = conn.cursor()
    for r in rows:
        try:
            cur.execute(
                """
                INSERT INTO game_officials (game_id, ref_name, ref_id, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (game_id, ref_name) DO UPDATE SET
                    ref_id = excluded.ref_id,
                    updated_at = excluded.updated_at
                """,
                (str(r["game_id"]), str(r["ref_name"]),
                 r.get("ref_id"), int(r.get("updated_at") or 0)),
            )
        except Exception as e:
            print(f"upsert_game_officials: row failed: {e}")
    conn.commit()


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


def upsert_team_stats(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_stats (
                team_id, pts_fb, pts_off_tov,
                opp_fga, opp_fg_pct, opp_fg3a, opp_fg3_pct, opp_tov, opp_stl, opp_blk,
                opp_pts_paint, opp_pts_fb, opp_pts_off_tov,
                opp_def_rating_last5, opp_blk_last5, opp_stl_last5,
                lg_pts_fb, lg_pts_off_tov, lg_fga, lg_fg_pct, lg_fg3a, lg_tov, lg_stl,
                foul_rate_season, foul_rate_last5, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (team_id) DO UPDATE SET
                pts_fb = excluded.pts_fb,
                pts_off_tov = excluded.pts_off_tov,
                opp_fga = excluded.opp_fga,
                opp_fg_pct = excluded.opp_fg_pct,
                opp_fg3a = excluded.opp_fg3a,
                opp_fg3_pct = excluded.opp_fg3_pct,
                opp_tov = excluded.opp_tov,
                opp_stl = excluded.opp_stl,
                opp_blk = excluded.opp_blk,
                opp_pts_paint = excluded.opp_pts_paint,
                opp_pts_fb = excluded.opp_pts_fb,
                opp_pts_off_tov = excluded.opp_pts_off_tov,
                opp_def_rating_last5 = excluded.opp_def_rating_last5,
                opp_blk_last5 = excluded.opp_blk_last5,
                opp_stl_last5 = excluded.opp_stl_last5,
                lg_pts_fb = excluded.lg_pts_fb,
                lg_pts_off_tov = excluded.lg_pts_off_tov,
                lg_fga = excluded.lg_fga,
                lg_fg_pct = excluded.lg_fg_pct,
                lg_fg3a = excluded.lg_fg3a,
                lg_tov = excluded.lg_tov,
                lg_stl = excluded.lg_stl,
                foul_rate_season = excluded.foul_rate_season,
                foul_rate_last5 = excluded.foul_rate_last5,
                updated_at = excluded.updated_at
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


def upsert_ref_stats(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO referee_stats (ref_name, games, foul_rate, home_win_pct, pace, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ref_name) DO UPDATE SET
                games = excluded.games,
                foul_rate = excluded.foul_rate,
                home_win_pct = excluded.home_win_pct,
                pace = excluded.pace,
                updated_at = excluded.updated_at
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


def upsert_team_foul_rates(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_foul_rates (team_id, foul_rate_season, foul_rate_last5, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE SET
                foul_rate_season = excluded.foul_rate_season,
                foul_rate_last5 = excluded.foul_rate_last5,
                updated_at = excluded.updated_at
            """,
            (
                int(r["team_id"]),
                float(r.get("foul_rate_season", 0.0)),
                float(r.get("foul_rate_last5", 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_rolling_dvp(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO dvp_rolling (team_id, "window", pts, reb, ast, fg3m, stl, blk, tov, gp, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, "window") DO UPDATE SET
                pts = excluded.pts,
                reb = excluded.reb,
                ast = excluded.ast,
                fg3m = excluded.fg3m,
                stl = excluded.stl,
                blk = excluded.blk,
                tov = excluded.tov,
                gp = excluded.gp,
                updated_at = excluded.updated_at
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


def _synergy_frame(**kwargs):
    last_error = None
    for mode in ('Totals', 'PerGame'):
        try:
            frame = synergyplaytypes.SynergyPlayTypes(
                **kwargs, per_mode_simple=mode, timeout=15).get_data_frames()[0]
            if frame.empty:
                raise RuntimeError('empty synergy response')
            return frame
        except Exception as exc:
            last_error = exc
    raise RuntimeError('synergy endpoint unavailable in both modes') from last_error


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

    failures = []
    for pt, (pct_key, ppp_key) in play_type_map.items():
        try:
            df = _synergy_frame(
                season=season,
                play_type_nullable=pt,
                player_or_team_abbreviation='P',
                type_grouping_nullable='offensive',
            )
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
            failures.append(pt)
            print(f"compute_synergy_play_types: fetch failed for {pt}: {e}")
            time.sleep(1.0)

    if failures:
        raise RuntimeError(f'incomplete synergy play types: {failures}')

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
    """Select court status explicitly; NBA returns OFF before ON."""
    from concurrent.futures import ThreadPoolExecutor
    from nba_api.stats.endpoints import teamplayeronoffdetails
    from pathlib import Path
    import json

    def fetch_team(team):
        tid = int(team['id'])
        cache = Path(__file__).resolve().parents[1] / 'logs' / 'precompute_cache' / f'onoff_v2_{season}_{tid}.json'
        if cache.exists():
            return json.loads(cache.read_text()), None
        error = None
        for endpoint, extra in ((teamplayeronoffsummary.TeamPlayerOnOffSummary, {}),
                                (teamplayeronoffdetails.TeamPlayerOnOffDetails, {'measure_type_detailed_defense': 'Advanced'})):
            try:
                frames = endpoint(team_id=tid, season=season, timeout=10, **extra).get_data_frames()
                on, off = {}, {}
                for frame in frames:
                    if frame.empty or 'COURT_STATUS' not in frame:
                        continue
                    for _, row in frame.iterrows():
                        pid = int(row.get('VS_PLAYER_ID', row.get('PLAYER_ID', 0)))
                        status = str(row['COURT_STATUS']).strip().lower()
                        if not pid:
                            continue
                        if status.startswith('off'):
                            off[pid] = float(row['NET_RATING'])
                        elif status.startswith('on'):
                            on[pid] = float(row['NET_RATING'])
                if not on or not off:
                    raise RuntimeError('missing explicit on/off frames')
                rows = [{'player_id': pid, 'on_court_net_rating': rating,
                         'off_court_net_rating': off.get(pid, 0.),
                         'on_off_differential': rating - off.get(pid, 0.)}
                        for pid, rating in on.items()]
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rows))
                return rows, None
            except Exception as exc:
                error = type(exc).__name__
        return [], (tid, error)

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        # map retains team order, preserving the existing deterministic
        # last-team fallback for players who changed teams during a season.
        for team_rows, failure in pool.map(fetch_team, teams.get_teams()):
            rows.extend(team_rows)
            if failure:
                failures.append(failure)
    if failures:
        raise RuntimeError(f'incomplete on/off sources: {failures}')
    return rows


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
    """Collect all historical players' zones in one league request.

    Sum makes and attempts across traded-team rows. Corner 3 is already a
    subtotal in this endpoint; do not also add Left/Right Corner 3.
    """
    from nba_api.stats.endpoints import leaguedashplayershotlocations
    from math import isfinite
    frame = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season, per_mode_detailed='Totals', timeout=30,
    ).get_data_frames()[0]
    if frame.empty:
        raise RuntimeError(f'empty shot-zone response for {season}')
    zones = {'rim': 'Restricted Area', 'paint': 'In The Paint (Non-RA)',
             'midrange': 'Mid-Range', 'corner3': 'Corner 3',
             'above_break3': 'Above the Break 3'}
    selected = set(player_ids) if player_ids is not None else None
    totals = {}
    for _, row in frame.iterrows():
        pid = int(row[('', 'PLAYER_ID')])
        if selected is not None and pid not in selected:
            continue
        acc = totals.setdefault(pid, {k: [0.0, 0.0] for k in zones})
        for key, label in zones.items():
            for i, stat in enumerate(('FGM', 'FGA')):
                value = float(row.get((label, stat), 0.0) or 0.0)
                # NBA uses null/NaN for zones with no attempts. Treat
                # those empty counts as zero before summing the player.
                acc[key][i] += value if isfinite(value) else 0.0
    results = []
    for pid, zone in totals.items():
        total = sum(v[1] for v in zone.values())
        if total <= 0:
            continue
        out = {'player_id': pid}
        for key, (fgm, fga) in zone.items():
            out[key + '_fga_pct'] = fga / total
            out[key + '_fg_pct'] = fgm / fga if fga else 0.0
        results.append(out)
    return results


def compute_quarter_splits(season: str, player_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Four league requests replace hundreds of per-player requests."""
    selected = set(player_ids) if player_ids is not None else None
    results = {}
    for period in range(1, 5):
        frame = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, period=period, per_mode_detailed='PerGame', timeout=30,
        ).get_data_frames()[0]
        time.sleep(1.2)
        if frame.empty:
            raise RuntimeError(f'empty quarter {period} response for {season}')
        for _, row in frame.iterrows():
            pid = int(row['PLAYER_ID'])
            if selected is not None and pid not in selected:
                continue
            out = results.setdefault(pid, {'player_id': pid, 'q1_avg': 0.0,
                'q2_avg': 0.0, 'q3_avg': 0.0, 'q4_avg': 0.0, 'q4_min_pg': 0.0})
            out[f'q{period}_avg'] = float(row['PTS'])
            if period == 4:
                out['q4_min_pg'] = float(row['MIN'])
    return list(results.values())


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

    failures = []
    for pt, key in play_type_map.items():
        try:
            df = _synergy_frame(
                season=season,
                play_type_nullable=pt,
                player_or_team_abbreviation='T',
                type_grouping_nullable='defensive',
            )
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
            failures.append(pt)
            print(f"compute_synergy_team_defense: fetch failed for {pt}: {e}")
            time.sleep(1.0)

    if failures:
        raise RuntimeError(f'incomplete synergy play types: {failures}')

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

def upsert_advanced_player_stats(
    conn: psycopg.Connection,
    rows: list[dict[str, Any]],
    updated_at: int,
    season: str | None = None,
) -> None:
    """Persist advanced stats per (player, season).

    ``season`` defaults to the current season for backward compatibility
    with callers that haven't been updated yet. Multi-season backfill
    callers MUST pass ``season=`` explicitly so each row is tagged with
    the correct year — otherwise multi-season runs would just keep
    overwriting the same (player_id, current_season) row.
    """
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_advanced_stats (player_id, season, usg_pct, ts_pct, efg_pct, ast_pct,
               oreb_pct, dreb_pct, reb_pct, pie,
               off_rating, def_rating, pace, net_rating,
               age, height_inches, weight, years_experience, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                usg_pct = excluded.usg_pct,
                ts_pct = excluded.ts_pct,
                efg_pct = excluded.efg_pct,
                ast_pct = excluded.ast_pct,
                oreb_pct = excluded.oreb_pct,
                dreb_pct = excluded.dreb_pct,
                reb_pct = excluded.reb_pct,
                pie = excluded.pie,
                off_rating = excluded.off_rating,
                def_rating = excluded.def_rating,
                pace = excluded.pace,
                net_rating = excluded.net_rating,
                age = excluded.age,
                height_inches = excluded.height_inches,
                weight = excluded.weight,
                years_experience = excluded.years_experience,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
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


def upsert_clutch_stats(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_clutch_stats (player_id, season, clutch_pts_pg, clutch_fg_pct, clutch_fg3_pct,
               clutch_fta_pg, clutch_plus_minus, clutch_min_pg, clutch_games, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                clutch_pts_pg = excluded.clutch_pts_pg,
                clutch_fg_pct = excluded.clutch_fg_pct,
                clutch_fg3_pct = excluded.clutch_fg3_pct,
                clutch_fta_pg = excluded.clutch_fta_pg,
                clutch_plus_minus = excluded.clutch_plus_minus,
                clutch_min_pg = excluded.clutch_min_pg,
                clutch_games = excluded.clutch_games,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('clutch_pts_pg', 0.0)), float(r.get('clutch_fg_pct', 0.45)),
                float(r.get('clutch_fg3_pct', 0.33)), float(r.get('clutch_fta_pg', 0.0)),
                float(r.get('clutch_plus_minus', 0.0)), float(r.get('clutch_min_pg', 0.0)),
                int(r.get('clutch_games', 0)), int(updated_at),
            ),
        )
    conn.commit()


def upsert_hustle_stats(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_hustle_stats (player_id, season, contested_shots_pg, deflections_pg,
               charges_drawn_pg, screen_assists_pg, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                contested_shots_pg = excluded.contested_shots_pg,
                deflections_pg = excluded.deflections_pg,
                charges_drawn_pg = excluded.charges_drawn_pg,
                screen_assists_pg = excluded.screen_assists_pg,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('contested_shots_pg', 3.0)), float(r.get('deflections_pg', 1.0)),
                float(r.get('charges_drawn_pg', 0.1)), float(r.get('screen_assists_pg', 0.5)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_shot_profile(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_shot_profile (player_id, season, open_shot_fg_pct, open_shot_freq,
               tight_shot_fg_pct, tight_shot_freq,
               catch_shoot_fg_pct, catch_shoot_freq,
               pullup_fg_pct, pullup_freq, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                open_shot_fg_pct = excluded.open_shot_fg_pct,
                open_shot_freq = excluded.open_shot_freq,
                tight_shot_fg_pct = excluded.tight_shot_fg_pct,
                tight_shot_freq = excluded.tight_shot_freq,
                catch_shoot_fg_pct = excluded.catch_shoot_fg_pct,
                catch_shoot_freq = excluded.catch_shoot_freq,
                pullup_fg_pct = excluded.pullup_fg_pct,
                pullup_freq = excluded.pullup_freq,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('open_shot_fg_pct', 0.50)), float(r.get('open_shot_freq', 0.30)),
                float(r.get('tight_shot_fg_pct', 0.38)), float(r.get('tight_shot_freq', 0.15)),
                float(r.get('catch_shoot_fg_pct', 0.40)), float(r.get('catch_shoot_freq', 0.25)),
                float(r.get('pullup_fg_pct', 0.40)), float(r.get('pullup_freq', 0.20)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_play_types(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_play_types (player_id, season, iso_poss_pct, iso_ppp,
               pnr_bh_poss_pct, pnr_bh_ppp,
               pnr_roll_poss_pct, pnr_roll_ppp,
               spotup_poss_pct, spotup_ppp,
               transition_poss_pct, transition_ppp,
               postup_poss_pct, cut_poss_pct, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                iso_poss_pct = excluded.iso_poss_pct,
                iso_ppp = excluded.iso_ppp,
                pnr_bh_poss_pct = excluded.pnr_bh_poss_pct,
                pnr_bh_ppp = excluded.pnr_bh_ppp,
                pnr_roll_poss_pct = excluded.pnr_roll_poss_pct,
                pnr_roll_ppp = excluded.pnr_roll_ppp,
                spotup_poss_pct = excluded.spotup_poss_pct,
                spotup_ppp = excluded.spotup_ppp,
                transition_poss_pct = excluded.transition_poss_pct,
                transition_ppp = excluded.transition_ppp,
                postup_poss_pct = excluded.postup_poss_pct,
                cut_poss_pct = excluded.cut_poss_pct,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
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


def upsert_on_off(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_on_off (player_id, season, on_court_net_rating, off_court_net_rating,
               on_off_differential, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                on_court_net_rating = excluded.on_court_net_rating,
                off_court_net_rating = excluded.off_court_net_rating,
                on_off_differential = excluded.on_off_differential,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('on_court_net_rating', 0.0)),
                float(r.get('off_court_net_rating', 0.0)),
                float(r.get('on_off_differential', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_shot_zones(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_shot_zones (player_id, season, rim_fga_pct, rim_fg_pct,
               paint_fga_pct, paint_fg_pct,
               midrange_fga_pct, midrange_fg_pct,
               corner3_fga_pct, corner3_fg_pct,
               above_break3_fga_pct, above_break3_fg_pct, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                rim_fga_pct = excluded.rim_fga_pct,
                rim_fg_pct = excluded.rim_fg_pct,
                paint_fga_pct = excluded.paint_fga_pct,
                paint_fg_pct = excluded.paint_fg_pct,
                midrange_fga_pct = excluded.midrange_fga_pct,
                midrange_fg_pct = excluded.midrange_fg_pct,
                corner3_fga_pct = excluded.corner3_fga_pct,
                corner3_fg_pct = excluded.corner3_fg_pct,
                above_break3_fga_pct = excluded.above_break3_fga_pct,
                above_break3_fg_pct = excluded.above_break3_fg_pct,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('rim_fga_pct', 0.25)), float(r.get('rim_fg_pct', 0.62)),
                float(r.get('paint_fga_pct', 0.30)), float(r.get('paint_fg_pct', 0.55)),
                float(r.get('midrange_fga_pct', 0.20)), float(r.get('midrange_fg_pct', 0.42)),
                float(r.get('corner3_fga_pct', 0.10)), float(r.get('corner3_fg_pct', 0.38)),
                float(r.get('above_break3_fga_pct', 0.25)), float(r.get('above_break3_fg_pct', 0.35)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_quarter_splits(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_quarter_splits (player_id, season, q1_avg, q2_avg, q3_avg, q4_avg, q4_min_pg, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                q1_avg = excluded.q1_avg,
                q2_avg = excluded.q2_avg,
                q3_avg = excluded.q3_avg,
                q4_avg = excluded.q4_avg,
                q4_min_pg = excluded.q4_min_pg,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('q1_avg', 0.0)), float(r.get('q2_avg', 0.0)),
                float(r.get('q3_avg', 0.0)), float(r.get('q4_avg', 0.0)),
                float(r.get('q4_min_pg', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def upsert_opp_shot_zones(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_opp_shot_zones (team_id, rim_fg_pct_allowed, paint_fg_pct_allowed,
               midrange_fg_pct_allowed, corner3_fg_pct_allowed,
               above_break3_fg_pct_allowed, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE SET
                rim_fg_pct_allowed = excluded.rim_fg_pct_allowed,
                paint_fg_pct_allowed = excluded.paint_fg_pct_allowed,
                midrange_fg_pct_allowed = excluded.midrange_fg_pct_allowed,
                corner3_fg_pct_allowed = excluded.corner3_fg_pct_allowed,
                above_break3_fg_pct_allowed = excluded.above_break3_fg_pct_allowed,
                updated_at = excluded.updated_at
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


def upsert_synergy_defense(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_synergy_defense (team_id, pnr_ppp_allowed, iso_ppp_allowed,
               spotup_ppp_allowed, transition_ppp_allowed,
               postup_ppp_allowed, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE SET
                pnr_ppp_allowed = excluded.pnr_ppp_allowed,
                iso_ppp_allowed = excluded.iso_ppp_allowed,
                spotup_ppp_allowed = excluded.spotup_ppp_allowed,
                transition_ppp_allowed = excluded.transition_ppp_allowed,
                postup_ppp_allowed = excluded.postup_ppp_allowed,
                updated_at = excluded.updated_at
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


def upsert_player_tracking(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_tracking_stats (player_id, season, avg_speed, avg_speed_off, avg_speed_def,
               dist_miles, dist_miles_off, dist_miles_def,
               touches_pg, time_of_poss_pg, avg_drib_per_touch,
               paint_touches_pg, elbow_touches_pg,
               passes_made_pg, potential_ast_pg, secondary_ast_pg, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                avg_speed = excluded.avg_speed,
                avg_speed_off = excluded.avg_speed_off,
                avg_speed_def = excluded.avg_speed_def,
                dist_miles = excluded.dist_miles,
                dist_miles_off = excluded.dist_miles_off,
                dist_miles_def = excluded.dist_miles_def,
                touches_pg = excluded.touches_pg,
                time_of_poss_pg = excluded.time_of_poss_pg,
                avg_drib_per_touch = excluded.avg_drib_per_touch,
                paint_touches_pg = excluded.paint_touches_pg,
                elbow_touches_pg = excluded.elbow_touches_pg,
                passes_made_pg = excluded.passes_made_pg,
                potential_ast_pg = excluded.potential_ast_pg,
                secondary_ast_pg = excluded.secondary_ast_pg,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
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


def upsert_team_standings(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_standings (team_id, win_pct, wins, losses, conf_rank, games_back,
               home_win_pct, road_win_pct, current_streak, l10_wins,
               pts_pg, opp_pts_pg, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE SET
                win_pct = excluded.win_pct,
                wins = excluded.wins,
                losses = excluded.losses,
                conf_rank = excluded.conf_rank,
                games_back = excluded.games_back,
                home_win_pct = excluded.home_win_pct,
                road_win_pct = excluded.road_win_pct,
                current_streak = excluded.current_streak,
                l10_wins = excluded.l10_wins,
                pts_pg = excluded.pts_pg,
                opp_pts_pg = excluded.opp_pts_pg,
                updated_at = excluded.updated_at
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


def upsert_player_scoring_breakdown(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_scoring_breakdown (player_id, season, pct_pts_3pt, pct_pts_paint, pct_pts_ft,
               pct_pts_midrange, pct_uast_fgm, pct_ast_fgm, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                pct_pts_3pt = excluded.pct_pts_3pt,
                pct_pts_paint = excluded.pct_pts_paint,
                pct_pts_ft = excluded.pct_pts_ft,
                pct_pts_midrange = excluded.pct_pts_midrange,
                pct_uast_fgm = excluded.pct_uast_fgm,
                pct_ast_fgm = excluded.pct_ast_fgm,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('pct_pts_3pt', 0.0)), float(r.get('pct_pts_paint', 0.0)),
                float(r.get('pct_pts_ft', 0.0)), float(r.get('pct_pts_midrange', 0.0)),
                float(r.get('pct_uast_fgm', 0.0)), float(r.get('pct_ast_fgm', 0.0)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_player_vs_opponent(season: str = '2024-25', max_players: int = 400) -> list[dict[str, Any]]:
    """Fetch each player's historical splits vs each opponent team via game log aggregation."""
    from nba_api.stats.endpoints import leaguegamelog
    frame = leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation='P', timeout=60,
    ).get_data_frames()[0]
    if frame.empty:
        raise RuntimeError(f'empty player game log for {season}')
    abbrev_to_id = {t['abbreviation']: t['id'] for t in teams.get_teams()}
    frame = frame.copy()
    frame['OPP_ABBREV'] = frame['MATCHUP'].str.split().str[-1]
    results = []
    for (pid, opp), group in frame.groupby(['PLAYER_ID', 'OPP_ABBREV']):
        opp_id = abbrev_to_id.get(opp)
        if not opp_id:
            continue
        fga, fta, pts = (float(group[c].mean()) for c in ('FGA', 'FTA', 'PTS'))
        denom = 2 * (fga + .44 * fta)
        results.append({'player_id': int(pid), 'opponent_team_id': opp_id,
            'gp': len(group), 'avg_stat_pts': pts,
            'fg_pct': float(group['FG_PCT'].mean()),
            'ts_pct': pts / denom if denom > 0 else .55,
            'avg_min': float(group['MIN'].mean())})

    return results


def upsert_player_vs_opponent(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_vs_opponent (player_id, opponent_team_id, gp, avg_stat_pts, fg_pct, ts_pct, avg_min)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, opponent_team_id) DO UPDATE SET
                gp = excluded.gp,
                avg_stat_pts = excluded.avg_stat_pts,
                fg_pct = excluded.fg_pct,
                ts_pct = excluded.ts_pct,
                avg_min = excluded.avg_min
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


def compute_team_rest_splits(season: str = '2024-25', team_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Fetch how each team performs on B2B vs rested (for opponent context)."""
    from nba_api.stats.endpoints import teamdashboardbygeneralsplits

    all_teams = teams.get_teams()
    results: list[dict[str, Any]] = []

    def _sf(val, default=0.0):
        try:
            return float(val) if val is not None and str(val) not in ('nan', '') else default
        except Exception:
            return default

    def _find_rest_frame(frames_list):
        for frame in frames_list:
            if frame.empty:
                continue
            if 'GROUP_SET' in frame.columns:
                gsets = [str(v).upper() for v in frame['GROUP_SET'].unique()]
                if any('REST' in g for g in gsets):
                    return frame
            if 'GROUP_VALUE' in frame.columns:
                gvals = [str(v).upper() for v in frame['GROUP_VALUE'].unique()]
                if any('REST' in g for g in gvals):
                    return frame
        return None

    def _fetch_team(t):
        tid = int(t['id'])
        if team_ids is not None and tid not in team_ids:
            return None
        try:
            # The Base measure type has PTS / FGA / etc. but lacks the
            # rate stats we actually want here (DEF_RATING, PACE). Pull
            # Advanced too and merge on GROUP_VALUE so b2b_def_rating /
            # b2b_pace / rested_def_rating / rested_pace land as real
            # values rather than the placeholder defaults that gave every
            # team identical features and zero training importance.
            frames = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=tid,
                season=season,
                per_mode_detailed='PerGame', timeout=10,
            ).get_data_frames()
            time.sleep(1.2)
            adv_frames = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=tid,
                season=season,
                per_mode_detailed='PerGame', timeout=10,
                measure_type_detailed_defense='Advanced',
            ).get_data_frames()
            time.sleep(1.2)

            rest_df = _find_rest_frame(frames)
            adv_rest_df = _find_rest_frame(adv_frames)

            # Merge advanced columns onto base on GROUP_VALUE so both PTS
            # (for b2b_pts_allowed) and DEF_RATING/PACE are accessible.
            if rest_df is not None and adv_rest_df is not None and not adv_rest_df.empty:
                if 'GROUP_VALUE' in rest_df.columns and 'GROUP_VALUE' in adv_rest_df.columns:
                    keep_cols = ['GROUP_VALUE'] + [
                        c for c in adv_rest_df.columns
                        if c not in rest_df.columns and c in (
                            'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE',
                        )
                    ]
                    rest_df = rest_df.merge(
                        adv_rest_df[keep_cols], on='GROUP_VALUE', how='left',
                    )

            if rest_df is None or rest_df.empty:
                return None

            # filter for rest-day rows. The NBA API's actual GROUP_VALUE
            # strings are "0 Days Rest", "1 Days Rest", ..., "6+ Days Rest"
            # — the previous regex looked for "0 REST"/"1 REST" without
            # the intervening "DAYS" word and never matched anything,
            # which is why every team's rest-split features defaulted to
            # the same placeholder values (112/100/115/110/100) and had
            # zero importance in trained models. Match by the leading
            # digit instead so we're robust to phrasing changes.
            gv_col = 'GROUP_VALUE' if 'GROUP_VALUE' in rest_df.columns else rest_df.columns[1]
            _gv_upper = rest_df[gv_col].astype(str).str.upper()
            b2b_rows = rest_df[_gv_upper.str.contains(
                r'^(?:0|1)\s*DAYS?\s*REST|BACK\s*TO\s*BACK|^B2B', na=False, regex=True,
            )]
            rested_rows = rest_df[_gv_upper.str.contains(
                r'^(?:[2-9]|1[0-9])\s*\+?\s*DAYS?\s*REST', na=False, regex=True,
            )]

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

            return {
                'team_id': tid,
                'b2b_def_rating': b2b_def,
                'b2b_pace': b2b_pace,
                'b2b_pts_allowed': b2b_pts,
                'rested_def_rating': rest_def,
                'rested_pace': rest_pace,
            }
        except Exception as e:
            print(f"compute_team_rest_splits: team {tid} failed: {e}")
            time.sleep(1.2)

    from concurrent.futures import ThreadPoolExecutor
    requested = [t for t in all_teams if team_ids is None or t['id'] in team_ids]
    # Bound concurrency and request timeouts so one unavailable team's
    # dashboard cannot hold up every other team's independent source.
    with ThreadPoolExecutor(max_workers=3) as pool:
        return [row for row in pool.map(_fetch_team, requested) if row is not None]


def upsert_team_rest_splits(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_rest_splits (team_id, b2b_def_rating, b2b_pace, b2b_pts_allowed,
               rested_def_rating, rested_pace, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE SET
                b2b_def_rating = excluded.b2b_def_rating,
                b2b_pace = excluded.b2b_pace,
                b2b_pts_allowed = excluded.b2b_pts_allowed,
                rested_def_rating = excluded.rested_def_rating,
                rested_pace = excluded.rested_pace,
                updated_at = excluded.updated_at
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


from functools import lru_cache


@lru_cache(maxsize=40)
def _season_player_averages(season: str):
    """Immutable historical averages, cached on disk across crawl restarts."""
    from pathlib import Path
    import json
    cache = Path(__file__).resolve().parents[1] / 'logs' / 'precompute_cache' / f'player_averages_{season}.json'
    if cache.exists():
        rows = json.loads(cache.read_text())
        return {int(row['PLAYER_ID']): row for row in rows}
    print(f"[yoy] collecting historical player averages {season}", flush=True)
    frame = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed='PerGame', timeout=20,
    ).get_data_frames()[0]
    time.sleep(1.2)
    if frame.empty:
        raise RuntimeError(f'empty player averages for {season}')
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(frame.to_json(orient='records'))
    return {int(row['PLAYER_ID']): row.to_dict() for _, row in frame.iterrows()}


@lru_cache(maxsize=40)
def _season_player_members(season: str):
    """The bio endpoint supplies exact membership if the base endpoint fails."""
    try:
        return frozenset(_season_player_averages(season))
    except Exception:
        print(f'[yoy] using bio membership for {season}', flush=True)
        frame = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=season, per_mode_simple='PerGame', timeout=20,
        ).get_data_frames()[0]
        if frame.empty:
            raise RuntimeError(f'empty historical membership for {season}')
        return frozenset(int(row['PLAYER_ID']) for _, row in frame.iterrows() if float(row.get('GP', 0)) > 0)


def compute_player_yoy(season: str = '2024-25', max_players: int = 400) -> list[dict[str, Any]]:
    """Compute season-specific YoY values without looking into later seasons.

    Count actual seasons played from league snapshots. The former career
    endpoint always picked the latest two seasons, even for historical jobs.
    """
    year = int(season[:4])
    current = _season_player_averages(season)
    previous = _season_player_averages(f'{year-1}-{str(year)[2:]}')
    try:
        bio = commonallplayers.CommonAllPlayers(season=season, is_only_current_season=0, timeout=10).get_data_frames()[0]
        first_years = [int(row['FROM_YEAR']) for _, row in bio.iterrows()
                       if int(row['PERSON_ID']) in current]
        if not first_years:
            raise RuntimeError('missing debut years')
        earliest = min(first_years)
    except Exception:
        # For the 2022+ rebuild cohort the full NBA stats era covers every
        # career. Fetch that entire era rather than guessing career lengths.
        if year < 2022:
            raise
        earliest = 1996
        print('[yoy] debut endpoint unavailable; scanning full stats era', flush=True)
    counts = {pid: 0 for pid in current}
    # League snapshots go back to 1996; every player in the rebuild's
    # 2022-26 seasons debuted after that.
    if earliest < 1996:
        raise ValueError('career season count requires pre-1996 source data')
    for y in range(earliest, year+1):
        members = _season_player_members(f'{y}-{str(y+1)[2:]}')
        for pid in counts.keys() & members:
            counts[pid] += 1
    def ts(row):
        denom = 2 * (float(row.get('FGA', 0)) + .44 * float(row.get('FTA', 0)))
        return float(row.get('PTS', 0)) / denom if denom > 0 else .55
    results = []
    for pid, row in current.items():
        prior = previous.get(pid)
        # If the immediately previous season was missed, use the latest
        # season actually played, matching the career endpoint's intention.
        if prior is None and counts[pid] > 1:
            for y in range(year-2, earliest-1, -1):
                prior = _season_player_averages(f'{y}-{str(y+1)[2:]}').get(pid)
                if prior is not None:
                    break
        results.append({'player_id': pid,
            'yoy_pts_change': float(row['PTS']) - float(prior['PTS']) if prior else 0.,
            'yoy_ts_change': ts(row) - ts(prior) if prior else 0.,
            'yoy_usage_change': (float(row['FGA']) - float(prior['FGA'])) / max(float(prior['FGA']), 1.) if prior else 0.,
            'seasons_in_league': counts[pid]})
    return results


def upsert_player_yoy(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int, *, season: str | None = None) -> None:
    eff_season = season or compute_current_season()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO player_yoy_stats (player_id, season, yoy_pts_change, yoy_ts_change, yoy_usage_change,
               seasons_in_league, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season) DO UPDATE SET
                yoy_pts_change = excluded.yoy_pts_change,
                yoy_ts_change = excluded.yoy_ts_change,
                yoy_usage_change = excluded.yoy_usage_change,
                seasons_in_league = excluded.seasons_in_league,
                updated_at = excluded.updated_at
            """,
            (
                int(r['player_id']), eff_season,
                float(r.get('yoy_pts_change', 0.0)),
                float(r.get('yoy_ts_change', 0.0)),
                float(r.get('yoy_usage_change', 0.0)),
                int(r.get('seasons_in_league', 1)),
                int(updated_at),
            ),
        )
    conn.commit()


def compute_team_home_away_splits(season: str = '2024-25') -> list[dict[str, Any]]:
    """Use two league location splits rather than thirty team dashboards."""
    by_location = {}
    for location in ('Home', 'Road'):
        frame = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense='Advanced',
            location_nullable=location, per_mode_detailed='PerGame', timeout=30,
        ).get_data_frames()[0]
        time.sleep(1.2)
        by_location[location] = {int(row['TEAM_ID']): float(row['DEF_RATING']) for _, row in frame.iterrows()}
    home, away = by_location['Home'], by_location['Road']
    if len(home.keys() & away.keys()) != 30:
        raise RuntimeError(f'incomplete home/away splits for {season}')
    return [{'team_id': tid, 'home_def_rating': home[tid], 'away_def_rating': away[tid],
             'home_away_def_split': home[tid] - away[tid]} for tid in sorted(home.keys() & away.keys())]


def upsert_team_home_away_splits(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_home_away_splits (team_id, home_def_rating, away_def_rating, home_away_def_split, updated_at)
            VALUES (%(team_id)s, %(home_def_rating)s, %(away_def_rating)s, %(home_away_def_split)s, %(updated_at)s)
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


def upsert_lineup_stats(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_lineup_stats
                (team_id, top_lineup_net_rating, bench_net_rating, bench_strength, lineup_continuity, lineups_played_count, updated_at)
            VALUES
                (%(team_id)s, %(top_lineup_net_rating)s, %(bench_net_rating)s, %(bench_strength)s, %(lineup_continuity)s, %(lineups_played_count)s, %(updated_at)s)
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
            raise RuntimeError(f'Injury coverage incomplete for team {tid}') from _e
    return rows


def upsert_injury_status(conn: psycopg.Connection, rows: list[dict[str, Any]], updated_at: int) -> None:
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO team_injury_status (team_id, key_players_out, total_players_out, total_impact, updated_at)
            VALUES (%(team_id)s, %(key_players_out)s, %(total_players_out)s, %(total_impact)s, %(updated_at)s)
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

    conn = db.connect(db_path)
    ensure_tables(conn)

    dvp = scrape_dvp_by_position()
    upsert_dvp(conn, dvp, updated_at=updated_at)

    defenders = compute_special_defenders(season=season)
    upsert_defenders(conn, defenders, updated_at=updated_at)

    ref_rows = compute_referee_stats_from_history(db_path=db_path)
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
        upsert_advanced_player_stats(conn, adv_rows, updated_at=updated_at, season=season)
        print(f"  advanced player stats: {len(adv_rows)} rows")
    except Exception as _e:
        print(f"advanced player stats failed: {_e}")

    clutch_rows: list[dict[str, Any]] = []
    try:
        print("computing clutch stats...")
        clutch_rows = compute_clutch_stats(season=season)
        upsert_clutch_stats(conn, clutch_rows, updated_at=updated_at, season=season)
        print(f"  clutch stats: {len(clutch_rows)} rows")
    except Exception as _e:
        print(f"clutch stats failed: {_e}")

    hustle_rows: list[dict[str, Any]] = []
    try:
        print("computing hustle stats...")
        hustle_rows = compute_hustle_stats(season=season)
        upsert_hustle_stats(conn, hustle_rows, updated_at=updated_at, season=season)
        print(f"  hustle stats: {len(hustle_rows)} rows")
    except Exception as _e:
        print(f"hustle stats failed: {_e}")

    shot_profile_rows: list[dict[str, Any]] = []
    try:
        print("computing shot profile...")
        shot_profile_rows = compute_shot_profile(season=season)
        upsert_shot_profile(conn, shot_profile_rows, updated_at=updated_at, season=season)
        print(f"  shot profile: {len(shot_profile_rows)} rows")
    except Exception as _e:
        print(f"shot profile failed: {_e}")

    play_type_rows: list[dict[str, Any]] = []
    try:
        print("computing synergy play types...")
        play_type_rows = compute_synergy_play_types(season=season)
        upsert_play_types(conn, play_type_rows, updated_at=updated_at, season=season)
        print(f"  play types: {len(play_type_rows)} rows")
    except Exception as _e:
        print(f"synergy play types failed: {_e}")

    on_off_rows: list[dict[str, Any]] = []
    try:
        print("computing on/off ratings...")
        on_off_rows = compute_on_off_ratings(season=season)
        upsert_on_off(conn, on_off_rows, updated_at=updated_at, season=season)
        print(f"  on/off ratings: {len(on_off_rows)} rows")
    except Exception as _e:
        print(f"on/off ratings failed: {_e}")

    shot_zone_rows: list[dict[str, Any]] = []
    try:
        print("computing shot zone breakdown (up to 400 players)...")
        shot_zone_rows = compute_shot_zone_breakdown(season=season)
        upsert_shot_zones(conn, shot_zone_rows, updated_at=updated_at, season=season)
        print(f"  shot zones: {len(shot_zone_rows)} rows")
    except Exception as _e:
        print(f"shot zone breakdown failed: {_e}")

    quarter_split_rows: list[dict[str, Any]] = []
    try:
        print("computing quarter splits (up to 400 players)...")
        quarter_split_rows = compute_quarter_splits(season=season)
        upsert_quarter_splits(conn, quarter_split_rows, updated_at=updated_at, season=season)
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
        upsert_player_tracking(conn, tracking_rows, updated_at=updated_at, season=season)
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
        upsert_player_scoring_breakdown(conn, scoring_breakdown_rows, updated_at=updated_at, season=season)
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
        upsert_player_yoy(conn, player_yoy_rows, updated_at=updated_at, season=season)
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


