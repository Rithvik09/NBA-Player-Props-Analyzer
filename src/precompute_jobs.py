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

from nba_api.stats.endpoints import CommonTeamRoster, leaguedashplayerstats
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

    conn.close()
    return {"season": season, "updated_at": updated_at, "dvp_rows": len(dvp), "defender_rows": len(defenders)}


