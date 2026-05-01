"""
Odds API integration for tracking NBA player prop line movements.

Polls The Odds API for player prop lines, stores snapshots over time,
computes line movement features, and surfaces sharp action signals.
"""

import sqlite3
import logging
import time
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_implied(price: float) -> float:
    """Convert American odds to implied probability (0-1)."""
    if price > 0:
        return 100.0 / (price + 100.0)
    elif price < 0:
        return abs(price) / (abs(price) + 100.0)
    return 0.5


# ---------------------------------------------------------------------------
# OddsTracker
# ---------------------------------------------------------------------------

class OddsTracker:
    """Track NBA player prop line movements via The Odds API."""

    BASE_URL = "https://api.the-odds-api.com"
    SPORT = "basketball_nba"

    # Maps Odds API market keys → internal prop type names
    PROP_TYPE_MAP = {
        "player_points":                    "points",
        "player_rebounds":                   "rebounds",
        "player_assists":                    "assists",
        "player_threes":                     "three_pointers",
        "player_steals":                     "steals",
        "player_blocks":                     "blocks",
        "player_turnovers":                  "turnovers",
        "player_points_rebounds_assists":     "pts_ast_reb",
        "player_points_rebounds":             "pts_reb",
        "player_points_assists":             "pts_ast",
        "player_rebounds_assists":            "ast_reb",
        "player_double_double":              "double_double",
    }

    MARKETS = ",".join(PROP_TYPE_MAP.keys())

    def __init__(self, api_key: str | None = None, db_path: str = "basketball_data.db"):
        # ``api_key`` is only used for the live HTTP fetches; B3 callers
        # (auto_grade_pending → record_outcome) want the DB methods only
        # and shouldn't need to pass a fake key just to satisfy the ctor.
        self.api_key = api_key
        self.db_path = db_path
        self._init_db()

    # ---- DB setup ---------------------------------------------------------

    def _init_db(self):
        """Create line-tracking tables if they don't exist.

        Idempotent ALTER TABLE adds the closing-line columns + the
        prop_outcomes table introduced for CLV-as-training-signal (B3).
        Safe to run on existing DBs — skips columns/tables that exist.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS prop_line_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                player_id INTEGER,
                player_name TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                line REAL NOT NULL,
                over_price REAL,
                under_price REAL,
                snapshot_time TEXT NOT NULL,
                UNIQUE(game_id, player_name, prop_type, bookmaker, snapshot_time)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS prop_line_summary (
                game_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                player_id INTEGER,
                prop_type TEXT NOT NULL,
                opening_line REAL,
                current_line REAL,
                line_movement REAL,
                opening_over_price REAL,
                current_over_price REAL,
                price_movement REAL,
                num_snapshots INTEGER DEFAULT 0,
                first_seen TEXT,
                last_updated TEXT,
                consensus_line REAL,
                line_std REAL,
                sharp_action_score REAL DEFAULT 0.0,
                PRIMARY KEY(game_id, player_name, prop_type)
            )
        """)

        # B3 — closing-line capture for CLV. We add `closing_line` (the
        # consensus line right before tip) and `closing_line_at` (the
        # snapshot time we used) to prop_line_summary. Idempotent: skip
        # the ADD COLUMN if it's already there.
        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(prop_line_summary)")}
        for col, typ in (
            ("closing_line", "REAL"),
            ("closing_line_at", "TEXT"),
        ):
            if col not in existing_cols:
                c.execute(f"ALTER TABLE prop_line_summary ADD COLUMN {col} {typ}")

        # B3 — prop_outcomes is the join table that turns logged lines +
        # post-game results into CLV training data. One row per
        # (game, player, prop_type) once the game has finished and we've
        # observed both the closing line and the actual stat result.
        c.execute("""
            CREATE TABLE IF NOT EXISTS prop_outcomes (
                game_id TEXT NOT NULL,
                player_id INTEGER,
                player_name TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                observed_line REAL,        -- the line we made our prediction against
                closing_line REAL,         -- the line right before tipoff
                actual_result REAL,        -- the player's realised stat
                settled_at TEXT,           -- when we recorded the outcome
                PRIMARY KEY(game_id, player_name, prop_type)
            )
        """)

        conn.commit()
        conn.close()
        log.info("[odds] DB tables initialized")

    # ---- API calls --------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict | list | None:
        """Make a GET request and log quota usage."""
        params = params or {}
        params["apiKey"] = self.api_key
        resp = None
        try:
            resp = requests.get(url, params=params, timeout=30)
            used = resp.headers.get("x-requests-used", "?")
            remaining = resp.headers.get("x-requests-remaining", "?")
            log.info("[odds] API quota: %s used, %s remaining", used, remaining)
            # telemetry for /odds/status
            try:
                _LAST_POLL["quota_remaining"] = int(remaining) if remaining != "?" else None
            except ValueError:
                _LAST_POLL["quota_remaining"] = None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = resp.status_code if resp is not None else None
            if status == 429:
                log.warning("[odds] Rate limited — backing off")
            elif status == 401:
                log.error("[odds] Invalid API key — renew ODDS_API_KEY")
                _LAST_POLL["error"] = "401 unauthorized — key invalid or expired"
            else:
                log.error("[odds] HTTP error: %s", e)
            return None
        except requests.exceptions.RequestException as e:
            log.error("[odds] Request failed: %s", e)
            return None

    def fetch_upcoming_games(self) -> list[dict]:
        """Fetch upcoming NBA games with basic odds."""
        url = f"{self.BASE_URL}/v4/sports/{self.SPORT}/odds/"
        data = self._get(url, {
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
        })
        if not data:
            return []
        games = []
        for event in data:
            games.append({
                "id": event["id"],
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                "commence_time": event.get("commence_time", ""),
            })
        log.info(f"[odds] Found {len(games)} upcoming games")
        print(f"[odds] Found {len(games)} upcoming games")
        return games

    def fetch_player_props(self, event_id: str) -> list[dict]:
        """Fetch all player prop lines for a single game event."""
        url = f"{self.BASE_URL}/v4/sports/{self.SPORT}/events/{event_id}/odds"
        data = self._get(url, {
            "regions": "us",
            "markets": self.MARKETS,
            "oddsFormat": "american",
        })
        if not data:
            return []

        props = []
        bookmakers = data.get("bookmakers", [])
        for bk in bookmakers:
            bk_name = bk.get("key", bk.get("title", "unknown"))
            for market in bk.get("markets", []):
                market_key = market.get("key", "")
                prop_type = self.PROP_TYPE_MAP.get(market_key)
                if not prop_type:
                    continue

                # Group outcomes by player name to pair Over/Under
                player_outcomes: dict[str, dict] = {}
                for outcome in market.get("outcomes", []):
                    name = outcome.get("description", outcome.get("name", ""))
                    if not name:
                        continue
                    side = outcome.get("name", "").lower()  # "Over" or "Under"
                    point = outcome.get("point", 0.0)
                    price = outcome.get("price", 0)

                    if name not in player_outcomes:
                        player_outcomes[name] = {"line": point}
                    if "over" in side:
                        player_outcomes[name]["over_price"] = float(price)
                        player_outcomes[name]["line"] = float(point)
                    elif "under" in side:
                        player_outcomes[name]["under_price"] = float(price)

                for player_name, info in player_outcomes.items():
                    props.append({
                        "player_name": player_name,
                        "prop_type": prop_type,
                        "bookmaker": bk_name,
                        "line": info.get("line", 0.0),
                        "over_price": info.get("over_price", -110),
                        "under_price": info.get("under_price", -110),
                    })
        return props

    # ---- Snapshot + storage -----------------------------------------------

    # Tipoff window inside which a snapshot is treated as the "closing
    # line." 20 min is a balance between (a) waiting long enough for
    # late-breaking news to bake into the line and (b) running far
    # enough ahead of tip that the polling job doesn't race the game.
    CLOSING_WINDOW_MINUTES = 20

    def snapshot_all_games(self) -> int:
        """Poll all upcoming games and store line snapshots. Returns total lines stored."""
        games = self.fetch_upcoming_games()
        if not games:
            print("[odds] No upcoming games found")
            return 0

        total = 0
        closes_captured = 0
        for i, game in enumerate(games, 1):
            event_id = game["id"]
            commence_time = game.get("commence_time", "")
            props = self.fetch_player_props(event_id)
            if props:
                self._store_snapshot(event_id, props)
                # Update summaries
                seen = set()
                for p in props:
                    key = (event_id, p["player_name"], p["prop_type"])
                    if key not in seen:
                        self._update_summary(*key)
                        seen.add(key)
                total += len(props)

                # B3: if this game's tipoff is inside the closing window,
                # stamp the closing line on every (player, prop) we just
                # snapshotted. We do this *here* (rather than in a
                # separate cron) so the closing line and the snapshot it
                # came from are written atomically — no race where the
                # tipoff passes between snapshot and capture.
                if commence_time and self._inside_closing_window(commence_time):
                    try:
                        n = self.capture_closing_lines(
                            event_id, commence_time,
                            window_minutes=self.CLOSING_WINDOW_MINUTES,
                        )
                        closes_captured += n
                    except Exception as e:  # noqa: BLE001
                        log.warning("[odds] capture_closing_lines failed for %s: %s", event_id, e)

            print(f"[odds] Game {i}/{len(games)}: {game['away_team']} @ {game['home_team']} — {len(props)} lines")
            # Be polite to the API
            if i < len(games):
                time.sleep(1)

        # Surface to /odds/status so we can see the CLV pipeline working
        # without poking at SQLite directly.
        _LAST_POLL["closes_captured"] = int(closes_captured)

        print(f"[odds] Snapshotted {total} lines across {len(games)} games "
              f"(closes_captured={closes_captured})")
        log.info(
            "[odds] Snapshotted %d lines across %d games (closes_captured=%d)",
            total, len(games), closes_captured,
        )
        return total

    @staticmethod
    def _inside_closing_window(commence_time: str) -> bool:
        """Is ``commence_time`` (Odds-API ISO-with-Z) within
        ``CLOSING_WINDOW_MINUTES`` of *now*?

        A pure clock check — independent of any DB state. Returns False
        on parse failure so that a malformed timestamp can't cause us
        to spuriously stamp closing lines on every game.
        """
        try:
            # ``fromisoformat`` accepts ``2026-04-30T19:00:00+00:00`` but
            # not the trailing-Z form, so we swap manually.
            ts = commence_time.rstrip("Z")
            tipoff = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        except Exception:
            return False
        now = datetime.now(timezone.utc)
        delta = (tipoff - now).total_seconds() / 60.0
        # Capture if tipoff is in (-window, +window) — i.e. within the
        # window before tip OR up to the window after (in case the
        # poll lands a minute or two late).
        return -OddsTracker.CLOSING_WINDOW_MINUTES <= delta <= OddsTracker.CLOSING_WINDOW_MINUTES

    def _store_snapshot(self, game_id: str, props: list[dict]):
        """Insert line snapshots into prop_line_history."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for p in props:
            try:
                c.execute("""
                    INSERT OR IGNORE INTO prop_line_history
                    (game_id, player_name, prop_type, bookmaker, line, over_price, under_price, snapshot_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    p["player_name"],
                    p["prop_type"],
                    p["bookmaker"],
                    p["line"],
                    p.get("over_price"),
                    p.get("under_price"),
                    now,
                ))
            except sqlite3.Error as e:
                log.warning(f"[odds] Insert error: {e}")
        conn.commit()
        conn.close()

    def _update_summary(self, game_id: str, player_name: str, prop_type: str):
        """Recompute prop_line_summary for a given (game, player, prop) combo."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get all snapshots ordered by time
        c.execute("""
            SELECT line, over_price, under_price, bookmaker, snapshot_time
            FROM prop_line_history
            WHERE game_id = ? AND player_name = ? AND prop_type = ?
            ORDER BY snapshot_time ASC
        """, (game_id, player_name, prop_type))
        rows = c.fetchall()

        if not rows:
            conn.close()
            return

        # Opening = earliest snapshot (average across bookmakers at that time)
        first_time = rows[0]["snapshot_time"]
        opening_rows = [r for r in rows if r["snapshot_time"] == first_time]
        opening_line = sum(r["line"] for r in opening_rows) / len(opening_rows)
        opening_over = sum((r["over_price"] or -110) for r in opening_rows) / len(opening_rows)

        # Current = latest snapshot
        last_time = rows[-1]["snapshot_time"]
        current_rows = [r for r in rows if r["snapshot_time"] == last_time]
        current_line = sum(r["line"] for r in current_rows) / len(current_rows)
        current_over = sum((r["over_price"] or -110) for r in current_rows) / len(current_rows)

        line_movement = current_line - opening_line
        price_movement = current_over - opening_over

        # Consensus = average across all bookmakers at latest snapshot
        consensus_line = current_line  # already averaged above
        import statistics
        line_vals = [r["line"] for r in current_rows]
        line_std = statistics.stdev(line_vals) if len(line_vals) > 1 else 0.0

        # Distinct snapshot times
        distinct_times = len(set(r["snapshot_time"] for r in rows))

        # Sharp action score:
        # If price moved one direction but line moved opposite → sharp money
        # e.g., over price went MORE negative (more juice on over = public on over)
        #        but line went DOWN (books lowered it = sharp money on under)
        if abs(line_movement) > 0.25:
            # price_movement > 0 means over became less favorable (public on over)
            # line_movement < 0 means line dropped (sharp on under)
            if (price_movement > 0 and line_movement < 0) or \
               (price_movement < 0 and line_movement > 0):
                sharp_action_score = abs(line_movement) * 2.0
            else:
                sharp_action_score = abs(line_movement) * 0.5
        else:
            sharp_action_score = 0.0

        c.execute("""
            INSERT INTO prop_line_summary
            (game_id, player_name, prop_type, opening_line, current_line, line_movement,
             opening_over_price, current_over_price, price_movement,
             num_snapshots, first_seen, last_updated, consensus_line, line_std, sharp_action_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, player_name, prop_type) DO UPDATE SET
                current_line = excluded.current_line,
                line_movement = excluded.line_movement,
                current_over_price = excluded.current_over_price,
                price_movement = excluded.price_movement,
                num_snapshots = excluded.num_snapshots,
                last_updated = excluded.last_updated,
                consensus_line = excluded.consensus_line,
                line_std = excluded.line_std,
                sharp_action_score = excluded.sharp_action_score
        """, (
            game_id, player_name, prop_type,
            opening_line, current_line, line_movement,
            opening_over, current_over, price_movement,
            distinct_times, first_time, last_time,
            consensus_line, line_std, sharp_action_score,
        ))
        conn.commit()
        conn.close()

    # ---- B3: closing-line capture + outcomes (CLV groundwork) -------------

    def capture_closing_lines(
        self,
        game_id: str,
        tipoff_iso: str,
        window_minutes: int = 15,
    ) -> int:
        """Stamp the closing line on each (player, prop_type) for ``game_id``.

        For each (player, prop_type) tracked for this game, take the
        consensus line from the most recent snapshot strictly before
        ``tipoff_iso`` and within ``window_minutes`` of it, and write it
        to ``prop_line_summary.closing_line``.

        Why this is decoupled from the polling loop: the polling loop
        runs every 30-60 min, but for CLV we want the *latest* line we
        could have bet, not "the line at our last poll." A separate
        capture step that runs once near tip-off lets us snapshot more
        precisely without re-polling the whole market.

        Returns the number of (player, prop_type) rows stamped.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            # Find the latest snapshot per (player, prop_type) before tipoff.
            # Average across bookmakers at that snapshot for a consensus line.
            # Both sides of the time comparison are normalised through
            # SQLite's datetime() so we accept either snapshot format
            # ("YYYY-MM-DD HH:MM:SS") or the ISO-with-Z form returned by
            # the Odds API ("2026-04-30T19:00:00Z").
            rows = c.execute(
                """
                SELECT player_name, prop_type, snapshot_time, AVG(line) AS avg_line
                FROM prop_line_history
                WHERE game_id = ?
                  AND datetime(snapshot_time) <= datetime(?)
                  AND datetime(snapshot_time) >= datetime(?, '-' || ? || ' minutes')
                GROUP BY player_name, prop_type, snapshot_time
                ORDER BY player_name, prop_type, snapshot_time DESC
                """,
                (game_id, tipoff_iso, tipoff_iso, window_minutes),
            ).fetchall()

            # Take the most-recent snapshot per (player, prop)
            seen = set()
            stamped = 0
            for player_name, prop_type, snap_time, avg_line in rows:
                key = (player_name, prop_type)
                if key in seen:
                    continue
                seen.add(key)
                c.execute(
                    """
                    UPDATE prop_line_summary
                    SET closing_line = ?, closing_line_at = ?
                    WHERE game_id = ? AND player_name = ? AND prop_type = ?
                    """,
                    (float(avg_line), snap_time, game_id, player_name, prop_type),
                )
                stamped += 1
            conn.commit()
            log.info(
                "[odds] captured closing lines for %d (player, prop_type) on %s",
                stamped, game_id,
            )
            return stamped
        finally:
            conn.close()

    def record_outcome(
        self,
        game_id: str,
        player_name: str,
        prop_type: str,
        actual_result: float,
        observed_line: float | None = None,
        player_id: int | None = None,
    ) -> None:
        """Persist the post-game stat for one (game, player, prop) into
        ``prop_outcomes``, joining with the closing line we already have.

        Called from the post-game settlement job (or wherever the actual
        stat result becomes known). When enough rows accumulate, training
        can join ``prop_outcomes`` against ``prop_line_summary`` to build
        a CLV-aware feature set.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            closing_row = c.execute(
                """
                SELECT closing_line FROM prop_line_summary
                WHERE game_id = ? AND player_name = ? AND prop_type = ?
                """,
                (game_id, player_name, prop_type),
            ).fetchone()
            closing_line = float(closing_row[0]) if closing_row and closing_row[0] is not None else None

            now = datetime.now(timezone.utc).isoformat()
            c.execute(
                """
                INSERT OR REPLACE INTO prop_outcomes
                (game_id, player_id, player_name, prop_type, observed_line,
                 closing_line, actual_result, settled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id, player_id, player_name, prop_type,
                    observed_line, closing_line, float(actual_result), now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def clv_training_rows(self, prop_type: str | None = None) -> list[dict]:
        """Read settled outcomes joined with closing lines — the data that
        becomes B3's training signal once we've accumulated enough.

        Returns one dict per settled prop with fields suitable for CLV
        analysis: ``observed_line``, ``closing_line``, ``line_to_close_drift``,
        ``actual_result``, ``hit`` (1 if result > observed_line else 0).
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            sql = """
                SELECT game_id, player_id, player_name, prop_type,
                       observed_line, closing_line, actual_result, settled_at
                FROM prop_outcomes
                WHERE closing_line IS NOT NULL AND observed_line IS NOT NULL
            """
            args: list = []
            if prop_type is not None:
                sql += " AND prop_type = ?"
                args.append(prop_type)
            rows = c.execute(sql, args).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            (gid, pid, pname, ptype, obs, close_l, actual, settled_at) = r
            try:
                drift = float(close_l) - float(obs)
                hit = 1 if float(actual) > float(obs) else 0
            except (TypeError, ValueError):
                continue
            out.append({
                "game_id": gid,
                "player_id": pid,
                "player_name": pname,
                "prop_type": ptype,
                "observed_line": float(obs),
                "closing_line": float(close_l),
                "line_to_close_drift": drift,
                "actual_result": float(actual),
                "hit": hit,
                "settled_at": settled_at,
            })
        return out

    # ---- Feature extraction -----------------------------------------------

    def get_line_features(self, player_name: str, prop_type: str,
                          game_id: str | None = None) -> dict:
        """Get ML-ready features from line movement data."""
        empty = {
            "opening_line": 0.0,
            "current_line": 0.0,
            "line_movement": 0.0,
            "line_movement_pct": 0.0,
            "implied_over_prob": 0.5,
            "implied_under_prob": 0.5,
            "market_consensus_std": 0.0,
            "sharp_action_score": 0.0,
            "line_velocity": 0.0,
            "stale_line_flag": 0.0,
            "bookmaker_count": 0.0,
        }

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        if game_id:
            c.execute("""
                SELECT * FROM prop_line_summary
                WHERE player_name = ? AND prop_type = ? AND game_id = ?
            """, (player_name, prop_type, game_id))
        else:
            c.execute("""
                SELECT * FROM prop_line_summary
                WHERE player_name = ? AND prop_type = ?
                ORDER BY last_updated DESC LIMIT 1
            """, (player_name, prop_type))

        row = c.fetchone()
        if not row:
            conn.close()
            return empty

        # Count distinct bookmakers at latest snapshot
        c.execute("""
            SELECT COUNT(DISTINCT bookmaker) as cnt
            FROM prop_line_history
            WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """, (row["game_id"], player_name, prop_type))
        bk_count = c.fetchone()["cnt"]
        conn.close()

        opening = float(row["opening_line"] or 0)
        current = float(row["current_line"] or 0)
        movement = float(row["line_movement"] or 0)
        over_price = float(row["current_over_price"] or -110)
        num_snaps = int(row["num_snapshots"] or 0)

        # Hours between first and last snapshot
        try:
            t0 = datetime.strptime(row["first_seen"], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(row["last_updated"], "%Y-%m-%d %H:%M:%S")
            hours = max((t1 - t0).total_seconds() / 3600.0, 0.1)
        except (ValueError, TypeError):
            hours = 1.0

        return {
            "opening_line": opening,
            "current_line": current,
            "line_movement": movement,
            "line_movement_pct": movement / max(abs(opening), 0.1),
            "implied_over_prob": american_to_implied(over_price),
            "implied_under_prob": 1.0 - american_to_implied(over_price),
            "market_consensus_std": float(row["line_std"] or 0),
            "sharp_action_score": float(row["sharp_action_score"] or 0),
            "line_velocity": movement / hours,
            "stale_line_flag": 1.0 if num_snaps > 3 and abs(movement) < 0.5 else 0.0,
            "bookmaker_count": float(bk_count),
        }

    # ---- Signal engine ----------------------------------------------------

    def get_sharp_signals(self, model_prediction: float, player_name: str,
                          prop_type: str, game_id: str | None = None) -> dict:
        """
        Compare model prediction vs market line movement to surface actionable signals.

        Returns:
            signal_type: "sharp_fade" | "model_market_agree" | "stale_line" | "neutral"
            conviction: 0.0 - 1.0
            recommendation: "STRONG OVER" | "OVER" | "STRONG UNDER" | "UNDER" | "PASS"
            reasoning: human-readable explanation
        """
        feats = self.get_line_features(player_name, prop_type, game_id)
        current = feats["current_line"]
        opening = feats["opening_line"]
        movement = feats["line_movement"]
        sharp = feats["sharp_action_score"]

        if current == 0.0:
            return {
                "signal_type": "no_data",
                "conviction": 0.0,
                "model_edge": 0.0,
                "model_edge_pct": 0.0,
                "recommendation": "PASS",
                "reasoning": "No line data available for this player prop.",
            }

        model_edge = model_prediction - current
        model_edge_pct = model_edge / max(abs(current), 0.1)
        model_says_over = model_prediction > current
        line_moved_up = movement > 0.25
        line_moved_down = movement < -0.25
        stale = feats["stale_line_flag"] > 0

        signal_type = "neutral"
        conviction = 0.0
        recommendation = "PASS"
        reasoning = ""

        if model_says_over and line_moved_down:
            # Model says OVER, line moved DOWN → both agree on OVER
            signal_type = "model_market_agree"
            conviction = min(1.0, abs(model_edge_pct) * 2 + abs(movement) * 0.5)
            recommendation = "STRONG OVER" if conviction > 0.6 else "OVER"
            reasoning = (
                f"Model predicts {model_prediction:.1f} (OVER {current:.1f}). "
                f"Line dropped {abs(movement):.1f} pts from {opening:.1f} — "
                f"sharp money also on OVER. Strongest signal."
            )
        elif not model_says_over and line_moved_up:
            # Model says UNDER, line moved UP → both agree on UNDER
            signal_type = "model_market_agree"
            conviction = min(1.0, abs(model_edge_pct) * 2 + abs(movement) * 0.5)
            recommendation = "STRONG UNDER" if conviction > 0.6 else "UNDER"
            reasoning = (
                f"Model predicts {model_prediction:.1f} (UNDER {current:.1f}). "
                f"Line rose {abs(movement):.1f} pts from {opening:.1f} — "
                f"sharp money also on UNDER. Strongest signal."
            )
        elif model_says_over and line_moved_up:
            # Model says OVER but line moved UP (sharp took UNDER) → sharp fade
            signal_type = "sharp_fade"
            conviction = min(1.0, sharp * 0.3 + abs(movement) * 0.4)
            recommendation = "UNDER" if conviction > 0.4 else "PASS"
            reasoning = (
                f"Model predicts {model_prediction:.1f} (OVER {current:.1f}), "
                f"but line rose {abs(movement):.1f} pts — sharp money on UNDER. "
                f"Fading model in favor of sharp action."
            )
        elif not model_says_over and line_moved_down:
            # Model says UNDER but line dropped (sharp took OVER) → sharp fade
            signal_type = "sharp_fade"
            conviction = min(1.0, sharp * 0.3 + abs(movement) * 0.4)
            recommendation = "OVER" if conviction > 0.4 else "PASS"
            reasoning = (
                f"Model predicts {model_prediction:.1f} (UNDER {current:.1f}), "
                f"but line dropped {abs(movement):.1f} pts — sharp money on OVER. "
                f"Fading model in favor of sharp action."
            )
        elif stale and abs(model_edge) > 1.0:
            signal_type = "stale_line"
            conviction = min(1.0, abs(model_edge_pct) * 1.5)
            recommendation = "OVER" if model_says_over else "UNDER"
            reasoning = (
                f"Line hasn't moved ({opening:.1f} → {current:.1f}) despite "
                f"{feats['bookmaker_count']:.0f} bookmakers tracked. "
                f"Model sees {abs(model_edge):.1f} pt edge — possible stale line."
            )
        else:
            signal_type = "neutral"
            conviction = abs(model_edge_pct) * 0.5
            recommendation = "PASS"
            reasoning = (
                f"Model predicts {model_prediction:.1f} vs line {current:.1f}. "
                f"No strong market signal (movement: {movement:+.1f})."
            )

        return {
            "signal_type": signal_type,
            "conviction": round(conviction, 3),
            "model_edge": round(model_edge, 2),
            "model_edge_pct": round(model_edge_pct, 4),
            "recommendation": recommendation,
            "reasoning": reasoning,
        }


# ---------------------------------------------------------------------------
# Polling entry point
# ---------------------------------------------------------------------------

# Last-poll telemetry consulted by /odds/status. Dict so the health endpoint
# can distinguish "never polled" from "polled but got zero" from "auth failed".
_LAST_POLL: dict = {
    "time": None,
    "lines_stored": None,
    "error": None,
    "quota_remaining": None,
    # B3: how many (player, prop) closing lines were stamped on the most
    # recent poll. Stays None on polls where no game tipped during the
    # window, so we can distinguish "feature off" from "feature ran but
    # found nothing."
    "closes_captured": None,
}


def poll_odds(api_key: str, db_path: str = "basketball_data.db") -> int:
    """Single poll cycle — call on a cron/schedule every 30-60 min on game days."""
    tracker = OddsTracker(api_key, db_path)
    _LAST_POLL["time"] = datetime.now(timezone.utc).isoformat()
    try:
        count = tracker.snapshot_all_games()
        _LAST_POLL["lines_stored"] = int(count)
        _LAST_POLL["error"] = None
        log.info(f"[odds] Poll complete: {count} lines stored")
        return count
    except Exception as exc:  # noqa: BLE001
        _LAST_POLL["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("[odds] Poll failed")
        return 0


def last_poll_status() -> dict:
    """Snapshot of the most recent poll — used by the /odds/status endpoint."""
    return dict(_LAST_POLL)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python odds_tracker.py <API_KEY> [db_path]")
        sys.exit(1)
    key = sys.argv[1]
    db = sys.argv[2] if len(sys.argv) > 2 else "basketball_data.db"
    poll_odds(key, db)
