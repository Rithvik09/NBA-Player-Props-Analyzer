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

        refs: dict[str, dict] = {}
        refs_meta: dict[str, Any] = {'updated_at': None}
        team_foul: dict[int, dict] = {}
        dvp_rolling: dict[tuple, dict] = {}
        team_stats: dict[int, dict] = {}
        team_stats_meta: dict[str, Any] = {'updated_at': None}

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

            # --- referee stats ---
            try:
                cur.execute(
                    "SELECT ref_name, games, foul_rate, home_win_pct, pace FROM referee_stats"
                )
                ref_rows = cur.fetchall()
                ref_max_ts = None
                for (ref_name, games, foul_rate, home_win_pct, pace) in ref_rows:
                    refs[str(ref_name).lower()] = {
                        'games':        int(games) if games is not None else 0,
                        'foul_rate':    float(foul_rate) if foul_rate is not None else 0.0,
                        'home_win_pct': float(home_win_pct) if home_win_pct is not None else 0.5,
                        'pace':         float(pace) if pace is not None else 0.0,
                    }
                # updated_at is not selected here but refs_meta can stay None safely
            except Exception:
                pass

            # --- team foul rates ---
            try:
                cur.execute(
                    "SELECT team_id, foul_rate_season, foul_rate_last5 FROM team_foul_rates"
                )
                for (team_id, foul_rate_season, foul_rate_last5) in cur.fetchall():
                    team_foul[int(team_id)] = {
                        'foul_rate_season': float(foul_rate_season) if foul_rate_season is not None else 20.0,
                        'foul_rate_last5':  float(foul_rate_last5)  if foul_rate_last5  is not None else 20.0,
                    }
            except Exception:
                pass

            # --- rolling DVP ---
            try:
                cur.execute(
                    "SELECT team_id, window, pts, reb, ast, fg3m, stl, blk, tov, gp FROM dvp_rolling"
                )
                for (team_id, window, pts, reb, ast, fg3m, stl, blk, tov, gp) in cur.fetchall():
                    dvp_rolling[(int(team_id), int(window))] = {
                        'pts':  float(pts)  if pts  is not None else 0.0,
                        'reb':  float(reb)  if reb  is not None else 0.0,
                        'ast':  float(ast)  if ast  is not None else 0.0,
                        'fg3m': float(fg3m) if fg3m is not None else 0.0,
                        'stl':  float(stl)  if stl  is not None else 0.0,
                        'blk':  float(blk)  if blk  is not None else 0.0,
                        'tov':  float(tov)  if tov  is not None else 0.0,
                        'gp':   int(gp)     if gp   is not None else 0,
                    }
            except Exception:
                pass

            # --- team stats (rich offensive/defensive baseline) ---
            try:
                cur.execute(
                    """
                    SELECT team_id, pts_fb, pts_off_tov,
                           opp_fga, opp_fg_pct, opp_fg3a, opp_fg3_pct,
                           opp_tov, opp_stl, opp_blk, opp_pts_paint, opp_pts_fb, opp_pts_off_tov,
                           opp_def_rating_last5, opp_blk_last5, opp_stl_last5,
                           lg_pts_fb, lg_pts_off_tov, lg_fga, lg_fg_pct, lg_fg3a, lg_tov, lg_stl,
                           foul_rate_season, foul_rate_last5, updated_at
                    FROM team_stats
                    """
                )
                ts_max_ts = None
                for row in cur.fetchall():
                    (tid, pts_fb, pts_off_tov,
                     opp_fga, opp_fg_pct, opp_fg3a, opp_fg3_pct,
                     opp_tov, opp_stl, opp_blk, opp_pts_paint, opp_pts_fb, opp_pts_off_tov,
                     opp_def_rating_last5, opp_blk_last5, opp_stl_last5,
                     lg_pts_fb, lg_pts_off_tov, lg_fga, lg_fg_pct, lg_fg3a, lg_tov, lg_stl,
                     foul_rate_season, foul_rate_last5, updated_at) = row
                    team_stats[int(tid)] = {
                        'pts_fb':               float(pts_fb)              if pts_fb              is not None else 12.0,
                        'pts_off_tov':          float(pts_off_tov)         if pts_off_tov         is not None else 16.0,
                        'opp_fga':              float(opp_fga)             if opp_fga             is not None else 86.0,
                        'opp_fg_pct':           float(opp_fg_pct)          if opp_fg_pct          is not None else 0.47,
                        'opp_fg3a':             float(opp_fg3a)            if opp_fg3a            is not None else 35.0,
                        'opp_fg3_pct':          float(opp_fg3_pct)         if opp_fg3_pct         is not None else 0.36,
                        'opp_tov':              float(opp_tov)             if opp_tov             is not None else 14.0,
                        'opp_stl':              float(opp_stl)             if opp_stl             is not None else 7.0,
                        'opp_blk':              float(opp_blk)             if opp_blk             is not None else 5.0,
                        'opp_pts_paint':        float(opp_pts_paint)       if opp_pts_paint       is not None else 44.0,
                        'opp_pts_fb':           float(opp_pts_fb)          if opp_pts_fb          is not None else 12.0,
                        'opp_pts_off_tov':      float(opp_pts_off_tov)     if opp_pts_off_tov     is not None else 16.0,
                        'opp_def_rating_last5': float(opp_def_rating_last5) if opp_def_rating_last5 is not None else 110.0,
                        'opp_blk_last5':        float(opp_blk_last5)       if opp_blk_last5       is not None else 5.0,
                        'opp_stl_last5':        float(opp_stl_last5)       if opp_stl_last5       is not None else 7.0,
                        'lg_pts_fb':            float(lg_pts_fb)           if lg_pts_fb           is not None else 12.0,
                        'lg_pts_off_tov':       float(lg_pts_off_tov)      if lg_pts_off_tov      is not None else 16.0,
                        'lg_fga':               float(lg_fga)              if lg_fga              is not None else 86.0,
                        'lg_fg_pct':            float(lg_fg_pct)           if lg_fg_pct           is not None else 0.47,
                        'lg_fg3a':              float(lg_fg3a)             if lg_fg3a             is not None else 35.0,
                        'lg_tov':               float(lg_tov)              if lg_tov              is not None else 14.0,
                        'lg_stl':               float(lg_stl)              if lg_stl              is not None else 7.0,
                        'foul_rate_season':     float(foul_rate_season)    if foul_rate_season    is not None else 20.0,
                        'foul_rate_last5':      float(foul_rate_last5)     if foul_rate_last5     is not None else 20.0,
                    }
                    if updated_at is not None:
                        ts_max_ts = int(updated_at) if ts_max_ts is None else max(ts_max_ts, int(updated_at))
                team_stats_meta['updated_at'] = ts_max_ts
            except Exception:
                pass

        except Exception:
            # Tables may not exist yet; return empty structures.
            pass

        # ---- New enriched tables ----
        player_advanced: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, usg_pct, ts_pct, efg_pct, ast_pct,
                       oreb_pct, dreb_pct, reb_pct, pie,
                       off_rating, def_rating, pace, net_rating,
                       age, height_inches, weight, years_experience
                FROM player_advanced_stats
                """
            )
            for row in cur.fetchall():
                (pid, usg, ts, efg, ast_p, oreb, dreb, reb, pie,
                 offr, defr, pace, net, age, ht, wt, yrs) = row
                player_advanced[int(pid)] = {
                    'usg_pct_official':    float(usg)  if usg  is not None else 0.18,
                    'ts_pct_official':     float(ts)   if ts   is not None else 0.55,
                    'efg_pct_official':    float(efg)  if efg  is not None else 0.50,
                    'ast_pct_official':    float(ast_p) if ast_p is not None else 0.15,
                    'oreb_pct_official':   float(oreb) if oreb is not None else 0.05,
                    'dreb_pct_official':   float(dreb) if dreb is not None else 0.15,
                    'reb_pct_official':    float(reb)  if reb  is not None else 0.10,
                    'pie':                 float(pie)  if pie  is not None else 0.10,
                    'player_off_rating':   float(offr) if offr is not None else 110.0,
                    'player_def_rating':   float(defr) if defr is not None else 110.0,
                    'player_pace':         float(pace) if pace is not None else 100.0,
                    'net_rating_player':   float(net)  if net  is not None else 0.0,
                    'player_age':          float(age)  if age  is not None else 26.0,
                    'player_height_inches': float(ht)  if ht   is not None else 78.0,
                    'player_weight':       float(wt)   if wt   is not None else 220.0,
                    'years_experience':    float(yrs)  if yrs  is not None else 5.0,
                }
        except Exception:
            pass

        player_clutch: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, clutch_pts_pg, clutch_fg_pct, clutch_fg3_pct,
                       clutch_fta_pg, clutch_plus_minus, clutch_min_pg, clutch_games
                FROM player_clutch_stats
                """
            )
            for row in cur.fetchall():
                (pid, pts, fg, fg3, fta, pm, mn, gp) = row
                player_clutch[int(pid)] = {
                    'clutch_pts_per_game':  float(pts) if pts is not None else 0.0,
                    'clutch_fg_pct':        float(fg)  if fg  is not None else 0.45,
                    'clutch_fg3_pct':       float(fg3) if fg3 is not None else 0.33,
                    'clutch_fta_per_game':  float(fta) if fta is not None else 0.0,
                    'clutch_plus_minus':    float(pm)  if pm  is not None else 0.0,
                    'clutch_min_per_game':  float(mn)  if mn  is not None else 0.0,
                    'clutch_games':         int(gp)    if gp  is not None else 0,
                }
        except Exception:
            pass

        player_hustle: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, contested_shots_pg, deflections_pg,
                       charges_drawn_pg, screen_assists_pg
                FROM player_hustle_stats
                """
            )
            for row in cur.fetchall():
                (pid, cs, defl, chg, scr) = row
                player_hustle[int(pid)] = {
                    'contested_shots_per_game': float(cs)   if cs   is not None else 3.0,
                    'deflections_per_game':     float(defl) if defl is not None else 1.0,
                    'charges_drawn_per_game':   float(chg)  if chg  is not None else 0.1,
                    'screen_assists_per_game':  float(scr)  if scr  is not None else 0.5,
                }
        except Exception:
            pass

        player_shot_profile: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, open_shot_fg_pct, open_shot_freq,
                       tight_shot_fg_pct, tight_shot_freq,
                       catch_shoot_fg_pct, catch_shoot_freq,
                       pullup_fg_pct, pullup_freq
                FROM player_shot_profile
                """
            )
            for row in cur.fetchall():
                (pid, o_pct, o_freq, t_pct, t_freq, c_pct, c_freq, p_pct, p_freq) = row
                player_shot_profile[int(pid)] = {
                    'open_shot_fg_pct':    float(o_pct)  if o_pct  is not None else 0.50,
                    'open_shot_frequency': float(o_freq) if o_freq is not None else 0.30,
                    'tight_shot_fg_pct':   float(t_pct)  if t_pct  is not None else 0.38,
                    'tight_shot_frequency': float(t_freq) if t_freq is not None else 0.15,
                    'catch_shoot_fg_pct':  float(c_pct)  if c_pct  is not None else 0.40,
                    'catch_shoot_frequency': float(c_freq) if c_freq is not None else 0.25,
                    'pullup_fg_pct':       float(p_pct)  if p_pct  is not None else 0.40,
                    'pullup_frequency':    float(p_freq) if p_freq is not None else 0.20,
                }
        except Exception:
            pass

        player_play_types: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, iso_poss_pct, iso_ppp,
                       pnr_bh_poss_pct, pnr_bh_ppp,
                       pnr_roll_poss_pct, pnr_roll_ppp,
                       spotup_poss_pct, spotup_ppp,
                       transition_poss_pct, transition_ppp,
                       postup_poss_pct, cut_poss_pct
                FROM player_play_types
                """
            )
            for row in cur.fetchall():
                (pid, iso_pct, iso_ppp, pnr_pct, pnr_ppp, roll_pct, roll_ppp,
                 su_pct, su_ppp, tr_pct, tr_ppp, po_pct, cut_pct) = row
                player_play_types[int(pid)] = {
                    'iso_poss_pct':        float(iso_pct)  if iso_pct  is not None else 0.0,
                    'iso_ppp':             float(iso_ppp)  if iso_ppp  is not None else 0.9,
                    'pnr_bh_poss_pct':     float(pnr_pct)  if pnr_pct  is not None else 0.0,
                    'pnr_bh_ppp':          float(pnr_ppp)  if pnr_ppp  is not None else 0.9,
                    'pnr_roll_poss_pct':   float(roll_pct) if roll_pct is not None else 0.0,
                    'pnr_roll_ppp':        float(roll_ppp) if roll_ppp is not None else 0.9,
                    'spotup_poss_pct':     float(su_pct)   if su_pct   is not None else 0.0,
                    'spotup_ppp':          float(su_ppp)   if su_ppp   is not None else 1.0,
                    'transition_poss_pct': float(tr_pct)   if tr_pct   is not None else 0.0,
                    'transition_ppp':      float(tr_ppp)   if tr_ppp   is not None else 1.1,
                    'postup_poss_pct':     float(po_pct)   if po_pct   is not None else 0.0,
                    'cut_poss_pct':        float(cut_pct)  if cut_pct  is not None else 0.0,
                }
        except Exception:
            pass

        player_on_off: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, on_court_net_rating, off_court_net_rating, on_off_differential
                FROM player_on_off
                """
            )
            for row in cur.fetchall():
                (pid, on_r, off_r, diff) = row
                player_on_off[int(pid)] = {
                    'on_court_net_rating':  float(on_r)  if on_r  is not None else 0.0,
                    'off_court_net_rating': float(off_r) if off_r is not None else 0.0,
                    'on_off_differential':  float(diff)  if diff  is not None else 0.0,
                }
        except Exception:
            pass

        player_shot_zones: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, rim_fga_pct, rim_fg_pct,
                       paint_fga_pct, paint_fg_pct,
                       midrange_fga_pct, midrange_fg_pct,
                       corner3_fga_pct, corner3_fg_pct,
                       above_break3_fga_pct, above_break3_fg_pct
                FROM player_shot_zones
                """
            )
            for row in cur.fetchall():
                (pid, r_fa, r_fg, pa_fa, pa_fg, m_fa, m_fg,
                 c3_fa, c3_fg, ab_fa, ab_fg) = row
                player_shot_zones[int(pid)] = {
                    'rim_fga_pct':          float(r_fa)  if r_fa  is not None else 0.25,
                    'rim_fg_pct':           float(r_fg)  if r_fg  is not None else 0.62,
                    'paint_fga_pct':        float(pa_fa) if pa_fa is not None else 0.30,
                    'paint_fg_pct':         float(pa_fg) if pa_fg is not None else 0.55,
                    'midrange_fga_pct':     float(m_fa)  if m_fa  is not None else 0.20,
                    'midrange_fg_pct':      float(m_fg)  if m_fg  is not None else 0.42,
                    'corner3_fga_pct':      float(c3_fa) if c3_fa is not None else 0.10,
                    'corner3_fg_pct':       float(c3_fg) if c3_fg is not None else 0.38,
                    'above_break3_fga_pct': float(ab_fa) if ab_fa is not None else 0.25,
                    'above_break3_fg_pct':  float(ab_fg) if ab_fg is not None else 0.35,
                }
        except Exception:
            pass

        player_quarter_splits: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, q1_avg, q2_avg, q3_avg, q4_avg, q4_min_pg
                FROM player_quarter_splits
                """
            )
            for row in cur.fetchall():
                (pid, q1, q2, q3, q4, q4_min) = row
                player_quarter_splits[int(pid)] = {
                    'q1_avg':         float(q1)     if q1     is not None else 0.0,
                    'q2_avg':         float(q2)     if q2     is not None else 0.0,
                    'q3_avg':         float(q3)     if q3     is not None else 0.0,
                    'q4_avg':         float(q4)     if q4     is not None else 0.0,
                    'q4_min_per_game': float(q4_min) if q4_min is not None else 0.0,
                }
        except Exception:
            pass

        team_opp_shot_zones: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT team_id, rim_fg_pct_allowed, paint_fg_pct_allowed,
                       midrange_fg_pct_allowed, corner3_fg_pct_allowed,
                       above_break3_fg_pct_allowed
                FROM team_opp_shot_zones
                """
            )
            for row in cur.fetchall():
                (tid, rim, paint, mid, c3, ab) = row
                team_opp_shot_zones[int(tid)] = {
                    'rim_fg_pct_allowed':          float(rim)   if rim   is not None else 0.62,
                    'paint_fg_pct_allowed':        float(paint) if paint is not None else 0.55,
                    'midrange_fg_pct_allowed':     float(mid)   if mid   is not None else 0.42,
                    'corner3_fg_pct_allowed':      float(c3)    if c3    is not None else 0.38,
                    'above_break3_fg_pct_allowed': float(ab)    if ab    is not None else 0.35,
                }
        except Exception:
            pass

        team_synergy_defense: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT team_id, pnr_ppp_allowed, iso_ppp_allowed,
                       spotup_ppp_allowed, transition_ppp_allowed, postup_ppp_allowed
                FROM team_synergy_defense
                """
            )
            for row in cur.fetchall():
                (tid, pnr, iso, su, tr, po) = row
                team_synergy_defense[int(tid)] = {
                    'pnr_ppp_allowed':        float(pnr) if pnr is not None else 0.9,
                    'iso_ppp_allowed':        float(iso) if iso is not None else 0.9,
                    'spotup_ppp_allowed':     float(su)  if su  is not None else 1.0,
                    'transition_ppp_allowed': float(tr)  if tr  is not None else 1.1,
                    'postup_ppp_allowed':     float(po)  if po  is not None else 0.9,
                }
        except Exception:
            pass

        player_tracking: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, avg_speed, avg_speed_off, avg_speed_def,
                       dist_miles, dist_miles_off, dist_miles_def,
                       touches_pg, time_of_poss_pg, avg_drib_per_touch,
                       paint_touches_pg, elbow_touches_pg,
                       passes_made_pg, potential_ast_pg, secondary_ast_pg
                FROM player_tracking_stats
                """
            )
            for row in cur.fetchall():
                (pid, spd, spd_off, spd_def, dist, dist_off, dist_def,
                 touches, top, drib, paint_t, elbow_t, passes, pot_ast, sec_ast) = row
                player_tracking[int(pid)] = {
                    'tracking_avg_speed':         float(spd)     if spd     is not None else 4.5,
                    'tracking_avg_speed_off':     float(spd_off) if spd_off is not None else 4.8,
                    'tracking_avg_speed_def':     float(spd_def) if spd_def is not None else 4.2,
                    'tracking_dist_miles':        float(dist)    if dist    is not None else 2.5,
                    'tracking_dist_miles_off':    float(dist_off) if dist_off is not None else 1.3,
                    'tracking_dist_miles_def':    float(dist_def) if dist_def is not None else 1.2,
                    'tracking_touches_pg':        float(touches) if touches  is not None else 50.0,
                    'tracking_time_of_poss_pg':   float(top)     if top      is not None else 2.5,
                    'tracking_avg_drib_per_touch': float(drib)   if drib     is not None else 1.5,
                    'tracking_paint_touches_pg':  float(paint_t) if paint_t  is not None else 2.0,
                    'tracking_elbow_touches_pg':  float(elbow_t) if elbow_t  is not None else 1.5,
                    'tracking_passes_made_pg':    float(passes)  if passes   is not None else 30.0,
                    'tracking_potential_ast_pg':  float(pot_ast) if pot_ast  is not None else 5.0,
                    'tracking_secondary_ast_pg':  float(sec_ast) if sec_ast  is not None else 1.0,
                }
        except Exception:
            pass

        team_standings: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT team_id, win_pct, wins, losses, conf_rank, games_back,
                       home_win_pct, road_win_pct, current_streak, l10_wins,
                       pts_pg, opp_pts_pg
                FROM team_standings
                """
            )
            for row in cur.fetchall():
                (tid, win_pct, wins, losses, conf_rank, games_back,
                 home_wp, road_wp, streak, l10w, pts_pg, opp_pts_pg) = row
                team_standings[int(tid)] = {
                    'win_pct':        float(win_pct)    if win_pct    is not None else 0.5,
                    'wins':           int(wins)         if wins       is not None else 41,
                    'losses':         int(losses)       if losses     is not None else 41,
                    'conf_rank':      int(conf_rank)    if conf_rank  is not None else 8,
                    'games_back':     float(games_back) if games_back is not None else 5.0,
                    'home_win_pct':   float(home_wp)    if home_wp    is not None else 0.5,
                    'road_win_pct':   float(road_wp)    if road_wp    is not None else 0.5,
                    'current_streak': int(streak)       if streak     is not None else 0,
                    'l10_wins':       int(l10w)         if l10w       is not None else 5,
                    'pts_pg':         float(pts_pg)     if pts_pg     is not None else 112.0,
                    'opp_pts_pg':     float(opp_pts_pg) if opp_pts_pg is not None else 112.0,
                }
        except Exception:
            pass

        player_scoring_breakdown: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, pct_pts_3pt, pct_pts_paint, pct_pts_ft,
                       pct_pts_midrange, pct_uast_fgm, pct_ast_fgm
                FROM player_scoring_breakdown
                """
            )
            for row in cur.fetchall():
                (pid, p3, ppaint, pft, pmid, puast, past) = row
                player_scoring_breakdown[int(pid)] = {
                    'pct_pts_3pt':      float(p3)     if p3     is not None else 0.25,
                    'pct_pts_paint':    float(ppaint) if ppaint is not None else 0.30,
                    'pct_pts_ft':       float(pft)    if pft    is not None else 0.15,
                    'pct_pts_midrange': float(pmid)   if pmid   is not None else 0.20,
                    'pct_uast_fgm':     float(puast)  if puast  is not None else 0.40,
                    'pct_ast_fgm':      float(past)   if past   is not None else 0.60,
                }
        except Exception:
            pass

        player_vs_opponent: dict[tuple, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, opponent_team_id, gp, avg_stat_pts, fg_pct, ts_pct, avg_min
                FROM player_vs_opponent
                """
            )
            for row in cur.fetchall():
                (pid, opp_tid, gp, pts, fg, ts, mn) = row
                player_vs_opponent[(int(pid), int(opp_tid))] = {
                    'vs_opp_avg_pts': float(pts) if pts is not None else 0.0,
                    'vs_opp_fg_pct':  float(fg)  if fg  is not None else 0.45,
                    'vs_opp_ts_pct':  float(ts)  if ts  is not None else 0.55,
                    'vs_opp_gp':      int(gp)    if gp  is not None else 0,
                    'vs_opp_avg_min': float(mn)  if mn  is not None else 30.0,
                }
        except Exception:
            pass

        team_rest_splits: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT team_id, b2b_def_rating, b2b_pace, b2b_pts_allowed,
                       rested_def_rating, rested_pace
                FROM team_rest_splits
                """
            )
            for row in cur.fetchall():
                (tid, b2b_def, b2b_pace, b2b_pts, rest_def, rest_pace) = row
                b2b_def_f  = float(b2b_def)  if b2b_def  is not None else 112.0
                rest_def_f = float(rest_def) if rest_def is not None else 110.0
                team_rest_splits[int(tid)] = {
                    'opp_b2b_def_rating':        b2b_def_f,
                    'opp_b2b_pace':              float(b2b_pace)  if b2b_pace  is not None else 100.0,
                    'opp_b2b_pts_allowed':       float(b2b_pts)   if b2b_pts   is not None else 115.0,
                    'opp_rested_def_rating':     rest_def_f,
                    'opp_rested_pace':           float(rest_pace) if rest_pace is not None else 100.0,
                    'opp_rest_def_rating_delta': b2b_def_f - rest_def_f,
                }
        except Exception:
            pass

        player_yoy_stats: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT player_id, yoy_pts_change, yoy_ts_change, yoy_usage_change, seasons_in_league
                FROM player_yoy_stats
                """
            )
            for row in cur.fetchall():
                (pid, pts_ch, ts_ch, usg_ch, seasons) = row
                player_yoy_stats[int(pid)] = {
                    'yoy_pts_change':    float(pts_ch)  if pts_ch  is not None else 0.0,
                    'yoy_ts_change':     float(ts_ch)   if ts_ch   is not None else 0.0,
                    'yoy_usage_change':  float(usg_ch)  if usg_ch  is not None else 0.0,
                    'seasons_in_league': int(seasons)   if seasons is not None else 5,
                }
        except Exception:
            pass

        team_home_away_splits: dict[int, dict] = {}
        try:
            cur.execute(
                "SELECT team_id, home_def_rating, away_def_rating, home_away_def_split FROM team_home_away_splits"
            )
            for row in cur.fetchall():
                (tid, home_def, away_def, split) = row
                team_home_away_splits[int(tid)] = {
                    'home_def_rating':       float(home_def)  if home_def  is not None else 110.0,
                    'away_def_rating':       float(away_def)  if away_def  is not None else 110.0,
                    'home_away_def_split':   float(split)     if split     is not None else 0.0,
                }
        except Exception:
            pass

        team_lineup_stats: dict[int, dict] = {}
        try:
            cur.execute(
                """
                SELECT team_id, top_lineup_net_rating, bench_net_rating, bench_strength,
                       lineup_continuity, lineups_played_count
                FROM team_lineup_stats
                """
            )
            for row in cur.fetchall():
                (tid, top_nr, bench_nr, bench_str, continuity, count) = row
                team_lineup_stats[int(tid)] = {
                    'top_lineup_net_rating': float(top_nr)    if top_nr    is not None else 0.0,
                    'bench_net_rating':      float(bench_nr)  if bench_nr  is not None else 0.0,
                    'bench_strength':        float(bench_str) if bench_str is not None else 0.0,
                    'lineup_continuity':     float(continuity) if continuity is not None else 0.7,
                    'lineups_played_count':  int(count)       if count     is not None else 1,
                }
        except Exception:
            pass

        team_injury_status: dict[int, dict] = {}
        try:
            cur.execute(
                "SELECT team_id, key_players_out, total_players_out, total_impact FROM team_injury_status"
            )
            for row in cur.fetchall():
                (tid, key_out, total_out, impact) = row
                team_injury_status[int(tid)] = {
                    'key_players_out':   int(key_out)    if key_out   is not None else 0,
                    'total_players_out': int(total_out)  if total_out is not None else 0,
                    'total_impact':      float(impact)   if impact    is not None else 0.0,
                }
        except Exception:
            pass

        try:
            conn.close()
        except Exception:
            pass

        payload = {
            'dvp': dvp,
            'dvp_meta': dvp_meta,
            'dvp_pos_avgs': dvp_pos_avgs,
            'defenders': defenders,
            'defenders_meta': defenders_meta,
            'refs': refs,
            'refs_meta': refs_meta,
            'team_foul': team_foul,
            'dvp_rolling': dvp_rolling,
            'team_stats': team_stats,
            'team_stats_meta': team_stats_meta,
            'player_advanced': player_advanced,
            'player_clutch': player_clutch,
            'player_hustle': player_hustle,
            'player_shot_profile': player_shot_profile,
            'player_play_types': player_play_types,
            'player_on_off': player_on_off,
            'player_shot_zones': player_shot_zones,
            'player_quarter_splits': player_quarter_splits,
            'team_opp_shot_zones': team_opp_shot_zones,
            'team_synergy_defense': team_synergy_defense,
            'player_tracking': player_tracking,
            'team_standings': team_standings,
            'player_scoring_breakdown': player_scoring_breakdown,
            'player_vs_opponent': player_vs_opponent,
            'team_rest_splits': team_rest_splits,
            'player_yoy_stats': player_yoy_stats,
            'team_home_away_splits': team_home_away_splits,
            'team_lineup_stats': team_lineup_stats,
            'team_injury_status': team_injury_status,
        }
        self._cache = payload
        self._cache_at = now
        return payload

    def is_fresh(self, max_age_seconds: int = 60 * 60 * 26) -> bool:
        data = self.refresh()
        now = int(time.time())
        dvp_ts = data.get('dvp_meta', {}).get('updated_at') or 0
        def_ts = data.get('defenders_meta', {}).get('updated_at') or 0
        ts_ts  = data.get('team_stats_meta', {}).get('updated_at') or 0
        newest = max(int(dvp_ts), int(def_ts), int(ts_ts))
        return newest > 0 and (now - newest) < max_age_seconds


