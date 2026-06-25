# NBA Player Props Analyzer

A data-driven NBA player prop betting assistant. It ingests stats from the NBA API, FantasyPros, ESPN, Basketball-Reference, and (optionally) The Odds API, builds a **408-feature** machine learning pipeline, and serves predictions through a Flask web app with logging, grading, bankroll tracking, and line-movement monitoring.

---

## What it does

- Predicts whether a player will go **over or under** a prop line (points, assists, rebounds, combos, double-double, etc.)
- Estimates expected stat output via per-prop **XGBoost regressors**
- Blends ML output with a statistical baseline (recent form + season average)
- Tracks prediction accuracy over time and supports Kelly sizing / bankroll management
- Polls live prop lines when configured, storing movement history for sharp-action signals

---

## Architecture

| Layer | Technology |
|---|---|
| Web API | Flask (`src/app.py`) |
| Database | SQLite (`basketball_data.db`) |
| ML inference | `EnhancedMLPredictor` (`src/models.py`) |
| Feature schema | `NUMERIC_FEATURE_KEYS` in `src/ml_features.py` (408 features) |
| Precomputed cache | `PrecomputedStore` + `precompute_jobs.py` |
| Training | `scripts/train_models.py` |
| Odds polling | `src/odds_tracker.py` (background thread when `ODDS_API_KEY` is set) |

**Prediction blend (when models are loaded):**

| Output | Weight |
|---|---|
| ML regression / classifier | 75% |
| Statistical baseline (0.7 × recent avg + 0.3 × season avg) | 25% |

If per-prop models are missing, the app falls back to global models, then heuristics — it does not crash.

---

## Machine learning

### Models

- **Per-prop XGBoost regressors** — predict expected stat value (RMSE-optimized via Optuna)
- **Per-prop XGBoost classifiers** — predict over/under probability
- **Isotonic calibration** — calibrates classifier probabilities on a holdout set
- **Walk-forward validation** — reports AUC and Brier score per prop before saving

Falls back to `HistGradientBoosting` if XGBoost is unavailable.

### Supported prop types (14)

| Single stats | Combos | Binary |
|---|---|---|
| points, assists, rebounds | pts+reb, pts+ast, ast+reb | double_double |
| steals, blocks, turnovers | pts+ast+reb, stl+blk | triple_double |
| three_pointers | | |

Models are saved under `models/` as `reg_*.joblib`, `clf_raw_*.joblib`, and `clf_cal_*.joblib`, with metadata in `models/model_metadata.json`.

### Feature pipeline (408 features)

All features are defined in `src/ml_features.py` and built consistently in:

- `scripts/train_models.py` (training)
- `src/models.py` → `prepare_features()` (inference)
- `src/basketball_betting_helper.py` (live prop analysis)

**Coverage:**

| Category | Count | Notes |
|---|---|---|
| NBA stats, precompute, derived | ~397 | Wired to real data sources (see below) |
| Market / odds line movement | 11 | Populated at inference when `ODDS_API_KEY` is set and line history exists; trained as zeros until history accumulates |

The 11 odds features: `opening_line`, `current_line`, `line_movement`, `line_movement_pct`, `implied_over_prob`, `implied_under_prob`, `market_consensus_std`, `sharp_action_score`, `line_velocity`, `stale_line_flag`, `bookmaker_count`.

Feature groups include: rolling performance, schedule/fatigue, team/opp pace and ratings, DVP deltas, defender matchups, play types (Synergy), shot zones, tracking stats, on/off court, travel/arena, lineup depth, injuries, referee tendencies, year-over-year trends, playoff context, and more.

---

## Data sources

### Live / on-demand

| Source | Used for |
|---|---|
| **NBA API** (`nba_api`) | Player game logs, team stats, advanced metrics, clutch/hustle, play types, on/off, shot zones, quarter splits, tracking, lineups, standings, box scores |
| **ESPN** (scrape) | Injury reports via `InjuryTracker` |
| **FantasyPros** (scrape) | Defense vs. position (DVP) by PG/SG/SF/PF/C |
| **Basketball-Reference** (scrape) | Referee foul-rate tendencies |
| **The Odds API** (optional) | Player prop lines, line movement, consensus spread, sharp-action score |
| **Static arena data** (`src/arena_data.py`) | Altitude, capacity, travel distance, timezone change |

### Precomputed tables (SQLite)

Run `python3 scripts/update_precomputed.py` to refresh. Jobs live in `src/precompute_jobs.py`.

| Table | Source endpoint / method |
|---|---|
| `dvp_by_position` | FantasyPros DVP scrape |
| `team_special_defenders` | `LeagueDashPlayerStats` (Defense measure) |
| `referee_stats` | Basketball-Reference |
| `team_stats`, `team_foul_rates` | `LeagueDashTeamStats` |
| `dvp_rolling` | Rolling opponent defensive form |
| `player_advanced_stats` | `LeagueDashPlayerBioStats` / advanced splits |
| `player_clutch_stats` | `LeagueDashPlayerClutch` |
| `player_hustle_stats` | `LeagueHustleStatsPlayer` |
| `player_shot_profile` | `LeagueDashPlayerPtShot` |
| `player_play_types` | `SynergyPlayTypes` |
| `player_on_off` | `TeamPlayerOnOffSummary` |
| `player_shot_zones` | `PlayerDashboardByShootingSplits` |
| `player_quarter_splits` | `PlayerDashboardByGameSplits` |
| `player_tracking_stats` | `LeagueDashPtStats` |
| `player_scoring_breakdown` | Scoring breakdown endpoints |
| `player_vs_opponent` | Historical player vs. team splits |
| `player_yoy_stats` | Year-over-year development |
| `team_opp_shot_zones` | Opponent shot zone defense |
| `team_synergy_defense` | Synergy team defensive play types |
| `team_standings` | `LeagueStandingsV3` |
| `team_rest_splits` | Team performance on B2B vs. rested |
| `team_home_away_splits` | `TeamDashboardByGeneralSplits` (home/away DEF_RATING) |
| `team_lineup_stats` | `LeagueDashLineups` (bench strength, continuity) |
| `team_injury_status` | `InjuryTracker` (key players out, impact score) |
| `game_officials` | `BoxScoreSummaryV2` (per-game referee assignment) |
| `prop_line_history` | The Odds API polls (when configured) |

`PrecomputedStore` loads all of the above into memory and refreshes on a TTL (default ~26 hours).

---

## Quick start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional but recommended for training:
pip install xgboost optuna
```

### Configure (optional)

```bash
export ODDS_API_KEY="your_the_odds_api_key"   # enables line movement tracking
export ODDS_POLL_INTERVAL=3600                # poll every 60 min (default)
```

### Run precompute + train

```bash
# 1) Refresh all precomputed datasets
python3 scripts/update_precomputed.py --db basketball_data.db

# 2) Train per-prop models (4 seasons, 300 players)
caffeinate -i python3 -u scripts/train_models.py \
  --seasons 2022-23,2023-24,2024-25,2025-26 \
  --max-players 300 \
  --models-dir models \
  --db basketball_data.db
```

### Start the server

```bash
python3 run.py
# → http://127.0.0.1:5000
```

When `ODDS_API_KEY` is set, a background thread polls prop lines every 60 minutes and a close-line sweeper runs every 5 minutes on game days.

---

## API endpoints (selected)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main UI |
| `/analyze_prop` | POST | Analyze a player prop bet |
| `/search_players` | GET | Player search |
| `/get_player_stats/<id>` | GET | Player stats |
| `/accuracy` | GET | Model accuracy summary |
| `/logs` | GET | Prediction log |
| `/bankroll` | GET/POST | Bankroll & bet tracking |
| `/kelly` | POST | Kelly criterion sizing |
| `/injuries/<team>` | GET | Team injury report |
| `/odds/status` | GET | Odds poller status |
| `/healthz` | GET | Health check |
| `/job_status` | GET | Background precompute/train job status |
| `/train` | POST | Trigger training |
| `/retrain` | POST | Trigger full retrain |

---

## Background jobs

The app can auto-run daily precompute and training via background threads (enabled by default).

| Variable | Default | Description |
|---|---|---|
| `AUTO_JOBS` | `1` | Enable/disable auto jobs |
| `AUTO_JOBS_PRECOMPUTE_HOUR` / `MINUTE` | `5:15` | Daily precompute time |
| `AUTO_JOBS_TRAIN_HOUR` / `MINUTE` | `6:00` | Daily training time |
| `AUTO_JOBS_TRAIN_MODE` | `incremental` | `incremental`, `batch`, or `hybrid` |
| `AUTO_JOBS_TRAIN_SEASONS` | `2023-24,2024-25,2025-26` | Seasons for batch training |
| `AUTO_JOBS_TRAIN_MAX_PLAYERS` | `200` | Max players per training run |

For jobs that run even when Flask is stopped, see `scripts/macos.launchd.com.example.nba-props-analyzer.plist` or `scripts/cron.example.txt`.

### Recommended hybrid setup

- **Daily**: incremental model updates (fast, new games only)
- **Weekly**: full batch retrain with walk-forward AUC/Brier report

```bash
export AUTO_JOBS_TRAIN_MODE=hybrid
export AUTO_JOBS_BATCH_WEEKDAY=0          # Monday
export AUTO_JOBS_BATCH_SEASONS=2021-22,2022-23,2023-24,2024-25,2025-26
```

---

## Project structure

```
├── run.py                      # Flask entry point
├── basketball_data.db          # SQLite (stats, precompute, odds history)
├── models/                     # Trained per-prop joblib artifacts
├── scripts/
│   ├── train_models.py         # Full training pipeline
│   └── update_precomputed.py   # Daily precompute CLI
└── src/
    ├── app.py                  # Flask routes + background jobs
    ├── basketball_betting_helper.py  # Live prop analysis
    ├── models.py               # EnhancedMLPredictor
    ├── ml_features.py          # 408-feature schema
    ├── precompute_jobs.py      # All precompute jobs + DB schema
    ├── precomputed_store.py    # In-memory cache loader
    ├── odds_tracker.py         # The Odds API integration
    ├── injury_tracker.py       # ESPN injury scraping
    ├── arena_data.py           # Arena/travel static data
    └── data_collector.py       # Game log collection
```

---

## Notes

- **Retraining**: Models in `models/` are loaded automatically at startup. Retrain when you add features, refresh precompute data, or want updated season coverage.
- **Multiple model versions**: Train into different directories (`--models-dir models_v2`) to keep versions side by side.
- **Odds features**: The 11 market features need accumulated line history from live polling (or a paid historical odds API). Until then they default to zero at training time and populate gradually at inference.
- **Logs / cache / bytecode** are gitignored (see `.gitignore`).
