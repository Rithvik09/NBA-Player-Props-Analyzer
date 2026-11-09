# NBA Player Props Analyzer

A data-driven NBA player prop betting assistant. It ingests stats from the NBA API, FantasyPros, ESPN, Basketball-Reference, and (optionally) The Odds API, builds a **408-feature** machine learning pipeline, and serves predictions through a Flask web app with logging, grading, bankroll tracking, and line-movement monitoring.

---

## Current model readiness

The app still serves the legacy416-feature models in `models/`. Frozen prospective
studies and newer research candidates are separate; none of the recent research
models has been promoted. Passing feature tests does not establish an advantage
over market probabilities. See `PROSPECTIVE_POINTS_VALIDATION.md` for the dated
foundational audit, comparison evidence and remaining integration gaps.

Invalid core prediction inputs now return unavailable/PASS. Failed model inference
is labeled as a statistical fallback, returns PASS/LOW and is excluded from the
named-model prediction ledger. Nightly automatic retraining is disabled by default;
`NIGHTLY_RETRAIN=1` explicitly enables it. Serving artifact replacement requires a
separate deliberate training/promotion decision.

### Reproducible serving and database verification

Serving now records a content-addressed manifest (artifact and source hashes,
Python/library versions, feature policy and known training cutoffs). Each successful
model prediction stores its exact model inputs and output alongside that manifest
in PostgreSQL, in the same transaction as the prediction log. A failed evidence
write rolls back the prediction and returns `prediction_logged=false`.

Export a versioned manifest before starting a serving process:

```bash
.venv/bin/python -m scripts.serving_manifest --output logs/my_serving_release/manifest.json
.venv/bin/python -m scripts.serving_manifest --verify logs/my_serving_release/manifest.json
APP_BACKGROUND_TASKS=0 SERVING_MANIFEST_PATH=logs/my_serving_release/manifest.json .venv/bin/python run.py
```

`DATABASE_URL` must point to the intended database. `SERVING_MODEL_DIR` optionally
selects another artifact directory; use the same directory with the export command's
`--model-dir`. No artifact is promoted by exporting a manifest. Without a pinned
manifest, startup creates an observed identity; any subsequent artifact/source/runtime
drift fails verification. `GET /serving/provenance` exposes the active identity.
The sibling `.requirements.txt` records the observed inference package versions;
the manifest records Python. Keep the corresponding source revision and artifacts.

Replay a new logged prediction using its exact saved feature vector:

```bash
.venv/bin/python -m scripts.replay_served_prediction 123
```

Replay checks the hashes and exact model result. It does not refetch mutable source
data or certify historical training provenance. Old rows without evidence cannot
be replayed this way. Legacy training cutoffs remain unknown. The compatible
environment below resolves serialized-model version warnings. Confidence labels and
`edge` remain legacy heuristics, not validated price-aware expected returns.

The isolated integration suite uses real Flask, feature construction, loaded models
and PostgreSQL, with canned provider responses and HTTP blocked. Create a dedicated
local database named `codex_serving_test_<suffix>` and set `TEST_DATABASE_URL` to it;
the test setup resets its schema. Never use the application database for these tests.

```bash
APP_BACKGROUND_TASKS=0 .venv/bin/python -m pytest -q tests/test_serving_pipeline.py
```

On 2026-10-27 UTC, 149 targeted app/database/regression checks passed. Evidence is in
`logs/serving_pipeline_verification_01/`, including the HTTP round trip, exact replay,
manifest and frozen-artifact integrity check. Equality with the line now grades as
PUSH, excluded from win/loss correctness. `APP_BACKGROUND_TASKS=0` disables all Flask
startup jobs, but is not a general network sandbox for interactive analysis requests.

### Compatible legacy environment — verified 2026-10-27 UTC

Do not upgrade the existing `.venv`: frozen research continues to use it. A separate
clean environment loaded all 44 legacy joblib artifacts with no warnings using
scikit-learn 1.8.0 and XGBoost 3.2.0, matching their embedded serialization evidence.
The original environment used 1.6.1 and 3.4.1 respectively. The tested Python version
is 3.12.5 on macOS arm64; the original training Python version is unknown.

```bash
python3.12 -m venv .venv-serving-legacy
.venv-serving-legacy/bin/python -m pip install -r requirements.serving-legacy.lock
.venv-serving-legacy/bin/python -m pip check
.venv-serving-legacy/bin/python -m scripts.audit_legacy_runtime --output logs/new_legacy_check.json
```

The audit reads trusted local artifacts, checks their Git custody hashes, and tests
84 predictions across all 14 props. It refuses to overwrite its report. All 84
outputs matched the original environment exactly. The clean environment also
passed 150 app/database tests against the isolated local database, with external
connections blocked and sklearn version warnings treated as errors. A fresh-process
database replay and the previous environment's HTTP prediction matched exactly.
Evidence: `logs/legacy_runtime_recovery_01/`. The complete resolved package lock is
`requirements.serving-legacy.lock`; it leaves the main research lock unchanged.

For a deliberate future serving launch, use this environment with a newly exported
manifest from it. Runtime versions are part of the manifest identity, so an old
environment's manifest must not be reused. This verification did not restart or
deploy the app, change serving artifacts, or modify frozen validation.

**Provenance limit:** all artifact bytes match their last Git commits, but neither
Git dates nor embedded library versions establish training dates or dataset rows.
No exact legacy dataset hash, training cutoff, original Python version or complete
original environment was recovered. These remain explicitly unknown in the audit.
The batch trainer now records supplied-example hashes (including features, targets,
identities and order), outcome-date bounds, package versions, source hashes and
configuration in newly trained metadata. It does not invent history for old models
or certify that source features were available before each game. No retraining was
performed for this compatibility check.

The recovery follows the [scikit-learn persistence guidance](https://scikit-learn.org/stable/model_persistence.html)
and [XGBoost serialization guidance](https://github.com/dmlc/xgboost/blob/master/doc/tutorials/saving_model.rst).

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
| Database | PostgreSQL via psycopg 3 (`DATABASE_URL`, single entry point in `src/db.py`) |
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

### Precomputed tables (PostgreSQL)

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
# Optional, for hyperparameter search during training:
pip install optuna
```

`xgboost` is in `requirements.txt` because the shipped per-prop models in
`models/` are XGBoost boosters — without it they fail to deserialize and the app
silently falls back to the statistical baseline. Watch startup for
`models loaded from disk`; a run of `failed to load per-prop model for <prop>`
lines means the ML pipeline is not actually running.

### Configure

`DATABASE_URL` is required — every module reaches the database through
`src.db.connect()`, which reads it. The crawl still runs locally
(stats.nba.com and ESPN hang on datacenter IPs), so only storage is remote.

```bash
export DATABASE_URL="postgresql://user:password@host/nba_props"
```

Optional:

```bash
export ODDS_API_KEY="your_the_odds_api_key"   # enables line movement tracking
export ODDS_POLL_INTERVAL=3600                # poll every 60 min (default)

# Game-line (moneyline/spread/total) history capture. The Odds API bills
# markets × regions per call, so these govern what the capture spends:
# h2h,spreads,totals across us,eu is 6 credits a snapshot.
export GAME_LINE_SNAPSHOTS=open,close         # which daily snapshots run ("" disables)
export GAME_LINE_OPEN_HOUR=10                 # local hour the "open" snapshot may fire
export ODDS_REGIONS=us,eu                     # capture regions; "eu" is where Pinnacle is
export ODDS_QUOTA_FLOOR=50                    # refuse captures below this many credits

# Public/sharp betting splits from Action Network's scoreboard feed. Free,
# keyless and independent of ODDS_API_KEY, so it runs whether or not an Odds
# API account exists — but it is OFF until you set the flag, so no process
# reaches a live endpoint just by importing the app. Cadence is courtesy,
# not budget. A run with the flag unset says so at startup.
export BETTING_SPLITS_POLL=1                  # off by default; must be enabled explicitly
export BETTING_SPLITS_POLL_INTERVAL=1200      # seconds between polls (20 min default)
export BETTING_SPLITS_USER_AGENT="..."        # overrides the identity we send
```

Every `scripts/*.py` CLI takes `--db <dsn>`; omitting it falls back to
`DATABASE_URL`. Passing a `.db` file path fails fast with a clear error
rather than being handed to libpq as a database name.

### Run precompute + train

```bash
# 1) Refresh all precomputed datasets (--db defaults to $DATABASE_URL)
python3 scripts/update_precomputed.py

# 2) Train per-prop models (4 seasons, 300 players)
caffeinate -i python3 -u scripts/train_models.py \
  --seasons 2022-23,2023-24,2024-25,2025-26 \
  --max-players 300 \
  --models-dir models
```

### Start the server

```bash
python3 run.py
# → http://127.0.0.1:5000
```

When `ODDS_API_KEY` is set, a background thread polls prop lines every 60 minutes and a close-line sweeper runs every 5 minutes on game days. The sweeper also takes the two daily game-line snapshots (`open`, `close`) into `game_line_history` — twice a game day rather than hourly, because on the free 500-credit tier a 6-credit capture every hour would exhaust the month in under a week.

With `BETTING_SPLITS_POLL=1`, a third thread polls Action Network's
public-betting scoreboard every 20 minutes into `betting_splits_history`: per
book and per market, the share of tickets and the share of money on each side.
That endpoint is free, keyless and undocumented, so the poller is gated on no
API key — only on that flag, which must be set explicitly. It shouts at ERROR
whenever a poll parses but yields no splits, because an empty table otherwise
looks identical to a night with no NBA.

### Run the tests

The suite starts each run from an empty schema, which means
`DROP SCHEMA public CASCADE`. It reads `TEST_DATABASE_URL` for that — a
separate variable the application never looks at — so an exported
`DATABASE_URL` pointing at real data cannot be dropped by running the
tests. The target has to have `test` as a word in its name
(`nba_props_test`, not `nba_props_latest`) and live on a local host; the
reset also stands down under `--collect-only`, which editors run on their
own. conftest copies `TEST_DATABASE_URL` into `DATABASE_URL` for the code
under test.

```bash
createdb nba_props_test
TEST_DATABASE_URL=postgresql:///nba_props_test pytest -q
```

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
├── src/db.py                   # the only place that opens a DB connection
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
