

NBA Player Props Analyzer
Welcome to the NBA Player Props Analyzer project! 

This tool is designed to assist sports bettors in making data-driven decisions by leveraging advanced machine learning models, real-time data updates, and comprehensive basketball analytics.


The NBA Player Props Analyzer transforms sports betting from an intuition-driven activity into a data-driven process. Using historical and real-time basketball data, this system predicts player performance and provides betting recommendations tailored to specific games and players.

This project integrates data management, machine learning, and database-driven analytics to create a recommendation engine that goes beyond traditional betting tools.

Key Features
  Real-Time Data Updates: Fetches player and team stats directly from the NBA.com API.
  Predictive Modeling:
  Random Forest Classifier: Predicts whether a player’s stats will exceed a given betting line.
  Random Forest Regressor: Estimates player performance metrics (e.g., points, assists, rebounds).
  Comprehensive Analytics: Incorporates player trends, team dynamics, and matchup specifics for enhanced predictions.
  User-Friendly API: Provides endpoints for querying player stats and generating predictions.
  System Architecture
  Backend: Built with Flask, connecting to a SQLite database for structured data management.
  Database: Relational tables store player, team, and game data for efficient querying.
  Machine Learning: Models are trained on historical game logs, leveraging features like season averages, recent form, and matchup statistics.
  Logging: Tracks system events and ensures reliability during API calls and data processing.


Data Collection and Integration

  Source: NBA.com API (via nba_api library), ESPN (for injury data scraping).
  Data Points:
  Player stats: Points, assists, rebounds, steals, blocks, turnovers, three Pointers Made, field Goal Percentage, three Point Percentage,   
  free Throw Percentage, minutes Played
  Performance Metrics: Historical Data, Team Context, Opposing Team Context, Injuries, Matchup Analysis
  Team metrics: Offensive Rating, Defensive Rating, Wins, Losees, Rest Days
  Database Schema:
  Players table: Stores player information.
  Game Logs table: Tracks performance metrics.


Machine Learning Models

  Random Forest Classifier:
  Binary predictions: “Over” or “Under” a given betting line.
  Features: Recent performance trends, matchup-specific stats.
  
  Random Forest Regressor:
  Continuous predictions: Expected values for player stats.
  Features: Season averages, defensive matchups, team context

## Training “best available” models (offline)
The app can run with a fast heuristic model, but for best accuracy you can train persisted per-prop models and have the app use them automatically.

### 1) Update daily precomputed datasets (DVP + special defenders)

```bash
python3 scripts/update_precomputed.py --db basketball_data.db
```

### 2) Train models
This trains per-prop regressors and saves them into `models/` (plus `models/model_metadata.json`).

```bash
python3 scripts/train_models.py --season 2025-26 --max-players 200 --models-dir models --db basketball_data.db
```

### Runtime behavior
- If trained models exist: `/analyze_prop` uses **trained regressors** (and trained combos for combined props).
- If not: it falls back to the heuristic predictor (still uses all contextual factors and precomputed datasets).

### MODEL_MODE (runtime model selection)
The runtime predictor supports a hybrid fallback path:
- `MODEL_MODE=batch` (default): use **batch-trained models** first; if a prop model is missing, fall back to **incremental SGD** if available; otherwise fall back to heuristics.
- `MODEL_MODE=incremental`: use **incremental SGD** first; then batch-trained models; then heuristics.
- `MODEL_MODE=hybrid`: same as `batch` (batch-first, incremental fallback).

## Job status endpoint
If the Flask app is running, you can check background job state at:
- `GET /job_status` (shows last precompute/train/batch run timestamps + return codes)

## Auto-run daily background jobs (optional)
The app can automatically keep both of these fresh in the background:
- daily precomputed datasets (DVP + special defenders)
- daily model training

This is enabled by default via background threads. You can control it with env vars:
- `AUTO_JOBS=1|0` (default: 1)
- `AUTO_JOBS_PRECOMPUTE_HOUR` / `AUTO_JOBS_PRECOMPUTE_MINUTE` (default: 5:15)
- `AUTO_JOBS_TRAIN_HOUR` / `AUTO_JOBS_TRAIN_MINUTE` (default: 6:00)
- `AUTO_JOBS_TRAIN_MAX_PLAYERS` (default: 200)
- `AUTO_JOBS_TRAIN_MODE=incremental|batch|hybrid` (default: incremental)
- `AUTO_JOBS_TRAIN_SEASONS=2023-24,2024-25,2025-26` (only used when TRAIN_MODE=batch)
- `AUTO_JOBS_BATCH_WEEKLY=1|0` (default: 1 when TRAIN_MODE=hybrid, else 0)
- `AUTO_JOBS_BATCH_WEEKDAY=0..6` (0=Mon .. 6=Sun, default: 0)
- `AUTO_JOBS_BATCH_HOUR` / `AUTO_JOBS_BATCH_MINUTE` (default: 18:00)
- `AUTO_JOBS_BATCH_SEASONS=2021-22,2022-23,2023-24,2024-25,2025-26` (default: AUTO_JOBS_TRAIN_SEASONS)
- `AUTO_JOBS_BATCH_MAX_PLAYERS` (default: AUTO_JOBS_TRAIN_MAX_PLAYERS)
- `AUTO_JOBS_BATCH_MIN_INTERVAL_SECONDS` (default: ~6 days)
- `AUTO_JOBS_MIN_INTERVAL_SECONDS` (default: ~20h safety cooldown)

Notes:
- Jobs run via subprocess (`scripts/update_precomputed.py` and `scripts/train_models.py`) so they don’t block requests.
- For production, a cron job is still a good idea for predictability, but auto-jobs make local/dev “just work”.

### Recommended “hybrid” setup
- **Daily**: incremental model updates (fast, only new games)
- **Weekly**: batch retrain (full refresh + walk-forward AUC/Brier report)

Set:
- `AUTO_JOBS_TRAIN_MODE=hybrid`
- `AUTO_JOBS_TRAIN_HOUR=18` and `AUTO_JOBS_TRAIN_MINUTE=0` (daily incremental at 6pm)
- `AUTO_JOBS_BATCH_WEEKDAY=0` (Monday) and `AUTO_JOBS_BATCH_HOUR=18` `AUTO_JOBS_BATCH_MINUTE=0`
- `AUTO_JOBS_BATCH_SEASONS=2021-22,2022-23,2023-24,2024-25,2025-26`

## Always-on daily jobs (runs even if Flask is NOT running)
If you want daily refresh/training even when your Flask server is stopped, use a system scheduler.

### Option A: macOS `launchd` (recommended on Mac)
1) Copy `scripts/macos.launchd.com.example.nba-props-analyzer.plist` to:
   - `~/Library/LaunchAgents/com.example.nba-props-analyzer.daily-jobs.plist`
2) Edit the plist and replace all `/ABSOLUTE/PATH/...` placeholders with your real paths.
3) Load it:

```bash
launchctl load -w ~/Library/LaunchAgents/com.example.nba-props-analyzer.daily-jobs.plist
```

To unload:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.example.nba-props-analyzer.daily-jobs.plist
```

### Option B: cron (works on many systems)
See `scripts/cron.example.txt` and add an entry via:

```bash
crontab -e
```

### Important note on “20 seasons”
Training saves artifacts into `models/` and those are reused automatically. You only need to retrain when you want to update/refresh models.
If you want to keep multiple “versions” (e.g., 3 seasons vs 20 seasons), train into different folders:
- `--models-dir models_3seasons`
- `--models-dir models_20seasons`
