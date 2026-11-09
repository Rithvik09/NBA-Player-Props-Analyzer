## Expanded points validation — October 25, 2026

The v4 source repair and expanded free-data replay are complete: 80,264 player-games,
including 45,244 fitting rows (previously 6,894). Development selected 161 features,
retaining injury and shooting signals. Against the equally tuned expanded compact
baseline on the original 2,538 evaluation player-games, AUC improved 0.63016 →
0.64292 and log loss 0.66588 → 0.66102; the game-cluster intervals exclude zero.
Points RMSE improved 7.7494 → 7.6890. This passes the predefined research threshold,
with the limits of a previously exposed season and synthetic lines. Serving models
are unchanged; the seven defender features remain excluded.

See [POINTS_EXPANDED_RESEARCH.md](POINTS_EXPANDED_RESEARCH.md) for the full comparison,
repairs, uncertainty, feature dispositions and artifacts. The source contract is
now `point_in_time_v4`; the earlier v3 audit and candidate files are preserved.

## Feature utility follow-up — October 25, 2026

The development-selected points candidate now uses 183 verified features and
sigmoid calibration. It preserves injuries and shooting percentages. See
[POINTS_FEATURE_RESEARCH.md](POINTS_FEATURE_RESEARCH.md) for all-feature diagnostics,
group ablations, results, and the calibration control showing that most of the
probability gain comes from repairing calibration. Serving models are unchanged.

## Verified-only points candidate — October 25, 2026

Training has now run on the cached audited points dataset using 362 active
features. The seven defender fields, 14 source-blocked fields and 33 duplicate
slots are excluded from the fitted matrices. The saved candidate freezes its
column list, separate from the global feature registry. Existing serving models
are unchanged; no other prop model was retrained in this first run.

Artifacts: `logs/verified_points_v1/verified.joblib`, `compact_baseline.joblib`,
`manifest.json`, `report.json`, and `evaluation_predictions.csv`.
Reproduce in a new output directory with
`python -m scripts.train_verified_points --output logs/verified_points_new_run`.
The command refuses to overwrite an existing experiment.

The chronological split has 6,894 fitting player-games before January 2025,
2,116 calibration player-games in January–June 2025, and 2,538 evaluation
player-games in 2025–26. The evaluation season has been exposed in previous
research; this is a retrospective comparison, not a new blind test. Five
prior-only synthetic half-point lines per game are used, not historical odds.

| Metric | 362-feature candidate | 16-feature compact baseline |
|---|---:|---:|
| Calibrated AUC | 0.6299 | 0.6246 |
| Calibrated Brier (lower better) | 0.23874 | 0.23847 |
| Calibrated log loss (lower better) | 0.67414 | 0.67703 |
| Points RMSE | 7.7666 | 7.7941 |

The paired NBA-game bootstrap AUC-difference interval is [-0.0070, +0.0158].
Predictive improvement is not established; calibration Brier is slightly worse.
The candidate is not promoted. Source verification alone is not a utility test.

When a deferred feature group qualifies, create a new manifest/profile, collect
sufficient point-in-time training rows, retrain a separate candidate and compare
on matched chronological data. Never add a column to an existing fitted model
or backdate current qualification into past training rows. The existing defender
monitor now includes this training follow-up after qualification and adequate
coverage, with serving models kept unchanged.

---

> Current-season follow-up: the broader assignment ranker fails the unchanged gate on 36 new games (22/45 correct, 48.9%, below 100 required forecasts). It remains shadow-only. A separate cross-season roster-history fix improves historical opening-week assignment eligibility from 27 to 46. See SOURCE_BLOCKERS.md.

> October 25 update: the season-specific defender rule now passes the existing gate for 2023–24 through 2025–26, but fails for 2022–23 and current 2026–27. Four market capabilities have verified free live snapshots; historical training coverage remains blocked. See SOURCE_BLOCKERS.md for current results; older audit findings below describe earlier evidence.

# Feature contract for new training

The `point_in_time_v3` contract reviews all 416 ordered numeric columns. There
are 369 active columns, 33 duplicates/covered expressions, and 14 distinct
capabilities blocked on credible sources. Of the original 162 disabled columns,
115 now have implemented, sourced definitions. Schema order is unchanged.

Every decision, source, unit and definition lives in `src/audited_features.py`
and `src/recovered_features.py`. The full measured ledger is
`logs/full_feature_audit/features.csv`; `features.md` is the readable version.
Do not treat active as proof of predictive value or complete source coverage.

## Recovered capabilities

- Timestamped official injury reports: key absences, opponent lost-minute
  impact, actual usage when an absent star previously missed games, and medical
  return history. Reports after the prediction cutoff are excluded. Unknown
  identities, missing reports and insufficient comparable appearances remain
  missing, never healthy zero. The archive includes multiple publication hours
  and both hour-only and minute-based official filename conventions.
- Actual opponent shooting-zone percentages, player-zone interactions, defensive
  play-type efficiencies, league baselines, contested rebounds, touches, fouls
  drawn, clutch usage and early/late shot-clock frequencies. These use completed
  previous-season snapshots as explicitly lagged priors.
- Validated play-by-play scoring states, shot distances, and-one opportunities,
  substitution timing, garbage-time exposure and starting-unit exposure. Scoring
  and shots are checked against NBA boxscores; stint features require all player
  minutes to agree with the precision of the recorded source. Missing shot
  distances are not replaced with zero.
- Actual five-player lineups, prior starting-unit ratings, rotation continuity,
  position defense for players with known prior-season position membership,
  previous-season head-to-head, rest-conditioned opponent
  performance, schedule standings, arena history and broadcast metadata.
- Playoff/play-in flags and prior series state. Player history includes regular
  season and postseason; no current or future outcome enters the feature.
- Separately qualified prior-season game-script probability forecasters. Failed
  prior-period calibration/discrimination gates leave probabilities missing.
  These are feature sources, not new player-prop serving models.

The recovery uses five seasons of source snapshots, six seasons of player logs,
actual postseason logs, archived reports and independently validated NBA/ESPN
play-by-play. Opponent-zone endpoints and capped lineup downloads were repaired.
The initial scoring fixes preserve actual midrange, unassisted, fast-break and
turnover scoring fields. Snapshot collection time does not establish historical
publication time; prior-season snapshots are a modeling assumption.

## Missingness and shared behavior

Training and audited serving call the same builder. Player windows contain at
most 20 strictly earlier appearances; team histories also include games the
player missed. Target-date and future results are excluded. Postseason team
rates use explicitly estimated possessions from complete team boxes where
advanced rates are unavailable.

Unavailable active measurements remain native NaN through both regression and
classifier vectors and model alignment. Excluded schema slots stay zero for
compatibility. Zero is therefore a legitimate observed value only when its
source is present. The classifier's actual decision line remains the canonical
`line` input; the duplicate `current_line` slot stays excluded.

## What remains blocked

Seven defender fields now use actual matchup observations and a qualified,
prior-only expected-defender rule. This is a forecast, not a confirmed upcoming
assignment, and remains missing for unqualified seasons or sparse pairings.
The original rule failed on the expanded archive. The October 25 rule-selection
repair qualifies 2023–24 through 2025–26 at 52–54% assignment accuracy;
2022–23 and current 2026–27 remain below the unchanged 50% gate.
Historical coverage remains sparse and does not prove predictive improvement.
The switching field remains blocked: all 1,221,376 validated source rows report zero
for `switchesOn`, which does not establish a populated switching feed.
Three model-performance fields require logged
out-of-sample predictions and subsequent outcomes. One average shot-clock field
requires actual event-level times; bucket shares do not identify a mean. Nine
market fields require timestamped, decision-time game/prop quotes and objective
market definitions. Four now have verified free live observations, but lack
historical training coverage. These are distinct capabilities, not dismissed as unimportant.
The exact 14 reasons and all 33 duplicate explanations are in the registry and
activation ledger. No paid feed was purchased or synthetic market history used.

## Reproduce and inspect

With `DATABASE_URL` set to the local NBA database:

```sh
.venv/bin/python -m scripts.audit_all_features
```

The audit selects the top 50 players using previous-season minutes and replays
2022–23 through 2025–26. It writes per-feature and per-season coverage, ranges,
missing counts, constants, source/code fingerprints and versioned input datasets
under `logs/full_feature_audit`. The report asserts stable file inputs during
replay and unchanged serving-model hashes. Coverage measures this cohort, not
all NBA players. See `FEATURE_ACTIVATION.md` for the 162-column disposition.

## Training and serving boundaries

`scripts/train_models.py` defaults to the audited builder and writes candidates
to `logs/audited_models`. Missing dated season caches fail explicitly. Examples
and model metadata carry the policy; mixed-policy training is rejected.
Metadata opts candidate models into the matching serving adapter. Audited
models reject untagged legacy features and preserve native missingness.

Existing deployed player-prop models are untouched and remain on their legacy
contract until separately trained, evaluated and promoted. The old
`EnhancedMLPredictor.train` and incremental SGD paths remain legacy and must not
produce v2 artifacts. A fresh current-season cache is a deployment prerequisite;
this historical audit does not certify current roster/injury freshness. Live
injury observations are archived separately; the historical replay currently
uses the official PDF archive.

Feature recovery establishes a valid input path. Player-prop accuracy,
calibration, incremental feature value and profitability remain unmeasured for
this contract and require independent evaluation before promotion.

## Source-blocker follow-up

See `SOURCE_BLOCKERS.md` for the matchup-source validation, partial qualification
coverage, unpopulated switching field, and future prediction/market provenance
repairs. Source-qualified defender forecasts do not establish player-prop gains.
The former v2 report is retained as `logs/full_feature_audit/report_v2.json`.
