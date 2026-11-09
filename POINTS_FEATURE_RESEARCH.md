# Points feature investigation — October 25, 2026

The 362-feature model contains real signal, but verified availability does not
establish incremental predictive value. The strongest confirmed correction is
calibration. A new research candidate uses 183 verified features and shallower
trees, retaining injuries and shooting percentages. Serving models are unchanged.

## What the experiments found

- There are 6,894 fitting player-games, not 34,470 independent examples. Five
  synthetic lines per player-game increase classifier rows, not independent
  observations. The sample is the audited top-50 prior-season-minutes cohort,
  not the entire NBA population.
- Among 362 training columns, 96 pairs have absolute Pearson correlation above
  0.95 (at least 500 jointly observed rows). Only one column is missing in more
  than 50% of fitting rows, and none exceeds 90%. Missingness is not the main
  explanation for the aggregate result. Prior-season aggregates can be dense
  yet stale; availability alone does not establish freshness or usefulness.
- Across two development windows, 52 columns have positive permutation signal
  in both, 47 negative in both, 135 mixed, and 128 no positive signal. This is
  diagnostic evidence, not 47 proven harmful features: correlated inputs can
  substitute for each other, and permutations can break natural relationships.
- The old isotonic calibration worsened the full model's final-season log loss
  from 0.66629 to 0.67414 and Brier score from 0.23691 to 0.23874. It assigned
  near-zero over probabilities to 37 examples, four of which went over. Smooth
  sigmoid calibration beat raw and isotonic on a separate calibration-selection
  period and was selected before the final evaluation.

## Which features help or hurt

Individual permutation diagnostics use two seeds in each window, shuffle within
month at player-game level, keep all synthetic lines together, and recompute the
line-relative inputs. Larger positive loss increases indicate useful reliance.
The complete 362-row inventory is in
`logs/points_feature_investigation_v1/feature_verdicts.csv`; raw fold diagnostics
and retrained group ablations are alongside it.

Consistently helpful examples include season average, recent average, season
minutes, effective field-goal percentage, recent three-point percentage, points
per touch, team key players out, opponent injury impact, lineup continuity, and
games in the last seven days. Season average is the largest measured individual
signal. The candidate retains the shooting-percentage and injury examples.

Consistently negative permutation examples include opponent pick-and-roll PPP
allowed, usage with a star out, previous-season average against this opponent,
scoring-efficiency trend, and percentage of points from threes. These remain
research suspects rather than universally disabled fields. Group-based selection
can retain an individually negative field when its group helps; this experiment
does not claim a minimal or individually optimal 183-column set.

Retrained group ablations give stronger context:

- Adding clutch/quarter features to the compact classifier improved log loss in
  both windows. However, that 33-column combination did not win the subsequent
  joint feature-set/regularization comparison.
- Adding box-score extras, injuries, lagged player statistics, play types,
  scoring/play-by-play, and tracking/hustle each improved points RMSE in both
  windows. Their union plus the compact core gives 183 columns.
- Removing the broad team-matchup group from the full classifier improved log
  loss in both windows. Adding rotation features worsened RMSE in both windows.
- Shooting-zone/profile and schedule-context groups had mixed results. This
  does not mean shooting percentages are unhelpful: those live in the box-score
  group and showed positive individual signal.
- Interactions matter: the lagged-player group hurt classifier log loss alone,
  yet the combined 183-column profile with shallower trees won development.

Groups are operational source-based categories, not clean causal interventions.
Exact membership is frozen in the investigation protocol.

## New candidate and validation

`logs/selected_points_v2/candidate.joblib` freezes 183 columns for both outputs,
XGBoost depth 2, 250 trees, learning rate 0.04, minimum child weight 30 and L2
penalty 30. Defender and source-blocked features remain excluded.

Selection used two historical folds: fit before January 2024 or January 2025,
validate on January–February of that year. Four feature profiles and three
regularization settings were compared on mean log loss / RMSE. Because feature
groups were also selected on these folds, development scores are optimistic.
Calibration was fit on March–April 2025 and selected on May–June, then the chosen
sigmoid was refit on March–June. All decisions were written before evaluating
2025–26. That season was already exposed in earlier research, so these results
are retrospective, not a fresh blind test.

| Metric | Compact 16 v1 | Full 362 v1 | Selected 183 v2 |
|---|---:|---:|---:|
| AUC | 0.62460 | 0.62992 | 0.63237 |
| Brier (lower better) | 0.23847 | 0.23874 | 0.23690 |
| Log loss (lower better) | 0.67703 | 0.67414 | 0.66634 |
| Points RMSE | 7.79415 | 7.76659 | 7.74322 |
| Points MAE | 6.06264 | 6.04905 | 6.04004 |

Paired NBA-game bootstrap, keeping all players/lines together, gives new-minus-
full-v1 log-loss delta -0.00779, 95% interval [-0.01699, -0.00120], and Brier delta
-0.00184, interval [-0.00358, -0.00016]. These intervals do not account for all
research-selection uncertainty or repeated players across games.

The essential control: against the **uncalibrated** full v1, new log loss is
0.000056 worse and Brier just 0.000009 better, with both intervals spanning zero.
Against the uncalibrated compact model, AUC improves by 0.00780, but its interval
[-0.00387, 0.01772] spans zero. There is no established large feature-selection
advantage. Most demonstrated probability improvement repairs calibration; the
smaller model preserves signal with modestly better point-error measurements.
Do not claim betting profit from synthetic lines or promote automatically.

## Artifacts and checks

- `logs/points_feature_investigation_v1/`: all feature diagnostics, ablations,
  source grouping, input hashes and development-only group selection.
- `logs/selected_points_v2/`: model, frozen manifest, development results,
  evaluation predictions/report, calibration-control comparison and hashes.
- `scripts/investigate_points_features.py` and `scripts/train_selected_points.py`
  reproduce training in fresh output directories and refuse overwrites.
- 29 relevant tests passed, covering temporal separation, defender exclusions,
  candidate column isolation, calibration reload/extreme inputs, and clustered
  comparison direction, plus existing points/audit tests.

Next evidence should come from prospective real quoted lines and a broader
point-in-time training cohort. Keep collecting free forecasts and retain the
current qualification gates; do not tune further against this same final season.
