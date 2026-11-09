# Expanded points experiment — October 25, 2026

Built **80,264 player-games**, up from 11,548 (6.95×), using cached free NBA data. No paid calls or serving-model promotion.
Final fitting rows increased from 6,894 to 45,244; the rest of the dataset is reserved for development/calibration/evaluation.

## Repairs and feature decisions

- Corrected `usage_rate_with_star_out`: previously, games from before a trade could count as games without the current teammate. Earlier observed membership must now place both players on the same team. Fewer than three qualifying games stays missing.
- Normalized five opponent-matchup means to missing when there were no prior meetings. Counts remain zero; genuine observed zeros remain zero. The same saved candidate applies the normalization at inference.
- PIE is signed, so negative values are valid. [NBA glossary](https://www.nba.com/stats/help/glossary). Corrected the validator units without clipping or changing measurements.
- The 47 negative permutation signals were treated as suspects, not proof of broken calculations. Compared full, previous 183, and pruned alternatives. Compact-core fields remain protected so controls have identical inputs.
- Injury and shooting-percentage capabilities remain eligible; none were globally disabled merely because one permutation was negative.

## Dataset and controls

Expanded eligibility requires ten earlier season appearances and at least 12 mean minutes over the prior 20 appearances. Original top-50 prior-season-minute players are retained for an exactly matched comparison. Rookies can enter after the same prior-history requirements. Current-game minutes, points and future season totals never select rows.

Every original player-game was recovered with identical targets, dates, and compact inputs. The only changed matched source feature is the repaired absent-star usage; point-candidate matchup normalization is applied subsequently. Missingness remains native NaN. Seven unqualified defender features and all source-blocked fields remain excluded.

Each rich profile and the compact baseline get the same three regularization settings and a uniform-versus-quarter-weight choice for added players. A separate original-cohort compact control and an unweighted expanded compact control isolate the data-size effect. All use the same calibration-selection procedure. Feature/model selection uses Jan–Feb 2024 and 2025. Calibration fits Mar–Apr 2025, selects on May–Jun, then refits Mar–Jun. All choices use earlier data; the 2025–26 evaluation is nevertheless previously exposed research, not a fresh blind test.

The expanded cohort broadens the population, so the primary score uses the unchanged original cohort. Expanded-population and prior-minute subgroup scores are secondary. Lines remain prior-derived synthetic half-point lines, not historical bookmaker prices.

## Frozen selection

Candidate classifier: `['selected_pruned', 'expanded', 'regularized']`, 161 columns; calibration `raw`.
Candidate regressor: `['selected_pruned', 'expanded', 'original']`, 161 columns.
Overall development winner: classifier `['selected_pruned', 'expanded', 'regularized']`; regressor `['selected_pruned', 'expanded', 'original']`. Saved separately as `development_selected.joblib`; final outcomes do not change this choice.

## Results

### Matched original cohort: 2,538 player-games

| Model | AUC ↑ | Brier ↓ | Log loss ↓ | Points RMSE ↓ |
|---|---:|---:|---:|---:|
| Compact, original data | 0.62853 | 0.23765 | 0.66796 | 7.7683 |
| Compact, expanded data | 0.63016 | 0.23668 | 0.66588 | 7.7494 |
| Selected 161, expanded data | 0.64292 | 0.23428 | 0.66102 | 7.6890 |

Candidate minus equally tuned expanded compact: log-loss delta **-0.00486**, game-cluster 95% interval [-0.00798, -0.00191]; Brier delta -0.00240, interval [-0.00387, -0.00101].

### Expanded population: 20,357 player-games

| Model | AUC ↑ | Brier ↓ | Log loss ↓ | Points RMSE ↓ |
|---|---:|---:|---:|---:|
| Compact, original data | 0.63073 | 0.23821 | 0.66919 | 6.4918 |
| Compact, expanded data | 0.64145 | 0.23357 | 0.65936 | 6.2371 |
| Selected 161, expanded data | 0.65805 | 0.23003 | 0.65182 | 6.1626 |

Candidate minus equally tuned expanded compact: log-loss delta **-0.00753**, game-cluster 95% interval [-0.00897, -0.00599]; Brier delta -0.00354, interval [-0.00429, -0.00282].

## Interpretation

Meets the predeclared meaningful-improvement research threshold: **True**.

The threshold requires at least 0.002 log-loss improvement on the original cohort, a game-cluster confidence interval excluding zero, no worse Brier or RMSE there, and no worse Brier/log loss on the expanded population. This is not a betting-profit or deployment threshold.

A corrected feature is not necessarily a predictive feature. Inspect the development comparisons and the raw-probability controls in `report.json` before attributing gains to pruning rather than calibration. Correlated player histories, a single fitting seed, repeated research on this evaluation season, and multiple development choices limit the confidence intervals; a prospective real-quote comparison is still required.

## Reproduction and artifacts

- `scripts/build_expanded_points.py`: checkpointed free-data replay; immutable database snapshot and input hashes.
- `scripts/check_expanded_points.py`: original-row identity, compact-input parity, changed-field and range checks.
- `scripts/train_expanded_points.py --prepare` freezes the experiment protocol; running without `--prepare` trains/evaluates it. Development checkpoints support resumption.
- `logs/expanded_points_v4/`: manifest, per-player checkpoints, season datasets, quality and coverage reports. `build_sources/` preserves the exact source hashes used for generation, before the PIE units-only correction; `build_environment.json` records library versions.
- `logs/expanded_points_experiment_v4/`: saved candidates/controls, protocol, frozen selection, manifests/hashes, predictions, metrics, and all-feature dispositions.

Serving models are unchanged. The new source contract is `point_in_time_v4`; existing v3 research artifacts remain separate and are not silently retrained or overwritten.

## Additional verification

The gain persists before calibration: selected raw log loss 0.66102, versus compact 0.66590. The selected feature set retains injury impact, return-from-injury history, true shooting, effective FG%, recent 3P%, and points per touch. Training gain is included in the feature inventory as descriptive evidence, not a causal ablation.

The full regression suite passed 724 tests after the source-contract repair; the final focused suite passed 19 tests, including normalization, weighting, negative-PIE preservation and saved-model parity checks.

original_cohort paired game-cluster RMSE delta: -0.06045, 95% interval [-0.10801, -0.01055].

expanded paired game-cluster RMSE delta: -0.07450, 95% interval [-0.09065, -0.05853].

The calibration library emitted intermediate numerical warnings. Independent elementwise scalar-objective solvers converged and verified the selected compact calibrators; maximum probability difference 0.0000303, below the 0.001 check tolerance. The selected rich candidate uses raw probabilities. See calibration_numeric_check.json.
