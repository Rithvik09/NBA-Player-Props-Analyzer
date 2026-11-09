## Neutral shot-volume correction — 2026-10-27

Built and tested a separate correction in `logs/points_neutral_volume_01/` using
existing real quotes only. Original candidates, serving models and frozen
prospective validation remain unchanged. No credits spent or deployment.

The new model learns only a nonnegative standardized prior-shot-volume coefficient
with fixed ridge1 (bounded0..2). For each training-history cohort, its normalization
offset is solved using market probabilities and features, never outcome frequency,
so weighted mean corrected probability equals weighted mean market probability.
This prevents the prior model's learned blanket under adjustment. It does not
promise unbiased predictions after distribution shift. Maximum observed fitting
neutrality error across all walk-forward fits was1.1e-16. The fixed coefficient
was0.00413695. Average comparison probability correction shrank from the old
-0.752 percentage points to+0.011 points (fixed) /+0.008 points (walk-forward).

Short-history diagnostic coverage is now487 player-games/41 independent games,
with fitting restricted to earlier calendar dates and unchanged minimum100-row /
20-game gates per history cohort. Eligible-history coverage is245/20; aggregate
732/52. The fixed model uses the original1,016 pre-July2025 rows. The adaptive arm
refits before each of19 validation days, never using same-day outcomes. Earlier
validation outcomes may enter subsequent adaptive fits, as explicitly declared.
Both alternatives were fixed before this experiment's scoring; neither was selected
or promoted from the comparison. All2025-26 outcomes were previously exposed.
This repairs diagnostic coverage, not independent holdout availability, and does
not supply a fully supported independent three-way chronology.

| Model | All log loss | Short-history | Eligible-history |
|---|---:|---:|---:|
| Market |0.690973|0.691412|0.690101|
| Fixed neutral volume |0.690849|0.691383|0.689787|
| Walk-forward neutral volume |0.690819|0.691402|0.689661|
| Original traceable candidate |0.691598|0.692008|0.690785|

Game-cluster95% log-loss delta intervals versus market:
- Fixed overall:-0.000124, interval[-0.000299,+0.000040].
- Walk-forward overall:-0.000153, interval[-0.000371,+0.000057].
- Fixed short-history:-0.000028, interval[-0.000234,+0.000179].
- Walk-forward short-history:-0.000009, interval[-0.000234,+0.000206].
- Fixed eligible-history:-0.000314, interval[-0.000541,-0.000067].
- Walk-forward eligible-history:-0.000440, interval[-0.000860,-0.000029].

Shot volume shows a small exploratory benefit concentrated in longer-history
players. Overall and short-history intervals include no gain; the favorable
longer-history subgroup remains exposed-data research after multiple prior
analyses, not independent proof. Keep market as the reference and preserve frozen
validation; no evidence-based promotion established.

Seven focused tests passed, including severely under-heavy targets retaining
market neutrality, missing-data rejection and unchanged coverage gates. Exact
fresh-process refits reproduced the fixed model and all19 walk-forward forecast
blocks using preserved source and environment. Inputs, row indices per fold,
parameters, source snapshot, package lock, predictions and clustered metrics are
saved with hashes. Reproduce:

```bash
.venv-serving-legacy/bin/python -m scripts.replay_neutral_volume --output logs/points_neutral_volume_01
```

## Traceable correction diagnosis — 2026-10-27

Completed offline diagnosis in `logs/points_traceable_diagnosis_03/`. Frozen
validation and serving artifacts were hash-checked unchanged. No credits spent.
The original traceable candidate is preserved; no alternate candidate promoted.

Exact Shapley decomposition of the five fitted logit terms attributes the
candidate-minus-market log-loss gap as follows. Positive is harmful on the exposed
comparison; these are descriptive contributions, not independently established
feature effects:

| Term | Log-loss contribution |
|---|---:|
| Intercept (under bias) |+0.00040647|
| Scoring edge |+0.00010078|
| Shot volume |-0.00011940|
| History |+0.00004278|
| Rest |+0.00019471|

The intercept contributes65% of the net+0.00062534 gap. Training over frequency
was46.95% (46.25% under candidate weights), versus market49.85% (weighted49.46%).
Comparison over frequency was51.23% against market50.02%. The learned downward
correction transfers the earlier outcome imbalance into a later sample with a
reversed residual. Short-history share also shifts from21.36% to66.53%. Rest and
edge terms hurt both comparison cohorts. Shot volume helps both; history hurts
short-history but helps eligible-history. None of this establishes causality.

Seven fixed fits were declared before scoring: full, no-intercept, edge-only,
edge+rest, intercept-only, uniform-weight full, and stronger ridge4. All used
existing real quotes and training-only preprocessing.

| Arm | Comparison log loss |
|---|---:|
| Market |0.690973|
| Edge only |0.691075|
| Stronger ridge4 |0.691126|
| No intercept |0.691179|
| Edge+rest |0.691266|
| Intercept only |0.691366|
| Uniform-weight full |0.691392|
| Original/full |0.691598|

Simplifying/shrinking improves this diagnostic loss but no arm beats market.
The stronger-ridge delta versus market is+0.000153 with game-cluster95% interval
[-0.000196,+0.000496]. Do not choose edge-only based on these already-exposed
comparison results.

Chronological development used only pre-July2025 outcomes. The proposed first
fold had119 rows/8 fitting games and252 rows/18 validation games, below the
unchanged100-row/20-game gate. It was not fitted. The supported Jan–Jun2025
validation contains645 rows/40 games but only2 short-history rows/2 games.
Uniform-weight full had the smallest development loss; its delta versus market
was-0.000582 with interval[-0.001348,+0.000192], crossing zero. It also failed the
short-history support gate. The predefined rule therefore retains market; no
new correction is selected. The original failed run is preserved under `_01`;
`_02` exposed numerical backend warnings, and `_03` uses explicit contractions
with RuntimeWarnings fatal. All8 focused tests passed, including attribution
additivity, coefficient exclusion and line monotonicity.

## Traceable real-line replacement experiment — 2026-10-27 UTC

Trained a separate points-probability candidate in `.venv-serving-legacy`, using
only existing real quotes. The candidate is a four-input market-logit correction
plus intercept, fixed mean-loss ridge1, balanced history/game weights. A direct
compact control (existing19-column history/missingness contract, fixed sum-loss
ridge100) was retrained on identical rows. No new tuning or synthetic-line training.
This is not a replacement of all14 production prop models.

Training:1,016 unique player-games /66 games, Jan5,2024–Jun8,2025. Comparison:732
unique player-games /52 games after July1,2025. Short-history support:217/20 in
training and487/41 in comparison (rows/games). Eligible-history support:799/51
and245/20. Games do not overlap training/comparison; no protected2026-27 rows.
These historical outcomes were already exposed, so this is development evidence.

| Model | Log loss | Brier | AUC |
|---|---:|---:|---:|
| Market |0.690973|0.248917|0.530932|
| Traceable candidate |0.691598|0.249229|0.531339|
| 50/50 |0.693147|0.250000|0.500000|
| Retrained real-line compact |0.699727|0.253261|0.484430|
| Cached frozen compact |0.746087|0.273752|0.490499|

Candidate-minus-market log-loss delta:+0.000625; game-cluster95% interval
[-0.000545,+0.001785]. Candidate-minus-retrained-compact:-0.008129,
interval[-0.013008,-0.003111]. Thus the candidate improves over compact on this
reused comparison but does not establish incremental gain over market. No promotion.
Both fitted models passed14,640 fixed-market line-monotonicity comparisons each.

`logs/points_traceable_replacement_01/` preserves the dated plan, complete input
bundle, row assignments and quotes, training-example hash and outcome dates,
weights, exact source snapshot, package lock, model parameters/artifacts,
predictions, cohort metrics and clustered intervals. The frozen compact comparator
uses previously saved predictions joined1:1 by NBA game/player, checked against
matching dataset hash, labels and market probabilities; it was not reloaded under
a different library version or modified. New candidate and compact predictions
were reproduced exactly, including a fresh-process refit using the saved source
snapshot.10 relevant regression checks passed. No credits spent; serving and
frozen validation artifacts/source hashes remain unchanged.

Reproduce with:

```bash
.venv-serving-legacy/bin/python -m scripts.replay_traceable_points --output logs/points_traceable_replacement_01 --refit
```

The replay verifies the Python version, package lock and archived file hashes
before using the preserved source. Future experiments must use a new output
folder. Keeping this candidate separate preserves both frozen prospective studies.

## Legacy runtime recovery — 2026-10-27 UTC

The separate `.venv-serving-legacy` uses the artifact-recorded scikit-learn 1.8.0
and XGBoost 3.2.0, with all resolved dependencies in
`requirements.serving-legacy.lock`. All 44 artifacts loaded without warnings;
84 predictions across 14 props matched the old environment exactly. The clean
app/database suite passed 150 checks, with sklearn mismatch warnings fatal.
Fresh-process replay and the previous HTTP inference result matched exactly.
Evidence: `logs/legacy_runtime_recovery_01/`.

Frozen validation, serving artifacts and the original `.venv` are unchanged;
no credits, retraining or deployment. Artifact Git hashes and serialization
versions were recovered; exact legacy training rows/cutoff and original Python
remain unknown. Batch training now records actual supplied-example fingerprints,
date bounds, source hashes, configuration and packages for future runs. This
adds provenance recording, not retrospective certification or predictive gain.

## Serving pipeline verification — 2026-10-27 UTC

The application serving path now carries observed model/source/runtime manifests and
stores exact feature/result evidence atomically with each model prediction. New
scripts `serving_manifest` and `replay_served_prediction` export/check identities
and replay saved inputs. README documents setup and the isolated database test.
No research candidate was promoted, no model artifact was replaced, no service was
restarted, and no credits were spent. Frozen v1/v5 and defender hashes passed.

149 targeted checks passed with external connections blocked and a disposable local
PostgreSQL database. The real Flask/helper/feature/model/database flow passed;
provider inputs were canned fixtures, not a live feed validation. Tests cover exact
replay, PUSH grading, failed inference not entering the model ledger, malformed
input rejection, global background-job shutdown and atomic rollback on evidence
failure. Fresh-process replay is recorded separately. Evidence:
`logs/serving_pipeline_verification_01/`.

This is serving reproducibility, not evidence of improved predictive performance.
Legacy training cutoffs remain unknown; observed runtime pins do not recover the
original training environment. Legacy serialized-model compatibility warnings and
price-independent recommendation/confidence heuristics remain documented limits.

# Prospective points validation

## Foundational audit and serving failures — October 26, 2026

The investigation now distinguishes three different systems: the legacy app
serving path (`EnhancedMLPredictor`, models/), frozen v1/v5 prospective studies,
and offline research candidates. The app still uses416-feature legacy artifacts,
model_version1, with no feature-policy or training-cutoff metadata. Research
improvements have intentionally not been promoted, so app behavior is not a test
of those research fixes. The app also blends classifier output with a regression/
Gaussian estimate using hand-set weights; research metrics do not certify that
final served probability. HIGH/MEDIUM labels use distance-from-neutral heuristics,
not demonstrated current-season predictive reliability.

### Ground truth and construction checks

All1,748 historical real-line rows were independently matched to cached official
NBA player/game boxscores: zero missing identities, zero points/date mismatches,
no duplicate player-games, no protected prospective-season rows, no pushes, and
all binary labels exactly equal points>line. Provider IDs differ from NBA IDs;
existing qualified importer mappings were used, not naïve namespace equality.
Existing quote/provenance validation passed. This is not a new independent audit
of every provider timestamp: the previously corroborated timestamp importer
remains the source of that qualification.

Broader regression checks passed92 tests covering feature defaults and formulas,
future-outcome exclusion, prior-season snapshot selection, recovery/injury timing,
calibration precision, deduplication/monotonic training rules, prospective guards
and defender monitoring. This does not establish predictive usefulness. Earlier
controlled comparisons still find no demonstrated improvement over market, and
repeatedly exposed historical outcomes remain exploratory.

### Reproduced implementation bugs, now fixed locally

1. Missing variance produced NaN probability, then max/min clamping converted it
   to1.0. Missing mean could additionally leave predicted points nonfinite.
   Core inputs and model/blended outputs now receive explicit finite checks.
   Invalid inputs return unavailable/PASS with finite compatibility values; the
   UI hides those placeholder numbers rather than presenting a50% forecast.
2. Injected classifier failure returned LEAN UNDER/MEDIUM with no error or mode
   indication. Failure now returns explicit statistical_fallback, PASS/LOW and
   a sanitized error-type reason. Valid ML predictions keep the existing blend.
   Response status is propagated through the helper to the UI. Failed/unavailable
   inference is not inserted into the named-model prediction ledger.
3. App import unconditionally started a4AM scheduler that could overwrite serving
   models and recalibrate thresholds. Nightly training is now off by default;
   NIGHTLY_RETRAIN=1 is explicit opt-in. Manual training paths still exist; this
   gate prevents automatic replacement, not all possible deliberate replacement.

Evidence before/after: `logs/points_foundations_audit_02/report.json` and
`logs/points_foundations_audit_03/report.json`. The initial `_01` output failed
serialization of a NumPy integer and is not the completed report. Actual artifacts
load for all14 props. A24-case cached-feature serving replay found zero increasing
probabilities across10.5/20.5/30.5 lines; this is a narrow predict() contract check,
not a full live feature-builder parity test or a performance evaluation.

### Remaining prerequisites before claiming the app works reliably

- Establish one explicit serving-model manifest with source/feature policy,
  training cutoff and dependency versions, then replay the exact final served
  probability. Do not attach a new manifest to old artifacts without evidence.
- Artifact loading emits sklearn1.8→1.6.1 serialization warnings and XGBoost
  serialization warnings. Loading succeeds, but cross-version compatibility is
  not certified. Do not change this shared environment underneath frozen studies;
  use an isolated, pinned environment for a future model build/replay.
- Audit recommendation economics separately: `/analyze_prop` accepts a line but
  no offered odds, and `edge` is predicted-points distance/line, not expected
  betting return. OVER/UNDER recommendations are therefore not proof of value
  at a particular price. This was not silently rewritten during the audit.
- Retain market/no-correction as the research benchmark; no tested candidate
  justifies promotion. Preserve prospective monitoring to accumulate fresh
  evidence. Short-history selection remains under-supported.
- Full live HTTP/database integration and all14-prop statistical validation were
  not performed here. No app startup or external data call was needed; this audit
  covers offline data truth, points inference, source inspection and regression
  safeguards. The seven deferred defender features remain unqualified.

Four new serving/scheduler tests pass, plus92 broader checks; Node syntax and
DOM-stub checks cover unavailable/fallback UI branches. Python compilation and
whitespace checks pass. All frozen study source/model/protocol and serving model
artifact hashes remain unchanged. Local serving code has the fixes above; no
running service was restarted, candidate deployed, credits spent or data collected.
`completion.json` in the after-audit directory records scope, evidence and gaps.

## History-available simple correction controls — October 26, 2026

Completed six predeclared offline arms in `logs/points_simple_correction_01/`.
This repairs the research model's input-availability mismatch and controls the
history-group imbalance; it does not manufacture missing historical evidence or
resolve the two-case short-history selection shortage.

The new separate helper `src/points_simple_correction.py` uses four inputs:
prior scoring edge attenuated by n/(n+10), prior FGA per appearance, log(1+n), and
rest days; n is the existing available appearance count capped at20. The scoring
edge is (.7*recent_avg+.3*season_avg-line). Attenuation constants were fixed before
fitting, not selected on comparison outcomes. All four fields are finite for all
existing rows, including one-appearance cases. There is no imputation or use of
unavailable five/seven/fourteen-game features; missing required inputs fail closed.
Availability does not imply precise estimates or complete underlying history.

Each model has five coefficients including intercept, added to the market logit.
Standardization uses training weights only. The scoring-edge coefficient is
nonnegative, so over probability cannot increase as the line rises with market
probability held fixed. Intercept and all coefficients are penalized.

Two fitting modes are compared at fixed mean-loss ridge penalties0.25,1,4:
- Balanced pooled:50% objective weight for each history group, equal game weight
  within each group; no weights learned from the comparison population.
- Separate: different short/eligible models, each with equal game weights within
  its own cohort. Short data therefore cannot be dominated by longer histories.

Training remains the1,016 pre-July2025 real player-games:217 short-history rows
across20 games and799 eligible rows across51 games (66 distinct games in union).
Comparison is the same732 later rows across52 games:487 short/41 games and245
eligible/20 games. These are previously exposed historical outcomes. The earlier
selection period still contains only two short-history cases, so no new model or
penalty is selected; all six results are reported and market remains the fallback.

| Arm | Overall log loss | Short-history log loss | Eligible log loss |
|---|---:|---:|---:|
|Market|0.69097|0.69141|0.69010|
|Balanced, penalty0.25|0.69306|0.69342|0.69235|
|Separate, penalty0.25|0.69524|0.69565|0.69441|
|Balanced, penalty1|0.69160|0.69201|0.69078|
|Separate, penalty1|0.69207|0.69232|0.69158|
|Balanced, penalty4|0.69113|0.69155|0.69027|
|Separate, penalty4|0.69123|0.69159|0.69050|
|Previous compact correction|0.69744|0.69704|0.69825|
|Existing research compact|0.69703|0.69461|0.70185|
|Constant0.5|0.69315|0.69315|0.69315|

All six improve pooled loss versus the previous correction; none beats market
on pooled or cohort log loss. Strongest balanced shrinkage differs from market by
only +0.000153, paired game-cluster95% interval[-0.000196,+0.000496]. Its difference
versus prior correction is -0.006315 [-0.010795,-0.001869], but this is not evidence
of an edge over market. Mean absolute probability correction is just0.235 percentage
points (short0.258, eligible0.189), compared with3.209 points previously. Overall
AUC is0.53059 versus market0.53093. The benefit is largely consistent with avoiding
unreliable adjustments; simplification, weighting and shrinkage were bundled, so
this study does not causally attribute the improvement to one change alone.
Separate models do not outperform balanced models at any matched penalty on pooled
loss, consistent with the limited early-season evidence remaining a constraint.

No promotion or retrospective winner selection. Six focused tests pass; all six
arms pass14,640 adjacent-line checks each. Source/input hashes, existing correction
models, and frozen v1/v5/defender/serving hashes verified unchanged. No collection,
credits or prospective-record changes. Evidence is plan.json, report.json,
predictions.csv, six model-group files and integrity.json in the experiment folder.

## Why real-market corrections degrade predictions — October 26, 2026

Completed offline diagnostics in `logs/points_market_diagnosis_01/`, preserving
all previous models and comparisons. The evidence points primarily to overfitting
weak incremental signal with limited, seasonally uneven data. Feature coverage
shift compounds that problem; this is not proof that every feature is invalid or
that collecting any arbitrary additional games will fix it.

### 1. Direct evidence of overfitting

On the initial 371-row/26-game fit, compact ridge100 improves market log loss by
0.01645 but worsens the subsequent selection loss by 0.01258. Reducing regularization
to ridge1 increases the apparent training improvement to 0.03406 while increasing
the selection penalty to 0.06110. On the final 1,016-row refit, the same tradeoff
persists: ridge100 training improvement 0.00496 versus comparison penalty 0.00647;
ridge1 training improvement 0.00848 versus comparison penalty 0.01147.

Thirty whole-game bootstrap refits (including refitting preprocessing) produce a
median prediction standard deviation of 3.50 percentage points, larger than the
original median absolute correction of 2.87 points. In 95.8% of comparison rows,
the empirical 2.5–97.5% correction range crosses zero. This small ensemble is a
stability diagnostic, not a calibrated confidence interval for a selected strategy.

Thirty within-training-game label shuffles preserve each game's over rate while
breaking player-feature/outcome pairing. Median apparent training improvement is
0.00681; actual improvement is only 0.00496. Twenty-five of 30 shuffled fits achieve
at least that training gain. These shuffles are not a full market-conditional null
and the fraction is not a formal p-value, but they demonstrate that an improvement
of this size can readily arise from fitting unstable associations.

### 2. Limited independent games and changing population

Final training has 1,016 player-games but only 66 NBA game clusters. The shooting
model has 71 coefficient slots, 45 varying design columns and matrix rank 38;
these counts include redundant and constant columns, not 71 independent predictors.
The effective information is substantially less than 1,016 independent outcomes.
Initial selection has only two short-history cases, so separate early-season
correction selection remains unsupported.

Short-history share is 58.0% in the initial fit, 0.3% in selection, 21.4% in the
final combined fitting period and 66.5% in comparison. Consequently neither the
selection population nor final fitting population closely represents the later
short-history-heavy comparison.

A chronological training-prefix probe (same ridge100, same comparison games):

| Fitting games | Fitting rows | Comparison log loss |
|---|---:|---:|
|26|371|0.70155|
|40|595|0.69892|
|52|787|0.69684|
|66|1,016|0.69744|
|Market|—|0.69097|

More dated data helps initially but does not monotonically solve the problem.
Prefix size also changes season/history composition; this is not a controlled
sample-size-only experiment and cannot establish how much more data is needed.

### 3. Feature availability and weak incremental quality

Features are constructed under the existing prior-history missingness contract,
but availability changes dramatically between final fitting and comparison:

| Feature | Missing in fitting | Missing in comparison |
|---|---:|---:|
|Scoring standard deviation|9.4%|58.3%|
|Five-game trend / five-game minutes|9.4%|58.3%|
|Standardized edge|9.4%|58.3%|
|Fourteen-day rolling average|23.0%|75.0%|
|Seven-game minutes|15.5%|60.8%|

History appearances shift -1.36 fitting standard deviations; shooting-efficiency
attempt support shifts -1.12. Missingness reflects insufficient prior history,
not confirmed zero performance; filling it with zeros would not restore evidence.
Several feature correlations with outcome-minus-market residual are weak or
change sign across periods. Full feature-level measurements are in report.json.
These descriptive shifts do not isolate which individual feature causes harm.

Simpler controls, all fitted on the same real development quotes:

| Correction | Fitting log loss | Comparison log loss |
|---|---:|---:|
|Market unchanged|0.69387|0.69097|
|Intercept only|0.69232|0.69285|
|Four line-derived fields only|0.69171|0.69450|
|Small five-feature set plus line fields|0.69125|0.69570|
|Compact plus line fields|0.68891|0.69744|
|Compact + shooting plus line fields|0.68377|0.69991|

Increasing feature complexity consistently improves fitting loss while worsening
this comparison. Even the intercept-only correction fails to improve market.
The simple models were diagnostic probes, not selected replacements; do not choose
a new model on these already exposed comparison results.

### Conclusion and next design constraint

Overfitting has direct experimental support. Limited game diversity and major
history/coverage shifts are substantial contributors; the current features have
not demonstrated stable information beyond market. No new source-corruption bug
was established by these diagnostics. Keep the existing zero-correction fallback.
A future candidate should have very few predeclared, reliable inputs and must
show repeatable improvement on chronological periods representative of deployment,
including separately supported early-season cases, before a prospective freeze.
This is a design constraint, not authorization to spend or collect additional data.

Evidence: plan/report.json, low-dimensional diagnostic model files,
bootstrap_predictions.npz, integrity.json. Existing input and model hashes verified;
frozen v1/v5/defender/serving unchanged. No paid requests, collection, prospective
record edits or promotions. Seven relevant regression tests pass. All findings
remain exploratory on previously exposed outcomes.

## Real-market probability correction experiment — October 26, 2026

Implemented and ran a separate real-quote-only correction study in
`logs/points_market_correction_02/`. Prediction is
`sigmoid(logit(no_vig_market_probability) + learned_correction(features))`.
The market-logit coefficient is fixed at one; zero correction reproduces market.
The correction is a regularized linear model with training-only median imputation,
standardization, clipped standardized inputs, missingness indicators, and penalized
intercept. Line-derived coefficients preserve nonincreasing over probabilities
when the line rises while the supplied market probability is fixed. This constraint
does not certify arbitrary inconsistent market probabilities across different lines.

No synthetic labels or synthetic weighting enter fitting. Each real player-game
contributes exactly once, with weight one. Features compare compact alone against
compact plus reliable shooting/support/volume. Both use ridge penalties 1/10/100;
a direct compact classifier with no market offset uses the same three penalties,
real rows, preprocessing, and line constraints as a matched comparator.

### Chronological selection and limitations

- Fit: 371 real player-games across 26 games before January 2025.
- Select: 645 across 40 games, January–June 2025.
- Refit after selection: all 1,016 pre-July-2025 rows across 66 games.
- Compare: 732 rows across 52 games after July 2025; 487 short-history rows across
  41 games and 245 eligible-history rows across 20 games. Cohort game counts overlap.

Selection uses pooled log loss and includes unchanged market as a zero-correction
option. Ties prefer market, then stronger regularization. There are only two
short-history selection rows across two games; no separate short-history settings
are selected or claimed qualified. Fitting also has only 18 short-history and 11
eligible-history games, although pooled fitting exceeds 20. These are pooled
research models, not independently qualified per-cohort models. Comparison outcomes
were exposed in prior research; this remains retrospective, not fresh validation.

The selection period favors unchanged market: loss 0.69398 versus 0.70656 for the
best nonzero correction (compact, ridge100), 0.71109 for shooting ridge100, and
0.70522 for the best direct real-only compact (ridge100). Weaker regularization
performs worse. Selection and all refitted model files were saved before scoring
the later comparison. Market was selected; no correction was promoted.

### Later matched comparison

| Arm | Log loss (lower better) | AUC | Brier |
|---|---:|---:|---:|
| Market / selected zero correction |0.69097|0.53093|0.24892|
| Constant 0.5 |0.69315|0.50000|0.25000|
| Best earlier-selected nonzero correction: compact ridge100 |0.69744|0.51546|—|
| Shooting correction ridge100 |0.69991|0.51043|—|
| Real-only compact, selected ridge100 |0.69973|0.48443|0.25326|
| Previous corrected-training compact |0.69703|0.48546|0.25193|
| Original frozen compact, read-only retrospective scoring |0.74609|0.49050|0.27375|

The compact correction's log-loss disadvantage to market is +0.00647, paired
NBA-game-cluster 95% interval [+0.00176,+0.01132]. Shooting correction disadvantage
is +0.00894 [+0.00170,+0.01651]. These are descriptive intervals unadjusted for
multiple comparisons. Short/eligible compact correction loss is 0.69704/0.69825;
market is 0.69141/0.69010. Shooting correction is 0.70213/0.69548. The original
frozen compact is an additional legacy reference, not an equally trained control;
the real-only compact is the matched training control.

Anchoring predictions to market improves ranking relative to the matched direct
compact (AUC 0.51546 versus 0.48443), but the learned adjustments still worsen
market itself. Thus replacing the synthetic target does not establish incremental
predictive information. The selection-period fallback to no correction is the
appropriate result under this protocol, not evidence that a new model beats market.
The limited and seasonally uneven quote sample cannot establish that all possible
market-correction architectures would fail. No further tuning on the comparison
period or promotion is justified by this result.

Evidence: `plan.json`, `selection.json`, all development and refit models,
`report.json`, `predictions.csv`, `correction_uncertainty.json`, `integrity.json`.
The initial `_01` directory records an aborted initialization before any model
fit or comparison: tests caught row-count versus feature-count sizing, corrected
before the successful `_02` run. All nine fitted arms pass 14,640 adjacent-line
checks each; 16 focused tests pass. Frozen v1/v5/defender/serving hashes and all
research inputs/source/comparator hashes verified. No credits, collection,
prospective records, frozen sources or serving models changed.

## Training-target and shooting transfer investigation — October 26, 2026

Completed 12 predeclared research models in `logs/points_target_experiment_01/`:
three targets (prior-average-centered wide thresholds, fixed thresholds independent
of player averages, and direct next-game points regression), each with core,
core+efficiency/support, core+volume, and core+all shooting features. All use the
same 46,217 unique base outcomes across 2,609 games before October 2024, calibration
on October 2024–June 2025, and the same later comparison rows. Equal outcome weight,
real/synthetic deduplication, source checks and separate history cohorts remain.
Fixed thresholds are 0.5/5.5/10.5/15.5/20.5/25.5/30.5/40.5. Regression has no line
inputs; it predicts points, converted to over probability using a fixed 10-point
normal scale followed by the same calibration-only nonnegative slope/intercept.
These are controlled probes, not tuned probability-distribution models. Identical
hyperparameter values do not imply identical optimization across loss functions.

### Main finding: a useful scoring signal need not be a market-edge signal

On the exact same 487 short-history player-games (41 games), adding shooting to
the centered classifier improves synthetic-center AUC from 0.6002 to 0.6527 but
reduces real-line AUC from 0.5012 to 0.4868. The paired difference between those
shooting effects is -0.0669, game-cluster 95% interval [-0.1071, -0.0291]. Synthetic
and real thresholds disagree on the over label in 17.5% of these cases (12.7% in
the 245 eligible-history cases across 20 games). This is a threshold/label change,
not evidence that either correctly computed label is corrupted.

Recent efficiency predicts regression relative to a recent-scoring-based synthetic
line. For example, short-history smoothed true shooting has Spearman correlation
-0.314 with points minus synthetic center, but only +0.015 with points minus real
line. Its correlation with real line minus center is -0.662. The observed pattern
is consistent with real market lines already accounting for much of that shooting
signal. These correlations are descriptive; they do not prove bookmaker intent or
establish a single causal mechanism.

### Target changes do not resolve real-line ranking

Real-line AUC below uses raw scores, so a fitted zero calibration slope cannot
hide ranking. Core and added-shooting arms are compared within each target.

| Training target | Short core | Short +shooting | Eligible core | Eligible +shooting |
|---|---:|---:|---:|---:|
| Centered thresholds |0.5012|0.4868|0.4626|0.4358|
| Fixed thresholds |0.5076|0.4976|0.4554|0.4307|
| Direct points |0.4958|0.4810|0.4570|0.4369|
| Market |0.5235|—|0.5458|—|

The fixed-threshold and direct-points probes retain the shooting penalty. Thus
centering synthetic lines is not the whole explanation. Fixed-threshold short
shooting delta is -0.0101 [-0.0201, +0.0013]; direct-points short delta is -0.0148
[-0.0361, +0.0083]. These individual effects are uncertain. Across all three target
families, short-history shooting benefit transfers significantly worse to real
lines than to synthetic-center lines in the unadjusted descriptive intervals.

Direct-points shooting features nevertheless improve short-history scoring MAE
5.8701→5.6929: delta -0.1772 points [-0.3073, -0.0428]. Eligible MAE improves only
4.8530→4.8221, uncertain. Better average scoring error therefore does not establish
better ranking near market thresholds; the target and scoring metric matter.

Separating feature groups locates most of the classifier penalty in efficiency
and attempt-support fields, not per-game shot volume. Centered short AUC is
0.5012 core, 0.4875 with efficiency/support, 0.4991 with volume; eligible is
0.4626, 0.4407, 0.4622 respectively. This does not isolate individual efficiency
fields from their support fields. All twelve arms still lose market and constant
0.5 on calibrated cohort log loss. The direct-points core short-history calibrator
chooses slope zero, producing calibrated AUC 0.5 despite raw AUC 0.4958; calibration
cannot create ranking information. No arm is promoted.

### Interpretation and evidence limits

The data support a mismatch between learning scoring/reversion relationships and
learning incremental information beyond an actual market line. Repairing feature
semantics, widening synthetic lines, or optimizing absolute points alone does not
solve it. Keep scoring features available for points estimation research, but do
not assume synthetic-label AUC qualifies them for betting-edge ranking. A future
research design should evaluate incremental corrections to real market
probabilities using actual dated quotes, with fresh prospective validation after
freezing; this investigation does not authorize further collection or spending.

All outcomes here were already exposed in research. Intervals use 500 paired
game-cluster draws and are not adjusted for multiple comparisons. Historical
coverage/selection and sparse base real quotes (119) limit conclusions. More
synthetic thresholds are not independent outcomes. Regression's simple fixed
normal scale is not a full test of all possible distributional models.

Artifacts: `plan.json`, `report.json`, `predictions.csv`,
`representation_uncertainty.json`, `transfer_diagnostics.json`, and `integrity.json`
in the experiment directory. All twelve arms pass 14,640 adjacent-line comparisons
each. The two centered controls reproduce the preceding experiment exactly.
Thirteen focused tests pass; frozen v1/v5/defender/serving hashes, input hashes and
research source hashes match. No network collection, credits, serving change or
frozen validation edit occurred.

## Shooting and injury representation controls — October 26, 2026

Completed nine controlled research arms using existing data only, saved in
`logs/points_evidence_feature_experiment_01/`. No collection, credits, serving
change, or frozen-validation change. Legacy compact/full controls reproduce the
preceding fixed-wide experiment's metrics exactly. All nine models pass 14,640
adjacent-line comparisons each with zero violations. Eleven focused regression
tests pass. Frozen v1/v5/defender/serving and research source hashes verified.

### Construction findings and separate research fixes

The previous 48-field shooting group mixed percentages, scoring shares, play types
and touches; it did not isolate shooting efficiency. Prior-five zero-attempt
percentages were encoded as zero in 136 three-point and 152 free-throw cases among
1,748 real player-games. The short-history mask also discarded percentages below
five appearances irrespective of attempt count. `fta_per_game` and
`ft_attempts_per_game` duplicate each other.

New research sidecars rebuild prior-five and prior-20 attempt/make counts from
cached official logs, excluding the target game and later games, with game/player,
date and history-count alignment checks. Rates shrink toward priors fitted only
on pre-October-2024 base-training logs, with fixed pseudocounts and explicit
attempt support. Zero attempts yield a prior estimate, not observed zero accuracy.
Volume remains separate. A fully represented arm also removes the duplicate FTA
field. These transformations live in `src/points_evidence_representation.py` and
`scripts/build_evidence_representation.py`; frozen builders are unchanged.

Injury counts mix complete evidence and partial lower bounds. There are 132 team
and 147 opponent rows with zero known key absences but incomplete qualification.
This does not establish those players were healthy. The new representation only
exposes absence counts when a report exists and both unresolved identities and
unknown roles are zero; otherwise it preserves missingness. Genuine complete zero
remains zero. Comparison coverage is 60.0% team and 58.5% opponent. Shooting rate
estimates exist for all comparison rows, but that is not 100% observed evidence.
No injuries were invented or historical reports backfilled from later knowledge.

### Matched comparison

All arms use identical chronological splits, fixed wide synthetic lines, equal
weight per original outcome, real/synthetic deduplication, monotone line effects,
and calibration fitted only on the calibration period. Comparison has 487
short-history player-games across 41 games and 245 eligible-history player-games
across 20 games. Historical outcomes were previously exposed in research; these
are exploratory comparisons, not fresh validation.

| Arm | Short-history AUC | Eligible-history AUC | Eligible log loss |
|---|---:|---:|---:|
| Core / legacy compact |0.50117|0.46261|0.70185|
| Core + raw shooting |0.49327|0.43483|0.70149|
| Core + reliable shooting |0.48684|0.43583|0.70118|
| Core + legacy injury |0.49637|0.46975|0.70011|
| Core + qualified injury |0.49746|0.47930|0.70001|
| Core + both new groups |0.48503|0.45072|0.70003|
| Legacy full |0.49038|0.45974|0.70015|
| Full with revised representation |0.48802|0.47162|0.69956|
| Market |0.52346|0.54577|0.69010|
| Constant 0.5 |0.50000|0.50000|0.69315|

Qualified versus legacy injury eligible AUC delta is +0.00955, paired game-cluster
95% interval [-0.00366, +0.02608]. Revised versus legacy full delta is +0.01189
[-0.00590, +0.03430]. Neither establishes improvement. Reliable versus raw
shooting eligible delta is only +0.00100 [-0.00689, +0.00804]; short-history delta
is -0.00643 [-0.01416, +0.00224]. Against core alone, reliable shooting reduces AUC
in both cohorts. These 500-draw intervals are descriptive and unadjusted for
multiple comparisons. Full evidence is in `representation_uncertainty.json`.

Removing the two injury-unknown flags from compact leaves predictions identical,
so those flags do not explain the ranking failure in this experiment. Attempt
reliability fixes do not resolve the shooting ranking penalty either: construction
semantics and predictive usefulness are different questions. Incomplete injury
coverage remains a limitation; these data do not isolate a single causal reason
for poor ranking. All nine arms still lose to market and constant 0.5 on cohort
log loss. Keep the cleaner constructions for research, but do not activate or
promote them as validated predictive improvements. Frozen studies continue as-is.

## Synthetic-line and feature controls — October 26, 2026

Completed18 predeclared research models in `logs/points_line_feature_experiment_02/`
using existing data only. Base/calibration/comparison splits, equal outcome weights,
real/synthetic outcome deduplication and monotonicity constraints remain fixed from
the corrected-training experiment. All18 pass14,640 adjacent-line comparisons each
with zero violations. Eight focused tests pass. Frozen v1/v5/defender and serving
hashes are unchanged; no paid request, new collection or promotion occurred.

The first implementation run (`...experiment_01`) is preserved but superseded:
rounding center+offset before floor introduced51 differences from the original
narrow recipe in one season near integer boundaries. The corrected implementation
uses floor(center)+0.5+rounded offset. Both narrow controls now reproduce the prior
experiment's cohort log losses exactly. A regression test protects this equivalence.

### Synthetic sampling

The recipes are original offsets[-2,-1,0,1,2], fixed wide[-8,-4,0,4,8], and empirical
5/25/50/75/95 percentiles estimated ONLY from119 pre-October2024 real fitting
quotes (rounded offsets[-4,-2,-1,1,4]). No later quotes or outcomes determine those
recipes. Each outcome retains total weight1 after clipping/deduplicating lines.
Comparison actual lines inside the training recipe's span rise from439/732 (60.0%)
to707/732 (96.6%) for wide and593/732 (81.0%) for empirical. Span coverage is not a
claim that synthetic lines reproduce sportsbook selection or uncertainty.

| Arm | Short-history log loss | Eligible-history log loss |
|---|---:|---:|
| Narrow full |0.69859|0.70090|
| Wide full |0.69581|0.70015|
| Empirical full |0.69630|0.69982|
| Narrow compact |0.69480|0.70369|
| Wide compact |0.69461|0.70185|
| Empirical compact |0.69451|0.70159|
| Market |0.69141|0.69010|
|0.5|0.69315|0.69315|

Broader synthetic support modestly helps probability loss but does not solve the
ranking problem. Empirical compact eligible AUC improves0.45867→0.47135; paired
game-cluster95% delta interval[0.00544,0.01984]. It is still below chance and market.
Wide full short AUC improves0.48165→0.49038, but its interval includes zero.
No tested arm beats market or0.5 on cohort log loss.

### Retrained feature ablations

Six disjoint groups cover every noncompact field. Each is separately added to
compact and dropped from full under the predeclared fixed-wide recipe. These are
retrained matched comparisons, not inference-time masking. Exact membership is
in plan.json; groups include both original feature names and their correlated
relatives. Core compact fields remain in every arm.

| Added group versus wide compact | Short AUC change | Eligible AUC change |
|---|---:|---:|
| Injury extras |−0.00358|+0.00507|
| Shooting/scoring/play-type extras |−0.00738|−0.02264|
| Matchup extras |−0.00031|−0.00060|
| Role/tracking extras |+0.00608|+0.00354|
| History/trend extras |−0.00005|+0.00240|
| Other context |−0.00077|+0.01589|

All additive-group AUC95% intervals include zero. Thus no added feature group is
established as improving ranking on these data. Role/tracking is a tentative
positive direction; matchup adds little. Shooting/scoring/play-type extras hurt
point estimates in both cohorts, and removing that group from full improves AUC
by0.00803/0.01963, but those intervals also include zero. Removing injury extras
from full helps short AUC by0.00506, interval[0.00121,0.00942]; eligible improvement
is uncertain. This implicates the current representation/interactions, not the
value of injury information in principle. Do not delete individually useful fields
based on a grouped test. Other-context is a mixed group, not an identified single
feature improvement.

Uncertainty uses500 paired game-cluster bootstrap draws. These32 recipe/group
comparisons are exploratory, unadjusted for multiple testing, on exposed historical
research outcomes; they are not independent confirmation. No winning arm was
silently selected or promoted. Rankings and complete intervals are retained in
`ranking_uncertainty.json`; per-row probabilities are in `predictions.csv`.

## Corrected training comparison — October 26, 2026

User-authorized offline repair completed in `logs/points_training_repair_01/`.
No new data calls or credits were used. The separately saved repaired and compact
classifiers are research artifacts; frozen v1/v5/defender validation and serving
models remain unchanged. Nine focused tests passed and persisted candidates
reproduced the saved predictions.

### Repair contract

- Base fitting uses only outcomes before October1,2024. Calibration uses
  October1,2024–June30,2025. Comparison uses July1,2025 onward. Game identities are
  asserted disjoint across all three, including synthetic base outcomes. This is
  honest separation within the experiment, not a claim that previously researched
  comparison outcomes are fresh.
- There are46,217 unique base player-games across2,609 games:46,098 synthetic-source
  outcomes plus119 actual-line outcomes.117 overlapping synthetic outcomes were
  removed before expansion. Real lines replace the entire corresponding synthetic
  outcome block rather than adding another copy.
- Every original outcome has total sample weight1. Each synthetic line receives
  1/(number of synthetic lines); each actual-line row receives1. The real block now
  has119/46,217 total weight, rather than50%. Row weights range0.2–1.0; there is no
  source-block normalization or multiplier. This leaves limited real-line training
  support and does not pretend the synthetic lines became market data.
- Trees constrain decision_line negatively and all three derived edge fields
  positively. The calibrated transform has nonnegative slope. Both saved models
  passed14,640 adjacent-line comparisons on732 actual-line cases, with zero rises
  in over probability as the threshold increased. The scan covers actual line
  minus5 through plus5 in0.5 steps; the constraints provide the broader monotonic
  contract with other features fixed.
- Both arms have identical dates, outcomes, weights, hyperparameters and calibration
  procedure. A fixed intercept+nonnegative-slope calibrator is fitted per history
  cohort; no method/hyperparameter selection or refitting on comparison labels.
  Calibrators and both model files are saved before scoring comparison outcomes.
  Synthetic line generation remains unchanged to isolate the requested repairs.

### Coverage and results

Calibration has217 short-history cases/20 games and680 eligible cases/43 games.
Comparison has487 short cases/41 games and245 eligible cases/20 games. Counts
across cohorts can share games and should not be added as independent games.

| Comparison cohort | Repaired log loss | Compact log loss | Market |0.5|
|---|---:|---:|---:|---:|
| Short history |0.69859|0.69480|0.69141|0.69315|
| Eligible history |0.70090|0.70369|0.69010|0.69315|

The repaired model improved from previous development losses0.73694/0.73659,
but still does not beat market or0.5. Compact is now better for short histories.
Game-cluster95% intervals for repaired-minus-compact are short[0.00036,0.00734]
and eligible[-0.00641,0.00052]. Against market they are short[0.00097,0.01346]
and eligible[-0.00112,0.02207]. These are descriptive retrospective intervals,
not independent confirmation. Raw repaired losses before calibration are
0.71961/0.71101; the new disjoint calibration appropriately shrinks slopes to
0.357/0.313 rather than sharpening overfit short-history predictions.

The four repairs were applied together; the comparison does not isolate each
repair's contribution. Ranking remains weak: repaired AUC0.482/0.464. The gain
largely comes from removing overconfidence rather than establishing a predictive
edge. No promotion or new prospective study was started from these results.
Detailed metrics, paired uncertainty, per-row predictions, source/model hashes,
and training counts are retained in the experiment directory. Further historical
spending remains paused; the existing frozen prospective monitor continues.

## Training diagnosis and correction — October 26, 2026

**Correction to the prior coverage conclusion:** numerical row/game counts passed,
but the calibration fitting set was not disjoint from base-model training. The
prior comparison remains exposed development evidence; it is not a clean test of
calibration transfer. No paid requests were made during this investigation.
Detailed evidence: `logs/points_training_diagnosis_20261026/`.

### Confirmed training problems

1. **Base-training overlap in calibration.** 209/217 short-history calibration
   player-games were in the base synthetic training set. At the stricter game level,
   215/217 rows overlap (the base can also learn from other players in that game).
   Only2 short-history rows in2 games remain disjoint. Eligible calibration has
   156/799 overlapping rows;643 rows in40 games remain disjoint. Same outcomes at
   different synthetic/actual lines are still overlapping outcomes. The earlier
   fresh-holdout audit did not check this second boundary; that was a missed guard.
   Repaired AUC is0.959 on base-seen fitting rows versus0.522 on base-unseen fitting
   rows and0.487 on later short-history selection. Fitting calibration on the
   overfit predictions selected slope2.15, worsening short-history selection loss
   from0.737 to0.868. Calibration cannot fix a near-random ranking.
2. **Extreme weighting and duplicate outcomes.** Reconstructed v6 training has
   271,189 synthetic classifier rows from54,351 player-games and294 real rows.
   The trainer assigns the real block half the total weight: about922.4x weight per
   real row relative to a synthetic row. Of294 real player-games,286 also occur in
   the synthetic block. This corrects the earlier approximate459x claim for this
   exact v6 training reconstruction. It strongly concentrates fitting on a small
   real-line sample; it does not turn those rows into independent evidence.
3. **Wrong line distribution.** On identical732 later player-games, replacing the
   actual line with the synthetic training-style central line changes repaired
   log loss from about0.737 to0.673 (AUC0.626). Labels necessarily change with the
   line, so this isolates target-distribution sensitivity, not a causal proof that
   one feature caused the loss.276/732 actual lines lie over2.5 points away from
   the prior-average center used to generate synthetic lines.
4. **Nonmonotone probabilities.** With all player features fixed, moving the line
   from actual-minus1 to actual-plus1 increases the over probability in167/732
   repaired cases and169/732 compact cases. The current tree learners have no
   joint monotonic constraints on decision_line and its derived edge columns.
   This violates the intended probability semantics and needs correction in a
   new training experiment or a coherent conditional points distribution model.

### Feature findings and limits

All178 classifier inputs (174 player/context fields plus4 line-derived fields)
were inventoried for missingness, constants and range shift. None is constant on
this entire fitting population. Missing stddev/standardized_edge and several
shooting/trend fields rises from9% to58%; rolling14day missingness rises23% to75%.
The later population contains more short histories, so these aggregate shifts are
not proof of broken source wiring. They do show that broad history masks alone do
not establish that a learned branch transfers across season openings.

Masking injury fields at inference improves loss by0.00191; masking shooting
fields improves0.00274; masking matchup fields worsens0.00542. These are sensitivity
probes, not retrained ablations or evidence that injuries/shooting should be deleted.
They are much smaller than the approximately0.046 gap to market probabilities.
Prior all-row quote-time replay evidence for the original1,579 rows remains valid;
new169 rows passed the existing independent timing/identity importer. This pass
has not independently replayed all new feature values, and no new timestamp bug
was established. Archived quote timestamps still rely on provider metadata.

### Offline control and prevention

A fixed, strongly regularized real-line-only market-residual logistic probe, fitted
only on the earlier development partition, scores0.69144 short /0.69013 eligible,
versus market0.69141 /0.69010. It avoids the large loss of the old models by making
almost no adjustment; it establishes no improvement over market. No hyperparameter
search or promotion was performed. A separate numerical recheck reproduced these
values without matrix-multiply runtime warnings.

`src/calibration_independence.py` and `scripts/research_points_calibration_v3.py`
require a hash-locked base-training game inventory and reject any fitting or
selection game overlap before model loading/output creation. The actual inputs
are blocked on26 overlapping games; `gate_check.json` records the failure. Prior
runners, execution locks, failed runs and published research results remain preserved.
Seventeen focused calibration/independence tests passed. Frozen v1/v5/defender
sources, model hashes, existing evidence and serving artifacts remain unchanged.

The next valid repair uses forward out-of-fold base predictions for calibration
(or refits a new base strictly before the calibration period), deduplicates real
and synthetic outcomes, bounds the real-row weighting, and tests monotonic line
behavior. Test feature groups through matched retraining, not deletion based on
this selection set. With current data the existing frozen base lacks independent
short-history calibration support; acquiring more quotes for games it trained on
would not fix that. Future confirmation still requires a separately frozen study.

## Authorized calibration batch completed — October 26, 2026

The user authorized the exact twelve planned games within120 credits. All twelve
were collected with120 credits reserved; no replacements or extra games were
requested. The original paused proposal remains unchanged; the authorized copy and
reservation ledger are in `logs/real_points_paid_calibration_01/`.

Of1,135 book quotes,1,130 passed identity/timing qualification; five missing-provider-ID
quotes were rejected. After history/void/deduplication checks,169 new player-games
were added, totaling1,748 across118 games. All twelve planned games have usable
intended-cohort rows. Coverage gates passed:

| Cohort | Calibration fit rows/games | Selection rows/games |
|---|---:|---:|
| Short history |217/20|487/41|
| Eligible history |799/51|245/20|

The original calibration run stopped on a numerical optimizer failure and is
preserved in `logs/points_calibration_research_01/`. Float32 prediction arithmetic
lost the small perturbations used for finite-difference optimization. Separate v2
source/runner files use float64 arithmetic; regression tests cover that behavior.
The method families, data splits, nonnegative-slope constraint and selection rule
were unchanged. The execution lock is in `logs/early_season_calibration_execution_v2/`.

Results are in `logs/points_calibration_research_02/report.json`. Lower log loss is better:

| Development selection cohort | Repaired | Compact | Market |0.5|
|---|---:|---:|---:|---:|
| Short history |0.73694|0.75772|0.69141|0.69315|
| Eligible history |0.73659|0.74300|0.69010|0.69315|

Both short-history arms selected identity (no calibration adjustment); eligible
arms selected intercept+slope with only a tiny improvement. Repaired-minus-compact
log-loss95% game-cluster intervals include zero: short[-0.04558,0.00794],
eligible[-0.03051,0.01718]. Repaired-minus-market intervals are positive:
short[0.02124,0.07116], eligible[0.00709,0.08804]. These are descriptive intervals
on exposed development selection outcomes, not independent validation and not
adjusted for calibration selection. More calibration coverage did not establish a
useful model. No candidate was promoted or added to prospective serving.

Twenty-seven focused tests passed. Frozen v1/v5/defender model/source/prior-evidence
hashes and serving hashes match. Existing prospective monitoring continues unchanged.
No further historical spending is authorized by this completed batch.

## Dated calibration-only plan and runner — October 26, 2026

Spending remains explicitly paused. The locked proposal is
`logs/early_season_calibration_plan_v1/plan.json` (`plan_lock.json`). All twelve
NBA/provider identities were joined against free archived ESPN schedules and
official cached game logs. Prior provider attempts were checked, including uncertain
attempts. No DunkAPI request, account-balance request or credit spend occurred.

The new retrospective development boundary is July 1, 2025. This reuses all 1,579
qualified player-games in 106 games; it does not claim fresh historical validation.

| Cohort | Calibration fit rows / games | Selection rows / games | Missing games |
|---|---:|---:|---|
| 1–9 prior appearances | 140 / 14 | 439 / 35 | 6 fitting |
| 10+ prior appearances | 799 / 51 | 201 / 14 | 6 selection |

The fixed proposal adds six short-history fitting games on October 24–25, 2024,
and six eligible-history selection games on November 9–10, 2025. Selection used
chronological game IDs and prior-history support, never points or performance.
Cohorts may overlap within games; game counts across cohorts are not additive.
The minimum is 120 credits at 10 per game, equivalent to $0.30 wallet value at
DunkAPI's published 400 credits/dollar (https://dunkapi.com/pricing, checked October26).
This is not a balance confirmation or a guarantee of usable quotes. Every proposed
game must supply at least one qualified row in the deficient cohort; otherwise a
new dated plan is required. No automatic replacements are authorized.

`scripts/research_points_calibration.py` is a separate offline runner. It uses
hash-locked v6 repaired and compact base classifiers without retraining them,
fits identity/intercept/nonnegative-slope calibration on fitting rows only,
and selects on the later development split without refitting selection labels.
It saves separate calibrator parameters and research reports, never deployment
artifacts. Market and 0.5 baselines use the same selection rows. Historical research
exposure is explicit throughout. The original v8 fresh-holdout guard is unchanged.

Twenty focused tests passed, including a synthetic end-to-end successful run,
invalid-source/cohort/quote rejection, drift protection, no base-model fitting,
no selection-label refitting, and an enforced spending pause. The actual existing
inputs were rejected before loading a base model or creating an output directory:
two cohorts still lack independent games. Evidence: `readiness.json` in the plan
folder. Frozen v1/v5/defender sources, models and prior evidence, plus serving model
hashes, were verified unchanged. This is a tested runner, not a trained new candidate.

## Locked calibration-research design — October 26, 2026

`logs/early_season_calibration_design_v1/protocol.json` and `lock.json` record the
separate historical-development / future-prospective design. This is a design lock,
not an executable collection plan or a frozen trained candidate. Spending remains
paused. Existing v1/v5 monitoring continues unchanged.

Historical calibration compares identity, intercept-only and intercept/slope
adjustments for fixed repaired and compact models, selected on a later development
split. Each short-history and eligible-history calibration split retains the
100-player-game / 20-independent-game requirement. Historical results are explicitly
exposed research; fresh quotes do not create fresh outcomes. A separate future study
must freeze the selected candidate and compare on matched actual lines against the
compact control, no-vig market, and 0.5, with cohort-specific game-cluster uncertainty.

Before collecting: prepare an outcome-independent dated manifest, assess reusable
quotes and cost/coverage, verify remaining balance, and implement/test a distinct
calibration-only runner. The v8 fresh-holdout guard remains intact. Twenty-four new
games alone cannot establish two disjoint 20-game calibration splits; reuse and
short-history coverage must be measured before proposing a bounded batch.

## Precollection wiring audit — October 26, 2026

**Audit verdict: do not spend on the proposed fresh historical holdout yet.** No
additional provider requests or credits were used. The audit covers quote identity,
source qualification, feature reconstruction, market baselines, split exposure,
calibration support, collection budgets, training entry points and frozen artifacts.
Evidence is in `logs/points_precollection_audit_20261026/`.

### Verified

- Replayed all 174 prepared features on **all 1,579 real-line rows**, replacing the
  earlier 32-row sample. Every value/missingness mask matched its original quote-time
  construction. All archived quote cutoffs, lines and independent timing checks
  passed. This proves reproducibility, not predictive usefulness of every field.
- Checked all 76 archived game-download responses against their requested provider
  game identities; no mismatch found.
- Confirmed the market baseline uses paired actual over/under prices; model/market
  predictions share the same player/game/line rows. Frozen reference injury inputs
  and source hashes remain protected.
- Verified 45 frozen artifacts, serving models, v6 and v7 saved models. No frozen
  protocol, saved forecast, threshold or candidate was altered.
- 129 distinct focused tests passed: 126 in the pipeline suite plus 3 new training
  preflight tests (the 8-test gate suite repeats 5 collection tests).

### Blocking study-design finding: new quotes are not unseen outcomes

The existing v5 datasets used in training/development/evaluation already contain
5,222 distinct NBA games across 2022–23 through2025–26. Against 5,279 cached games,
57 are absent, but **none has eligible short-history players under the current
minimum-one-prior-appearance contract**. They are opening appearances. We cannot
turn newly purchased odds for previously used games into a fresh game-level holdout.
The entire2026–27 season remains excluded from historical training/calibration.

Historical early-season quotes can still support retrospective calibration research.
A genuine confirmation test requires a new frozen prospective study on future games,
with models/calibration locked before those games and no retroactive forecasts. It
must coexist with the unchanged v1/v5 studies. Early-season coverage must be checked
separately: current calibration has 2 short-history fitting rows across 2 games and 0
selection rows. Pooled sample counts do not establish readiness. The current eligible
evaluation subset also has only 14 independent games.

### Wiring defects repaired

Previously `calibration_support` and `promotion_gate` were report helpers, not mandatory
training entry checks. Historical v6/v7 research sources are preserved for reproducibility.
The new versioned runner, `scripts/research_points_v8.py`, calls
`src/points_training_preflight.py` **before output creation or model fitting**. It
checks disjoint/chronological game splits, protected-season exclusion, fresh-game
exposure, separate short/eligible calibration and evaluation coverage, labels, quote
alignment and actual-price market probabilities. A regression test proves malformed
coverage cannot reach classifier construction. Running it on the current data stopped
before creating an output directory, with the expected exposure/coverage blockers.
No v8 model was trained; successful training on a new qualifying dataset remains untested.

Paid collection previously used batch-local locks and attempt checks, so a new batch
could repeat a prior uncertain request. The collector now uses the shared free/paid
budget lock, checks attempts across months/batches/raw evidence, and validates each
response's provider game/date against the plan. New spending requires a hash-locked,
timestamped protocol with explicit cohorts, horizon, split roles and model-selection
rule. Its exposure inventory must include the existing historical datasets. Tests
prove unapproved or already-exposed plans stop before any network request. Completed
batches remain closed without requesting credentials or credits.

### Credit and deployment status

Existing ledgers remain 500 reserved free credits (including uncertain attempts) and
270 reserved paid credits. A live remaining wallet balance has not been verified;
these are ledger totals, not an account-balance claim. No new paid batch was opened.

The next valid design is historical early-season calibration research followed by
separately locked prospective confirmation. Closing-line archives and request-time,
60-minute or 15-minute forecasts must not be treated as interchangeable horizons.
The existing monitor continues its existing captures; the v7 market-anchored interface
requires both prices and is intentionally not plugged into the frozen two-argument
prediction interfaces. Neither a new prospective sidecar nor a serving change was
introduced by this audit.

## Why real-line probabilities remain weak — October 26, 2026

Completed a separate forensic audit and v7 research experiment. All frozen v1/v5
artifacts, v6 model files and serving models remain unchanged. No additional paid
calls occurred. The new experiment did **not** establish an improved deployable
model; the failed evaluation and all ablations are preserved.

### Findings by suspected cause

**Real lines / labels.** All 1,579 quote rows pass the archived timestamp, line and
cutoff alignment checks. Binary labels equal actual points greater than the line;
no target mismatches were found against official NBA player/game records. The
historical timing audit independently rechecked 72 game/source pairs against NBA
opening-period and ESPN jump-ball evidence. The earliest accepted quote-to-start
margin is 262 seconds. These checks verify recorded evidence; they cannot prove
that an external provider's capture-time metadata is accurate.

Actual bookmaker lines represent a substantially harder task than synthetic
history-centered lines. On exactly the same 640 players/games, actual outcomes and
quote-time features, v6 repaired raw log loss is **0.65596 on synthetic lines versus
0.73800 on real lines**; compact is 0.68143 versus 0.75630. Repeated synthetic lines
are not independent observations. This controlled probe isolates line construction,
so earlier synthetic success cannot be treated as evidence of market-beating skill.

**Feature timing.** A fixed, outcome-independent sample of 32 rows reproduced all
174 prepared features at the original archived quote cutoffs. Across overlapping
synthetic/real rows, only injury-related fields differ; synthetic rows normally use
one-hour-before-start cutoffs while real rows use their actual quote cutoffs. No
broad numerical/unit or target-alignment error appeared in these checks. These
findings do not certify every external feature source's historical timestamp.

**Training weight and duplicate evidence.** V6 gave only 294 real-line fitting
examples half the classification fitting weight, against tens of thousands of
historical player-games. A real row could carry roughly 459 times an average row's
weight. V6 repaired raw loss was 0.28754 on those real fitting rows and 0.73800 on
evaluation. A controlled v7 weighting sweep supports overfitting: with other
settings fixed, repaired raw evaluation losses were 0.71561, 0.71272, 0.74260 and
0.73981 for 0%, 1%, 10% and 50% real weight. These evaluation ablations were saved
only after candidate selection, and were not used to pick another winner.

The v7 pipeline removes 286 real fitting player-games from synthetic fitting data
before synthetic line expansion. This prevents counting the same outcome in both
fitting blocks at different cutoffs. Historical fitting contains 54,065 unique
player-games after that removal. The weighting sweep was selected using the
existing development split, not evaluation outcomes.

**Calibration and population shift.** V6 sigmoid slopes were only 0.059/0.068;
both predictions collapsed near 46.8% over. Calibration-fit and selection over
rates were 47.2% and 46.4%; evaluation was 50.9%. An under-biased near-constant
forecast is worse than50/50 when that base rate shifts. Calibration also has just
**two short-history cases in fitting and zero in selection**, while evaluation has
439. Pooled sample minimums hid this unsupported cohort.

### Separate repair experiment and failed result

`src/points_v7.py` adds two-sided de-vigged market probabilities, bounded
market-anchored shrinkage without a learned global intercept, and explicit mixture
weights. The market prices are genuine archived over/under prices. The controlled
sweep tests real fitting shares0%,1%,10%,50% in both174-feature and19-feature arms.
Raw and conventional sigmoid outputs are diagnostic only. Candidate selection uses
March-April calibration and May-June development; a market-only option is included.
Choices were saved before evaluation. The selected174 and compact19 classifiers
both used **0% real fitting weight**; real lines were used for calibration/selection.
They must not be described as classifiers fitted on actual lines.

The no-intercept shrinkage weights, refitted on the prescribed calibration period,
are0.4353 for selected174 and0.3342 for compact19. The experiment did not solve the
underlying weak signal:

| Arm | All640 log loss | Eligible201 log loss |
|---|---:|---:|
| Fixed50/50 | 0.693147 | 0.693147 |
| De-vigged market only | 0.690626 | 0.689646 |
| Previous v6 repaired | 0.696948 | 0.696587 |
| Development-selected v7 repaired | 0.697171 | 0.695904 |
| Equally selected v7 compact | 0.695911 | 0.697856 |

Overall there are46 independent games; the eligible cohort has14, and the439
short-history cases span35 (cohorts overlap in some games). Selected v7 minus market
log-loss difference is +0.006545, game-cluster95% interval [+0.002104,+0.011181].
Its eligible difference versus compact is -0.001951, interval[-0.006831,+0.003370].
Thus the apparent development gain did not carry forward. The observed market-only
advantage over50/50 is a benchmark result, not proof of a profitable betting edge.

`src/points_research_gate.py` adds explicit independent-validation, market/compact
improvement and separate calibration-cohort coverage checks. Applying it produces
`logs/points_v7_research/readiness.json`: not eligible for promotion; short-history
calibration unsupported. These are checks for new research, not edits to frozen
study thresholds. No market fallback or v7 output was deployed, and failed model
forecasts were not replaced by benchmark predictions after seeing outcomes.

The remaining requirement is a new dated validation design with enough early-season
real-line examples for both calibration and independent evaluation, plus evidence
of improvement beyond market prices. More tuning on this exposed640-row set would
not supply that evidence. Historical metered collection remains paused; the
unchanged prospective v1/v5 monitors continue.

Evidence: `logs/points_v7_research/diagnosis.json`, `quote_timing_audit.json`,
`line_domain_audit.json`, `experiment_plan.json`, `selection.json`,
`ablation_evaluation.json`, `report.json`, `readiness.json` and `verification.json`.
47 focused tests passed; saved predictions reproduced after model reload, all45
previously snapshotted frozen artifacts match, and v6/serving hashes are unchanged.

## First real-line controlled experiment — October 25, 2026

The user confirmed a $5 top-up and authorized collection of the missing games and
the controlled comparison. The experiment is complete; no model was deployed and
neither frozen prospective study was changed. All 45 recorded frozen artifacts
and serving model hashes match the pre-experiment snapshot. The existing v1/v5
monitors remain active; v6 has no prospective forecasts.

### Collection and qualification

`logs/real_points_paid_01/plan.json` prespecified eight additional games per deficient
split, selected by chronological date and provider event ID without looking at
player outcomes or model performance. The original batch made 24 requests under
a separate 240-credit ceiling, preserving the 500-credit free ledger. Twenty-one
games qualified. Games `0042400157` and `0042400222` lacked independently corroborated
opening jump balls; `0042400223` had conflicting scheduled starts. Alternate free
ESPN core feeds did not recover the missing opening events. All rejections and raw
responses remain archived; no timestamp rule was relaxed.

A separate dated plan in `logs/real_points_paid_replacements_02/plan.json` selected
three replacement games only after free timing checks. Its ceiling was 30 credits.
The earlier two-replacement draft was superseded before spending any credits.
Total: **27 downloads, 270 estimated/reserved credits, 24 new qualified independent
games and 373 new player-games**. The provider did not expose a remaining wallet
balance in these responses. Both paid batches are closed; further metered history
collection is paused. This authorization does not change the ongoing Odds API caps.

The merged dataset `logs/real_points_merged_05/real_points.joblib` contains 1,579
unique player-games across 106 games. Prior rows and the evaluation cohort remain
exactly unchanged. Every split meets both the 100-player-game and 20-game gates:

| Split | Player-games | Independent games |
|---|---:|---:|
| Fitting | 294 | 20 |
| Calibration fit | 322 | 20 |
| Calibration selection | 323 | 20 |
| Exposed retrospective evaluation | 640 | 46 |

### Fixed comparison and results

The plan was saved in `logs/real_points_paid_01/experiment_plan.json` before model
fitting or evaluation. `scripts/train_points_v6.py` fits repaired174 and compact19
using the same chronological samples, hyperparameters and real/synthetic weighting.
Both new arms omit the always-unavailable 30-appearance column; original compact16
and frozen161 retain their original columns and reconstructed injury semantics.
Calibration selected sigmoid for both new arms using only the prescribed calibration
splits. The classifier fits 54,351 historical player-games expanded into synthetic
lines plus 294 actual-line examples; the two classifier blocks receive equal total
fitting weight. Point regressors use historical point outcomes, not fabricated odds.

Primary population: at least ten prior appearances, **201 player-games / 14 games**.
Short-history cases remain separate: **439 player-games / 35 games**. Some games
contain both cohorts, so their game counts do not sum to the overall 46.

| Model | Eligible log loss | All640 log loss | All640 Brier | All640 points RMSE |
|---|---:|---:|---:|---:|
| Original frozen161 | 0.712897 | 0.729496 | 0.266992 | 7.0885 |
| Original compact16 | 0.728151 | 0.750405 | 0.275617 | 7.3491 |
| Repaired174, real-line trained | 0.696587 | 0.696948 | 0.251895 | 6.8270 |
| Equally retrained compact19 | 0.696811 | 0.697493 | 0.252166 | 7.1957 |

Lower log loss, Brier and RMSE are better. For repaired minus retrained compact,
the primary log-loss difference is **-0.000224**, with game-cluster paired 95%
interval **[-0.002146, +0.001580]**. Across all640 rows it is -0.000545,
interval [-0.001887, +0.000822]. Short-history difference is -0.000691,
interval [-0.002397, +0.001084]. None establishes an advantage over the equally
trained compact control. Against the original compact, the eligible difference is
-0.031565, interval [-0.053910, -0.008756], but that does not isolate the benefit of
extra features from the training/calibration changes shared by both new models.

The repaired model's overall AUC is 0.4825, and its overall log loss remains worse
than the fixed50/50 benchmark (0.693147). The improved loss against older references
therefore does not establish useful discrimination, profitability or a betting edge.
This is an already exposed retrospective research set, not fresh prospective proof.
The predeclared primary comparison did not pass; no promotion or evaluation-driven
retuning was performed.

Evidence: `logs/points_v6_real_01/{protocol.json,report.json,predictions.csv,
interpretation.json,repaired.joblib,compact_retrained.joblib}` and
`logs/real_points_merged_05/{feature_manifest.json,reference_inputs.joblib,readiness.json}`.
The full ordered manifest records observed feature counts by split; unresolved
injury roles and incomplete windows retain the missing-feature policy below.
42 focused regression tests passed, serialized model predictions reproduce the
saved evaluation results, and all frozen artifacts passed final hash checks.

## Pretraining comparison repairs — October 25, 2026, late evening

This section supersedes older historical coverage/budget descriptions below.
No payment, new candidate fit, deployment, or change to either frozen prospective
study was made. The next real-line experiment is `scripts/train_points_v6.py`;
the existing v5 synthetic candidate and its prospective protocol remain frozen.

### Comparison check

Frozen161 and the original compact model now receive a separate reference frame
with their original three injury fields reconstructed at each archived quote
cutoff. New candidates receive repaired injury evidence. The reference frame
must match the dataset hash and player/game/date order; all other input columns
must be identical and recorded reconstruction source hashes must match. Both arms
use exactly the same real lines and outcomes. Regression RMSE now also uses those
same real-line player-games, rather than a different synthetic cohort.

`logs/points_pretraining_v6/comparison_check.json` records 1,206 aligned rows and
26 outcome-independently selected replays through the original frozen feature
builder, with all three injury fields matching. Original semantics differ from
the repaired inputs on 305 team-count rows and 296 opponent-count/impact rows.
The check also demonstrates that the trainer refuses insufficient split coverage
before creating an output directory or fitting. This is a comparison-integrity
check, not a claim of predictive improvement.

### Missing-feature policy for the next candidate

The versioned ordered 174-column manifest and per-column, per-split observed
counts are in `logs/points_pretraining_v6/manifest.json`. The manifest excludes
`rolling_30day_avg`: the historical contract caps history at 20 appearances, so a
30-appearance window cannot be populated. The legacy rolling column names refer
to appearances, not calendar-day windows. No remaining candidate column is
entirely missing in any of the four current real-line splits; no infinities were
found. This does not mean every row has every feature.

- Require at least one strictly prior current-season appearance. Exclude
  zero-history cases; keep 1–9-appearance cases in a separately reported cohort.
- Incomplete slope, workload, volatility and shooting windows remain NaN at their
  declared minimum history. Retain history length and short-history indicators.
  Do not pad history with future games or silently substitute a shorter window.
- Missing or unavailable injury reports mean unknown, not a healthy roster.
  Retain report availability, total confirmed Out count, unresolved identities,
  unknown roles and the known key-player lower bound as distinct evidence.
- A published report can support a zero count; missing evidence cannot. Unknown
  roles leave the full key-player count/impact unavailable. The original zero-role
  fallback is retained only when reproducing original frozen-model semantics.
- Use model-native NaN handling. Do not globally fill missing inputs with zero,
  apply future-derived means, or retrofit today's source qualification into past
  rows. Historical feature evidence must be available at the quote cutoff.
- Exclude unqualified deferred groups, including the seven defender features.
  Adding qualified groups later requires a new manifest, dated training support,
  separate output directory and controlled comparison.

Current 1,206-row coverage: team/opponent reports are available for 88.31%/87.06%;
full key-player counts are observed for 61.28%/60.28%. True shooting and recent
FG percentage are observed for 63.85%, with incomplete history left unknown.
These are coverage limitations, not repaired factual observations.

### Remaining data gate and next experiment

Canonical NBA team IDs fix the public archive's LA/Los Angeles Clippers mismatch
without accepting swapped home/away identities. Replaying existing raw evidence
recovered 25 player-games across two additional games, with no credits spent.
New evidence is in `logs/real_points_public_02/` and
`logs/real_points_merged_04/`; prior outputs are preserved.

| Chronological split | Player-games | Independent games | Required games |
|---|---:|---:|---:|
| Fit | 179 | 12 | 20 |
| Calibration fit | 190 | 12 | 20 |
| Calibration selection | 197 | 12 | 20 |
| Exposed retrospective evaluation | 640 | 46 | 20 |

The dataset totals 1,206 player-games across 82 games. We still require at least
24 dated earlier games: eight in each of the first three splits. All splits must
also retain at least 100 distinct player-games after validation; book duplicates
never increase sample size. A paid balance is not a guarantee of qualified games.
The full 500-credit free allowance remains reserved, including uncertain attempts.
No extra historical collection or top-up was performed in this repair pass.

Once coverage qualifies, run the separately versioned trainer with the merged
real dataset and its matching reference sidecar. Repaired and retrained compact
controls use identical fitting/calibration windows; unchanged original models
remain reference arms. Real-line classification fitting is balanced with the
existing synthetic fitting block; calibration and evaluation use actual lines.
The points regressor still fits historical point outcomes. Report matched losses,
independent games and game-cluster uncertainty. This exposed retrospective set
cannot establish fresh prospective success. Passing minimum counts does not
establish that the larger model beats compact.

### Monitoring

The defender monitor now archives raw responses before parsing and distinguishes
an unpublished empty matchup table from an error or incorrect assignment. The
latest free refresh has 42 completed boxscore games; 17 prospective defender
forecasts await matchup publication. The original 45 held-out cases remain
separate (48.89% assignment accuracy, 13.33% minutes baseline), and qualification
has not passed. Thresholds and frozen ranker are unchanged. The original points
study and separate v5 study continue under the shared unchanged quote budget.

## Separate repaired candidate — October25, 2026 evening

The original study, its models, source files, protocol and saved forecasts remain
frozen. `logs/prospective_points_v5/` is a separate paired study using
`logs/points_v5_synthetic/repaired.joblib`. Its contract is implemented in
`src/points_v5.py`; no serving models were replaced.

Incomplete slope/workload/shooting windows are unavailable in the new contract,
with explicit history length and short-history indicators. Short-history examples
now appear in fitting, rather than relying solely on models trained after ten
appearances. Four rebuilt seasons contain92,962 player-games, including12,698
opening-season examples. Fitting beforeJanuary2025 uses54,351 player-games,
including9,107 short-history examples. The remaining seasons/periods retain
chronological calibration and retrospective evaluation roles.

Injury handling now retains report availability, total Out count, unresolved
identities, unknown roles, and the known key-player lower bound. Dated fresh NBA
rosters resolve new players; absent role history remains unknown. In the archived
Lakers example, Cameron Carr was absent from the legacy name lookup, despite an
actual report being present. The new resolver identifies both listed Out players,
retains two known absences and one unknown role, and does not falsely convert the
full key-player count to zero. This is an evidence-handling repair, not proof the
unknown rookie role has been recovered.

Initial retrospective results still use synthetic lines. A controlled follow-up
restored the original injury semantics for the frozen reference inputs:

| Population | Frozen161 logloss | Repaired logloss | Paired difference95% interval |
|---|---:|---:|---|
| 3,315 short-history player-games |0.65214|0.64340|[-0.01276,-0.00512]|
|20,357 eligible-history player-games|0.65184|0.65149|[-0.00101,+0.00033]|

These are previously exposed2025–26 research outcomes, not independent
prospective confirmation. The main training report scores all models on common
revised injury inputs; prefer `true_short_input_comparison.json` and
`true_frozen_input_comparison.json` for comparisons against original frozen inputs.
No improvement is established for the eligible-history cohort. The14 exposed
October25 forecasts were not training/calibration examples or new prospective
candidate forecasts.

`scripts.monitor_prospective_points` continues the v1 workflow and invokes
`scripts.validate_points_v5`. The sidecar uses the exact archived quote, feature
cutoff, source snapshots and roster from v1, requires a fresh quote and a new
prediction before tipoff, and saves both reference probabilities alongside the
repaired prediction. Past forecasts are never backfilled as prospective. It
reports separate comparisons against both frozen models. The first100 eligible
settled matched forecasts are the primary checkpoint; short histories stay a
separate cohort. A sidecar failure does not stop original collection.

While the repaired study is active, the monitor may capture early-season warmup
games so the repair can be tested prospectively. The v1 eligibility definition,
one-attempt/day rule,20-credit Odds API monthly ceiling and50-credit account
reserve remain unchanged. Both studies share one quote purchase. No new odds
credits were spent during implementation; local reservations remained6.

### Real-line training: implemented, source qualification still blocked

The user supplied a local DunkAPI key. Five game downloads returned463 raw
player/book quotes. The strict importer accepted zero:382 rejected timestamps
were after official scheduled tipoff,15 lacked a unique NBA player/date match,
and14 distinct player-game selections had no earlier current-season appearance.
Another52 later-book duplicates were skipped after selecting those player-games;
they are not independent examples. The first-game failures do not mean those
players lacked prior NBA careers. Late scheduled-tipoff timestamps do not prove
actual play had started, but we have not verified actual tipoff independently and
will not silently loosen the pregame rule.

The provider advertises [historical props](https://dunkapi.com/docs/get-player-props)
and a [500-credit free allowance](https://dunkapi.com/pricing). Its live no-charge
estimate returned1,000 credits for the requested multi-season player range, so
that request could not be funded. Smaller game requests cost10 each. Five
successful downloads used50 estimated credits; the local ledger conservatively
reserves60 including the failed first attempt. The collector now checks the
actual estimate before any metered request and enforces a200-credit local cap.
No subscription or wallet purchase was made. Further spending is paused pending
a source with qualified pregame timestamps.

`scripts.build_real_points_training` joins real player names/dates uniquely to
NBA games, checks official teams and tipoff, reconstructs features at quote time,
deduplicates player-games before outcome-based void checks, and excludes the
entire prospective season. `scripts.train_points_v5 --real-data ...` supports
real-line fitting and calibration with separate chronological samples, but
requires at least100 rows and20 games in each real-data split. The current
synthetic candidate is **not real-line trained**. A future real-line model must
be saved and validated as another frozen version.

Evidence: `logs/real_points_pilot_02/report.json`, raw responses under
`logs/real_points_sources/`, and candidate reports under
`logs/points_v5_synthetic/`. Twenty-six targeted tests passed, including partial
injury evidence, future/stale roster rejection, incomplete-window handling,
protected-season exclusion, quote timing and immutable original forecasts.
An independent numerical calibration check reproduced the saved sigmoid to
within0.000020 probability. The combined monitor ran successfully: original14
forecasts remain settled, no games remained to capture, and the repaired study
has zero fresh prospective forecasts so far. Its historical improvement must not
be represented as a prospective result.

Started October 25, 2026 at 20:32 UTC, before Lakers–Jazz's official 21:00 UTC tipoff. The first batch contains **14 paired player-game forecasts**, observed approximately 27 minutes before tipoff. All are early-season **warmup**, separate from the primary eligible cohort. No result is available at launch.

The frozen 161-feature candidate (`development_selected.joblib`) and equally trained 16-feature compact control (`compact_expanded.joblib`) come from `logs/expanded_points_experiment_v4`. They predict the same actual bookmaker points line. Existing serving models remain unchanged. Calibration and feature selections remain frozen.

## Evidence and controls

- `logs/prospective_points_v1/protocol.json` records hashes and the analysis plan, frozen before forecasts. A preflight dependency failure occurred before any collection; its earlier protocol is retained as `aborted_preflight_protocol.json`.
- `forecasts/` contains exclusively created paired predictions, exact lines/prices/timestamps, complete feature vectors, missing-source flags, input hashes, and source-run paths. No forecast is backdated or overwritten.
- `runs/` retains raw quotes, official schedule, roster, injury observation, and database/player-log inputs. Predictions must finish before official tipoff, within 10 minutes of quote collection and 15 minutes of provider market observation.
- Select the lexicographically first available bookmaker for each player, independently of price, predicted edge, and outcome. First saved forecast per player/game/horizon wins. Multiple books do not create additional independent samples.
- Primary eligibility matches training: ten earlier current-season appearances, plus prior20 mean minutes at least12 unless in the original prior-season top50 cohort. Short histories and other out-of-cohort players remain labeled warmup.
- Real completed NBA results settle both predictions together. DNPs are void, unresolved players remain missing, and pushes count only toward points-error metrics.

## Comparison

Primary evaluation uses actual run/request-time quotes. `--horizon 60m` (50–70 minutes) and `--horizon 15m` (10–20 minutes) support separate timing studies, with strict window checks. The four-hour monitor cannot guarantee those windows; they must not be represented as collected when absent. This command-line workflow does not yet intercept individual web-app prediction requests.

Report paired log loss, Brier score, AUC, and points RMSE. Game-cluster bootstrap intervals account for correlated players in the same game. The first checkpoint is **100 eligible settled player-games** in a horizon. Prespecified research success also requires at least20 games, a log-loss improvement of at least0.002 with its paired95% interval below zero, and nonworse Brier and RMSE. Subsequent checkpoints are exploratory; no automatic deployment or retuning. Warmup outcomes cannot satisfy the primary gate. No ROI or closing-line-value claim is made without suitable evidence.

## Collection and operation

The first snapshot cost2 free-plan credits. Cumulative local October pilot use is6 of20 reserved credits; the provider reported491 account credits remaining. The unchanged collector enforces a20-credit/month local cap and50-credit account reserve. At most one collection attempt per UTC day; the monitor conserves credits until the next game has training-eligible players. This limited sampling may take several weeks or longer to reach100 eligible observations.

After refreshing the existing free current-season caches, the recurring monitor runs:

```sh
DATABASE_URL=postgresql://rithv@localhost/nba_props .venv/bin/python -m scripts.monitor_prospective_points
```

Grade/report without spending quote credits:

```sh
.venv/bin/python -m scripts.validate_prospective_points
```

The current result is `logs/prospective_points_v1/report.json`. Quote failures are conservatively reserved against the budget; inspect durable attempts before considering a retry. Frozen model/source hash drift deliberately stops new runs for review. A new feature contract or model belongs in a separately versioned prospective study, never an edited existing protocol.

Validation:13 focused tests passed, including pregame/future/stale guards, horizon checks, immutable records, duplicate sample rejection, DNP/push/missing-result handling, and paired warmup separation. Settlement parsing was also checked against a completed official NBA boxscore. All14 paired predictions independently reproduced exactly from archived feature vectors and real quote lines; protocol/input hashes and pregame timestamps were verified.

The existing four-hour heartbeat is active under “Collect NBA data and validate points models.” It retains defender qualification work and now runs points grading/eligible collection. It notifies on the first completed warmup results,100 eligible settled forecasts, later100-case checkpoints, or persistent failures requiring action; routine unchanged states remain quiet.

### Historical quote import repair (October 25, 2026 ET)

The original zero-accepted result conflated source validity, model history eligibility,
and duplicate bookmaker observations. The importer used scheduled start as actual tipoff.
A separate research qualification now verifies the opening NBA period marker against
ESPN's opening jump-ball UTC timestamp, requires agreement within 60 seconds, and uses
the earlier bound. It does not grant a blanket late-quote grace period or change frozen
prospective rules. NBA's legacy description says EST even during daylight time; Eastern
local conversion is accepted only with the independent UTC corroboration.

Provider player IDs are resolved through the historical ESPN event roster, then matched
uniquely to NBA name, team and date; event teams and scheduled starts must also agree.
Carlton/Bub Carrington has a missing provider ID: official CommonPlayerInfo PLAYERCODE
`carlton_carrington`, display name and NBA ID verify the alias, then the historical ESPN
roster supplies the provider identity. No fuzzy name matching is used.

Results: 462/463 bookmaker quotes qualify, representing 79 player-games across five
games. The incomplete Jaxson Hayes quote lacks under odds and remains rejected.
65 player-games across four games have feature-ready current-season history. The 14
opening-night player-games remain valid source records but cannot enter the frozen v5
feature policy with zero current-season appearances. Quotes and previous rejection
reports are retained. This is not enough data for the unchanged minimum of 100 rows
and 20 independent games in each chronological split; no new model was trained.

Evidence: `logs/real_points_qualified_v2_final/qualified.json`, official timing and ESPN
raw responses in `logs/real_points_sources/`, and `logs/real_points_training_v2/`.
The original v1 and v5 source/model/protocol hashes were verified unchanged. No further
DunkAPI or Odds API credits were used. These near-tipoff historical quotes do not establish
60-minute or 15-minute prospective availability or independently audit the provider's
archival capture clock. Continue the frozen prospective studies unchanged.

### Expanded free historical collection (October 25, 2026 ET)

Collected 14 additional games under the existing 200-credit monthly ceiling (now 200
reserved; no paid purchases and no Odds API calls). Selection was fixed before collection:
first available provider event ID on each of 14 prespecified dates spanning the four
chronological splits. The plan and free ESPN schedules are archived in
`logs/real_points_batch_01/plan.json`. Results were not used to choose games.

Combined archive: 1,740 bookmaker quotes; 1,730 pass source qualification, representing
301 player-games across 19 games. Ten quotes remain rejected: nine unresolved missing
provider identities (Moe Wagner, B.J. Boston Jr., Jaylin Williams), one missing under price.
Fifteen zero-current-season-history cases are retained as qualified source records but
excluded from the existing feature policy. Feature-ready dataset: 286 player-games in
18 independent games. No bookmaker duplicates are counted as extra player-games.

| Split | Player-games | Independent games | Required |
|---|---:|---:|---|
| Fit before January 2025 | 77 | 5 | >=100 rows and >=20 games |
| Calibration March–April 2025 | 81 | 5 | >=100 rows and >=20 games |
| Selection May–June 2025 | 62 | 4 | >=100 rows and >=20 games |
| Evaluation 2025–26 | 66 | 4 | >=100 rows and >=20 games |

At least 62 further usable games are needed, before allowing for source rejection or
coverage gaps. At 10 credits/game that is at least 620 more credits. With the existing
200-credit monthly collection cap this requires at least four further allowance periods,
and provider free-credit refill timing must also be respected. A calendar-month local
reset is not evidence of an account refill. No model training or threshold relaxation
was performed. The 20-game floor is a minimum readiness check, not proof of statistical
power or representativeness; these results remain retrospective research.

Alternative-source research: DunkAPI advertises recurring 500 free credits, but its
`get_prop_history` shortcut lacks quote capture timestamps in the documented response
and defaults to a cross-book median. It cannot replace the timestamped bookmaker records.
OddsLink free excludes props/history; its history plan starts at $99/month. PropLine's
$99 historical export starts April 2026, too late for the fitting/calibration splits.
Public repositories found in the search offered synthetic odds or collector code, not a
verified replacement archive spanning all required periods. No alternative was activated.
Sources: https://dunkapi.com/pricing ; https://dunkapi.com/docs/get-prop-history ;
https://oddslink.app/pricing/ ; https://prop-line.com/historical-backfill .

Reproducible commands: `python -m scripts.collect_real_points_batch` resumes this batch
without re-buying completed requests; `python -m scripts.prepare_real_points_batch`
archives free evidence and builds separate outputs; `python -m
scripts.report_real_points_readiness logs/real_points_batch_01/training/real_points.joblib`
checks independent-game coverage and verifies both frozen studies. The finished batch
is not automatically extended on a credit reset. Subsequent batches need new dated
plans and output directories, preserving this evidence. Twenty-six focused tests passed.

### Full free-credit allowance authorized (October 25, 2026 ET)

The 200-credit ceiling was an agent-imposed conservative research limit, not a provider
restriction. The user explicitly authorized the remaining free credits. The shared
historical collector ceiling is now 500 credits; the existing reservations and attempts
were preserved. No subscription or wallet purchase was made. A no-charge estimate
returned 10 credits/game but exposed no account balance, so the ledger still includes
10 credits conservatively reserved for the earlier failed wide query. It now records
500 reserved credits: 49 successfully archived games (490 credits) plus that uncertain
10-credit reservation. Do not interpret the ledger as a verified provider balance.

Batch02 added 30 games using 300 more credits. Its prespecified dates span the four
chronological splits; empty selection dates were replaced using separately archived
supplement plans. The initial provider HTTP429 on a no-charge estimate prompted slower
requests: 13 seconds per game accounts for both estimate and fetch calls under the
low-balance rate limit. Attempts were retained, not erased. Combined raw archive:
4,499 bookmaker quotes across 49 games. No Odds API credits were spent.

The earlier four-further-month estimate is superseded by this authorization. The monitor
instructions now reflect the 500-credit ceiling and repaired source qualification;
calendar resets still do not establish that provider free credits have refilled. The
frozen v1/v5 studies, their predictions, model artifacts, cutoff rules and prospective
collection budget remain unchanged. Historical outputs are in `logs/real_points_batch_02/`.

Batch02 final results: 4,480 qualified bookmaker quotes representing 764 unique
player-games; 19 rejected for unresolved identities/missing odds. Fifteen zero-history
cases remain excluded from the existing feature policy. The separate feature dataset
now contains 749 usable player-games across 48 independent games:

| Split | Player-games | Independent games |
|---|---:|---:|
| Fit | 179 | 12 |
| Calibration | 190 | 12 |
| Selection | 197 | 12 |
| Evaluation | 183 | 12 |

Every split passes the 100-row floor but still needs eight more independent games.
At least 32 additional usable games (320 credits at current per-game cost) remain.
The next verified 500-credit free refill could cover that minimum if enough fetched
games qualify. Do not treat this as proven predictive gain or sufficient power by itself.
Read `logs/real_points_batch_02/training/readiness.json` for exact counts, dataset hash
and successful frozen-study verification. Twenty-seven focused tests passed, including
allowing the final request at 490 credits and blocking the next metered call at 500.
No model training or deployment was attempted while the game-count gates remain unmet.

### Additional free public archive (October 25, 2026 ET)

Found and imported six raw bookmaker CSV snapshots from
https://github.com/mostgood1/NBA-Betting . Files are pinned to an immutable Git tree and
SHA256 hashes in `logs/real_points_source_search/public_archive_manifest.json`; the
archive collector source was inspected and saved without executing it. Provider timestamps
and local snapshot timestamps are retained. These are third-party archived prices;
NBA checks independently validate identities/start times, not sportsbook price authenticity.

The new importer requires complete Over/Under pairs at the same book, line and snapshot;
rejects capture at/after provider start; requires timezone-aware source updates no later
than capture and no more than15 minutes old. Official NBA scheduled-start validation
is then applied, with unique NBA player/date resolution and home/away checks. No timing
guard was relaxed. One record per player-game is chosen without looking at outcomes.
In-game observations, incomplete sides, stale/future timestamps and ambiguous names
are retained in rejection reports. The201 home/away-label mismatches remain excluded
rather than guessed; nine selected cases have no current-season history.

The public source contributes432 usable player-games from32 additional games, all in
2025–26. Merging with the existing research archive preserves existing records for any
overlap and keeps explicit per-row source provenance. Final dataset:1,181 player-games
across80 games; source files and prior datasets were preserved.

| Split | Player-games | Independent games | Ready |
|---|---:|---:|---|
| Fit |179|12|No|
| Calibration |190|12|No|
| Selection |197|12|No|
| Evaluation |615|44|Yes|

Even though32 additional games were found, they cannot replace the required eight games
in each earlier split. Remaining gap:24 games, before allowing for source failures.
Do not move evaluation games into training or change chronology to make this pass.
No model was trained and both frozen v1/v5 hashes passed verification. No metered data
calls, paid purchases, or protected-season imports occurred.

Evidence: `logs/real_points_public_01/training/report.json`,
`logs/real_points_merged_03/{real_points.joblib,provenance.json,readiness.json}`.
Source survey: `logs/real_points_source_search/source_audit.json`. New sources checked
included Statyx (history paid), Flash Props (NBA history begins in protectedOctober2026),
OddsPapi (documented history beginsJanuary2026), BettingPros (current free odds advertised,
required historical coverage unverified), and multiple public repositories. No verified
free source for the remaining pre-2025/March–June2025 gaps was found in this search.
