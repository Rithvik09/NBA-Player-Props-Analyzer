# Current-season defender investigation — 2026-10-25

## Recurring monitor update — October 25, 22:51 UTC

`python -m scripts.monitor_defenders` now performs incremental free collection,
archives each raw source response and prior player cache, verifies frozen hashes,
grades immutable shadow forecasts, and saves new pregame assignments without
rerunning the opened holdout. Run it before `scripts.monitor_prospective_points`.
It preserves pending, missing-source, DNP and rejected cases, deduplicates the
holdout/prospective union, and keeps assignment gates separate from seven-field
coverage qualification. Evidence is in
`logs/source_blocked/assignment_ranker/monitor_runs/` and `monitor_report.json`.

This check found no newly completed games: 36 cached games through October24,
17 defender forecasts still pending, no source failures, no new accepted shadow
forecasts, and unchanged45-case holdout results. Frozen model/source hashes match.
Fourteen focused defender/points tests passed. No qualification, retraining,
promotion, or additional odds-credit use occurred.

**The seven defender features remain unqualified for 2026–27. No gate was lowered.**
The new work improves evidence and coverage but does not yet justify activation.

## New frozen assignment model, fresh season test

The earlier approach needed a strong previous matchup between the same offensive
player and opponent. A separate defender-assignment ranker now also uses earlier
boxscore roles, defending history against similar offensive roles, recent minutes,
and pair history. It never uses target-game lineups, minutes or assignments.
Candidates exclude the offensive player himself, including after trades.
Unknown guarding roles stay missing. The player-prop models are untouched.

The ranker was fit on outcomes before October 2024. Its acceptance cutoff was
selected on October 2024–September 2026 data; this is explicitly **development**,
not a fresh holdout. The selection rule maximized coverage subject to >=55%
development accuracy, >=100 cases and baseline+5 percentage points. Model and
source hashes were frozen before inspecting the final current-season result.
A self-defender identity bug was corrected before any holdout result was read;
the preliminary artifacts were retained unopened for traceability.

All 36 completed games from October 20–24 were fetched from free official NBA
endpoints and validated against 812 player boxscores. The one timed-out game
succeeded on retry. This added 7,215 matchup rows. No odds credits or paid feeds
were used for this investigation.

| Current-season replay | Previous rule | Broader frozen ranker |
|---|---:|---:|
| Forecasts accepted | 8 | 45 |
| Available player matchups | 805 | 805 |
| Assignment coverage | 0.99% | 5.59% |
| Correct assignments | 4 | 22 |
| Accuracy | 50.0% | 48.9% |
| Minutes baseline on accepted cases | 12.5% | 13.3% |
| Pass unchanged qualification? | No | No |

The ranker increases assignment coverage **5.6×**, but fails both the >=100
forecast requirement and >=50% accuracy requirement. The baseline+5-point gate
also remains unchanged. The candidate confidence cutoff is a separate development
choice; it does not replace or lower any qualification threshold. Do not retune
on this now-opened holdout and describe the same period as independent validation.

The frozen model also produced **17 genuine pregame shadow assignments across
five upcoming October 25 games**, with prediction timestamp, NBA game ID,
official tipoff, model hash and fresh roster provenance saved. These are pending
results, not successful validation, and do not reach serving features.

Evidence: `logs/source_blocked/assignment_ranker/frozen.json`, `holdout.json`,
`shadow_20261025T180755.json`, and `logs/source_blocked/current_matchup_crawl.json`.
Research entry points: `scripts/research_defender_assignment.py` and
`scripts/evaluate_current_defenders.py`. Both refuse to overwrite a frozen or
opened experiment. Further qualification needs additional unseen games with the
model/cutoff unchanged, or a new model evaluated on a different future period.

## A real train/serve coverage mismatch fixed

Qualification used earlier NBA appearances across season boundaries, but the
feature builder used only appearances in the target season when determining the
expected defender's team. Opening-week rows consequently lost legitimate prior
roster evidence. The builder now merges the prior-season and current-season logs
and takes only appearances strictly before the decision date. A later transfer
still overrides the earlier team. A regression test covers both cases.

Current-season player logs and validated matchup observations are cached for
future feature construction. This does not override failed qualification.

Measured opening-week assignment coverage in the three qualified historical
seasons rises from **27 to 46 eligible assignments (+70%)** across 3,461 player
matchups. These are eligibility counts, not proof every downstream trait is
available. Evidence: `logs/source_blocked/opening_week_roster_repair.json`.
The full regression suite passed **712 tests** (481 warnings). The four-season
11,548-row audit still has 135–153 observations per defender field: that audit
skips each player's first ten appearances, whereas this repair primarily helps
opening week. No active column is wholly missing or constant and no fraction
violates its bounds. Serving-model hashes are unchanged. An independent replay
reproduced every frozen current-season prediction and metric exactly after
import, including safeguards against ingesting holdout games twice.

---

# Free-data follow-up — 2026-10-25

This section supersedes the dated results below. No subscription was purchased.

## Free collection is running as a bounded, one-shot command

Two snapshots for Lakers at Jazz (NBA `0022600112`) cost **4 credits**, leaving
**493**. The first preserved nine game totals but rejected player props because
roster evidence was refreshed after capture. The second, after fixing ordering,
produced **27 paired points quotes and nine totals**, with no identity rejection.
Across 16 distinct player/line decisions, replay produced bookmaker count,
implied Over probability and game total for all 16, and line dispersion for 15.
These are capture-time source probes, not 60/15-minute or actual-user-request
prediction evaluations. Those three validation horizons remain distinct.

`python -m scripts.capture_free_market` samples one event, one region and two
markets. Durable reservations cap this pilot at 20 credits/month and preserve
50 account credits. Failures consume reservations conservatively. It does not
schedule polling. Raw and canonical evidence and replay live in
`logs/source_blocked/free_market/`. Main models and training activation are unchanged.

Provider `last_update` means last seen by the provider, not when a bookmaker
changed its price. The canonical rows preserve this distinction. Such timestamps
can establish snapshot freshness but cannot establish a stale unchanged price.
First seen is not a certified opening. Neither is silently promoted into one.

## Defender rules: partial recovery, current season still fails

We compared 20 predefined recent-game/decay/concentration rules. For each target
season, selection used October–December of the previous season; qualification
used the following January–April. Same-day outcomes were withheld. The same
selection helper now runs in qualification and feature construction, with the
same regular-season-only assignment history and roster evidence.

| Target season | Held-out forecasts | Assignment accuracy | Qualified |
|---|---:|---:|---|
| 2022–23 | 981 | 47.4% | No |
| 2023–24 | 274 | 54.0% | Yes |
| 2024–25 | 427 | 52.5% | Yes |
| 2025–26 | 296 | 52.4% | Yes |
| 2026–27 | 147 | 49.0% | No |

The existing gates (100 observations, 50% accuracy, baseline +5 percentage points)
were not relaxed. Qualified rules are stored per season in the evidence bundle.
Their Wilson lower bounds are below 50%; these are preliminary assignment gates,
not strong statistical proof or evidence of improved player-prop predictions.
The seven fields can populate only qualifying historical rows with sufficient
measurements. **They are not qualified for the current season.** Missing values
remain missing; no actual target-game defender was used as a pregame assignment.

The seven fields are `primary_defender_score01`, `def_rating_individual`,
`primary_defender_rating`, `primary_defender_age`, `defender_size_mismatch`,
`defender_recent_form`, and `primary_defender_active`.

Rebuild sequence after adding matchup sources:

```sh
python -m scripts.build_defender_evidence
python -m scripts.qualify_defender_rules --promote
python -m scripts.audit_all_features
```

Regression suite: **707 passed** (481 warnings). Independent replay of 201
real forecasts found zero differences between research and feature construction.

Final stable-source replay: 11,548 player-games, all 416 slots checked. Each defender feature has 135–153 observed rows (1.17–1.32% overall). No active column is wholly missing or constant; no fraction violates its bounds. Current-season defender qualification still fails. Serving-model hashes are unchanged. Detailed counts: logs/source_blocked/report.json.

## All 14 remaining capabilities

| Feature | Actual disposition | Is payment the missing requirement? |
|---|---|---|
| `implied_game_total` | Free live source captured and replayed; no historical training coverage yet. | Paid archive can accelerate backfill; free forward collection works. |
| `implied_over_prob` | Paired real Over/Under prices captured and replayed at the exact decision line. | Same: paid for immediate historical coverage, not live access. |
| `market_consensus_std` | Multiple real books captured; observed in 15/16 probe decisions. | Same. |
| `bookmaker_count` | Verified books captured; observed in 16/16 probe decisions. | Same. |
| `opening_line` | Snapshot feed does not certify true opening; first-seen quotes are rejected as openings. | Paying for snapshots alone does not prove opening provenance. |
| `line_movement` | Current definition compares same-book certified opening to decision quote; opening evidence absent. | Requires qualified opening history, not just any paid feed. |
| `line_velocity` | Current adapter measures opening-to-current rate; blocked by the same opening provenance requirement. Interval movement exists separately in sharp research. | A verified timestamped history may help; paid snapshots alone do not establish true opening. |
| `stale_line_flag` | Provider last-seen timestamps do not establish unchanged-quote age; suppressed for this source. | Needs verified quote-change history or repeated observations with a separately validated definition. No paid source verified for this definition. |
| `sharp_action_score` | Signed components and three-horizon validation implemented; no qualifying held-out paired prediction dataset. | Validation/data accumulation is required even with paid odds. |
| `model_accuracy_player` | Logging and strict OOS adapter exist; no genuine eligible prediction ledger. Requires 20 non-push games per player/prop/model. | No. Must accumulate genuine predictions and grades. |
| `avg_prediction_error_player` | Same ledger requirement; at least 20 graded forecasts. | No. |
| `calibration_score_player` | Same ledger requirement; at least 50 non-push forecasts. | No. |
| `defender_switching_frequency` | All switching values zero in 1,214,161 validated matchup rows; schema presence is not a working source. | No populated free or paid source verified. |
| `avg_shot_clock_time` | No verified shot-level shot-clock observations; game clock and bucket shares cannot recover the exact mean. | No suitable free or paid source verified. |

The 14 remain excluded from historical model training. Free capture fixes access
for four capabilities; it does not fill earlier seasons or prove incremental utility.

[The Odds API pricing](https://the-odds-api.com/), checked October 25, lists
500 free credits without historical access, $30/month for 20,000 credits with
history, and $59/month for 100,000. No purchase is needed to continue the bounded
forward trial. Historical player-prop coverage begins May 3, 2023 according to
[the historical data documentation](https://the-odds-api.com/historical-odds-data/);
it cannot fill earlier dates merely by upgrading.

---

# Expanded-source follow-up — 2026-10-24

This section supersedes the earlier partial-archive results below.

The public [NBA matchup archive](https://huggingface.co/datasets/cdechoch/nba-data-archive)
added 5,957 games after validation against cached NBA player/team boxscores.
319 overlapping games agreed exactly; 167 conflicting games retained our original
observations, 87 lacked local boxscores, and 37 failed identity validation.
Source hashes and rejection reasons are in `logs/source_blocked/archive_import.json`.
The resulting evidence has 6,450 games and 1,214,161 rows. All switching counts
remain zero. This is broad matchup coverage, not a populated switching feed.

The larger sample invalidated the earlier defender qualification:

| Target season | Earlier forecasts | Accuracy | Qualified |
|---|---:|---:|---|
| 2022–23 | 3,187 | 44.1% | No |
| 2023–24 | 3,096 | 42.2% | No |
| 2024–25 | 3,159 | 47.9% | No |
| 2025–26 | 3,045 | 44.7% | No |
| 2026–27 | 2,447 | 42.5% | No |

The unchanged 50% gate remains enforced. Seven implemented defender paths now
produce missing values for every audited season. They must not be described as
validated or usable merely because the registry labels their paths active.
There are 14 source-blocked columns plus these seven failed qualification paths.

Validation of the current implementation: **700 tests passed** (481 warnings),
including future-input rejection and separate 60-minute, 15-minute and request-time
evaluation. Log: `logs/source_blocked/final_followup_suite.log`.

The final stable-source replay covered 11,548 player-games. Serving-model hashes
are unchanged. Exactly seven conditional defender fields are wholly missing;
no observed active field is constant or violates its fraction bounds.
Final report: `logs/source_blocked/report.json`.

## Capture and replay repairs

- Prediction logs now retain the NBA game ID, zoned tipoff, and schedule source
  at prediction time. NBA CDN lookup has a date/team-validated ESPN fallback
  only when the official scoreboard supplied the NBA ID. ESPN IDs remain separate.
  Missing schedule provenance stays unknown. Automatic grading now honors a
  known game ID rather than falling through to a neighboring date.
- Prop capture keeps missing prices null, rejects invalid prices, and never pairs
  Over/Under at different lines. Ambiguous multiple lines for the same main market
  are excluded. Legacy summary heuristics remain unsuitable for audited features.
- Added a strict full-game total adapter and a canonical history replay command.
  It reports observed feature counts without activating columns.
- Read-only database inspection found zero logged predictions, no prop/game odds
  tables, and 1,323 existing game-level betting-split rows. No historical
  predictions or timestamps were fabricated.

Canonical history JSON files are lists. Game and player IDs must be verified NBA
IDs before replay. Quote rows require `game_id`, `player_id`, `prop_type`,
`bookmaker`, `line`, `published_at`, `observed_at`, `game_start`, `source`, and
optional `over_price`, `under_price`, `is_opening`, `is_alternate`. Only a
source-certified market opening may set `is_opening`; first seen is insufficient.
Game totals use `market: totals`, no team/alternate/partial-game qualifiers.
Predictions require the fields documented in `src/source_evidence.py`.

```sh
.venv/bin/python -m scripts.export_prediction_history --output logs/source_blocked/prediction_history.json
.venv/bin/python -m scripts.audit_source_history --decisions decisions.json --quotes quotes.json --totals totals.json --predictions predictions.json --output logs/source_blocked/history_replay.json
```

## Data acquisition still required

No paid requests were made or subscriptions purchased. Official NBA CDN probes
returned HTTP errors; ESPN schedule access succeeded. No validated shot-level
shot-clock source was found. Bucket shares still do not establish the exact mean.

Provider options checked on 2026-10-24:

- [The Odds API](https://the-odds-api.com/) lists historical access at $30/month
  for 20,000 credits. Its documented prop archive begins May 3, 2023. A two-snapshot
  1,230-game season for one market and one region is about 24,600 credits before
  discovery calls, missing coverage and retries. This estimate is not a purchase.
- [DunkAPI](https://dunkapi.com/pricing) offers a free 500-credit tier and paid
  plans starting at $19/month. Historical endpoints warrant an authenticated
  sample check; closing-only history cannot establish intraday movement.
- [PropLine](https://prop-line.com/docs) advertises a $99 backfill pass with a
  trailing two-year window. Its documented opening is first observed, which
  does not satisfy our certified-opening definition. NBA depth remains unverified.

`sharp_action_score` now has an agreed research definition: signed Over/Under
pressure combining money/ticket imbalance, reverse movement, and incremental
market information. Game-level context is separate. Evaluation horizons are
60 minutes, 15 minutes, and the actual request timestamp. Components and a
held-out validation harness are implemented, but no qualifying holdout data or
trained score exists yet. Details: `SHARP_ACTION_RESEARCH.md`.

Further free-source checks found that The Odds Gap serves a fixed September
15–21, 2026 sample without a key (HTTP 200); its recent archive is paid. That
sample window does not fill the 2022–26 NBA training history. Its published
price is $10 for the first month, then $25/month, with archive start June 2026:
https://theoddsgap.com/data . A GitHub basketball-data index advertised detailed
odds in its README, but the named file returned 404 and its NBA directory had
no odds/line/prop files. Documentation claims alone were not accepted as data.

---

## Earlier partial-archive audit (historical; superseded)

# Source-blocker follow-up

Seven of the former 21 blocked columns now have conditional source paths.
The v3 candidate contract has 369 active columns, 33 duplicates and 14 remaining
source blockers. No serving player-prop model was trained or replaced.

## Actual defender observations, qualified forecast

`BoxScoreMatchupsV3` supplies NBA offensive/defensive player identities, tracked
partial possessions and matchup scoring. The collector validates game identity,
player identity, team opposition and nonnegative numeric measurements against
cached dated NBA boxscores. The local archive contains 493 measured games and
92,848 matchup rows. The endpoint subsequently returned repeated timeouts; the
bounded crawler stopped rather than declaring the requested crawl complete.

The regular-season-only expected-defender rule uses only earlier player/opponent matchups
within 400 days. It requires at least 20 partial possessions, a leading defender
share of at least 40%, a lead of at least 15 percentage points, and latest earlier
NBA team membership matching the opponent. This is inferred recent membership,
not a certified roster or a confirmed future assignment. Injury status is
reported separately; unavailable defenders are not silently replaced by a guess.

Regular-season forecasts are enabled only if the preceding regular season's
backtest has at least
100 forecasts, accuracy >=50%, and a margin of at least five percentage points
above selecting the opponent's highest-minute player. These are preliminary
source-qualification gates on a partial archive, not independent proof of model
utility or validation of every team's assignment behavior.

| Target season | Earlier forecasts | Assignment accuracy | Minutes baseline | Qualified |
|---|---:|---:|---:|---|
| 2022–23 | 294 | 59.5% | 15.3% | Yes |
| 2023–24 | 20 | 15.0% | 5.0% | No |
| 2024–25 | 17 | 52.9% | 0.0% | No |
| 2025–26 | 1 | 0.0% | 0.0% | No |
| 2026–27 | 9 | 55.6% | 22.2% | No |

Postseason and unknown game types remain missing. The current archive qualifies
only 2022–23; it does not establish coverage for the latest season.

Seven conditional fields use that forecast: prior official on-court defensive
rating, lagged age, height difference, confirmed availability, actual tracked
matchup scoring allowed, recent scoring-allowed change, and an observed
scoring-efficiency percentile. The percentile is descriptive, not a calibrated
probability. Official on-court rating and individual matchup scoring are distinct
measurements. Minimum matchup samples and comparison cohorts are required.
Unknown measurements stay native NaN. They are not replaced by team averages.

Evidence: `logs/source_blocked/defender_qualification.json`,
`defender_backtest.json`, and `defender_evidence.joblib`.

## Why switching stays blocked

Every collected `switchesOn` field is zero across all 92,848 rows. Merely seeing
a schema column does not establish actual coverage. The capability remains
blocked until a populated tracking source is verified. Lineup changes and the
number of different defenders are not counts of defensive switches.

## Prediction history: provenance repaired, observations still needed

New predictions record timezone-aware observation time, model version,
feature policy and a conservative model training-outcome cutoff. New grades
record when their outcome was actually entered. The audited training CLI writes
its latest outcome-data boundary into metadata; legacy model cutoffs are not
guessed from artifact file times. Existing metadata without this boundary stays
unknown. These repairs take effect when updated application code is running;
no historical prediction ledger was manufactured.

The pure adapter in `src/source_evidence.py` rejects training outcomes later
than prediction time, in-game/retrospective predictions, future or unknown
outcome availability, unzoned timestamps and other model versions. It keeps one
latest pregame prediction per game. MAE and binary directional accuracy require
20 eligible games; calibration uses 1 minus five-bin expected calibration error
and requires 50 eligible non-push games. Pushes are excluded from binary scores.
These three features remain blocked until there is qualifying real history and
an explicit activation audit. Existing pilot probabilities do not establish a
matching, served out-of-sample prediction ledger.

## Market history: capture repaired, historical feed still needed

Quote storage now preserves provider publication time separately from our
observation time, for both prop and game lines. Old rows are not assigned made-up
publication times. The pure adapter checks both timestamps against the decision
cutoff and tipoff, excludes alternate lines, compares movement within the same
book, uses explicit source-marked opening quotes, and keeps normalized over
probability tied to the actual decision line. It cannot safely consume a latest
summary or synthetic training lines as if they were archived prices.

The market features remain blocked until trustworthy timestamped quotes are
available and joined to validated NBA game/player identities. The capture and
adapter repairs are infrastructure, not historical observations or automatic
activation. Paid polling was not enabled and no Odds API credits were spent.

The provider documents historical snapshots as a paid-plan capability, with
player props beginning May 3, 2023:
https://the-odds-api.com/historical-odds-data/

Even a paid quote archive does not establish `sharp_action_score`: that field
still needs an objective definition/source. Line movement, tickets/money
imbalance, and an arbitrary multiplier are not proof of professional action.

## Exact shot-clock mean

The official shot-chart schema exposes game-clock minutes/seconds, not actual
shot-clock time at each shot. The already-active early/late bucket frequencies
are useful but do not identify the exact mean. Possession elapsed time is also
insufficient because of offensive rebounds, fouls and clock resets. This field
requires a validated event-level shot-clock source and remains blocked.

## Reproduce

```sh
.venv/bin/python -m scripts.crawl_defender_matchups
.venv/bin/python -m scripts.build_defender_evidence
.venv/bin/python -m scripts.audit_all_features
```

Run these sequentially: the coverage audit requires stable source files.
The matchup crawler has bounded batches, short timeouts and a failure circuit
breaker. Resume does not discard already-validated game files. Qualification
must be rebuilt after adding data; a successful forecast in one season does not
enable other seasons.

## Final measured validation

691 tests passed. The stable-source replay covered 11,548 player-games; serving-model hashes are unchanged. No active column is wholly missing or constant, and no observed fraction violates [0,1]. The seven new defender inputs are observed in only 79–91 rows each (0.68–0.79%), all in the qualifying 2022–23 season. These are conditional source paths, not broadly usable latest-season coverage. Per-feature counts are in `logs/source_blocked/report.json`. The leftover SQLite database also has no prediction or prop-quote rows to recover.
