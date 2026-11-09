> Free pilot started October 25: two one-event snapshots cost four credits. The validated snapshot contains 27 paired points quotes and nine game totals. This is capture-time source verification, not a qualified 60-minute, 15-minute or request-time predictive holdout. Provider last-seen timestamps are not certified price-change times or true openings.

# Sharp-action research contract

Agreed with the user on 2026-10-24. Experimental; not enabled in serving.

## Meaning and scope

A signed signal: positive supports Over, negative supports Under. It combines
three research hypotheses: informed-action indicators, reverse movement, and
market pressure that improves predictions. It does not identify who placed bets.
Game-level pressure is permitted as context, but must be labeled and evaluated
separately from player-prop evidence. A higher game total does not mechanically
imply a player's Over, especially for steals, turnovers or rebounds.

Primary evaluation: 60 minutes before tipoff. Separate checks: 15 minutes before
tipoff and the actual user-request timestamp. Never pool these horizons into one
headline result. Requests at or after tipoff are outside this pregame contract.
`requested_at` is now recorded when analysis begins, separately from the time
its prediction is logged. Request evaluation requires that exact timestamp;
quotes collected afterwards are unavailable at the request cutoff.

## Evidence and limitations

[Action Network's methodology](https://www.actionnetwork.com/education/sports-betting-sharp-money-professional-picks/)
describes money/ticket differences and reverse movement as indicators, alongside
other evidence. It is an operator's methodology, not an independent guarantee.
An [NBA totals study](https://journals.sagepub.com/doi/10.1177/152700250000100205)
found that line changes improved forecast accuracy. A
[college-basketball study](https://www.jvlone.com/sportsdocs/GamblingBasketbal2013.pdf)
found that information content varied with market profile. Neither result
establishes incremental predictive value for today's NBA player props.

Consequences for this implementation:

- Keep money/ticket gap, prior-split reverse movement, money-confirmed movement,
  same-line de-vigged price change, and cross-book movement direction separate.
- Movement compares the same book at two observed times. It is an interval change,
  not a claimed opening-line change. Earlier ticket percentages determine whether
  subsequent movement is reverse; later cumulative shares cannot rewrite history.
- Deduplicate syndication by underlying book. Do not assume multiple vendors are
  independent books or label unnamed Action book IDs as specific sportsbooks.
- Preserve missing data. Unknown volume prevents sample-size confidence claims.
  Zero on both sides of a split is treated as unavailable by the feed diagnostic.
- Quote movement requires known publication and observation times. A fresh fetch
  of an old quote does not prove a fresh market price. Without publication time,
  a split may contribute a descriptive gap but not confirmed quote movement.
- Keep injury/news timing, market, book, and time-to-tipoff in the research data.
  Public injury news can move prices without demonstrating private information.
- No default assertion that a named book is always sharp. Any book weighting or
  lead/lag feature must be learned on an earlier period and tested separately.

## Score construction and validation

The proposed interpretable score is
`100 * (candidate_over_probability - baseline_over_probability)`: the incremental
percentage-point shift toward Over after incorporating qualified evidence.
For example, 0.55 to 0.58 is +3, not a 58% probability of professional action.
The baseline must already contain existing market features. This avoids counting
ordinary line movement a second time and calling that added information.

No equal-weight composite or trained weights have been substituted for evidence.
When paired data exists, fit a regularized candidate on an earlier development
period, freeze choices, then compare on later untouched dates. Run ablations:
baseline, +prop evidence, +game context, and both. Keep each prop family and
horizon visible. Fit preprocessing, book weights and missingness treatment only
on development data. A score fed back into another model needs out-of-fold or
strictly earlier predictions; never use its fitted training probabilities.

`src/sharp_evidence.py` implements the components, numeric signed conversion, and
an initial acceptance gate. The gate rejects future inputs, overlapping
selection outcomes, duplicate decisions, invalid probabilities and mixed
horizons. It requires at least 200 distinct games, 30 game dates, and 50 outcomes
on each side; these are minimum safeguards, not a power calculation. It uses a
paired bootstrap grouped by Eastern game date and requires the 95% interval to
show lower log loss and no Brier deterioration. Additional per-market stability,
reliability plots and ablations remain required before activation. A passing
helper result alone never activates a model or feature.

Closing-line value is supporting evidence, evaluated after the decision and kept
out of feature construction. Compare the same book/line and de-vigged prices;
report line CLV separately from price CLV. Profitability after prices and costs
is a separate question. No outcome or CLV claim has been measured here.

```sh
.venv/bin/python -m scripts.evaluate_sharp_evidence --predictions paired_holdout_60m.json --horizon 60 --selection-outcomes-through 2025-10-01T00:00:00Z --output logs/source_blocked/sharp_60m.json
# Repeat with separate files and --horizon 15 or --horizon request.
```

Input rows need `game_id`, `player_id`, `prop_type`, `cutoff`, `game_start`,
`input_latest_at`, `training_outcomes_through`, `result_recorded_at`,
`over_outcome` (0/1; exclude pushes), `baseline_probability` and
`candidate_probability`; request-time rows also require `requested_at`.
The selection boundary and baseline contents must be documented by the caller.

## Measured source availability

The existing database contains 1,323 game-level betting-split observations,
zero logged predictions, and no prop/game quote tables. A fresh free Action
Network sample contained eight games and 42 valid paired total-market splits.
14 book/game pairs were absent or failed pairing checks. Publication timestamps
and populated stake/ticket counts were unavailable. Some games had already
started and were correctly excluded from pregame component calculation.
The diagnostic retains provider IDs; it is not silently joined to NBA training
rows. See `logs/source_blocked/action_pressure_diagnostic.json`.

There is currently no qualifying paired holdout dataset. All three evaluation
horizons are supported, but empirical validation cannot yet be reported.
See `SOURCE_BLOCKERS.md` for free-source findings and paid alternatives; no
subscriptions or paid API calls were made.
