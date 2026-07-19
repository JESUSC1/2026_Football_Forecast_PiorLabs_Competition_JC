# Submission Compliance Implementation

**Date:** 2026-07-18
**Fixture:** Spain vs Argentina, 2026-07-19

## Implemented

- Pinned `results.csv` to upstream commit
  `80f408d2c93ba4f9e06a2c7cdc5effb05fea9680`.
- Enforced the normalized full SHA-256
  `7f2a1026c1e78d825b58deba2555d81331b0a80752c57e8e0a3332d1350e5d4c`.
- Added `data-manifest.json` with source, row count, checksum, and purpose.
- Added the explicit `competition/final-fixtures.csv` template and removed
  wall-clock-dependent fixture selection.
- Added a pre-fixture training cutoff so no result at or after the requested
  fixture date can enter training.
- Generated World Cup state across historical editions instead of exposing
  tournament features only in 2026.
- Defined knockout state deterministically after 48 group matches for 32-team
  editions and 72 group matches for the expanded 2026 edition.
- Added pre-match timestamp, duplicate-source, and decimal-odds validation for
  market snapshots.
- Made nested blend metrics and data provenance fields code-generated in
  `artifacts/ensemble_config.json`.
- Added an exact tested Python dependency lock and documented TabPFN browser
  authentication.
- Expanded the automated suite from 6 to 11 tests.

## Verification

- The pinned loader returned 49,520 rows through 2026-07-19 and matched the full
  expected checksum.
- Full rolling evaluation was rerun on the corrected feature matrix.
- A separate production run reproduced the post-evaluation Final CSV exactly.
- All tests passed.
- `git diff --check` passed.
- The optional v2.6/v3 focused benchmark refresh reached the daily API limit.
  Its existing artifact is explicitly revision-labelled as pre-correction; it is
  not used by the production v3 fit or Final CSV.

## Final model configuration

- TabPFN v3: 0.8489460285 model-blend weight.
- LightGBM: 0.1077292397 model-blend weight.
- Poisson: 0.0433247317 model-blend weight.
- Temperature: 1.0374973678.
- Market consensus: fixed 0.85 final-stage weight, explicitly documented as a
  judgment choice because historical market OOF data was unavailable.

## Canonical submission

```csv
date,home_team,away_team,p_home_win,p_draw,p_away_win
2026-07-19,Spain,Argentina,0.42009285153303505,0.31109835875415126,0.26880878971281374
```

The file uses TabPFN as the dominant model component, has the required six
columns and full country names, contains strictly interior probabilities, and
sums to one within machine precision.
