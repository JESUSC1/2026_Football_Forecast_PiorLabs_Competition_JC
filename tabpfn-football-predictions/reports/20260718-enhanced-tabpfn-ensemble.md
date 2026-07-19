# Enhanced TabPFN Football Ensemble — Implementation Report

**Date:** 2026-07-18  
**Competition:** Prior Labs World Cup Football Prediction  
**Final fixture:** Spain vs Argentina, 2026-07-19  
**Metric:** Three-class regulation-time log-loss (lower is better)

## Outcome

The original single-file TabPFN template was refactored into a reproducible,
leakage-safe modeling pipeline with 53 engineered features, rolling-origin
evaluation, LightGBM and Poisson ensemble members, probability calibration,
and a saved market-consensus input.

Canonical submission (`../final-predictions.csv`):

```csv
date,home_team,away_team,p_home_win,p_draw,p_away_win
2026-07-19,Spain,Argentina,0.41772418342418227,0.3114810330968958,0.270794783478922
```

The row was validated to use the exact six-column schema, contain probabilities
strictly between zero and one, and sum to exactly one within machine precision.

## Models tried

### TabPFN baseline

- Original 26-feature template.
- Latest API default, which resolves to TabPFN v3.
- Up to 10,000 recent completed matches, starting in 2014.
- Served as the baseline for the focused feature/version benchmark.

### Enhanced TabPFN

- 53 leakage-safe pre-match features.
- Explicitly benchmarked with `model_path="v2.6_default"` and
  `model_path="v3_default"`.
- v3 with enhanced features was selected for the full evaluation.
- v3 thinking mode (`medium`, optimized for log-loss) was tried once on the
  Final fit. Its market-blended output was very close to ordinary v3, but it
  was not selected because ordinary v3 had complete rolling validation.

### LightGBM

- Multiclass gradient-boosted trees on the same enhanced feature matrix.
- 350 estimators, learning rate 0.025, 15 leaves, maximum depth 5, and L1/L2
  regularization.
- Added useful model diversity and received a nonzero optimized blend weight.

### Poisson goal model

- Uses exponentially weighted attacking and defensive goal rates to estimate
  expected home and away goals.
- Converts the joint score distribution into home-win/draw/away-win
  probabilities.
- Weaker alone, but retained a small positive blend weight on broad validation.

### Constrained ensemble and calibration

- TabPFN, LightGBM, and Poisson probabilities were aligned to the fixed class
  order `home_win`, `draw`, `away_win`.
- Ensemble weights were optimized on pooled out-of-fold log-loss with
  nonnegative weights constrained to sum to one.
- Temperature scaling was accepted only because it improved rolling OOF loss.
- Learned weights: TabPFN 0.8870966, LightGBM 0.0643383, Poisson 0.0485651.
- Learned temperature: 1.0372414.

### Final market component

- Three saved 90-minute price snapshots: BetMGM, Oddschecker, and Kalshi.
- Each source was converted to margin-free probabilities and the component-wise
  median was normalized.
- The Final uses 85% saved market consensus and 15% calibrated model ensemble.
- Inputs, retrieval time, and URLs are stored in `../final-market-odds.csv`.

## Features added

- Standard and fast Elo ratings.
- Separate exponentially weighted attack and defence form.
- Points/form windows of 3, 5, 10, and 20 matches.
- Recent win, draw, goals-for, goals-against, and goal-difference rates.
- Opponent-strength and strength-of-schedule features.
- Rest-day difference and short-rest knockout proxies.
- Current-World-Cup games, points, goals, goals allowed, clean sheets, and
  recent knockout form.
- Poisson expected goals and its three outcome probabilities.
- Existing neutral venue, tournament importance, experience, rest, streak,
  Elo, form, and head-to-head features were retained.

All features are constructed before applying the current match result. Because
the source contains dates but not kickoff times, results are applied in daily
batches: no match can consume another result from the same date.

## Evaluation design

- Monthly rolling-origin folds from January 2024 through June 2026.
- 2,631 total out-of-fold matches across 28 non-empty folds.
- Training for a fold includes only matches dated before that fold.
- Separate daily expanding evaluation for all 36 completed competition matches.
- Earlier competition dates can inform later dates; same-day results cannot.
- All labels use the score after 90 minutes, excluding extra time and penalties.

Final clean metrics after correcting regulation-time labels and same-day leakage:

| Model | Rolling OOF log-loss | 36-match competition log-loss |
|---|---:|---:|
| Enhanced TabPFN v3 | 0.8581 | 0.8193 |
| LightGBM | 0.8686 | 0.8235 |
| Poisson | 0.9775 | 0.9665 |
| Calibrated broad ensemble | **0.8567** | 0.8266 |

The saved OOF CSV is the source of truth for exact recomputation. Exact rolling
scores are 0.8581371999, 0.8685994778, and 0.9774886018 respectively.

The broad ensemble was retained because it had the best pooled rolling loss.
TabPFN alone was strongest on the much smaller competition window. The Final's
85% market weight makes this model-choice difference small in the submitted row.

## Focused TabPFN benchmark

March, May, and June 2026 were used to compare v2.6/v3 and baseline/enhanced
features. The clean benchmark is stored in `artifacts/version_benchmark.csv`.

| Month | v2.6 baseline | v2.6 enhanced | v3 baseline | v3 enhanced |
|---|---:|---:|---:|---:|
| 2026-03 | 0.9405 | 0.9313 | 0.9379 | **0.9301** |
| 2026-05 | 0.5228 | 0.5194 | 0.5298 | **0.5057** |
| 2026-06 | 0.8699 | 0.8705 | 0.8704 | **0.8669** |

Enhanced v3 won all three representative folds and was promoted. Legacy v2 and
v2.5 were intentionally not tested because the focused budget favored the two
current plausible candidates.

## Data corrections

The upstream results file records post-extra-time scores for some knockout
matches, but the competition scores regulation time. Sourced overrides are
stored in `regulation-time-overrides.csv`:

| Match | Raw result | Regulation result | Competition label |
|---|---:|---:|---|
| Belgium vs Senegal | 3-2 | 2-2 | draw |
| Argentina vs Cape Verde | 3-2 | 1-1 | draw |
| Norway vs England | 1-2 | 1-1 | draw |
| Argentina vs Switzerland | 3-1 | 1-1 | draw |

These corrections are applied in memory; `results.csv` remains the reproducible
raw source cache. `../previous-matches-ground-truth.csv` contains the corrected
one-hot labels for all 36 completed matches.

## Reproduction

```bash
conda activate football_forecast
cd tabpfn-football-predictions

# Tests
python -m pytest -q

# Focused v2.6/v3 benchmark
python benchmark_versions.py

# Full rolling evaluation and Final generation
python predict.py --model-version v3 --evaluate \
  --output ../enhanced-final-predictions.csv

# Normal Final generation using saved ensemble configuration
python predict.py --model-version v3 \
  --output ../final-predictions.csv
```

On macOS, LightGBM requires `llvm-openmp` from conda-forge. The root
`API_Key_TabPFN` is ignored by Git and never logged. The supplied value was not
in the browser-JWT format accepted by `tabpfn-client`, so completed experiments
used the client's supported cached browser authentication.

## Files and artifacts

| Path | Purpose |
|---|---|
| `features.py` | Chronological feature generation and score overrides |
| `models.py` | Model adapters, class alignment, blending, calibration |
| `evaluation.py` | Monthly and competition rolling evaluation |
| `predict.py` | Submission entry point and CLI |
| `benchmark_versions.py` | Focused TabPFN version/feature comparison |
| `regulation-time-overrides.csv` | Sourced 90-minute corrections |
| `artifacts/fold_metrics.csv` | Per-fold log-loss |
| `artifacts/oof_predictions.csv` | 2,631 aligned OOF predictions |
| `artifacts/competition_predictions.csv` | 36 daily expanding predictions |
| `artifacts/ensemble_config.json` | Selected weights and temperature |
| `artifacts/version_benchmark.csv` | Clean focused benchmark |

## Limitations

- The competition window has only 36 matches, so its score has high variance.
- No historical bookmaker-odds archive was available for rolling folds; market
  data is therefore a reproducible Final-only component.
- The Poisson model is deliberately simple and does not use player-level xG.
- No lineups, injuries, suspensions, travel, weather, or player availability
  data are included.
- `results.csv` has dates but no kickoff timestamps, requiring conservative
  same-date batching.
