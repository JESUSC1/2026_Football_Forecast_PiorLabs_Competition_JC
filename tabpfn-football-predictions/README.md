# TabPFN Football Prediction — WC 2026 Competition Entry

Competition entry for the [Prior Labs World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/worldcup). Predicts 90-minute regulation-time outcomes (home win / draw / away win) for international football matches using an ensemble of TabPFN v3, LightGBM, and a Poisson goal model.

**Final submission:** Spain vs Argentina, 2026-07-19  
**Submitted file:** `../enhanced-final-predictions.csv`  
**Pipeline:** 85% market consensus + 15% calibrated model ensemble (TabPFN 88.7%, LightGBM 6.4%, Poisson 4.9%)

---

## Results

| Model | Rolling OOF log-loss (2,631 matches) | Competition log-loss (36 matches) |
|---|---|---|
| Enhanced TabPFN v3 | 0.8581 | **0.8193** |
| LightGBM | 0.8686 | 0.8235 |
| Poisson | 0.9775 | 0.9665 |
| Calibrated ensemble | **0.8567** | 0.8266 |
| Leaderboard leader | — | 0.8240 |

**Nested CV result (honest, unbiased):** ensemble 0.8331 vs TabPFN alone 0.8301 on 993 held-out matches. The blend provides no measurable benefit over plain TabPFN; see `artifacts/ensemble_config.json` for details.

---

## Setup

```bash
conda create -n football_forecast python=3.11 pip -y
conda activate football_forecast
pip install -r requirements.txt
# macOS only (LightGBM dependency):
conda install -c conda-forge llvm-openmp
```

---

## Reproduce the Final prediction

```bash
# 1. Download the dataset (first run only)
python predict.py --refresh

# 2. Generate the Final prediction using saved ensemble config and market odds
python predict.py --model-version v3 --output ../final-predictions.csv
```

`artifacts/ensemble_config.json` contains the saved blend weights and temperature from the rolling evaluation. `../final-market-odds.csv` contains the three bookmaker snapshots used to compute the 85% market component.

To regenerate the ensemble weights and run the full rolling evaluation (slow — calls TabPFN API):

```bash
python predict.py --model-version v3 --evaluate --output ../final-predictions.csv
```

---

## Reproduce the LightGBM analysis

```bash
# Evaluate LightGBM on the 36-match competition window and generate the report
python predict_lgbm.py --evaluate --report
```

---

## Architecture

### Feature pipeline (`features.py`)

Single chronological forward pass through all ~47,000 matches. Every feature is computed before applying the current match result — no data leakage by construction. Same-day matches are batched so no match can consume another result from the same date.

53 features total, including:

- **ELO ratings** (standard + fast-decay variant): home/away ELO, ELO diff, fast ELO diff
- **Form** (3, 5, 10, 20-match windows): points per game, goal rates, win rates, EWM-weighted versions
- **Attack/defence ratings**: exponentially weighted goals scored and conceded
- **Strength of schedule**: opponent ELO adjustment
- **Head-to-head**: win/draw rate, average goal difference (last 14 meetings)
- **Rest**: days since last match, rest difference, short-rest proxy
- **Current-tournament form**: games, points, goals, clean sheets, knockout points (WC 2026)
- **Poisson model outputs**: expected goals and derived home/draw/away probabilities
- **Match context**: neutral venue flag, tournament importance score

**Regulation-time corrections:** `regulation-time-overrides.csv` supplies sourced 90-minute scores for four WC 2026 knockout matches where `results.csv` records post-extra-time scorelines. These are applied at load time so ELO updates, form, and streak features are not contaminated by extra-time goals.

### Models (`models.py`)

| Model | Configuration |
|---|---|
| TabPFN v3 | `ignore_pretraining_limits=True`, `random_state=42`, up to 10,000 recent matches |
| LightGBM | 350 estimators, lr=0.025, `num_leaves=15`, `max_depth=5`, L1/L2 regularisation |
| Poisson | Exponentially weighted attack/defence rates; joint score distribution |

Class order is always aligned to `[home_win, draw, away_win]` before blending.

### Ensemble (`models.py`)

Blend weights are optimised on pooled OOF predictions (nonneg, sum-to-one). Temperature scaling is applied after blending if it reduces OOF log-loss. Saved config in `artifacts/ensemble_config.json`.

**Nested CV:** weights and temperature are also fitted on folds before 2025-07-01 and scored on folds from 2025-07-01 onward to give an unbiased generalization estimate (stored in `nested_eval_log_loss`).

### Evaluation (`evaluation.py`)

- **Rolling-origin folds:** one fold per calendar month, Jan 2024 – Jun 2026. Each fold trains on all matches before that month (up to 10,000 most recent) and tests on that month's matches.
- **Competition folds:** daily expanding evaluation for every completed match in the WC 2026 window (Jun 27 – Jul 15), training on all pre-match data so earlier rounds can inform later ones.
- **Meta-parameter split (`META_SPLIT = 2025-07-01`):** ensemble weights and temperature are fitted on folds before this date and evaluated on folds after it for an unbiased nested estimate.

### Final market blend (`predict.py`)

For fixtures where market odds are available in `../final-market-odds.csv`, the prediction is:

```
final = (1 − MARKET_WEIGHT) × model_ensemble + MARKET_WEIGHT × market_consensus
```

`MARKET_WEIGHT = 0.85` for the WC 2026 Final. Odds are converted to margin-free probabilities by normalising the reciprocals across all sources, then taking the component-wise median. This weight was chosen by judgment — no historical odds archive was available to optimise it on rolling folds.

---

## Files

| File | Purpose |
|---|---|
| `predict.py` | Main CLI entry point |
| `features.py` | Leakage-safe chronological feature engineering |
| `models.py` | Model adapters, class alignment, blending, calibration |
| `evaluation.py` | Monthly and competition rolling evaluation, nested CV |
| `benchmark_versions.py` | TabPFN v2.6 vs v3, baseline vs enhanced comparison |
| `predict_lgbm.py` | Standalone LightGBM exploration with feature importance and report generation |
| `regulation-time-overrides.csv` | Sourced 90-minute scores for 4 ET knockout matches |
| `artifacts/ensemble_config.json` | Saved blend weights, temperature, and evaluation metrics |
| `artifacts/version_benchmark.csv` | v2.6/v3 × baseline/enhanced log-loss comparison |
| `tests/test_pipeline.py` | Unit tests: leakage, class alignment, schema validation |
| `reports/` | Analysis reports from this competition session |

---

## Key findings and caveats

1. **TabPFN alone is competitive with the ensemble.** Nested CV shows the blend (0.8331) does not beat TabPFN alone (0.8301) on the held-out period. The high TabPFN weight (88.7%) in the blend reflects this.

2. **`results.csv` records post-extra-time scores.** Four WC 2026 knockout matches went to extra time; their regulation-time scores are corrected via `regulation-time-overrides.csv`. Historical knockout matches in other editions (WC 2018, Euro 2020/2024, Copa América) may have the same issue and are not corrected.

3. **Market data is Final-only.** No historical odds archive was available for rolling folds, so the 85% market weight is a judgment call. For future competitions, collecting historical closing odds would let this weight be learned empirically.

4. **36-match competition window has high variance.** Any log-loss computed on 36 matches should not be used to draw strong model-selection conclusions. The rolling OOF over 2,631 matches is the authoritative benchmark.

---

## Data source

Match results: [martj42/international\_results](https://github.com/martj42/international_results) (`results.csv`, ~47,000 international matches). Downloaded automatically on first run via `--refresh`. Not committed to this repository.
