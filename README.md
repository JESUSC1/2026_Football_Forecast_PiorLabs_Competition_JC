# WC 2026 Football Prediction — Prior Labs Competition Entry

Predicts 90-minute regulation-time outcomes (home win / draw / away win) for international football matches using an ensemble of TabPFN v3, LightGBM, and a Poisson goal model.

**Competition:** [Prior Labs World Cup Game Outcome Prediction](https://ux.priorlabs.ai/worldcup)  
**Final submission:** `enhanced-final-predictions.csv`  
**Match:** Spain vs Argentina, 2026-07-19 · Spain **41.8%** · Draw **31.1%** · Argentina **27.1%**

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# macOS only (LightGBM): conda install -c conda-forge llvm-openmp
```

## Reproduce the Final prediction

```bash
# Download dataset (~47k matches) from public source
python predict.py --refresh

# Generate prediction using saved ensemble config + market odds
python predict.py --model-version v3 --output enhanced-final-predictions.csv
```

To also regenerate ensemble weights via full rolling evaluation (slow — calls TabPFN API):

```bash
python predict.py --model-version v3 --evaluate --output enhanced-final-predictions.csv
```

## Run tests

```bash
python -m pytest tests/ -q
```

---

## Results

| Model | Rolling OOF log-loss (2,631 matches) | Competition log-loss (36 matches) |
|---|---|---|
| Enhanced TabPFN v3 | 0.8581 | **0.8193** |
| LightGBM | 0.8686 | 0.8235 |
| Poisson | 0.9775 | 0.9665 |
| Calibrated ensemble | **0.8567** *(optimistic)* | 0.8266 |
| Nested CV — honest estimate | **0.8331** | — |
| Leaderboard leader | — | 0.8240 |

Nested CV (weights fitted on Jan 2024 – Jun 2025, scored on Jul 2025 – Jun 2026) shows the blend (0.8331) does not beat TabPFN alone (0.8301) on held-out data. The pipeline weights TabPFN at 88.7%.

**Final prediction pipeline:**  
85% margin-free market consensus (BetMGM + Oddschecker + Kalshi) + 15% calibrated model ensemble.

---

## Repository layout

```
.
├── predict.py                  # Main entry point — generates submission CSV
├── features.py                 # 53 leakage-safe chronological features
├── models.py                   # TabPFN, LightGBM, Poisson adapters; ensemble blending
├── evaluation.py               # Monthly rolling-origin folds; nested CV
├── benchmark_versions.py       # TabPFN v2.6 vs v3 comparison
├── predict_lgbm.py             # Standalone LightGBM exploration + report generation
├── requirements.txt
├── regulation-time-overrides.csv   # Sourced 90-min scores for 4 ET knockout matches
├── enhanced-final-predictions.csv  # Submitted prediction
├── final-market-odds.csv           # Bookmaker odds snapshot used in the blend
├── previous-matches-ground-truth.csv  # Competition ground truth (36 matches)
├── artifacts/
│   ├── ensemble_config.json    # Saved blend weights, temperature, and evaluation metrics
│   └── version_benchmark.csv  # v2.6/v3 × baseline/enhanced log-loss comparison
├── reports/                    # Analysis reports from this competition session
│   ├── 20260718-lgbm-tabpfn-analysis.md
│   ├── 20260718-enhanced-tabpfn-ensemble.md
│   └── 20260718-claude-analysis-review.md
├── tests/
│   └── test_pipeline.py        # Leakage, class alignment, schema, market tests
└── competition/                # Competition reference documents
    ├── TabPFN Football Prediction Competition —  (rules)
    └── Submit_Predictions.md
```

---

## How it works

### Features (`features.py`)
Single chronological forward pass — every feature is computed before applying the current result. Same-day matches are batched to prevent leakage. 53 features total:

- **ELO** (standard + fast-decay): home/away ratings, ELO diff
- **Form** (3/5/10/20-match windows): points per game, goal rates, win rates, exponentially weighted versions
- **Attack/defence ratings**: exponentially weighted goal rates per team
- **Strength of schedule**: opponent-adjusted form
- **Head-to-head**: win/draw rate and average goal difference
- **Rest**: days since last match, rest difference, short-rest proxy
- **Current-tournament form**: WC 2026 games, points, goals, clean sheets, knockout record
- **Poisson model outputs**: expected goals, derived win/draw/loss probabilities
- **Match context**: neutral venue, tournament importance

**Regulation-time corrections:** `regulation-time-overrides.csv` supplies sourced 90-minute scores for four WC 2026 knockout matches where `results.csv` records post-extra-time scorelines. Applied at load time so ELO, form, and streak are not contaminated by extra-time goals.

### Models (`models.py`)

| Model | Config |
|---|---|
| TabPFN v3 | `ignore_pretraining_limits=True`, `random_state=42`, up to 10,000 matches |
| LightGBM | 350 trees, lr=0.025, `num_leaves=15`, `max_depth=5`, L1+L2 |
| Poisson | Exponentially weighted attack/defence; joint score distribution |

### Ensemble
Weights optimised on pooled OOF log-loss (nonnegative, sum-to-one). Temperature scaling applied if it reduces OOF loss. Saved in `artifacts/ensemble_config.json`.

### Evaluation (`evaluation.py`)
- Monthly rolling-origin folds: Jan 2024 – Jun 2026 (28 folds, 2,631 matches)
- Competition folds: daily expanding evaluation for each WC 2026 match
- Nested CV: `META_SPLIT = 2025-07-01` — weights fitted on earlier folds, scored on later folds for an unbiased generalization estimate

---

## Key caveats

1. **`results.csv` includes extra-time scores.** Only four known WC 2026 matches are corrected. Historical knockouts (WC 2018, Euro 2020/2024, etc.) may still contain mislabelled training examples.
2. **Market weight not validated.** The 85% market weight was chosen by judgment — no historical odds archive was available for rolling folds.
3. **Nested CV shows no clear ensemble benefit.** TabPFN alone (0.8301) is marginally better than the blend (0.8331) on the held-out period.
4. **36-match competition window is high-variance.** Use rolling OOF metrics for model selection.

---

## Data source

`results.csv` — [martj42/international\_results](https://github.com/martj42/international_results) (~47,000 international matches). Fetched automatically by `python predict.py --refresh`. Not committed; SHA-256 prefix of the version used: `7f2a1026c1e78d82` (49,520 rows, last date 2026-07-19).
