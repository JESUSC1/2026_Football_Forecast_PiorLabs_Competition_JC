# WC 2026 Football Prediction — Prior Labs Competition Entry

Predicts 90-minute regulation-time outcomes (home win / draw / away win) for international football matches using an ensemble of TabPFN v3, LightGBM, and a Poisson goal model.

**Competition:** [Prior Labs World Cup Game Outcome Prediction](https://ux.priorlabs.ai/worldcup)  
**Final submission:** `enhanced-final-predictions.csv`  
**Match:** Spain vs Argentina, 2026-07-19 · Spain **42.0%** · Draw **31.1%** · Argentina **26.9%**

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# macOS only (LightGBM): conda install -c conda-forge llvm-openmp
```

For the exact tested Python package set, use `pip install -r requirements-lock.txt`.
The lock reflects the macOS/Python 3.11 environment used for the submission.
Alternatively, recreate the named environment with
`conda env create -f environment.yml`.

TabPFN requires authentication. Sign in once through the supported
`tabpfn-client` browser flow, or place a browser-issued JWT in the local
`API_Key_TabPFN` file. The file is ignored by Git. Other API-key formats are not
silently substituted for a valid cached client token.

## Reproduce the Final prediction

```bash
# Download the dataset from the pinned upstream Git commit and verify SHA-256
python predict.py --refresh

# Generate the explicit Final fixture independent of the current date
python predict.py --model-version v3 \
  --fixtures competition/final-fixtures.csv \
  --as-of 2026-07-19 \
  --output enhanced-final-predictions.csv
```

To also regenerate ensemble weights via full rolling evaluation (slow — calls TabPFN API):

```bash
python predict.py --model-version v3 --evaluate \
  --fixtures competition/final-fixtures.csv \
  --as-of 2026-07-19 \
  --output enhanced-final-predictions.csv
```

## Run tests

```bash
python -m pytest tests/ -q
```

---

## Results

| Model | Rolling OOF log-loss (2,631 matches) | Competition log-loss (36 matches) |
|---|---|---|
| Enhanced TabPFN v3 | 0.8582 | **0.8287** |
| LightGBM | 0.8679 | 0.8336 |
| Poisson | 0.9775 | 0.9665 |
| Calibrated ensemble | **0.8567** *(optimistic)* | 0.8350 |
| Nested CV — honest estimate | **0.8331** | — |
| Leaderboard leader | — | 0.8240 |

Nested CV (weights fitted on Jan 2024 – Jun 2025, scored on Jul 2025 – Jun 2026) shows the blend (0.8331) does not beat TabPFN alone (0.8311) on held-out data. The production model blend weights TabPFN at 84.9%.

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
├── requirements-lock.txt       # Exact tested Python package versions
├── environment.yml             # Exact Python/OpenMP environment entry point
├── data-manifest.json          # Pinned dataset commit and full checksum
├── regulation-time-overrides.csv   # Sourced 90-min scores for 4 ET knockout matches
├── enhanced-final-predictions.csv  # Submitted prediction
├── final-market-odds.csv           # Bookmaker odds snapshot used in the blend
├── previous-matches-ground-truth.csv  # Competition ground truth (36 matches)
├── artifacts/
│   ├── ensemble_config.json    # Saved blend weights, temperature, and evaluation metrics
│   └── version_benchmark.csv  # Historical v2.6/v3 comparison; revision-labelled
├── reports/                    # Analysis reports from this competition session
│   ├── 20260718-lgbm-tabpfn-analysis.md
│   ├── 20260718-enhanced-tabpfn-ensemble.md
│   └── 20260718-claude-analysis-review.md
├── tests/
│   └── test_pipeline.py        # Leakage, class alignment, schema, market tests
└── competition/                # Competition reference documents
    ├── TabPFN Football Prediction Competition —  (rules)
    ├── Submit_Predictions.md
    └── final-fixtures.csv       # Explicit Final template for deterministic reruns
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
- **Current-tournament form**: edition-scoped games, points, goals, clean sheets, and knockout record. Historical World Cups provide nonzero training examples; 32-team editions use 48 group fixtures and the expanded 2026 edition uses 72.
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
3. **Nested CV shows no clear ensemble benefit.** TabPFN alone (0.8311) is marginally better than the blend (0.8331) on the held-out period.
4. **36-match competition window is high-variance.** Use rolling OOF metrics for model selection.

---

## Data source

`results.csv` — [martj42/international\_results](https://github.com/martj42/international_results), pinned to commit `80f408d2c93ba4f9e06a2c7cdc5effb05fea9680`. The loader normalizes the CSV and requires full SHA-256 `7f2a1026c1e78d825b58deba2555d81331b0a80752c57e8e0a3332d1350e5d4` (49,520 rows, last fixture date 2026-07-19). See `data-manifest.json`.
