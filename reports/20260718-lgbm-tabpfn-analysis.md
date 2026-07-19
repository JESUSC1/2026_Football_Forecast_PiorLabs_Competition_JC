# WC 2026 Final — Full Analysis & Submission Guide

> **Historical report — superseded.** Metrics, weights, paths, and the Final row
> in this document predate the pinned-data and historical-tournament compliance
> corrections. Use `20260718-enhanced-tabpfn-ensemble.md`,
> `20260718-submission-compliance-implementation.md`, and
> `enhanced-final-predictions.csv` as the authoritative artifacts.

**Session:** 2026-07-18  
**Competition:** Prior Labs World Cup 2026 Prediction  
**Match:** Spain vs Argentina, 2026-07-19 (neutral venue)  
**Deadline:** 16:00 UTC 2026-07-19  
**Leaderboard target:** log-loss < 0.824 (dominik_scheuer)

---

## What Was Built

Two independent modeling sessions ran in parallel. Both converged on the same final prediction file.

### Files generated

| File | Source | Description |
|---|---|---|
| `predictions_20260718.csv` | Original `predict.py` (TabPFN) | Single-model TabPFN baseline |
| `predictions_lgbm_20260718.csv` | `predict_lgbm.py` | LightGBM with 38 features |
| `predictions_ensemble_20260718.csv` | `predict_lgbm.py` | Equal-weight 50/50 TabPFN + LightGBM |
| `predictions_calibrated_20260718.csv` | `predict_lgbm.py` | ELO-calibrated 80/20 TabPFN + LightGBM |
| `../enhanced-final-predictions.csv` | Codex pipeline | **Recommended submission** |
| `../final-predictions.csv` | Codex pipeline | Same as above |
| `regulation-time-overrides.csv` | Manual sourcing | Regulation scores for 4 ET knockout matches |
| `../final-market-odds.csv` | Manual collection | Raw decimal odds from 3 bookmakers |
| `artifacts/ensemble_config.json` | Codex pipeline | Learned weights and temperature |
| `artifacts/oof_predictions.csv` | Codex pipeline | 2,631 rolling OOF predictions |
| `artifacts/competition_predictions.csv` | Codex pipeline | Per-match predictions for 36 competition matches |
| `artifacts/fold_metrics.csv` | Codex pipeline | Per-fold log-loss across 28 monthly folds |
| `artifacts/version_benchmark.csv` | Codex pipeline | v2.6 vs v3, baseline vs enhanced features |

---

## Session 1: LightGBM Baseline (predict_lgbm.py)

**Goal:** Mirror `predict.py` using LightGBM; evaluate on 36 competition matches; add tournament-specific features.

### Features (38 total)

Standard features inherited from `predict.py`: `elo_diff`, `home_elo`, `away_elo`, form windows (5/10), H2H stats, rest, streak, `neutral`, `importance`.

Tournament-specific features added (12):

| Feature | Description |
|---|---|
| `{home,away}_tourn_n` | Matches played in current tournament edition |
| `{home,away}_tourn_winrate` | Win rate in current tournament |
| `{home,away}_tourn_drawrate` | Draw rate in current tournament |
| `{home,away}_tourn_gf` / `_ga` | Goals scored/conceded per game in tournament |
| `{home,away}_tourn_cs` | Clean sheet rate in tournament |

Tournament editions are block-detected via a >365-day gap in the same tournament name.

### Evaluation results

| Configuration | Competition log-loss (36 matches) | Accuracy |
|---|---|---|
| LightGBM (17 features, no tournament) | 0.7782 | 61% |
| LightGBM (38 features, with tournament) | 0.8501 | 64% |
| Leaderboard leader | 0.8240 | — |

**Important caveat (Codex conclusion):** The 0.7782 result is from a fixed train/test split on the 36-match competition window — not comparable to a proper rolling evaluation. It benefits from the fixed training window happening to capture the right patterns for this specific 36-match sequence. Codex's rolling OOF over 2,631 matches is the authoritative benchmark.

**Tournament feature distribution shift:** Adding tournament features improved accuracy but worsened log-loss. The model was trained on pre-competition data where all tournament features are zero; at inference they suddenly become nonzero. This creates miscalibration on early-round matches. For the Final (where both teams have 7 WC 2026 matches in the data), the features are informative, but the distribution shift penalty dominates in the backtest.

### Feature importance (LightGBM, gain)

| Rank | Feature | Gain % | Split % |
|---|---|---|---|
| 1 | elo_diff | 21.6% | 5.8% |
| 2 | h2h_gd | 6.1% | 3.9% |
| 3 | home_elo | 4.1% | 4.6% |
| 4 | gd10_diff | 4.0% | 3.8% |
| 5 | away_elo | 3.8% | 4.5% |
| 6 | away_played | 3.7% | 4.5% |
| 7 | home_played | 3.5% | 4.5% |
| 8 | form10_diff | 2.7% | 3.1% |
| 9 | home_tourn_ga | 2.6% | 3.2% |
| 10 | home_tourn_gf | 2.6% | 3.4% |

ELO alone accounts for 21.6% of gain. `home_played`/`away_played` rank high — they proxy for team experience and data reliability, not just match count.

### Match-by-match results (38-feature model)

| Date | Home | Away | Predicted | Actual | Log-loss | ✓/✗ |
|---|---|---|---|---|---|---|
| 2026-06-27 | Algeria | Austria | draw | draw | 1.0315 | ✓ |
| 2026-06-27 | Jordan | Argentina | away_win | away_win | 0.1091 | ✓ |
| 2026-06-27 | Colombia | Portugal | home_win | draw | 1.3092 | ✗ |
| 2026-06-27 | DR Congo | Uzbekistan | draw | home_win | 1.1113 | ✗ |
| 2026-06-27 | Panama | England | away_win | away_win | 0.1503 | ✓ |
| 2026-06-27 | Croatia | Ghana | home_win | home_win | 0.4331 | ✓ |
| 2026-06-28 | South Africa | Canada | away_win | away_win | 0.4906 | ✓ |
| 2026-06-29 | Brazil | Japan | home_win | home_win | 0.7367 | ✓ |
| 2026-06-29 | Germany | Paraguay | home_win | draw | 2.9643 | ✗ |
| 2026-06-29 | Netherlands | Morocco | away_win | draw | 1.6053 | ✗ |
| 2026-06-30 | Ivory Coast | Norway | draw | away_win | 1.0055 | ✗ |
| 2026-06-30 | France | Sweden | home_win | home_win | 0.3618 | ✓ |
| 2026-06-30 | Mexico | Ecuador | home_win | home_win | 0.4048 | ✓ |
| 2026-07-01 | England | DR Congo | home_win | home_win | 0.1656 | ✓ |
| 2026-07-01 | Belgium | Senegal | home_win | draw | 1.4985 | ✗ |
| 2026-07-01 | United States | Bosnia and Herzegovina | home_win | home_win | 0.1820 | ✓ |
| 2026-07-02 | Spain | Austria | home_win | home_win | 0.2542 | ✓ |
| 2026-07-02 | Portugal | Croatia | home_win | home_win | 0.1919 | ✓ |
| 2026-07-02 | Switzerland | Algeria | home_win | home_win | 0.7577 | ✓ |
| 2026-07-03 | Australia | Egypt | home_win | draw | 1.0102 | ✗ |
| 2026-07-03 | Argentina | Cape Verde | home_win | draw | 3.1803 | ✗ |
| 2026-07-03 | Colombia | Ghana | home_win | home_win | 0.1818 | ✓ |
| 2026-07-04 | Canada | Morocco | away_win | away_win | 0.6523 | ✓ |
| 2026-07-04 | Paraguay | France | draw | away_win | 1.3254 | ✗ |
| 2026-07-05 | Brazil | Norway | home_win | away_win | 1.4374 | ✗ |
| 2026-07-05 | Mexico | England | away_win | away_win | 0.4105 | ✓ |
| 2026-07-06 | Portugal | Spain | away_win | away_win | 0.2951 | ✓ |
| 2026-07-06 | United States | Belgium | away_win | away_win | 0.7245 | ✓ |
| 2026-07-07 | Argentina | Egypt | home_win | home_win | 0.1940 | ✓ |
| 2026-07-07 | Switzerland | Colombia | draw | draw | 0.7172 | ✓ |
| 2026-07-09 | France | Morocco | draw | home_win | 1.1185 | ✗ |
| 2026-07-10 | Spain | Belgium | home_win | home_win | 0.3881 | ✓ |
| 2026-07-11 | Norway | England | away_win | draw | 1.3749 | ✗ |
| 2026-07-11 | Argentina | Switzerland | home_win | draw | 1.7584 | ✗ |
| 2026-07-14 | France | Spain | away_win | away_win | 0.3833 | ✓ |
| 2026-07-15 | England | Argentina | away_win | away_win | 0.6890 | ✓ |

---

## Session 2: Codex Pipeline (features.py / models.py / evaluation.py)

**Goal:** Full modular refactor with rolling-origin evaluation, Poisson model, OOF ensemble weight optimization, and temperature calibration.

### Architecture

| Module | Role |
|---|---|
| `features.py` | Chronological feature generation; applies `regulation-time-overrides.csv` at load time |
| `models.py` | TabPFN, LightGBM, Poisson adapters; class-order alignment; ensemble weight optimization; temperature scaling |
| `evaluation.py` | Monthly rolling-origin folds; competition window evaluation |
| `predict.py` | CLI entry point (`--model-version`, `--evaluate`, `--thinking`, `--output`) |
| `benchmark_versions.py` | Focused v2.6 vs v3, baseline vs enhanced comparison |

### Features (53 total)

All 26 base features from the original template, plus 27 enhanced features:

| Feature group | Features added |
|---|---|
| Fast-decay ELO | `elo_fast_diff`, `home_elo_fast`, `away_elo_fast` |
| Attack/defense ratings | `attack_diff`, `defense_diff` (exponentially weighted) |
| Strength of schedule | `strength_schedule_diff` |
| Wider form windows | `form3_diff`, `form20_diff`, `draw_rate10_diff` |
| EWM form | `ewm_points_diff`, `ewm_gf_diff`, `ewm_ga_diff`, `ewm_gd_diff` |
| Rest | `rest_diff`, `home_extra_time_proxy`, `away_extra_time_proxy` |
| WC 2026 in-tournament | `wc_games_diff`, `wc_points_diff`, `wc_gf_diff`, `wc_ga_diff`, `wc_clean_sheet_diff`, `wc_knockout_points_diff` |
| Poisson outputs | `poisson_home_xg`, `poisson_away_xg`, `poisson_p_home_win`, `poisson_p_draw`, `poisson_p_away_win` |

### Regulation-time correction (how Codex handles it)

`load_data()` in `features.py` applies `regulation-time-overrides.csv` **before** computing any features. This means ELO updates, form windows, and streak calculations for the 4 known ET knockout matches use corrected regulation-time scores, not the ET scorelines in `results.csv`.

### Models and ensemble

Three models trained and blended:

| Model | Configuration |
|---|---|
| TabPFN v3 | `ignore_pretraining_limits=True`; up to 10,000 recent matches |
| LightGBM | 350 estimators, lr=0.025, max_leaves=15, max_depth=5, L1+L2 regularization |
| Poisson | Exponentially weighted attack/defense rates; joint score distribution |

Ensemble weights and temperature optimized on pooled OOF predictions (nonnegative, sum-to-one constraint):

```json
{
  "weights": { "tabpfn": 0.887, "lightgbm": 0.064, "poisson": 0.049 },
  "temperature": 1.037
}
```

TabPFN receives 88.7% of the weight. This empirically confirms that LightGBM's contribution should be small — consistent with the ELO-calibration analysis that found LightGBM biased toward Argentina.

### TabPFN version benchmark (v2.6 vs v3, baseline vs enhanced)

| Month | v2.6 baseline | v2.6 enhanced | v3 baseline | v3 enhanced |
|---|---|---|---|---|
| 2026-03 (143 matches) | 0.9405 | 0.9313 | 0.9379 | **0.9301** |
| 2026-05 (26 matches) | 0.5228 | 0.5194 | 0.5298 | **0.5057** |
| 2026-06 (207 matches) | 0.8699 | 0.8705 | 0.8704 | **0.8669** |

Enhanced v3 wins all three folds. Margins are small (0.006–0.024), confirming features matter more than model version.

### Authoritative rolling evaluation (28 folds, 2,631 matches, Jan 2024 – Jun 2026)

| Model | Rolling OOF log-loss | Competition log-loss (36 matches) |
|---|---|---|
| Enhanced TabPFN v3 | **0.8581** | **0.8193** |
| LightGBM | 0.8686 | 0.8235 |
| Poisson | 0.9775 | 0.9665 |
| Calibrated ensemble | **0.8567** | 0.8266 |

These are the **authoritative benchmarks**. The rolling OOF (2,631 matches, 28 folds) is far more reliable than the single 36-match competition window. On the competition window, TabPFN alone (0.8193) narrowly beats the ensemble (0.8266), but this could be noise — the rolling OOF correctly favors the ensemble.

The ensemble log-loss (0.8266) is above the leaderboard leader (0.8240), so the pipeline is competitive but not definitively ahead on this window.

### Final market blend

Three bookmakers' closing 90-minute odds were collected and converted to margin-free probabilities:

| Bookmaker | Spain | Draw | Argentina | Vig |
|---|---|---|---|---|
| BetMGM | 40.9% | 31.4% | 27.7% | 6.2% |
| Oddschecker | 41.3% | 31.6% | 27.1% | 5.4% |
| Kalshi | 41.7% | 32.0% | 26.2% | 3.0% |
| **Consensus** | **41.3%** | **31.7%** | **27.0%** | — |

Final prediction = 85% market consensus + 15% calibrated model ensemble:

```
Spain 41.8%   Draw 31.1%   Argentina 27.1%
```

The 85% market weight is a **judgment call** — no historical odds archive was available for the rolling folds, so this weight was not validated empirically. It reflects the reasoning that market prices aggregate vastly more information (squad fitness, tactical reports, injury news) than the model can access from match results alone.

---

## Data Quality: Regulation-Time Label Errors

This is the most structurally important finding from both sessions.

### Known overrides (WC 2026)

`results.csv` records post-extra-time final scores for knockout matches. Four WC 2026 matches are corrected in `regulation-time-overrides.csv`:

| Match | results.csv | 90-min regulation | Label |
|---|---|---|---|
| Belgium vs Senegal (Jul 1) | 4–2 Belgium | 2–2 | draw |
| Argentina vs Cape Verde (Jul 3) | 3–2 Argentina | 1–1 | draw |
| Norway vs England (Jul 11) | 2–1 Norway | 1–1 | draw |
| Argentina vs Switzerland (Jul 11) | 3–1 Argentina | 1–1 | draw |

The Codex pipeline applies these corrections at load time, so ELO updates and form windows use regulation scores for these matches. `predict_lgbm.py` does **not** apply these corrections — it uses raw ET scores for feature computation.

### Unknown historical errors

The corrections above cover only matches where external ground-truth was available (WC 2026 with `previous-matches-ground-truth.csv`). Historical knockout matches in `results.csv` may have the same problem:

| Tournament | Knockout matches in data (2015+) |
|---|---|
| FIFA World Cup | 64 (2018) + 64 (2022) + 104 (2026 partial) |
| UEFA Euro | 51 per tournament (2016, 2020, 2024) |
| Copa América | 12–16 knockout matches per edition |
| AFC Asian Cup | 16 knockout matches per edition |

Every knockout match in extra time is a potential mislabelled training example. There is no automated way to detect these from `results.csv` alone — the file does not distinguish regulation from ET goals. This affects both **training labels** (what the model learns to predict) and **ELO/form features** (ET wins inflate the winning team's strength).

---

## Final Prediction Comparison

| Source | Spain | Draw | Argentina | Notes |
|---|---|---|---|---|
| **enhanced-final-predictions.csv** | **41.8%** | **31.1%** | **27.1%** | **SUBMIT THIS** |
| Market consensus | 41.3% | 31.7% | 27.0% | 3-book vig-removed average |
| TabPFN v3 (predictions_20260718.csv) | 39.4% | 30.6% | 30.0% | Clean; within 2% of ELO baseline |
| ELO-implied | 38.2% | 28.0% | 33.8% | Pure ELO + 28% draw base rate |
| Calibrated 80/20 blend | 35.7% | 26.2% | 38.1% | Manual approximation to Codex weights |
| Equal-weight ensemble | 30.0% | 19.7% | 50.3% | **Rejected** — Argentina-biased |
| LightGBM only | 20.5% | 8.8% | 70.7% | **Rejected** — ET-inflated |

### Why the LightGBM and 50/50 ensemble were rejected

`results.csv` records Argentina's WC 2026 knockout wins including ET goals. Without the regulation-time override applied to feature computation, LightGBM sees:

| Stat | LightGBM view (ET included) | Regulation reality |
|---|---|---|
| WC 2026 record | 7W 0D 0L | 3W 2D 0L |
| Win streak | 14 consecutive | Broken Jul 11 (Swiss draw) |
| Tourn win rate | 100% | 43% |

LightGBM's 70.7% Argentina prediction follows directly from these inflated inputs. The equal-weight ensemble inherits this bias — LightGBM's extreme 70.7% overwhelms TabPFN's 30.0% even after averaging.

### Log-loss sensitivity under each outcome

| File | If Spain wins | If Draw | If Argentina wins |
|---|---|---|---|
| enhanced-final-predictions.csv | **0.873** | 1.166 | 1.306 |
| Market consensus | 0.884 | 1.149 | 1.309 |
| TabPFN (predictions_20260718.csv) | 0.930 | 1.186 | **1.204** |
| Calibrated 80/20 blend | 1.031 | 1.339 | 0.964 |
| Equal-weight ensemble | 1.204 | 1.625 | 0.687 |
| LightGBM only | 1.585 | 2.430 | **0.347** |

The market-informed file minimizes expected log-loss if Spain wins or draws, which the market and ELO both assign as more likely. LightGBM only wins the log-loss comparison if Argentina wins — an outcome the market prices at ~27%.

### Submission instructions

```bash
# enhanced-final-predictions.csv has the correct schema (no extra columns)
# Copy directly to the submission location:
cp enhanced-final-predictions.csv submission.csv

# TabPFN file has an extra 'predicted' column — strip it if submitting that instead:
cut -d',' -f1-3,5-7 tabpfn-football-predictions/predictions_20260718.csv > submission.csv
```

---

## Alignment with Competition Rules

| Rule | Status |
|---|---|
| §6: TabPFN must be in pipeline | ✅ TabPFN receives 88.7% ensemble weight |
| §6: Ensemble allowed | ✅ Explicitly permitted |
| §4: 90-minute result only | ✅ Regulation-time overrides applied at load time; draw probability elevated |
| §5: Probabilities in (0,1) summing to 1 | ✅ Enforced by `normalize_probabilities()` in `models.py` |
| §6: Reproducible from public data | ✅ `results.csv` from martj42; `random_state=42` throughout |

---

## Known Limitations and Honest Caveats

1. **Meta-parameter overfitting — quantified.** Ensemble weights and temperature were fitted and scored on the **same pooled OOF predictions**. Nested CV (weights fitted on 2024-01–2025-06, scored on 2025-07–2026-06, 993 matches) gives an unbiased estimate:

   | Method | Log-loss on eval period | Notes |
   |---|---|---|
   | TabPFN alone | 0.8301 | No meta-parameters |
   | Ensemble (weights from same eval period) | 0.8288 | Optimistic — overfit |
   | **Ensemble (weights from earlier folds)** | **0.8331** | **Honest — unbiased** |

   Bias from fitting-and-scoring on the same OOF: **−0.0043** (smaller than expected). More importantly, the nested ensemble (0.8331) is **worse than TabPFN alone (0.8301)** on the held-out period. The blend provides no measurable benefit over plain TabPFN when evaluated honestly. This is consistent with the competition window result (TabPFN 0.8193 vs Ensemble 0.8266). The production weights are kept because they use all available OOF data, but **TabPFN alone is the stronger single-model choice**.

2. **Market weight not validated.** The 85% market weight was chosen by judgment, not by optimizing on rolling folds. No historical bookmaker-odds archive was available to backtest this blend. The weight reflects the belief that market prices encode information (injuries, squad fitness, recent training reports) unavailable to the model.

3. **ET label contamination in training data.** Only the 4 known WC 2026 ET matches are corrected. Historical knockouts in WC 2018, Euro 2020, Copa América, AFCON, etc. likely contain additional mislabelled examples. The magnitude of this bias is unknown but probably small relative to the 10,000-match training window — knockout matches are a minority.

4. **36-match competition window is high-variance.** Any log-loss computed on 36 matches has wide confidence intervals. The earlier 0.7782 result (Session 1 LightGBM) is not comparable to the rolling OOF and should not be cited as evidence of superiority.

5. **No squad or lineup data.** The model has no access to injuries, suspensions, lineup choices, or travel schedules — all of which influence match outcomes.

---

## Next Steps (Prioritized)

### Immediate (before deadline)

- [ ] Submit `enhanced-final-predictions.csv` — already validated, correct schema.

### High priority (extend pipeline to future competitions)

**1. ✅ Apply regulation-time overrides to `predict_lgbm.py` feature computation.**  
`predict_lgbm.py`'s `load_data()` now applies `regulation-time-overrides.csv` before computing any features — the same pattern used by `features.py`. ELO updates and form windows for the 4 known ET knockout matches now use regulation-time scores in both pipelines.

**2. Build a historical ET-correction dataset.**  
Systematically identify knockout matches in `results.csv` that ended in extra time, using an external source (Wikipedia tournament pages, football-data.co.uk, or API-Football). Expand `regulation-time-overrides.csv` to cover WC 2018, Euro 2016/2020/2024, Copa América, AFCON. Estimated scope: 50–100 additional corrections across 10 years of knockout data. This removes the biggest structural bias in the training features and labels.

**3. ✅ Nested cross-validation for ensemble weights — implemented.**  
`evaluation.py` now splits OOF at `META_SPLIT = 2025-07-01`: weights and temperature are fitted on folds before that date and scored on folds after. The honest nested log-loss is 0.8331 vs 0.8301 for TabPFN alone — the ensemble does not improve over plain TabPFN on the held-out period. The bias from fitting-and-scoring on the same OOF was −0.0043 (smaller than expected). Production weights unchanged; use nested metrics for any future generalization claims.

### Medium priority (calibration and validation)

**4. Validate the market weight empirically.**  
Collect historical closing odds from football-data.co.uk for World Cups and major tournaments (available back to ~2008). Add market-implied probabilities as features in the rolling OOF. Optimize the model-vs-market blend weight on the same monthly folds. Replace the judgment-call 85% with an evidence-based weight.

**5. Temperature calibration on a proper held-out set.**  
Currently temperature is fitted on the same OOF predictions as the weights. Move temperature fitting to the held-out reporting folds (same fix as #3). Alternatively, use `sklearn.calibration.CalibratedClassifierCV` with isotonic regression before the ensemble step.

**6. Add Copa América / AFCON tournament features.**  
The current WC-only in-tournament features (`wc_games_diff`, `wc_points_diff`, etc.) are specialized for World Cups. For Copa América 2027, AFCON 2027, and Euro 2028 predictions, equivalent features tracking the current tournament edition's form would be needed. The block-detection logic in `predict_lgbm.py` generalizes; the feature names in `features.py` need to be parameterized.

### Longer-term

**7. Squad and lineup availability.**  
A suspension or key injury changes match dynamics significantly. Possible sources: Transfermarkt squad lists (public scraping), FBref lineup archives, or API-Football injury feeds. Adding a binary `key_player_missing` feature, even a rough one, has been shown to improve calibration in knockout tournaments.

**8. Competition-specific neutral venue correction.**  
The WC 2026 is played across USA/Canada/Mexico. Some matches are more "neutral" than others — e.g., Mexico vs a Central American team in Mexico City is not truly neutral. A geo-distance feature or host-nation proximity score could sharpen the neutral-venue handling beyond the current binary flag.
