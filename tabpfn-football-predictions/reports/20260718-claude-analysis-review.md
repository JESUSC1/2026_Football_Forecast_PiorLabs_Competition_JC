# Review of Claude's LightGBM Analysis and Recommended Next Steps

**Reviewed:** 2026-07-18  
**Primary reference:** `20260718-lgbm-tabpfn-analysis.md`  
**Post-fix reference:** `20260718-enhanced-tabpfn-ensemble.md`

## Bottom line

Claude's qualitative recommendation was sound: do not use the old LightGBM-only
or 50/50 forecasts for Spain vs Argentina. The old tree model's 70.7% Argentina
probability was both badly calibrated and contaminated by post-extra-time scores.
The market-informed forecast remains the strongest practical Final submission.

Several numerical results in the older report should not be used for model
selection. The authoritative post-fix evaluation is the leakage-safe enhanced
pipeline: TabPFN v3 scored 0.8581 on 2,631 rolling OOF matches and 0.8193 on the
36-match competition window; LightGBM scored 0.8686 and 0.8235 respectively.

## Findings that remain useful

- Elo is the dominant structured signal and should remain in every model.
- Regulation-time labels are essential; extra time and penalties must not update
  form, streak, Elo, attack/defence, or tournament state as wins.
- LightGBM supplies some diversity but should receive a small weight, not control
  the Final prediction.
- Tournament features can be useful, but must be constructed for historical
  tournament editions during training rather than appearing only at inference.
- The market, Elo, and TabPFN agree directionally that Spain is a modest favorite.
- The old equal-weight ensemble is dominated for this decision and should not be
  submitted.

## Findings superseded by the clean rerun

### The 0.7782 LightGBM score is not comparable

The older experiment used a different feature set, hyperparameters, training
procedure, and state-update implementation. It was not the final monthly
rolling-origin protocol. Its headline claim that LightGBM beat the 0.824
leaderboard should therefore be treated as exploratory, not as evidence of
generalization.

Under the common enhanced pipeline, post-correction LightGBM scored 0.8686 on
broad rolling OOF data and 0.8235 on the small competition window. The latter is
promising, but 36 matches are too few to establish a durable edge.

### The tournament-feature diagnosis was implementation-specific

Claude correctly observed that features which are zero throughout training and
nonzero at prediction time create distribution shift. The current generator
instead produces tournament state chronologically across historical editions.
The earlier `+0.072` result does not tell us whether the corrected tournament
features help the new pipeline; that requires a controlled ablation.

### Regulation-time contamination was broader than the Final prediction

Correcting four competition matches fixed the known Final-path state. Historical
knockout matches in `results.csv` can still contain post-extra-time scores and
therefore contaminate training labels and features. A comprehensive regulation-
time source is still needed before calling the entire historical pipeline clean.

### Calibration and blend selection are now improved, but not fully nested

The current convex weights and temperature are fitted on pooled OOF predictions.
Their reported 0.8567 score is evaluated on those same meta-training predictions,
so it is mildly optimistic. A nested temporal evaluation is needed to estimate
the true benefit of weight fitting and temperature scaling.

## Final forecast assessment

The canonical row is:

```csv
2026-07-19,Spain,Argentina,0.41772418342418227,0.3114810330968958,0.270794783478922
```

This is effectively a market forecast: 85% market consensus and 15% calibrated
model ensemble. That is defensible for predictive accuracy, but the 85% weight
was not learned from historical market OOF data. It should be documented as a
fixed judgment call, not a validation-selected parameter.

The saved market file also uses secondary article URLs rather than direct,
machine-verifiable bookmaker snapshots. The three observations may not be fully
independent, and the row labelled Kalshi is reconstructed from a third-party
article. This does not invalidate the probabilities, but weakens provenance and
reproducibility.

The rules allow ensembles as long as TabPFN is part of the pipeline and require
the input data or public URLs to be reproducible. They do not explicitly prohibit
market odds in the supplied rule text.

## Recommended next steps

### Before the Final deadline

1. Refresh 90-minute three-way odds close to submission time. Preserve the raw
   snapshot, UTC retrieval time, exact source URL, and margin-removal calculation.
   Prefer direct sources and avoid mixing qualification/advance markets with the
   regulation-time 1X2 market.
2. Check confirmed lineups, goalkeeper status, suspensions, and material injuries.
   Only change the forecast when information is sourced before the deadline and
   can be preserved reproducibly.
3. Re-run the submission generator and schema tests from the pinned files. Confirm
   the uploaded file is exactly `date,home_team,away_team,p_home_win,p_draw,
   p_away_win`, has one fixture, uses full country names, and sums to one.
4. Keep the current canonical forecast unless the refreshed market moves
   materially. Do not switch to the old LightGBM or equal-weight files.
5. Archive the exact uploaded CSV, code commit, environment lock, market snapshot,
   and model configuration together. The current loose dependency pins should be
   supplemented with an exported exact environment for reproducibility.

### Highest-value modeling work after submission

1. Build a historical regulation-time results table for all knockout matches,
   not only the four known competition corrections. Recompute labels and every
   chronological state feature from that table.
2. Obtain historical closing 1X2 odds and evaluate market-only, model-only, and
   model-plus-market forecasts on the same rolling folds. Learn or predeclare the
   market weight from those folds instead of fixing it at 85%.
3. Use nested rolling-origin meta-validation: fit blend weights and temperature
   on earlier OOF periods, then score them on later untouched periods. Compare
   against TabPFN alone and simple fixed blends.
4. Run feature-group ablations for tournament form, H2H, Poisson, Elo variants,
   opponent adjustment, and short-rest proxies. Retain groups only when they help
   several temporal blocks, not just the 36-match competition sample.
5. Tune LightGBM conservatively with rolling folds. Test stronger shrinkage,
   shallower trees, minimum leaf sizes, and class-independent calibration. H2H and
   streak features deserve special scrutiny because they can encourage unstable
   splits on small samples.
6. Evaluate recency-window length and regime weighting. International team quality
   changes across managers and player generations, so a fixed 10,000-match window
   should be compared with time decay and shorter calendar windows.
7. Add uncertainty reporting across folds, seeds, feature ablations, and TabPFN
   versions. A mean log-loss difference of a few thousandths is not actionable
   without its temporal variability.

## Priority order

For this Final: refreshed and auditable market data, submission validation, then
lineup news. For the reusable system: comprehensive regulation-time history,
historical odds validation, nested calibration/blending, then feature and
LightGBM tuning.
