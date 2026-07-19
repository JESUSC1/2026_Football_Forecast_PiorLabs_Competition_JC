"""Rolling-origin evaluation for football outcome models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import DATA_COMMIT, DATA_SHA256, ENHANCED_FEATURES, TRAIN_START
from models import (
    CLASS_NAMES, apply_temperature, blend_probabilities, fit_blend_weights,
    fit_lightgbm, fit_tabpfn, fit_temperature, multiclass_log_loss,
    poisson_probabilities, predict_lightgbm, predict_tabpfn,
)

MAX_TRAIN = 10_000
EVALUATION_START = pd.Timestamp("2024-01-01")
EVALUATION_END = pd.Timestamp("2026-07-01")
COMPETITION_START = pd.Timestamp("2026-06-27")
# Temporal split for nested meta-parameter evaluation.
# Weights and temperature are fitted on folds BEFORE this date and scored on
# folds ON OR AFTER it, giving an unbiased estimate of the blend benefit.
META_SPLIT = pd.Timestamp("2025-07-01")


def rolling_folds(frame: pd.DataFrame):
    played = frame[frame.outcome.notna() & (frame.date >= TRAIN_START)]
    for month in pd.period_range("2024-01", "2026-06", freq="M"):
        test = played[(played.date >= month.start_time) & (played.date < (month + 1).start_time)]
        train = played[played.date < month.start_time].tail(MAX_TRAIN)
        if len(test) and set(train.outcome.unique()) == set(CLASS_NAMES):
            yield str(month), train, test


def competition_folds(frame: pd.DataFrame):
    """Daily expanding folds for every completed match in the competition window."""
    played = frame[frame.outcome.notna() & (frame.date >= TRAIN_START)]
    competition = played[played.date >= COMPETITION_START]
    for day in sorted(competition.date.dt.normalize().unique()):
        test = competition[competition.date.dt.normalize() == day]
        train = played[played.date < day].tail(MAX_TRAIN)
        if len(test):
            yield pd.Timestamp(day).strftime("%Y-%m-%d"), train, test


def evaluate(frame: pd.DataFrame, model_version: str, thinking: str, output_dir: str = "artifacts"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    competition_rows = []
    metrics = []

    for fold_name, train, test in rolling_folds(frame):
        X_train, X_test = train[ENHANCED_FEATURES], test[ENHANCED_FEATURES]
        tabpfn = fit_tabpfn(X_train, train.outcome.to_numpy(), model_version, thinking)
        lightgbm = fit_lightgbm(X_train, train.outcome.to_numpy())
        predictions = {
            "tabpfn": predict_tabpfn(tabpfn, X_test),
            "lightgbm": predict_lightgbm(lightgbm, X_test),
            "poisson": poisson_probabilities(test),
        }
        for model_name, probs in predictions.items():
            metrics.append({"fold": fold_name, "model": model_name, "matches": len(test), "log_loss": multiclass_log_loss(test.outcome, probs)})
        for i, (_, row) in enumerate(test.iterrows()):
            item = {"fold": fold_name, "date": row.date.strftime("%Y-%m-%d"), "home_team": row.home_team, "away_team": row.away_team, "outcome": row.outcome}
            for model_name, probs in predictions.items():
                for j, label in enumerate(CLASS_NAMES):
                    item[f"{model_name}_{label}"] = probs[i, j]
            prediction_rows.append(item)

    for fold_name, train, test in competition_folds(frame):
        X_train, X_test = train[ENHANCED_FEATURES], test[ENHANCED_FEATURES]
        tabpfn = fit_tabpfn(X_train, train.outcome.to_numpy(), model_version, thinking)
        lightgbm = fit_lightgbm(X_train, train.outcome.to_numpy())
        predictions = {
            "tabpfn": predict_tabpfn(tabpfn, X_test),
            "lightgbm": predict_lightgbm(lightgbm, X_test),
            "poisson": poisson_probabilities(test),
        }
        for i, (_, row) in enumerate(test.iterrows()):
            item = {"fold": fold_name, "date": row.date.strftime("%Y-%m-%d"), "home_team": row.home_team, "away_team": row.away_team, "outcome": row.outcome}
            for model_name, probs in predictions.items():
                for j, label in enumerate(CLASS_NAMES):
                    item[f"{model_name}_{label}"] = probs[i, j]
            competition_rows.append(item)

    oof = pd.DataFrame(prediction_rows)
    if oof.empty:
        raise RuntimeError("No rolling folds were produced")

    oof["date"] = pd.to_datetime(oof["date"])
    model_names = ("tabpfn", "lightgbm", "poisson")

    # ── Nested meta-parameter evaluation (unbiased) ───────────────────────────
    # Fit blend weights and temperature on folds before META_SPLIT; score on
    # folds from META_SPLIT onward. This avoids the optimistic bias that arises
    # when meta-params are both fitted and evaluated on the same OOF pool.
    meta_train = oof[oof["date"] < META_SPLIT]
    meta_eval  = oof[oof["date"] >= META_SPLIT]
    nested_eval_loss = None
    nested_weights = None
    nested_temp = None
    tabpfn_alone_eval_loss = None
    if len(meta_train) and len(meta_eval):
        mt_probs = [meta_train[[f"{n}_{l}" for l in CLASS_NAMES]].to_numpy() for n in model_names]
        nested_weights = fit_blend_weights(meta_train.outcome.to_numpy(), mt_probs)
        nested_blend_mt = blend_probabilities(mt_probs, nested_weights)
        nested_temp, _, _ = fit_temperature(meta_train.outcome.to_numpy(), nested_blend_mt)
        me_probs = [meta_eval[[f"{n}_{l}" for l in CLASS_NAMES]].to_numpy() for n in model_names]
        nested_blend_me = apply_temperature(blend_probabilities(me_probs, nested_weights), nested_temp)
        nested_eval_loss = multiclass_log_loss(meta_eval.outcome.to_numpy(), nested_blend_me)
        tabpfn_alone_eval_loss = multiclass_log_loss(meta_eval.outcome.to_numpy(), me_probs[0])

    # ── Production weights: fit on all OOF for best estimates ─────────────────
    model_probs = [oof[[f"{name}_{label}" for label in CLASS_NAMES]].to_numpy() for name in model_names]
    weights = fit_blend_weights(oof.outcome.to_numpy(), model_probs)
    blended = blend_probabilities(model_probs, weights)
    temperature, raw_loss, calibrated_loss = fit_temperature(oof.outcome.to_numpy(), blended)
    calibrated = apply_temperature(blended, temperature)
    for j, label in enumerate(CLASS_NAMES):
        oof[f"ensemble_{label}"] = calibrated[:, j]

    competition = pd.DataFrame(competition_rows)
    competition_model_probs = [competition[[f"{name}_{label}" for label in CLASS_NAMES]].to_numpy() for name in model_names]
    competition_blend = apply_temperature(blend_probabilities(competition_model_probs, weights), temperature)
    for j, label in enumerate(CLASS_NAMES):
        competition[f"ensemble_{label}"] = competition_blend[:, j]
    summary = {
        "model_version": model_version, "thinking": thinking,
        "market_weight": 0.85,
        "weights": dict(zip(model_names, weights.tolist())),
        "temperature": temperature,
        "oof_log_loss": raw_loss,
        "calibrated_oof_log_loss": calibrated_loss,
        "nested_eval_log_loss": nested_eval_loss,
        "nested_meta_split": str(META_SPLIT.date()),
        "nested_meta_train_matches": len(meta_train),
        "nested_meta_eval_matches": len(meta_eval),
        "nested_meta_train_weights": dict(zip(model_names, nested_weights.tolist())) if nested_weights is not None else None,
        "nested_meta_train_temperature": nested_temp,
        "tabpfn_alone_eval_log_loss": tabpfn_alone_eval_loss,
        "blend_benefit_over_tabpfn_honest": (
            tabpfn_alone_eval_loss - nested_eval_loss
            if tabpfn_alone_eval_loss is not None else None
        ),
        "results_csv_rows": len(frame),
        "results_csv_last_date": str(frame.date.max().date()),
        "results_csv_sha256": DATA_SHA256,
        "results_source_commit": DATA_COMMIT,
        "competition_matches": len(competition),
        "competition_log_loss": multiclass_log_loss(
            competition.outcome, competition[[f"ensemble_{x}" for x in CLASS_NAMES]].to_numpy()
        ) if len(competition) else None,
    }
    oof.to_csv(output / "oof_predictions.csv", index=False)
    competition.to_csv(output / "competition_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "fold_metrics.csv", index=False)
    (output / "ensemble_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
