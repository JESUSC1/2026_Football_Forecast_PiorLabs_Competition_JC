"""Model adapters, calibration, and probability ensembling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss

CLASS_NAMES = np.array(["home_win", "draw", "away_win"])
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
EPSILON = 1e-9


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    probs = np.clip(probs, EPSILON, 1 - EPSILON)
    return probs / probs.sum(axis=1, keepdims=True)


def aligned_probabilities(probs: np.ndarray, classes) -> np.ndarray:
    result = np.zeros((len(probs), len(CLASS_NAMES)), dtype=float)
    for source_index, label in enumerate(classes):
        result[:, CLASS_TO_INT[str(label)]] = probs[:, source_index]
    return normalize_probabilities(result)


def load_api_token(path: str | Path) -> bool:
    """Load a raw token without logging or returning its value."""
    token_path = Path(path)
    if not token_path.exists():
        return False
    token = token_path.read_text(encoding="utf-8").strip()
    # tabpfn-client.set_access_token expects the browser-issued JWT. Other API
    # key types must not replace a valid token already cached by the client.
    if not token or not token.startswith("eyJ") or token.count(".") != 2:
        return False
    from tabpfn_client import set_access_token
    set_access_token(token)
    return True


def fit_tabpfn(X, y, model_version: str = "v3", thinking: str = "off"):
    from tabpfn_client import TabPFNClassifier
    kwargs = {
        "model_path": f"{model_version}_default",
        "random_state": 42,
        "ignore_pretraining_limits": True,
    }
    if thinking != "off":
        if model_version != "v3":
            raise ValueError("Thinking mode is supported only for TabPFN v3")
        kwargs.update(
            thinking_effort=thinking,
            thinking_metric="log_loss",
            thinking_timeout_s=600 if thinking == "medium" else 1200,
        )
    model = TabPFNClassifier(**kwargs)
    model.fit(X, y, description="Chronological international football outcome prediction")
    return model


def predict_tabpfn(model, X) -> np.ndarray:
    return aligned_probabilities(model.predict_proba(X), model.classes_)


def fit_lightgbm(X, y):
    from lightgbm import LGBMClassifier
    encoded = np.array([CLASS_TO_INT[value] for value in y], dtype=int)
    model = LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=350,
        learning_rate=0.025, num_leaves=15, max_depth=5,
        min_child_samples=35, subsample=0.85, colsample_bytree=0.8,
        reg_alpha=0.2, reg_lambda=1.0, random_state=42, verbosity=-1,
    )
    model.fit(X, encoded)
    return model


def predict_lightgbm(model, X) -> np.ndarray:
    labels = [CLASS_NAMES[int(i)] for i in model.classes_]
    return aligned_probabilities(model.predict_proba(X), labels)


def poisson_probabilities(frame: pd.DataFrame) -> np.ndarray:
    return normalize_probabilities(frame[[
        "poisson_p_home_win", "poisson_p_draw", "poisson_p_away_win"
    ]].to_numpy())


def multiclass_log_loss(y, probs) -> float:
    encoded = np.array([CLASS_TO_INT[value] for value in y], dtype=int)
    return float(log_loss(encoded, normalize_probabilities(probs), labels=[0, 1, 2]))


def fit_blend_weights(y, predictions: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(predictions, axis=1)
    encoded = np.array([CLASS_TO_INT[value] for value in y], dtype=int)

    def objective(weights):
        return log_loss(encoded, normalize_probabilities(np.einsum("nmc,m->nc", stacked, weights)), labels=[0, 1, 2])

    count = len(predictions)
    result = minimize(
        objective, np.full(count, 1 / count), method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    if not result.success:
        raise RuntimeError(f"Blend optimization failed: {result.message}")
    return result.x / result.x.sum()


def blend_probabilities(predictions: list[np.ndarray], weights) -> np.ndarray:
    return normalize_probabilities(np.einsum("nmc,m->nc", np.stack(predictions, axis=1), np.asarray(weights)))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, EPSILON, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return normalize_probabilities(values)


def fit_temperature(y, probs: np.ndarray) -> tuple[float, float, float]:
    baseline = multiclass_log_loss(y, probs)
    result = minimize_scalar(
        lambda t: multiclass_log_loss(y, apply_temperature(probs, t)),
        bounds=(0.5, 2.5), method="bounded",
    )
    calibrated = float(result.fun)
    return (float(result.x), baseline, calibrated) if calibrated < baseline else (1.0, baseline, baseline)
