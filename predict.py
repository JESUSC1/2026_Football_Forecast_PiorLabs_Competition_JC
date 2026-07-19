"""Train the enhanced ensemble and write competition-format predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import evaluate
from features import ENHANCED_FEATURES, TRAIN_START, build_features, load_data
from models import (
    CLASS_NAMES, apply_temperature, blend_probabilities, fit_lightgbm,
    fit_tabpfn, load_api_token, normalize_probabilities, poisson_probabilities,
    predict_lightgbm, predict_tabpfn,
)

MAX_TRAIN = 10_000
DEFAULT_WEIGHTS = {"tabpfn": 0.50, "lightgbm": 0.30, "poisson": 0.20}
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MARKET_WEIGHT = 0.85
ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", choices=("v2.6", "v3"), default="v3")
    parser.add_argument("--thinking", choices=("off", "medium", "high"), default="off")
    parser.add_argument("--evaluate", action="store_true", help="Run rolling-origin evaluation before fitting")
    parser.add_argument("--refresh", action="store_true", help="Refresh results.csv from its public source")
    parser.add_argument("--output", help="Output CSV path; defaults to predictions_YYYYMMDD.csv")
    return parser.parse_args()


def load_config(path: Path = Path("artifacts/ensemble_config.json")):
    if not path.exists():
        return DEFAULT_WEIGHTS, DEFAULT_TEMPERATURE
    config = json.loads(path.read_text(encoding="utf-8"))
    weights = config.get("weights", DEFAULT_WEIGHTS)
    if set(weights) != set(DEFAULT_WEIGHTS) or not np.isclose(sum(weights.values()), 1):
        raise ValueError(f"Invalid ensemble weights in {path}")
    return weights, float(config.get("temperature", DEFAULT_TEMPERATURE))


def market_consensus(path: Path, fixture: pd.Series) -> np.ndarray | None:
    if not path.exists():
        return None
    odds = pd.read_csv(path)
    selected = odds[
        (odds.date.astype(str) == fixture.date.strftime("%Y-%m-%d"))
        & (odds.home_team == fixture.home_team)
        & (odds.away_team == fixture.away_team)
    ]
    if selected.empty:
        return None
    raw = 1 / selected[["home_decimal_odds", "draw_decimal_odds", "away_decimal_odds"]].to_numpy(dtype=float)
    source_probs = normalize_probabilities(raw)
    return normalize_probabilities(np.median(source_probs, axis=0, keepdims=True))[0]


def validate_submission(output: pd.DataFrame):
    expected = ["date", "home_team", "away_team", "p_home_win", "p_draw", "p_away_win"]
    if list(output.columns) != expected:
        raise ValueError(f"Invalid output columns: {list(output.columns)}")
    probs = output[expected[3:]].to_numpy(dtype=float)
    if not np.all((probs > 0) & (probs < 1)):
        raise ValueError("Every probability must be strictly between zero and one")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Probability rows must sum to one")


def main():
    args = parse_args()
    if args.thinking != "off" and args.model_version != "v3":
        raise SystemExit("--thinking requires --model-version v3")
    load_api_token(ROOT / "API_Key_TabPFN")

    raw = load_data(refresh=args.refresh)
    features = build_features(raw)
    if args.evaluate:
        summary = evaluate(features, args.model_version, args.thinking)
        print(json.dumps(summary, indent=2))

    today = pd.Timestamp.now().normalize()
    played = features[features.outcome.notna() & (features.date >= TRAIN_START)].tail(MAX_TRAIN)
    future = features[features.home_score.isna() & (features.date > today)].sort_values("date")
    if future.empty:
        raise SystemExit("No future fixtures with missing scores were found")

    X_train, X_future = played[ENHANCED_FEATURES], future[ENHANCED_FEATURES]
    tabpfn = fit_tabpfn(X_train, played.outcome.to_numpy(), args.model_version, args.thinking)
    lightgbm = fit_lightgbm(X_train, played.outcome.to_numpy())
    components = [
        predict_tabpfn(tabpfn, X_future),
        predict_lightgbm(lightgbm, X_future),
        poisson_probabilities(future),
    ]
    weights, temperature = load_config()
    model_probs = blend_probabilities(components, [weights[name] for name in ("tabpfn", "lightgbm", "poisson")])
    model_probs = apply_temperature(model_probs, temperature)

    final_probs = model_probs.copy()
    market_path = ROOT / "final-market-odds.csv"
    for i, (_, fixture) in enumerate(future.iterrows()):
        market = market_consensus(market_path, fixture)
        if market is not None:
            final_probs[i] = normalize_probabilities(
                ((1 - DEFAULT_MARKET_WEIGHT) * model_probs[i] + DEFAULT_MARKET_WEIGHT * market).reshape(1, -1)
            )[0]

    output = future[["date", "home_team", "away_team"]].copy()
    output["date"] = output.date.dt.strftime("%Y-%m-%d")
    for j, column in enumerate(("p_home_win", "p_draw", "p_away_win")):
        output[column] = final_probs[:, j]
    validate_submission(output)

    destination = Path(args.output) if args.output else Path(f"predictions_{pd.Timestamp.now():%Y%m%d}.csv")
    output.to_csv(destination, index=False)
    print(f"Wrote {len(output)} fixture prediction(s) to {destination}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
