from pathlib import Path

import numpy as np
import pandas as pd

from features import ENHANCED_FEATURES, build_features, load_data
from models import CLASS_NAMES, aligned_probabilities, normalize_probabilities
from predict import market_consensus, validate_submission


def test_features_are_complete_and_pre_match():
    raw = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": "home_win"},
        {"date": pd.Timestamp("2026-01-02"), "home_team": "A", "away_team": "C", "home_score": np.nan, "away_score": np.nan, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": np.nan},
    ])
    featured = build_features(raw)
    assert not featured[ENHANCED_FEATURES].isna().any().any()
    assert featured.loc[0, "home_played"] == 0
    assert featured.loc[1, "home_played"] == 1
    assert featured.loc[1, "home_streak"] == 1


def test_draw_does_not_extend_win_streak():
    raw = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": "draw"},
        {"date": pd.Timestamp("2026-01-02"), "home_team": "A", "away_team": "C", "home_score": np.nan, "away_score": np.nan, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": np.nan},
    ])
    assert build_features(raw).loc[1, "home_streak"] == 0


def test_same_date_matches_do_not_leak_results():
    raw = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": "home_win"},
        {"date": pd.Timestamp("2026-01-01"), "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": "home_win"},
        {"date": pd.Timestamp("2026-01-02"), "home_team": "A", "away_team": "D", "home_score": np.nan, "away_score": np.nan, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": np.nan},
    ])
    featured = build_features(raw)
    assert featured.loc[1, "home_played"] == 0
    assert featured.loc[2, "home_played"] == 2


def test_class_alignment_and_normalization():
    values = aligned_probabilities(np.array([[0.2, 0.5, 0.3]]), ["away_win", "home_win", "draw"])
    assert np.allclose(values, [[0.5, 0.3, 0.2]])
    assert np.allclose(normalize_probabilities(values).sum(axis=1), 1)


def test_submission_schema_and_ranges():
    frame = pd.DataFrame([["2026-07-19", "Spain", "Argentina", 0.41, 0.315, 0.275]], columns=["date", "home_team", "away_team", "p_home_win", "p_draw", "p_away_win"])
    validate_submission(frame)


def test_market_consensus_is_margin_free():
    fixture = pd.Series({"date": pd.Timestamp("2026-07-19"), "home_team": "Spain", "away_team": "Argentina"})
    result = market_consensus(Path(__file__).parents[1] / "final-market-odds.csv", fixture)
    assert result is not None
    assert np.isclose(result.sum(), 1)
    assert result[0] > result[2]
