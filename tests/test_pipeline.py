from pathlib import Path

import numpy as np
import pandas as pd

from features import DATA_COMMIT, DATA_SHA256, ENHANCED_FEATURES, RAW_URL, build_features, load_data
from models import CLASS_NAMES, aligned_probabilities, normalize_probabilities
from predict import market_consensus, select_fixtures, validate_submission


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


def test_canonical_submission_matches_final_template():
    root = Path(__file__).parents[1]
    submission = pd.read_csv(root / "enhanced-final-predictions.csv")
    template = pd.read_csv(root / "competition" / "final-fixtures.csv")
    validate_submission(submission)
    assert submission[["date", "home_team", "away_team"]].astype(str).to_dict("records") == template.astype(str).to_dict("records")


def test_market_consensus_is_margin_free():
    fixture = pd.Series({"date": pd.Timestamp("2026-07-19"), "home_team": "Spain", "away_team": "Argentina"})
    result = market_consensus(Path(__file__).parents[1] / "final-market-odds.csv", fixture)
    assert result is not None
    assert np.isclose(result.sum(), 1)
    assert result[0] > result[2]


def test_dataset_source_is_pinned_and_has_full_checksum():
    assert DATA_COMMIT in RAW_URL
    assert "/master/" not in RAW_URL
    assert len(DATA_SHA256) == 64


def test_historical_world_cup_rows_receive_tournament_state():
    raw = pd.DataFrame([
        {"date": pd.Timestamp("2018-06-14"), "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": 1, "importance": 60, "outcome": "home_win"},
        {"date": pd.Timestamp("2018-06-15"), "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": 1, "importance": 60, "outcome": "draw"},
    ])
    featured = build_features(raw)
    assert featured.loc[1, "wc_games_diff"] == 1
    assert featured.loc[1, "wc_points_diff"] > 0


def test_explicit_fixture_selection_is_not_clock_dependent(tmp_path):
    raw = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B", "home_score": np.nan, "away_score": np.nan, "tournament": "Friendly", "neutral": 1, "importance": 20, "outcome": np.nan},
    ])
    featured = build_features(raw)
    template = tmp_path / "fixtures.csv"
    pd.DataFrame([{"date": "2020-01-01", "home_team": "A", "away_team": "B"}]).to_csv(template, index=False)
    selected = select_fixtures(featured, template)
    assert len(selected) == 1
    assert selected.iloc[0].home_team == "A"


def test_post_kickoff_market_snapshot_is_rejected(tmp_path):
    odds = tmp_path / "odds.csv"
    pd.DataFrame([{
        "date": "2026-07-19", "home_team": "Spain", "away_team": "Argentina",
        "source": "late", "retrieved_at_utc": "2026-07-19T00:00:01Z",
        "home_decimal_odds": 2.3, "draw_decimal_odds": 3.0, "away_decimal_odds": 3.4,
    }]).to_csv(odds, index=False)
    fixture = pd.Series({"date": pd.Timestamp("2026-07-19"), "home_team": "Spain", "away_team": "Argentina"})
    with np.testing.assert_raises(ValueError):
        market_consensus(odds, fixture)
