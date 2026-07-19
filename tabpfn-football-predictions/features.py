"""Leakage-safe chronological features for international football matches."""
from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = "results.csv"
RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
REGULATION_OVERRIDES = "regulation-time-overrides.csv"
HOME_ADVANTAGE = 65.0
TRAIN_START = pd.Timestamp("2014-01-01")
WORLD_CUP_2026_START = pd.Timestamp("2026-06-11")

BASE_FEATURES = [
    "elo_diff", "home_elo", "away_elo", "form5_diff", "form10_diff",
    "home_form5", "away_form5", "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played", "h2h_n", "h2h_home_winrate",
    "h2h_draw_rate", "h2h_gd", "neutral", "importance",
]

ENHANCED_FEATURES = BASE_FEATURES + [
    "elo_fast_diff", "home_elo_fast", "away_elo_fast",
    "attack_diff", "defense_diff", "strength_schedule_diff",
    "form3_diff", "form20_diff", "draw_rate10_diff",
    "ewm_points_diff", "ewm_gf_diff", "ewm_ga_diff", "ewm_gd_diff",
    "rest_diff", "home_extra_time_proxy", "away_extra_time_proxy",
    "wc_games_diff", "wc_points_diff", "wc_gf_diff", "wc_ga_diff",
    "wc_clean_sheet_diff", "wc_knockout_points_diff",
    "poisson_home_xg", "poisson_away_xg",
    "poisson_p_home_win", "poisson_p_draw", "poisson_p_away_win",
]


def tournament_importance(name: str) -> float:
    text = str(name).lower()
    if "world cup" in text and "qual" not in text:
        return 60.0
    if "confederations" in text:
        return 50.0
    if any(k in text for k in ("uefa euro", "copa am", "african cup", "asian cup", "gold cup", "nations league", "oceania nations")):
        return 45.0
    if "qualif" in text:
        return 35.0
    if "friendly" in text:
        return 20.0
    return 30.0


def load_data(refresh: bool = False, path: str = DATA) -> pd.DataFrame:
    if refresh or not os.path.exists(path):
        frame = pd.read_csv(RAW_URL)
        frame.to_csv(path, index=False)
    else:
        frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date"], kind="stable").reset_index(drop=True)
    frame["neutral"] = frame["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    override_path = Path(__file__).resolve().with_name(REGULATION_OVERRIDES)
    if override_path.exists():
        overrides = pd.read_csv(override_path)
        overrides["date"] = pd.to_datetime(overrides["date"])
        scores = overrides.set_index(["date", "home_team", "away_team"])[["regulation_home_score", "regulation_away_score"]]
        indexed = frame.set_index(["date", "home_team", "away_team"])
        matches = indexed.index.intersection(scores.index)
        indexed.loc[matches, "home_score"] = scores.loc[matches, "regulation_home_score"].to_numpy()
        indexed.loc[matches, "away_score"] = scores.loc[matches, "regulation_away_score"].to_numpy()
        frame = indexed.reset_index().sort_values("date", kind="stable").reset_index(drop=True)
    frame["outcome"] = np.select(
        [frame.home_score > frame.away_score, frame.home_score < frame.away_score],
        ["home_win", "away_win"], default="draw",
    )
    frame.loc[frame.home_score.isna() | frame.away_score.isna(), "outcome"] = np.nan
    frame["importance"] = frame.tournament.map(tournament_importance)
    return frame


def _mean(records, index: int, n: int, default: float) -> float:
    values = [row[index] for row in records[-n:]]
    return float(np.mean(values)) if values else default


def _ewm(records, index: int, alpha: float, default: float) -> float:
    if not records:
        return default
    value = float(records[0][index])
    for row in records[1:]:
        value = alpha * float(row[index]) + (1 - alpha) * value
    return value


def _poisson_probabilities(home_xg: float, away_xg: float, max_goals: int = 10) -> tuple[float, float, float]:
    hp = np.array([math.exp(-home_xg) * home_xg**i / math.factorial(i) for i in range(max_goals + 1)])
    ap = np.array([math.exp(-away_xg) * away_xg**i / math.factorial(i) for i in range(max_goals + 1)])
    grid = np.outer(hp, ap)
    probs = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
    probs /= probs.sum()
    return tuple(float(x) for x in probs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build pre-kickoff features in one forward pass; rows never see their own result."""
    elo = defaultdict(lambda: 1500.0)
    elo_fast = defaultdict(lambda: 1500.0)
    records = defaultdict(list)  # points, gf, ga, won, drawn, opponent elo
    last_date: dict[str, pd.Timestamp] = {}
    h2h = defaultdict(list)
    world_cup = defaultdict(list)
    rows: list[dict[str, float]] = []

    def team(team: str):
        rec = records[team]
        streak = 0
        for item in reversed(rec):
            if item[0] != 3:
                break
            streak += 1
        return {
            "elo": elo[team], "elo_fast": elo_fast[team],
            "form3": _mean(rec, 0, 3, 1.3), "form5": _mean(rec, 0, 5, 1.3),
            "form10": _mean(rec, 0, 10, 1.3), "form20": _mean(rec, 0, 20, 1.3),
            "win10": _mean(rec, 3, 10, 0.33), "draw10": _mean(rec, 4, 10, 0.27),
            "gf5": _mean(rec, 1, 5, 1.2), "ga5": _mean(rec, 2, 5, 1.2),
            "gd10": _mean([(0, r[1] - r[2]) for r in rec], 1, 10, 0.0),
            "ewp": _ewm(rec, 0, 0.25, 1.3), "ewgf": _ewm(rec, 1, 0.25, 1.2),
            "ewga": _ewm(rec, 2, 0.25, 1.2),
            "sos": _mean(rec, 5, 10, 1500.0), "streak": streak, "played": len(rec),
        }

    def wc(team_name: str):
        rec = world_cup[team_name]
        return {
            "games": len(rec), "points": _mean(rec, 0, 20, 1.3),
            "gf": _mean(rec, 1, 20, 1.2), "ga": _mean(rec, 2, 20, 1.2),
            "clean": _mean(rec, 3, 20, 0.3), "ko": _mean(rec, 0, 5, 1.3),
        }

    def apply_result(match, h, a, advantage):
        if pd.isna(match.home_score) or pd.isna(match.away_score):
            return
        home, away = match.home_team, match.away_team
        gd = float(match.home_score - match.away_score)
        score = 1.0 if gd > 0 else 0.0 if gd < 0 else 0.5
        expected = 1 / (1 + 10 ** ((a["elo"] - h["elo"] - advantage) / 400))
        multiplier = 1.0 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
        delta = match.importance * multiplier * (score - expected)
        elo[home] += delta; elo[away] -= delta
        fast_delta = min(80.0, match.importance * 1.35) * multiplier * (score - expected)
        elo_fast[home] += fast_delta; elo_fast[away] -= fast_delta
        hp = 3 if gd > 0 else 1 if gd == 0 else 0
        ap = 3 if gd < 0 else 1 if gd == 0 else 0
        records[home].append((hp, match.home_score, match.away_score, float(gd > 0), float(gd == 0), a["elo"]))
        records[away].append((ap, match.away_score, match.home_score, float(gd < 0), float(gd == 0), h["elo"]))
        last_date[home] = last_date[away] = match.date
        winner = home if gd > 0 else away if gd < 0 else "draw"
        h2h[tuple(sorted((home, away)))].append((home, gd, winner))
        if match.tournament == "FIFA World Cup" and match.date >= WORLD_CUP_2026_START:
            world_cup[home].append((hp, match.home_score, match.away_score, float(match.away_score == 0)))
            world_cup[away].append((ap, match.away_score, match.home_score, float(match.home_score == 0)))

    pending = []
    current_date = None
    for match in df.itertuples():
        if current_date is None:
            current_date = match.date
        elif match.date != current_date:
            for prior in pending:
                apply_result(*prior)
            pending = []
            current_date = match.date
        home, away = match.home_team, match.away_team
        h, a = team(home), team(away)
        hw, aw = wc(home), wc(away)
        advantage = HOME_ADVANTAGE * (1 - match.neutral)
        pair = h2h[tuple(sorted((home, away)))]
        if pair:
            h2h_n = len(pair)
            h2h_wr = sum(w == home for _, _, w in pair) / h2h_n
            h2h_dr = sum(w == "draw" for _, _, w in pair) / h2h_n
            h2h_gd = float(np.mean([gd if prior_home == home else -gd for prior_home, gd, _ in pair]))
        else:
            h2h_n, h2h_wr, h2h_dr, h2h_gd = 0, 0.5, 0.25, 0.0

        home_xg = float(np.clip((h["ewgf"] + a["ewga"]) / 2 + advantage / 500, 0.15, 4.0))
        away_xg = float(np.clip((a["ewgf"] + h["ewga"]) / 2, 0.15, 4.0))
        ph, pd_, pa = _poisson_probabilities(home_xg, away_xg)
        home_rest = min((match.date - last_date[home]).days, 90) if home in last_date else 30
        away_rest = min((match.date - last_date[away]).days, 90) if away in last_date else 30
        row = {
            "elo_diff": h["elo"] + advantage - a["elo"], "home_elo": h["elo"], "away_elo": a["elo"],
            "form5_diff": h["form5"] - a["form5"], "form10_diff": h["form10"] - a["form10"],
            "home_form5": h["form5"], "away_form5": a["form5"],
            "home_winrate": h["win10"], "away_winrate": a["win10"],
            "home_gf5": h["gf5"], "away_gf5": a["gf5"], "home_ga5": h["ga5"], "away_ga5": a["ga5"],
            "gd10_diff": h["gd10"] - a["gd10"], "home_streak": h["streak"], "away_streak": a["streak"],
            "home_rest": home_rest, "away_rest": away_rest, "home_played": h["played"], "away_played": a["played"],
            "h2h_n": h2h_n, "h2h_home_winrate": h2h_wr, "h2h_draw_rate": h2h_dr, "h2h_gd": h2h_gd,
            "elo_fast_diff": h["elo_fast"] + advantage - a["elo_fast"], "home_elo_fast": h["elo_fast"], "away_elo_fast": a["elo_fast"],
            "attack_diff": h["ewgf"] - a["ewgf"], "defense_diff": a["ewga"] - h["ewga"],
            "strength_schedule_diff": h["sos"] - a["sos"], "form3_diff": h["form3"] - a["form3"],
            "form20_diff": h["form20"] - a["form20"], "draw_rate10_diff": h["draw10"] - a["draw10"],
            "ewm_points_diff": h["ewp"] - a["ewp"], "ewm_gf_diff": h["ewgf"] - a["ewgf"],
            "ewm_ga_diff": h["ewga"] - a["ewga"], "ewm_gd_diff": (h["ewgf"] - h["ewga"]) - (a["ewgf"] - a["ewga"]),
            "rest_diff": home_rest - away_rest,
            "home_extra_time_proxy": float(home_rest <= 4 and match.importance >= 60),
            "away_extra_time_proxy": float(away_rest <= 4 and match.importance >= 60),
            "wc_games_diff": hw["games"] - aw["games"], "wc_points_diff": hw["points"] - aw["points"],
            "wc_gf_diff": hw["gf"] - aw["gf"], "wc_ga_diff": hw["ga"] - aw["ga"],
            "wc_clean_sheet_diff": hw["clean"] - aw["clean"], "wc_knockout_points_diff": hw["ko"] - aw["ko"],
            "poisson_home_xg": home_xg, "poisson_away_xg": away_xg,
            "poisson_p_home_win": ph, "poisson_p_draw": pd_, "poisson_p_away_win": pa,
        }
        rows.append(row)
        pending.append((match, h, a, advantage))

    for prior in pending:
        apply_result(*prior)

    return df.join(pd.DataFrame(rows, index=df.index))
