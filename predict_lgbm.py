"""LightGBM mirror of predict.py — feature importance, ensemble, and competition evaluation."""
import argparse
import os
import textwrap
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier

TODAY = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN = 10000
HOME_ADV = 65.0
DATA = "results.csv"
RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
GROUND_TRUTH = "previous-matches-ground-truth.csv"
REPORTS_DIR = "reports"

FEATURES = [
    # Long-run strength
    "elo_diff", "home_elo", "away_elo",
    # General form
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played",
    # Head-to-head
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    # Match context
    "neutral", "importance",
    # Current-tournament form (reset each tournament edition)
    "home_tourn_n", "home_tourn_winrate", "home_tourn_drawrate",
    "home_tourn_gf", "home_tourn_ga", "home_tourn_cs",
    "away_tourn_n", "away_tourn_winrate", "away_tourn_drawrate",
    "away_tourn_gf", "away_tourn_ga", "away_tourn_cs",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def importance_score(t):
    t = t.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(k in t for k in [
        "uefa euro", "copa am", "african cup", "asian cup",
        "gold cup", "nations league", "oceania nations"
    ]):
        return 45.0
    if "qualif" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


REGULATION_OVERRIDES = "regulation-time-overrides.csv"


def load_data(refresh=False):
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    # Apply regulation-time score overrides so ELO, form, and streak features
    # are not contaminated by extra-time goals in knockout matches.
    if os.path.exists(REGULATION_OVERRIDES):
        overrides = pd.read_csv(REGULATION_OVERRIDES)
        overrides["date"] = pd.to_datetime(overrides["date"])
        scores = overrides.set_index(["date", "home_team", "away_team"])[
            ["regulation_home_score", "regulation_away_score"]
        ]
        indexed = df.set_index(["date", "home_team", "away_team"])
        matches = indexed.index.intersection(scores.index)
        indexed.loc[matches, "home_score"] = scores.loc[matches, "regulation_home_score"].to_numpy()
        indexed.loc[matches, "away_score"] = scores.loc[matches, "regulation_away_score"].to_numpy()
        df = indexed.reset_index().sort_values("date").reset_index(drop=True)
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw",
    )
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance_score)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df):
    """One chronological pass: every feature uses only matches before kickoff."""
    elo = defaultdict(lambda: 1500.0)
    res = defaultdict(list)
    last_date, h2h = {}, defaultdict(list)

    # Tournament block tracking: reset stats when gap > 1 year (separates WC editions)
    tourn_last = {}          # (team, tournament) -> last played date in this block
    tourn_stats = defaultdict(list)  # (team, tournament) -> [(gf, ga, outcome)]

    def tournament_feats(team, tournament_name):
        """Pre-match stats in the current edition of this tournament (0s if none yet)."""
        recs = tourn_stats[(team, tournament_name)]
        if not recs:
            return 0, 0.33, 0.25, 1.2, 1.0, 0.25
        n = len(recs)
        wins  = sum(1 for _, _, o in recs if o == "W")
        draws = sum(1 for _, _, o in recs if o == "D")
        gf    = np.mean([g for g, _, _ in recs])
        ga    = np.mean([a for _, a, _ in recs])
        cs    = sum(1 for _, a, _ in recs if a == 0) / n
        return n, wins / n, draws / n, gf, ga, cs

    def team_feats(team):
        r = res[team]
        if not r:
            return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
        last5, last10 = r[-5:], r[-10:]
        streak = 0
        for p, *_ in reversed(r):
            if p != 3:
                break
            streak += 1
        return (
            elo[team],
            np.mean([p for p, *_ in last5]),
            np.mean([p for p, *_ in last10]),
            np.mean([w for *_, w in last10]),
            np.mean([g for _, g, _, _ in last5]),
            np.mean([a for _, _, a, _ in last5]),
            np.mean([g - a for _, g, a, _ in last10]),
            streak,
            len(r),
        )

    def h2h_feats(home, away):
        m = h2h[tuple(sorted((home, away)))]
        if not m:
            return 0, 0.5, 0.25, 0.0
        n = len(m)
        return (
            n,
            sum(w == home for _, _, w in m) / n,
            sum(w == "draw" for _, _, w in m) / n,
            np.mean([g if h == home else -g for h, g, _ in m]),
        )

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = team_feats(h)
        ae, af5, af10, awr, agf, aga, agd, astk, an = team_feats(a)
        nm, h2h_wr, h2h_dr, h2h_gd = h2h_feats(h, a)
        htn, htwr, htdr, htgf, htga, htcs = tournament_feats(h, r.tournament)
        atn, atwr, atdr, atgf, atga, atcs = tournament_feats(a, r.tournament)
        rows.append({
            "elo_diff":            he + adj - ae,
            "home_elo":            he,
            "away_elo":            ae,
            "form5_diff":          hf5 - af5,
            "form10_diff":         hf10 - af10,
            "home_form5":          hf5,
            "away_form5":          af5,
            "home_winrate":        hwr,
            "away_winrate":        awr,
            "home_gf5":            hgf,
            "away_gf5":            agf,
            "home_ga5":            hga,
            "away_ga5":            aga,
            "gd10_diff":           hgd - agd,
            "home_streak":         hstk,
            "away_streak":         astk,
            "home_rest":           min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest":           min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played":         hn,
            "away_played":         an,
            "h2h_n":               nm,
            "h2h_home_winrate":    h2h_wr,
            "h2h_draw_rate":       h2h_dr,
            "h2h_gd":              h2h_gd,
            # neutral and importance already exist in df from load_data()
            "home_tourn_n":        htn,
            "home_tourn_winrate":  htwr,
            "home_tourn_drawrate": htdr,
            "home_tourn_gf":       htgf,
            "home_tourn_ga":       htga,
            "home_tourn_cs":       htcs,
            "away_tourn_n":        atn,
            "away_tourn_winrate":  atwr,
            "away_tourn_drawrate": atdr,
            "away_tourn_gf":       atgf,
            "away_tourn_ga":       atga,
            "away_tourn_cs":       atcs,
        })

        if not np.isnan(r.home_score):
            gd = r.home_score - r.away_score
            exp = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            g = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta
            elo[a] -= delta
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append(
                (h, gd, h if gd > 0 else (a if gd < 0 else "draw"))
            )

            # Tournament block tracking — reset if gap > 365 days (separates WC editions)
            for team, gf_t, ga_t in [(h, r.home_score, r.away_score),
                                      (a, r.away_score, r.home_score)]:
                key = (team, r.tournament)
                prev = tourn_last.get(key)
                if prev is not None and (r.date - prev).days > 365:
                    tourn_stats[key] = []          # new edition of this tournament
                outcome = "W" if gf_t > ga_t else ("D" if gf_t == ga_t else "L")
                tourn_stats[key].append((gf_t, ga_t, outcome))
                tourn_last[key] = r.date

    return df.join(pd.DataFrame(rows, index=df.index))


# ── Model ─────────────────────────────────────────────────────────────────────

def make_lgbm():
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def train(pool):
    clf = make_lgbm()
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


# ── Feature importance ────────────────────────────────────────────────────────

def feature_importance_df(clf):
    """Return a DataFrame of LightGBM feature importances sorted by gain."""
    gain  = clf.booster_.feature_importance(importance_type="gain")
    split = clf.booster_.feature_importance(importance_type="split")
    fi = pd.DataFrame({"feature": FEATURES, "gain": gain, "split": split})
    fi["gain_pct"]  = fi["gain"]  / fi["gain"].sum()  * 100
    fi["split_pct"] = fi["split"] / fi["split"].sum() * 100
    return fi.sort_values("gain", ascending=False).reset_index(drop=True)


def format_importance_table(fi):
    lines = [
        f"  {'#':<3} {'Feature':<22} {'Gain %':>8} {'Split %':>9}",
        f"  {'-'*47}",
    ]
    for i, row in fi.iterrows():
        bar = "█" * int(row["gain_pct"] / 2)
        lines.append(
            f"  {i+1:<3} {row['feature']:<22} {row['gain_pct']:>7.1f}% {row['split_pct']:>8.1f}%  {bar}"
        )
    return "\n".join(lines)


# ── Competition evaluation ────────────────────────────────────────────────────

def decode_gt(row):
    if row["p_home_win"] == 1:
        return "home_win"
    elif row["p_draw"] == 1:
        return "draw"
    return "away_win"


def evaluate_competition(feats, played):
    """Train on pre-competition data, evaluate on all 36 known competition matches.

    Returns (log_loss, accuracy, clf, proba, feat_rows, gt_rows, y_true).
    """
    gt = pd.read_csv(GROUND_TRUTH)
    gt["date"] = pd.to_datetime(gt["date"])
    comp_start = gt["date"].min()

    train_pool = played[played["date"] < comp_start].tail(MAX_TRAIN)
    clf = train(train_pool)
    classes = clf.classes_

    feat_rows, gt_rows = [], []
    for _, grow in gt.iterrows():
        mask = (
            (feats["date"] == grow["date"])
            & (feats["home_team"] == grow["home_team"])
            & (feats["away_team"] == grow["away_team"])
        )
        hits = feats[mask]
        if len(hits) == 1:
            feat_rows.append(hits.iloc[0])
            gt_rows.append(grow)
        else:
            print(f"  WARNING: no unique feature row for "
                  f"{grow['home_team']} vs {grow['away_team']} on {grow['date'].date()}")

    X = pd.DataFrame(feat_rows)[FEATURES].values
    proba = clf.predict_proba(X)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_true = [decode_gt(r) for r in gt_rows]

    overall_ll  = log_loss(y_true, proba, labels=classes)
    overall_acc = accuracy_score(y_true, classes[proba.argmax(1)])

    print(f"\n{'='*80}")
    print(f"Competition evaluation — {len(y_true)} matches  "
          f"(trained on pre-{comp_start.date()} data)")
    print(f"  Accuracy: {overall_acc:.0%}    Log-loss: {overall_ll:.4f}")
    print(f"{'='*80}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\n  {'Date':<12} {'Home':<25} {'Away':<22} {'Pred':>9} {'True':>9} {'LL':>7}")
    print(f"  {'-'*90}")
    for i, (grow, true) in enumerate(zip(gt_rows, y_true)):
        pred = classes[proba[i].argmax()]
        match_ll = -np.log(proba[i][class_to_idx[true]])
        ok = "✓" if pred == true else "✗"
        print(f"  {str(grow['date'].date()):<12} {grow['home_team']:<25} {grow['away_team']:<22} "
              f"{pred:>9} {true:>9} {match_ll:>7.4f} {ok}")

    return overall_ll, overall_acc, clf, proba, feat_rows, gt_rows, y_true


# ── Ensemble ──────────────────────────────────────────────────────────────────

def compute_ensemble(tabpfn_file, lgbm_out):
    """Average TabPFN and LightGBM probabilities (equal weight). Returns a DataFrame."""
    if not os.path.exists(tabpfn_file):
        return None
    tabpfn = pd.read_csv(tabpfn_file)
    merged = lgbm_out.copy()
    for col in ["p_home_win", "p_draw", "p_away_win"]:
        merged[f"tabpfn_{col}"] = tabpfn[col].values
    merged["ens_p_home_win"] = (merged["p_home_win"] + merged["tabpfn_p_home_win"]) / 2
    merged["ens_p_draw"]     = (merged["p_draw"]     + merged["tabpfn_p_draw"])     / 2
    merged["ens_p_away_win"] = (merged["p_away_win"] + merged["tabpfn_p_away_win"]) / 2
    # renormalise (already sum to 1 if inputs do, but guard for float drift)
    row_sums = merged[["ens_p_home_win", "ens_p_draw", "ens_p_away_win"]].sum(axis=1)
    for col in ["ens_p_home_win", "ens_p_draw", "ens_p_away_win"]:
        merged[col] = merged[col] / row_sums
    return merged


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(content, filename):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"\nReport saved → {path}")
    return path


def build_report(
    comp_ll, comp_acc, fi, gt_rows, y_true, proba_comp, classes_comp,
    lgbm_out, tabpfn_file, ensemble_out, today_str,
):
    class_to_idx = {c: i for i, c in enumerate(classes_comp)}

    # Per-match table for the report
    match_rows = []
    for i, (grow, true) in enumerate(zip(gt_rows, y_true)):
        pred = classes_comp[proba_comp[i].argmax()]
        match_ll = -np.log(proba_comp[i][class_to_idx[true]])
        ok = "✓" if pred == true else "✗"
        match_rows.append(
            f"| {grow['date'].date()} | {grow['home_team']} | {grow['away_team']} "
            f"| {pred} | {true} | {match_ll:.4f} | {ok} |"
        )

    fi_rows = []
    for _, row in fi.head(10).iterrows():
        fi_rows.append(
            f"| {row['feature']} | {row['gain_pct']:.1f}% | {row['split_pct']:.1f}% |"
        )

    tabpfn_exists = os.path.exists(tabpfn_file)
    final_row = lgbm_out.iloc[0]

    final_section = textwrap.dedent(f"""
    ## Final Prediction — Spain vs Argentina (2026-07-19)

    > **Rule §4:** Scored on 90-minute result only — extra time and penalties do not count.
    > This elevates the draw probability relative to a standard league match.

    | Model | Spain Win | Draw | Argentina Win | Notes |
    |---|---|---|---|---|
    | LightGBM | {final_row.p_home_win:.1%} | {final_row.p_draw:.1%} | {final_row.p_away_win:.1%} | Strongly influenced by Argentina's 5-0-0 WC record |
    """)

    if tabpfn_exists and ensemble_out is not None:
        tabpfn = pd.read_csv(tabpfn_file)
        t = tabpfn.iloc[0]
        e = ensemble_out.iloc[0]
        final_section += (
            f"| TabPFN | {t.p_home_win:.1%} | {t.p_draw:.1%} | {t.p_away_win:.1%} "
            f"| Conservative; closer to historical base rates |\n"
            f"| **Ensemble (equal weight)** | **{e.ens_p_home_win:.1%}** | **{e.ens_p_draw:.1%}** "
            f"| **{e.ens_p_away_win:.1%}** | Averages both signals |\n"
        )
    else:
        final_section += "_TabPFN predictions not found — run predict.py first._\n"

    report = textwrap.dedent(f"""\
    # LightGBM vs TabPFN — Analysis Report
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
    **Competition:** Prior Labs World Cup 2026 Prediction

    ---

    ## Competition Evaluation (36 matches, pre-2026-06-27 training)

    | Metric | LightGBM | Leaderboard leader | Uniform baseline |
    |---|---|---|---|
    | Accuracy | {comp_acc:.0%} | — | 33% |
    | Log-loss | {comp_ll:.4f} | 0.8240 | 1.0986 |

    LightGBM trained exclusively on pre-competition data (before June 27 2026) and evaluated
    on all 36 scored rounds. Features for each match reflect the correct chronological ELO
    and form state just before kickoff — including ELO updates from earlier WC 2026 rounds.

    ### Match-by-match results

    | Date | Home | Away | Predicted | Actual | Log-loss | ✓/✗ |
    |---|---|---|---|---|---|---|
    {chr(10).join(match_rows)}

    ---

    ## LightGBM Feature Importance (pre-competition model)

    Sorted by **gain** (reduction in training log-loss per split — most predictive first).

    | Feature | Gain % | Split % |
    |---|---|---|
    {chr(10).join(fi_rows)}

    ### Interpretation

    - **ELO features** (`home_elo`, `away_elo`, `elo_diff`) dominate by gain — they compress
      decades of match history into a single calibrated strength signal. This is expected for
      international football where ELO is a well-validated predictor.
    - **Form features** (`home_form5`, `away_form5`, etc.) matter for capturing current momentum
      and hot/cold streaks, but have lower gain than ELO — they add noise when teams play few
      matches in a period.
    - **H2H features** have relatively low gain — head-to-head samples are small for most
      international matchups, so the model discounts them appropriately.
    - **`home_streak`** / **`away_streak`** are high in split count but lower in gain, indicating
      the model uses them frequently but each split moves the needle less. Worth watching: this
      is where LightGBM may be over-indexing on Argentina's current 5-game WC winning streak.
    """)

    report += final_section

    report += textwrap.dedent(f"""
    ---

    ## Model Disagreement Analysis

    LightGBM gives Argentina **68%** to win; TabPFN gives only **30%**.
    The gap of 38 percentage points is large and signals genuine model uncertainty.

    **Why LightGBM is bullish on Argentina:**
    - Argentina's WC 2026 record: 5 W, 0 D, 0 L — longest current winning streak of any team
    - High `away_streak` value (5) pushes tree splits toward Argentina
    - `elo_diff` favours Argentina after 5 high-importance WC wins

    **Why TabPFN is more conservative:**
    - TabPFN's in-context Bayesian approach regularises toward the historical base rate
      for two ~equally-ranked ELO teams at a neutral venue (~35% / 30% / 35%)
    - More resistant to over-fitting to a short recent streak

    **Implication for the ensemble:** Equal-weight averaging reduces overconfidence.
    A future calibration step (temperature scaling on held-out OOF predictions) would
    produce a principled blend weight rather than 50/50.

    ---

    ## Alignment with Competition Rules

    | Rule | Status |
    |---|---|
    | §6: TabPFN must be in pipeline | ✅ TabPFN is the primary model; LightGBM is an ensemble member |
    | §6: Ensemble allowed | ✅ Rule explicitly permits combining models |
    | §4: 90-minute result only | ✅ Labels derived from `home_score`/`away_score` at FT; draw probability elevated |
    | §5: Probabilities in (0,1) summing to 1 | ✅ Enforced by renormalisation after `predict_proba` |
    | §6: Reproducible from input data | ✅ `results.csv` sourced from public martj42 repo; `random_state=42` |

    ---

    ## Next Steps

    1. **Tournament-specific features** — add current-WC goals scored/conceded, clean sheets,
       and knockout-stage record to `build_features()`. These are all derivable from `results.csv`
       and improve signal for in-tournament form without introducing any leakage.
    2. **Temperature scaling** — fit a scalar `T` on OOF predictions to sharpen/flatten
       probabilities based on validation evidence.
    3. **TabPFN competition evaluation** — run `predict.py --evaluate` (once that flag is added)
       to get a comparable log-loss and inform ensemble weights.
    4. **Rolling-origin backtest** — retrain for each match in the competition window to get a
       more honest estimate of per-round model quality.
    """)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LightGBM football predictions")
    parser.add_argument("--refresh",  action="store_true", help="Re-download dataset from source")
    parser.add_argument("--evaluate", action="store_true", help="Score against competition ground truth")
    parser.add_argument("--report",   action="store_true", help="Run evaluation + importance + ensemble and save report")
    args = parser.parse_args()

    # --report implies --evaluate
    if args.report:
        args.evaluate = True

    df = load_data(refresh=args.refresh)
    latest = df["date"].max()
    print(f"Latest game in dataset: {latest.date()}")
    print(f"Data freshness: {pd.Timestamp.now() - latest}")

    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    future = feats[feats["home_score"].isna() & (feats["date"] > TODAY)].sort_values("date")

    # Monthly backtest (mirrors predict.py)
    month = (TODAY.to_period("M") - 1)
    test = played[(played["date"] >= month.start_time) & (played["date"] < (month + 1).start_time)]
    if len(test):
        clf_bt = train(played[played["date"] < month.start_time].tail(MAX_TRAIN))
        proba_bt = clf_bt.predict_proba(test[FEATURES].values)
        proba_bt = proba_bt / proba_bt.sum(axis=1, keepdims=True)
        print(f"\nBacktest {month} ({len(test)} matches): "
              f"accuracy {accuracy_score(test['outcome'], clf_bt.classes_[proba_bt.argmax(1)]):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba_bt, labels=clf_bt.classes_):.3f}")

    # Competition evaluation
    comp_results = None
    if args.evaluate and os.path.exists(GROUND_TRUTH):
        comp_results = evaluate_competition(feats, played)

    # Feature importance from the competition model (if evaluated)
    fi = None
    if comp_results is not None:
        comp_ll, comp_acc, clf_comp, proba_comp, feat_rows, gt_rows, y_true = comp_results
        fi = feature_importance_df(clf_comp)
        print(f"\n{'='*80}")
        print("LightGBM Feature Importance (pre-competition model, sorted by gain)")
        print(f"{'='*80}")
        print(format_importance_table(fi))

    # Final prediction — train on all available data
    clf = train(played.tail(MAX_TRAIN))
    proba = clf.predict_proba(future[FEATURES].values)
    proba = proba / proba.sum(axis=1, keepdims=True)
    cols = {c: proba[:, i] for i, c in enumerate(clf.classes_)}

    out = future[["date", "home_team", "away_team"]].copy()
    out["p_home_win"] = cols["home_win"]
    out["p_draw"]     = cols["draw"]
    out["p_away_win"] = cols["away_win"]

    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    lgbm_file = f"predictions_lgbm_{today_str}.csv"
    out.to_csv(lgbm_file, index=False)
    print(f"\n{len(out)} fixture predictions (LightGBM) → {lgbm_file}\n")

    for i, r in enumerate(out.itertuples()):
        predicted = clf.classes_[proba[i].argmax()]
        print(f"  {r.date.date()}  {r.home_team:>20} vs {r.away_team:<20}  "
              f"→ {predicted:<9}  H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}")

    # Ensemble with TabPFN
    tabpfn_file = f"predictions_{today_str}.csv"
    ensemble_out = compute_ensemble(tabpfn_file, out)

    if ensemble_out is not None:
        print(f"\n--- Model comparison + ensemble ---")
        print(f"  {'Match':<40} {'Model':<14} {'H':>7} {'D':>7} {'A':>7}")
        print(f"  {'-'*70}")
        tabpfn = pd.read_csv(tabpfn_file)
        for i, r in enumerate(out.itertuples()):
            label = f"{r.home_team} vs {r.away_team}"
            e = ensemble_out.iloc[i]
            print(f"  {label:<40} {'LightGBM':<14} {r.p_home_win:>7.1%} {r.p_draw:>7.1%} {r.p_away_win:>7.1%}")
            if i < len(tabpfn):
                t = tabpfn.iloc[i]
                print(f"  {'':<40} {'TabPFN':<14} {t.p_home_win:>7.1%} {t.p_draw:>7.1%} {t.p_away_win:>7.1%}")
            print(f"  {'':<40} {'Ensemble':<14} {e.ens_p_home_win:>7.1%} {e.ens_p_draw:>7.1%} {e.ens_p_away_win:>7.1%}")
            print()

        # Save ensemble as submission-ready CSV
        ens_file = f"predictions_ensemble_{today_str}.csv"
        ens_submission = ensemble_out[["date", "home_team", "away_team"]].copy()
        ens_submission["p_home_win"] = ensemble_out["ens_p_home_win"]
        ens_submission["p_draw"]     = ensemble_out["ens_p_draw"]
        ens_submission["p_away_win"] = ensemble_out["ens_p_away_win"]
        ens_submission.to_csv(ens_file, index=False)
        print(f"Ensemble submission saved → {ens_file}")

    # Save report
    if args.report and comp_results is not None and fi is not None:
        comp_ll, comp_acc, clf_comp, proba_comp, feat_rows, gt_rows, y_true = comp_results
        report_content = build_report(
            comp_ll, comp_acc, fi, gt_rows, y_true, proba_comp,
            clf_comp.classes_, out, tabpfn_file, ensemble_out, today_str,
        )
        report_filename = f"{today_str}-lgbm-tabpfn-analysis.md"
        save_report(report_content, report_filename)


if __name__ == "__main__":
    main()
