"""Focused TabPFN version/feature benchmark on representative chronological folds."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import BASE_FEATURES, ENHANCED_FEATURES, build_features, load_data
from models import fit_tabpfn, load_api_token, multiclass_log_loss, predict_tabpfn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=["2026-03", "2026-05", "2026-06"])
    parser.add_argument("--output", default="artifacts/version_benchmark.csv")
    args = parser.parse_args()
    load_api_token(Path(__file__).resolve().parent.parent / "API_Key_TabPFN")
    frame = build_features(load_data())
    rows = []
    for month_text in args.months:
        month = pd.Period(month_text, freq="M")
        train = frame[frame.outcome.notna() & (frame.date < month.start_time)].tail(10_000)
        test = frame[frame.outcome.notna() & (frame.date >= month.start_time) & (frame.date < (month + 1).start_time)]
        if test.empty:
            print({"month": month_text, "skipped": "no completed matches"})
            continue
        for version in ("v2.6", "v3"):
            for feature_name, columns in (("baseline", BASE_FEATURES), ("enhanced", ENHANCED_FEATURES)):
                model = fit_tabpfn(train[columns], train.outcome.to_numpy(), version, "off")
                loss = multiclass_log_loss(test.outcome, predict_tabpfn(model, test[columns]))
                item = {"month": month_text, "model_version": version, "features": feature_name, "matches": len(test), "log_loss": loss}
                rows.append(item)
                print(item)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
