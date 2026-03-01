

# interpret.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from data import load_data, split_data
from pipeline import TARGET_COL

ARTIFACT_DIR = Path("artifacts")


def permutation_importance_report(csv_path: str, seed: int = 42, n_repeats: int = 5):
    df = load_data(csv_path)
    splits = split_data(df, random_state=seed)

    model = joblib.load(ARTIFACT_DIR / "model.joblib")

    X_valid = splits.valid.drop(columns=[TARGET_COL])
    y_valid = (splits.valid[TARGET_COL].astype(str).str.strip().isin(["1", "2"])).astype(int).values


    result = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="average_precision",
        n_jobs=-1,
    )

    importances = result.importances_mean
    cols = list(X_valid.columns)

    ranked = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)
    top = [{"feature": f, "importance": float(im)} for f, im in ranked[:20]]

    (ARTIFACT_DIR / "feature_importance.json").write_text(json.dumps(top, indent=2))
    print("Saved artifacts/feature_importance.json")
    for row in top[:10]:
        print(row)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    permutation_importance_report(args.csv, seed=args.seed)