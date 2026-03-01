

# cv_evaluate.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from data import load_data
from pipeline import build_preprocessor, TARGET_COL
from model import build_random_forest
from src.experiments.decision import to_binary_high_risk


def cv_pr_auc(csv_path: str, k: int = 5, seed: int = 42, group_col: str = "Reference Number") -> float:
    df = load_data(csv_path)

    y = to_binary_high_risk(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])

    preprocessor, _, _ = build_preprocessor(df)
    clf = build_random_forest(random_state=seed)
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

    scores = []

    if group_col in df.columns:
        cv = GroupKFold(n_splits=k)
        splits = cv.split(X, y, groups=df[group_col])
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = cv.split(X, y)

    for tr, te in splits:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        scores.append(average_precision_score(yte, prob))

    mean_score = float(np.mean(scores))
    print(f"{k}-fold CV PR-AUC: {mean_score:.3f}  (folds={['%.3f'%s for s in scores]})")
    return mean_score


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cv_pr_auc(args.csv, k=args.k, seed=args.seed)