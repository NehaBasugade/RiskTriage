

# src/evaluate.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
)

from data import load_data, split_data
from src.experiments.decision import to_binary_high_risk
from pipeline import TARGET_COL

ARTIFACT_DIR = Path("artifacts")


def evaluate(csv_path: str, seed: int = 42):
    df = load_data(csv_path)
    splits = split_data(df, random_state=seed)

    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    cfg = json.loads((ARTIFACT_DIR / "threshold.json").read_text())

    thr_u = float(cfg["threshold_urgent"])
    thr_r = float(cfg["threshold_review"])

    X_test = splits.test.drop(columns=[TARGET_COL])
    y_test = to_binary_high_risk(splits.test[TARGET_COL])

    prob = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, prob))

    # URGENT metrics
    pred_u = (prob >= thr_u).astype(int)
    pu = float(precision_score(y_test, pred_u, zero_division=0))
    ru = float(recall_score(y_test, pred_u, zero_division=0))
    rate_u = float((pred_u == 1).mean())

    # REVIEW metrics (includes urgent+review)
    pred_r = (prob >= thr_r).astype(int)
    pr = float(precision_score(y_test, pred_r, zero_division=0))
    rr = float(recall_score(y_test, pred_r, zero_division=0))
    rate_r = float((pred_r == 1).mean())

    # Confusion matrix for URGENT (most important)
    cm = confusion_matrix(y_test, pred_u)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LOW", "URGENT"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (URGENT thr={thr_u:.2f})")
    plt.savefig(ARTIFACT_DIR / "confusion_matrix_urgent.png", dpi=160, bbox_inches="tight")
    plt.close()

    # PR curve
    p, r, _ = precision_recall_curve(y_test, prob)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AP={pr_auc:.3f})")
    plt.savefig(ARTIFACT_DIR / "pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    out = {
        "test_pr_auc": pr_auc,
        "threshold_urgent": thr_u,
        "threshold_review": thr_r,
        "test_urgent": {"precision": pu, "recall": ru, "flag_rate": rate_u},
        "test_review": {"precision": pr, "recall": rr, "flag_rate": rate_r},
    }
    (ARTIFACT_DIR / "test_metrics.json").write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print("Saved plots to artifacts/: pr_curve.png, confusion_matrix_urgent.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    evaluate(args.csv, seed=args.seed)