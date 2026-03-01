# src/experiments/exp004_compare_models.py
print("[exp004] starting imports...")
from src.legacy.train import TimeHHMMToHour, RareCategoryGrouper, SpatialBinner
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_score, recall_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.experiments.exp_common import (
    FEATURE_COLS, DROP_ALWAYS, RANDOM_STATE,
    ALERT_RATE_URGENT, ALERT_RATE_REVIEW,
    to_binary_high_risk, build_preprocessor, triage_preds,
    expected_calibration_error, get_split
)
print("[exp004] sklearn imported.")

def eval_one(name: str, pipe: Pipeline, X_valid: pd.DataFrame, y_valid: np.ndarray):
    proba = pipe.predict_proba(X_valid)[:, 1]

    pr_auc = float(average_precision_score(y_valid, proba))
    brier = float(brier_score_loss(y_valid, proba))
    ece = float(expected_calibration_error(y_valid, proba, n_bins=10))

    pred_u, pred_r, t_u, t_r = triage_preds(proba, ALERT_RATE_URGENT, ALERT_RATE_REVIEW)

    urgent_precision = float(precision_score(y_valid, pred_u, zero_division=0))
    urgent_recall = float(recall_score(y_valid, pred_u, zero_division=0))
    review_precision = float(precision_score(y_valid, pred_r, zero_division=0))
    review_recall = float(recall_score(y_valid, pred_r, zero_division=0))

    out = {
        "model": name,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece_10bin": ece,
        "urgent_precision": urgent_precision,
        "urgent_recall": urgent_recall,
        "urgent_flag_rate": float(pred_u.mean()),
        "review_precision": review_precision,
        "review_recall": review_recall,
        "review_flag_rate": float(pred_r.mean()),
        "t_urgent": float(t_u),
        "t_review": float(t_r),
    }
    return proba, out


def main():
    print("[exp004] main() entered.")
    os.makedirs("runs/exp004", exist_ok=True)

    df = pd.read_csv("data/traffic.csv")

    # Hard drops locked
    for c in DROP_ALWAYS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Target locked
    if "Casualty Severity" not in df.columns:
        raise ValueError("Missing required target column: 'Casualty Severity'")

    df = df.copy()
    df["_y"] = to_binary_high_risk(df["Casualty Severity"])


    df_train, df_valid, df_test = get_split(df)

    # Drop target
    X_train = df_train.drop(columns=["_y"])
    X_valid = df_valid.drop(columns=["_y"])
    y_train = df_train["_y"].to_numpy()
    y_valid = df_valid["_y"].to_numpy()


    for c in DROP_ALWAYS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=[c])
        if c in X_valid.columns:
            X_valid = X_valid.drop(columns=[c])

    # === Time parsing ===
    time_fix = TimeHHMMToHour("Time (24hr)")
    X_train = time_fix.fit_transform(X_train)
    X_valid = time_fix.transform(X_valid)

    # === Rare collapsing ===
    rare_fix = RareCategoryGrouper(
        cols=["Road Surface", "Weather Conditions", "Type of Vehicle"],
        min_count=20,
    )
    X_train = rare_fix.fit_transform(X_train)
    X_valid = rare_fix.transform(X_valid)

    # === Spatial binning ===
    spatial_fix = SpatialBinner(
        e_col="Grid Ref: Easting",
        n_col="Grid Ref: Northing",
        bin_size=2000,
    )
    X_train = spatial_fix.fit_transform(X_train)
    X_valid = spatial_fix.transform(X_valid)

    # Now select final feature columns (locked)
    X_train = X_train[FEATURE_COLS].copy()
    X_valid = X_valid[FEATURE_COLS].copy()

    pre = build_preprocessor(X_train)

    models = [
        ("RF_current",
         RandomForestClassifier(
             n_estimators=150,
             max_depth=18,
             min_samples_split=2,
             min_samples_leaf=2,
             class_weight="balanced_subsample",
             random_state=RANDOM_STATE,
             n_jobs=-1
         )),
        ("LogReg_balanced",
         LogisticRegression(
             class_weight="balanced",
             max_iter=2000,
             solver="lbfgs"
         )),
        ("GradBoost",
         GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]

    results = []
    probas = {}

    for name, model in models:
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        proba, res = eval_one(name, pipe, X_valid, y_valid)
        results.append(res)
        probas[name] = proba

    res_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)

    print("\n=== Exp004 Model Comparison (VALID) ===")
    print(res_df[[
        "model", "pr_auc", "brier", "ece_10bin",
        "urgent_precision", "urgent_recall", "urgent_flag_rate",
        "review_precision", "review_recall", "review_flag_rate"
    ]].to_string(index=False))

    res_path = "runs/exp004/model_compare_valid.csv"
    res_df.to_csv(res_path, index=False)
    print(f"\nSaved: {res_path}")


    np.savez(
        "runs/exp004/probas_valid_base.npz",
        y_valid=y_valid,
        **{k: v for k, v in probas.items()}
    )
    print("Saved: runs/exp004/probas_valid_base.npz")


if __name__ == "__main__":
    main()