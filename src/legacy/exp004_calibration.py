# src/experiments/exp004_calibration.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.experiments.exp_common import (
    FEATURE_COLS, DROP_ALWAYS, RANDOM_STATE,
    ALERT_RATE_URGENT, ALERT_RATE_REVIEW,
    to_binary_high_risk, build_preprocessor, triage_preds,
    expected_calibration_error, get_split
)

from src.legacy.train import TimeHHMMToHour, RareCategoryGrouper, SpatialBinner


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # Hard drop ID columns (locked)
    for c in DROP_ALWAYS:
        if c in X.columns:
            X = X.drop(columns=[c])

    time_fix = TimeHHMMToHour("Time (24hr)")
    X = time_fix.fit_transform(X)

    rare_fix = RareCategoryGrouper(
        cols=["Road Surface", "Weather Conditions", "Type of Vehicle"],
        min_count=20,
    )
    X = rare_fix.fit_transform(X)

    spatial_fix = SpatialBinner(
        e_col="Grid Ref: Easting",
        n_col="Grid Ref: Northing",
        bin_size=2000,
    )
    X = spatial_fix.fit_transform(X)

    return X


def score(name: str, proba: np.ndarray, y_valid: np.ndarray):
    pr = float(average_precision_score(y_valid, proba))
    brier = float(brier_score_loss(y_valid, proba))
    ece = float(expected_calibration_error(y_valid, proba, n_bins=10))

    pred_u, pred_r, _, _ = triage_preds(proba, ALERT_RATE_URGENT, ALERT_RATE_REVIEW)
    out = {
        "model": name,
        "pr_auc": pr,
        "brier": brier,
        "ece_10bin": ece,
        "urgent_precision": float(precision_score(y_valid, pred_u, zero_division=0)),
        "urgent_recall": float(recall_score(y_valid, pred_u, zero_division=0)),
        "review_precision": float(precision_score(y_valid, pred_r, zero_division=0)),
        "review_recall": float(recall_score(y_valid, pred_r, zero_division=0)),
    }
    return out


def main():
    os.makedirs("runs/exp004", exist_ok=True)

    df = pd.read_csv("data/traffic.csv")

    if "Casualty Severity" not in df.columns:
        raise ValueError("Missing required target column: 'Casualty Severity'")

    df = df.copy()
    df["_y"] = to_binary_high_risk(df["Casualty Severity"])

    df_train, df_valid, _ = get_split(df)

    y_train = df_train["_y"].to_numpy()
    y_valid = df_valid["_y"].to_numpy()

    X_train_raw = df_train.drop(columns=["_y"])
    X_valid_raw = df_valid.drop(columns=["_y"])

    # Apply SAME feature engineering
    X_train = apply_feature_engineering(X_train_raw)
    X_valid = apply_feature_engineering(X_valid_raw)

    # Select locked final features
    X_train = X_train[FEATURE_COLS].copy()
    X_valid = X_valid[FEATURE_COLS].copy()

    pre = build_preprocessor(X_train)

    base_rf = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_split=2, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    base_gb = Pipeline([
        ("pre", pre),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    # Fit base models
    base_rf.fit(X_train, y_train)
    base_gb.fit(X_train, y_train)

    proba_rf = base_rf.predict_proba(X_valid)[:, 1]
    proba_gb = base_gb.predict_proba(X_valid)[:, 1]

    # Calibrate on TRAIN only via CV (no peeking at valid)
    rf_sig = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=5)
    rf_iso = CalibratedClassifierCV(estimator=base_rf, method="isotonic", cv=5)

    gb_sig = CalibratedClassifierCV(estimator=base_gb, method="sigmoid", cv=5)
    gb_iso = CalibratedClassifierCV(estimator=base_gb, method="isotonic", cv=5)

    rf_sig.fit(X_train, y_train)
    rf_iso.fit(X_train, y_train)
    gb_sig.fit(X_train, y_train)
    gb_iso.fit(X_train, y_train)

    proba_rf_sig = rf_sig.predict_proba(X_valid)[:, 1]
    proba_rf_iso = rf_iso.predict_proba(X_valid)[:, 1]
    proba_gb_sig = gb_sig.predict_proba(X_valid)[:, 1]
    proba_gb_iso = gb_iso.predict_proba(X_valid)[:, 1]

    rows = []
    rows.append(score("RF_base", proba_rf, y_valid))
    rows.append(score("RF_sigmoid", proba_rf_sig, y_valid))
    rows.append(score("RF_isotonic", proba_rf_iso, y_valid))
    rows.append(score("GB_base", proba_gb, y_valid))
    rows.append(score("GB_sigmoid", proba_gb_sig, y_valid))
    rows.append(score("GB_isotonic", proba_gb_iso, y_valid))

    out = pd.DataFrame(rows).sort_values(["pr_auc", "brier"], ascending=[False, True])
    print("\n=== Exp004 Calibration (VALID) ===")
    print(out.to_string(index=False))

    out.to_csv("runs/exp004/calibration_valid.csv", index=False)
    print("\nSaved: runs/exp004/calibration_valid.csv")

    # Save probas for stability script
    np.savez(
        "runs/exp004/probas_valid_calibrated.npz",
        y_valid=y_valid,
        RF_base=proba_rf,
        RF_sigmoid=proba_rf_sig,
        RF_isotonic=proba_rf_iso,
        GB_base=proba_gb,
        GB_sigmoid=proba_gb_sig,
        GB_isotonic=proba_gb_iso,
    )
    print("Saved: runs/exp004/probas_valid_calibrated.npz")

    # Reliability plot (single figure)
    plt.figure()
    plt.plot([0, 1], [0, 1])

    for name, proba in [
        ("RF_base", proba_rf),
        ("RF_sigmoid", proba_rf_sig),
        ("RF_isotonic", proba_rf_iso),
        ("GB_base", proba_gb),
    ]:
        frac_pos, mean_pred = calibration_curve(y_valid, proba, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", linestyle="-", label=name)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Curves (Validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_path = "runs/exp004/reliability_curve.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()