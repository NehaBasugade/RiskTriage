# src/train.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.experiments.data import load_data, split_data
from src.pipeline import build_preprocessor, TARGET_COL
from src.model import build_random_forest
from src.experiments.decision import to_binary_high_risk, threshold_by_alert_rate

# Phase 3: Permutation importance utilities
from src.legacy.feature_prune_exp003 import (
    PIRunConfig,
    compute_permutation_importance_pr_auc,
    save_pi_tables,
    suggest_prune_batch,
)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

GROUP_COL = "Reference Number"



# Logging


def log_split_fingerprint(train_df, valid_df, test_df, group_col: str = GROUP_COL) -> None:
    def fp(df):
        return {
            "rows": int(len(df)),
            "unique_groups": int(df[group_col].nunique()) if group_col in df.columns else None,
        }

    def overlap_count(a, b):
        if group_col not in a.columns or group_col not in b.columns:
            return None
        ga = set(a[group_col].astype(str).unique())
        gb = set(b[group_col].astype(str).unique())
        return int(len(ga & gb))

    overlaps = {
        "train_valid": overlap_count(train_df, valid_df),
        "train_test": overlap_count(train_df, test_df),
        "valid_test": overlap_count(valid_df, test_df),
    }

    print("\n" + "=" * 88)
    print("EVAL QUALITY WARNING: PROVISIONAL METRICS (Phase 2 = implementation validation)")
    print("Do NOT interpret these numbers as final. Phase 3 will re-check evaluation setup.")
    print("-" * 88)
    print(f"SPLIT_FINGERPRINT train={fp(train_df)} valid={fp(valid_df)} test={fp(test_df)} overlaps={overlaps}")
    print("=" * 88 + "\n")



# Feature Transformers


class TimeHHMMToHour(BaseEstimator, TransformerMixin):
    def __init__(self, time_col: str, out_col: str = "TimeHour", drop_raw: bool = True):
        self.time_col = time_col
        self.out_col = out_col
        self.drop_raw = drop_raw

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        s = pd.to_numeric(X[self.time_col], errors="coerce")
        hour = np.floor(s / 100.0)
        hour = hour.where((hour >= 0) & (hour <= 23), np.nan)
        X[self.out_col] = hour.astype("float")
        if self.drop_raw:
            X = X.drop(columns=[self.time_col])
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str], min_count: int = 20, other_label: str = "Other"):
        self.cols = cols
        self.min_count = min_count
        self.other_label = other_label
        self.keep_values_ = {}

    def fit(self, X, y=None):
        for c in self.cols:
            if c not in X.columns:
                continue
            vc = X[c].astype("string").value_counts(dropna=False)
            keep = set(vc[vc >= self.min_count].index.astype("string"))
            self.keep_values_[c] = keep
        return self

    def transform(self, X):
        X = X.copy()
        for c, keep in self.keep_values_.items():
            if c not in X.columns:
                continue
            s = X[c].astype("string")
            X[c] = s.where(s.isna() | s.isin(keep), self.other_label)
        return X


class SpatialBinner(BaseEstimator, TransformerMixin):
    """
    Bin Easting/Northing into coarse grid cells.
    """
    def __init__(self, e_col: str, n_col: str, bin_size: int = 2000):
        self.e_col = e_col
        self.n_col = n_col
        self.bin_size = bin_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.e_col in X.columns:
            e = pd.to_numeric(X[self.e_col], errors="coerce")
            X["GridBin_E"] = (e // self.bin_size).astype("Int64")

        if self.n_col in X.columns:
            n = pd.to_numeric(X[self.n_col], errors="coerce")
            X["GridBin_N"] = (n // self.bin_size).astype("Int64")

        # Drop raw coordinates
        drop_cols = [c for c in [self.e_col, self.n_col] if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X



# Training


def train(
    csv_path: str,
    exp_id: str = "exp003_b0",
    alert_rate_urgent: float = 0.35,
    alert_rate_review: float = 0.60,
    random_state: int = 42,
):

    df = load_data(csv_path)
    splits = split_data(df, random_state=random_state)
    log_split_fingerprint(splits.train, splits.valid, splits.test)

    y_train = to_binary_high_risk(splits.train[TARGET_COL])
    y_valid = to_binary_high_risk(splits.valid[TARGET_COL])

    X_train = splits.train.drop(columns=[TARGET_COL])
    X_valid = splits.valid.drop(columns=[TARGET_COL])

    # Hard drop ID / non-signal columns 
    DROP_ALWAYS = ["Reference Number", "Vehicle Number", "Local Authority", "1st Road Class & No"]
    X_train = X_train.drop(columns=[c for c in DROP_ALWAYS if c in X_train.columns])
    X_valid = X_valid.drop(columns=[c for c in DROP_ALWAYS if c in X_valid.columns])

    # === Keep time parsing ===
    time_fix = TimeHHMMToHour("Time (24hr)")
    X_train = time_fix.fit_transform(X_train)
    X_valid = time_fix.transform(X_valid)

    # === Keep rare collapsing ===
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
        bin_size=2000,  # 2km grid
    )
    X_train = spatial_fix.fit_transform(X_train)
    X_valid = spatial_fix.transform(X_valid)

 
    print("\n[DEBUG] Raw columns entering preprocessor (after feature engineering):")
    print(sorted(X_train.columns))

    # Build model
    preprocessor, _, _ = build_preprocessor(X_train)
    clf = build_random_forest(random_state=random_state)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    # --- Validation predictions ---
    valid_prob = pipe.predict_proba(X_valid)[:, 1]
    valid_pr_auc = float(average_precision_score(y_valid, valid_prob))

    thr_urgent = threshold_by_alert_rate(valid_prob, alert_rate_urgent)
    thr_review = threshold_by_alert_rate(valid_prob, alert_rate_review)

    pred_u = (valid_prob >= thr_urgent).astype(int)
    pred_r = (valid_prob >= thr_review).astype(int)

    urgent_recall = float(recall_score(y_valid, pred_u))
    urgent_prec = float(precision_score(y_valid, pred_u))
    review_recall = float(recall_score(y_valid, pred_r))
    review_prec = float(precision_score(y_valid, pred_r))

    print("Saved artifacts to:", ARTIFACT_DIR.resolve())
    print(f"Validation PR-AUC: {valid_pr_auc:.4f}")
    print(f"URGENT: recall={urgent_recall:.3f} precision={urgent_prec:.3f} (flag_rate={alert_rate_urgent:.3f})")
    print(f"REVIEW: recall={review_recall:.3f} precision={review_prec:.3f} (flag_rate={alert_rate_review:.3f})")

  
    pi_cfg = PIRunConfig(
        n_repeats=10,
        random_state=42,
        n_jobs=1,
        agg="none",
    )

    grouped_pi, raw_pi = compute_permutation_importance_pr_auc(
        fitted_estimator=pipe,
        X_valid=X_valid,
        y_valid=y_valid,
        cfg=pi_cfg,
    )

    save_pi_tables(grouped_pi, raw_pi, out_dir="artifacts/pi", tag=exp_id)

    print("\n[PI] Top 10 features (higher = more important):")
    print(grouped_pi.head(10).to_string(index=False))

    print("\n[PI] Bottom 10 features (near-zero/negative = candidates to drop):")
    print(grouped_pi.tail(10).to_string(index=False))

    # Protect engineered features early (do not prune these in first rounds)
    protect = ["TimeHour", "GridBin_E", "GridBin_N"]

    suggested_drop = suggest_prune_batch(
        grouped_importance=grouped_pi,
        drop_frac=0.10,
        min_drop=1,
        max_drop=3,
        protect=protect,
    )

    print(f"\n[PI] Suggested prune batch (DO NOT APPLY YET): {suggested_drop}")
    print(f"[PI] Grouped PI CSV: artifacts/pi/{exp_id}_grouped.csv")
    print(f"[PI] Raw PI CSV: artifacts/pi/{exp_id}_raw.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--exp_id", type=str, default="exp003_b0")
    parser.add_argument("--alert_rate_urgent", type=float, default=0.35)
    parser.add_argument("--alert_rate_review", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        csv_path=args.csv,
        exp_id=args.exp_id,
        alert_rate_urgent=args.alert_rate_urgent,
        alert_rate_review=args.alert_rate_review,
        random_state=args.seed,
    )