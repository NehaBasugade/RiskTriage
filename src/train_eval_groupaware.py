from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import os, json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss

import matplotlib.pyplot as plt

# ---------------- locked config ----------------
DATA_PATH = os.path.join("data", "traffic.csv")
SPLIT_PATH = os.path.join("exp005_groupaware", "split_indices.json")

GROUP_COL = "Reference Number"
TARGET_COL = "Casualty Severity"

FINAL_FEATURES = [
    "1st Road Class",
    "Accident Date",
    "Age of Casualty",
    "Casualty Class",
    "GridBin_E",
    "GridBin_N",
    "Lighting Conditions",
    "Number of Vehicles",
    "Road Surface",
    "Sex of Casualty",
    "TimeHour",
    "Type of Vehicle",
    "Weather Conditions",
]

ALERT_RATE_URGENT = 0.35
ALERT_RATE_REVIEW = 0.60
RANDOM_STATE = 42


# ---------------- target mapping (locked) ----------------
def to_binary_high_risk(casualty_severity_series: pd.Series) -> np.ndarray:
    sev = casualty_severity_series.astype(str).str.strip()
    return np.where(sev.isin(["1", "2"]), 1, 0).astype(int)


# ---------------- FE (locked) ----------------
class TimeHHMMToHour(BaseEstimator, TransformerMixin):
    def __init__(self, timehour_col="TimeHour", candidate_time_cols=None):
        self.timehour_col = timehour_col
        # lock to your dataset's actual column name first
        self.candidate_time_cols = candidate_time_cols or [
            "Time (24hr)",
            "Time",
            "Accident Time",
            "Time of Accident",
            "Time (HHMM)",
            "Time (HH:MM)",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.timehour_col in X.columns:
            X[self.timehour_col] = pd.to_numeric(X[self.timehour_col], errors="coerce")
            return X

        src = next((c for c in self.candidate_time_cols if c in X.columns), None)
        if src is None:
            raise ValueError("TimeHour missing and no candidate time column found.")

        s = X[src].astype(str).str.strip()
        hour = pd.Series(np.nan, index=X.index)

        # "HH:MM"
        mask_colon = s.str.contains(":", na=False)
        hour.loc[mask_colon] = pd.to_numeric(s[mask_colon].str.split(":").str[0], errors="coerce")

        # "HHMM" numeric
        mask_digits = (~mask_colon) & s.str.fullmatch(r"\d{3,4}", na=False)
        hhmm = pd.to_numeric(s[mask_digits], errors="coerce")
        hour.loc[mask_digits] = np.floor(hhmm / 100.0)

        # "H" or "HH"
        mask_hour = (~mask_colon) & (~mask_digits) & s.str.fullmatch(r"\d{1,2}", na=False)
        hour.loc[mask_hour] = pd.to_numeric(s[mask_hour], errors="coerce")

        X[self.timehour_col] = hour
        return X


class SpatialBinner(BaseEstimator, TransformerMixin):
    def __init__(self, e_bin_col="GridBin_E", n_bin_col="GridBin_N", bin_meters=2000,
                 candidate_e_cols=None, candidate_n_cols=None):
        self.e_bin_col = e_bin_col
        self.n_bin_col = n_bin_col
        self.bin_meters = bin_meters
     
        self.candidate_e_cols = candidate_e_cols or ["Grid Ref: Easting", "Easting", "E", "Grid Ref E", "X"]
        self.candidate_n_cols = candidate_n_cols or ["Grid Ref: Northing", "Northing", "N", "Grid Ref N", "Y"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.e_bin_col in X.columns and self.n_bin_col in X.columns:
            X[self.e_bin_col] = pd.to_numeric(X[self.e_bin_col], errors="coerce")
            X[self.n_bin_col] = pd.to_numeric(X[self.n_bin_col], errors="coerce")
            return X

        e_src = next((c for c in self.candidate_e_cols if c in X.columns), None)
        n_src = next((c for c in self.candidate_n_cols if c in X.columns), None)
        if e_src is None or n_src is None:
            raise ValueError("GridBin_E/N missing and no easting/northing sources found.")

        e = pd.to_numeric(X[e_src], errors="coerce")
        n = pd.to_numeric(X[n_src], errors="coerce")

        X[self.e_bin_col] = np.floor(e / self.bin_meters)
        X[self.n_bin_col] = np.floor(n / self.bin_meters)
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Array-safe rare category grouper.
    Works with both DataFrame and numpy arrays (from ColumnTransformer).
    """
    def __init__(self, min_freq=20, rare_token="__RARE__"):
        self.min_freq = min_freq
        self.rare_token = rare_token
        self.keep_by_col_ = None

    def fit(self, X, y=None):
        X_arr = self._to_2d_array(X)
        self.keep_by_col_ = []
        for j in range(X_arr.shape[1]):
            s = pd.Series(X_arr[:, j]).astype(str).fillna("__MISSING__")
            vc = s.value_counts(dropna=False)
            keep = set(vc[vc >= self.min_freq].index.tolist())
            self.keep_by_col_.append(keep)
        return self

    def transform(self, X):
        X_arr = self._to_2d_array(X).copy()
        for j in range(X_arr.shape[1]):
            keep = self.keep_by_col_[j]
            s = pd.Series(X_arr[:, j]).astype(str).fillna("__MISSING__")
            X_arr[:, j] = np.where(s.isin(keep), s.values, self.rare_token)
        return X_arr

    @staticmethod
    def _to_2d_array(X):
        if isinstance(X, pd.DataFrame):
            return X.values
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


# ---------------- metrics helpers ----------------
def expected_calibration_error(y_true, p, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def thresholds_from_alert_rates(p_valid, alert_rate_urgent, alert_rate_review):
    p_valid = np.asarray(p_valid).astype(float)
    t_urgent = float(np.quantile(p_valid, 1.0 - alert_rate_urgent))
    t_review = float(np.quantile(p_valid, 1.0 - alert_rate_review))
    if t_review > t_urgent:
        t_review, t_urgent = t_urgent, t_review
    return t_review, t_urgent


def tier_metrics(y_true, p, t_review, t_urgent):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    pred_urgent = (p >= t_urgent)
    pred_alert = (p >= t_review)

    def prec_recall(mask):
        flagged = int(mask.sum())
        if flagged == 0:
            return 0.0, 0.0, 0.0
        tp = int((y_true[mask] == 1).sum())
        precision = tp / flagged
        recall = tp / max(1, int((y_true == 1).sum()))
        flag_rate = flagged / len(y_true)
        return float(precision), float(recall), float(flag_rate)

    pu, ru, fu = prec_recall(pred_urgent)
    pr, rr, fr = prec_recall(pred_alert)

    return {
        "t_review": float(t_review),
        "t_urgent": float(t_urgent),
        "URGENT_precision": pu,
        "URGENT_recall": ru,
        "URGENT_flag_rate": fu,
        "REVIEW_precision": pr,
        "REVIEW_recall": rr,
        "REVIEW_flag_rate": fr,
    }


def plot_reliability(y_true, p, out_png):
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=15, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Curve (Group-aware Test)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------------- build PREPROCESS ONLY ----------------
def build_preprocess(X: pd.DataFrame) -> Pipeline:
    numeric_cols = []
    cat_cols = []

    for c in FINAL_FEATURES:
        if c not in X.columns:
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    # force known numeric if present
    for c in ["Age of Casualty", "Number of Vehicles", "TimeHour", "GridBin_E", "GridBin_N"]:
        if c in FINAL_FEATURES and c in X.columns:
            if c not in numeric_cols:
                numeric_cols.append(c)
            if c in cat_cols:
                cat_cols.remove(c)

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(min_freq=20)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        [("num", num_pipe, numeric_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )

    # FE + Preprocess
    return Pipeline([
        ("timehour", TimeHHMMToHour(candidate_time_cols=["Time (24hr)"])),
        ("spatial", SpatialBinner(
            bin_meters=2000,
            candidate_e_cols=["Grid Ref: Easting"],
            candidate_n_cols=["Grid Ref: Northing"],
        )),
        ("pre", pre),
    ])

def fit_sigmoid_calibrator_groupcv(rf, Xtr, ytr, groups_tr, n_splits=5, random_state=42):
    """
    Manual group-aware Platt scaling:
    - GroupKFold -> out-of-fold base probs
    - Fit LogisticRegression on probs -> calibrated mapping
    Returns:
      rf_full (fit on all train),
      calibrator (LogisticRegression),
      oof_ap (sanity)
    """
    gkf = GroupKFold(n_splits=n_splits)

    oof_p = np.zeros(len(ytr), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xtr, ytr, groups_tr), 1):
        m = clone(rf)
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        oof_p[va_idx] = m.predict_proba(Xtr[va_idx])[:, 1]

   
    cal = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
    cal.fit(oof_p.reshape(-1, 1), ytr)

    # Fit base model on full train
    rf_full = clone(rf)
    rf_full.fit(Xtr, ytr)

    # sanity metric on OOF
    oof_ap = float(average_precision_score(ytr, oof_p))
    return rf_full, cal, oof_ap


# ---------------- main ----------------
assert os.path.exists(DATA_PATH), "Missing data/traffic.csv"
assert os.path.exists(SPLIT_PATH), "Missing exp005_groupaware/split_indices.json (run split_check.py first)"

df = pd.read_csv(DATA_PATH)
with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

idx_train = np.array(split["idx_train"], dtype=int)
idx_valid = np.array(split["idx_valid"], dtype=int)
idx_test = np.array(split["idx_test"], dtype=int)

y = to_binary_high_risk(df[TARGET_COL])
groups = df[GROUP_COL].astype(str).values
X = df.copy()

# 1) preprocess -> matrices
fe_pre = build_preprocess(X)
Xtr = fe_pre.fit_transform(X.iloc[idx_train], y[idx_train])
Xva = fe_pre.transform(X.iloc[idx_valid])
Xte = fe_pre.transform(X.iloc[idx_test])

# 2) locked model
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# 3) locked calibration
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Manual group-aware sigmoid calibration
rf_full, sigmoid_cal, oof_ap = fit_sigmoid_calibrator_groupcv(
    rf,
    Xtr, y[idx_train],
    groups[idx_train],
    n_splits=5,
    random_state=RANDOM_STATE
)
print("[CALIB] OOF PR-AUC (train only sanity):", round(oof_ap, 4))

# Base probs then calibrated probs
p_valid_base = rf_full.predict_proba(Xva)[:, 1]
p_test_base  = rf_full.predict_proba(Xte)[:, 1]

p_valid = sigmoid_cal.predict_proba(p_valid_base.reshape(-1, 1))[:, 1]
p_test  = sigmoid_cal.predict_proba(p_test_base.reshape(-1, 1))[:, 1]

# thresholds from VALID only 
t_review, t_urgent = thresholds_from_alert_rates(p_valid, ALERT_RATE_URGENT, ALERT_RATE_REVIEW)

# metrics on TEST
pr_auc = float(average_precision_score(y[idx_test], p_test))
brier = float(brier_score_loss(y[idx_test], p_test))
ece = float(expected_calibration_error(y[idx_test], p_test, n_bins=15))
tiers = tier_metrics(y[idx_test], p_test, t_review, t_urgent)

metrics = {
    "PR_AUC": pr_auc,
    "Brier": brier,
    "ECE": ece,
    **tiers,
}

print("[THRESHOLDS]")
print("t_review =", metrics["t_review"])
print("t_urgent =", metrics["t_urgent"])

print("\n[GROUP-AWARE TEST METRICS]")
print(json.dumps(metrics, indent=2))

os.makedirs("exp005_groupaware", exist_ok=True)
with open("exp005_groupaware/metrics_test.json", "w") as f:
    json.dump(metrics, f, indent=2)

plot_reliability(y[idx_test], p_test, "exp005_groupaware/reliability_test.png")
print("\n[SAVED] exp005_groupaware/metrics_test.json")
print("[SAVED] exp005_groupaware/reliability_test.png")


phase4 = {
    "PR_AUC": 0.360,
    "Brier": 0.176,
    "ECE": 0.076,
    "URGENT_precision": 0.365,
    "URGENT_recall": 0.543,
    "URGENT_flag_rate": 0.35,
    "REVIEW_precision": 0.292,
    "REVIEW_recall": 0.743,
    "REVIEW_flag_rate": 0.60,
}

keys = [
    "PR_AUC", "Brier", "ECE",
    "URGENT_precision", "URGENT_recall", "URGENT_flag_rate",
    "REVIEW_precision", "REVIEW_recall", "REVIEW_flag_rate",
]

print("\n[PHASE4 vs PHASE5]")
print(f"{'metric':<18} {'phase4':>10} {'phase5':>10} {'delta':>10}")
for k in keys:
    v4 = phase4.get(k, np.nan)
    v5 = metrics.get(k, np.nan)
    d = v5 - v4
    print(f"{k:<18} {v4:>10.3f} {v5:>10.3f} {d:>10.3f}")