# src/experiments/exp_common.py

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42

# Locked feature set (do not change)
FEATURE_COLS = [
    '1st Road Class',
    'Accident Date',
    'Age of Casualty',
    'Casualty Class',
    'GridBin_E',
    'GridBin_N',
    'Lighting Conditions',
    'Number of Vehicles',
    'Road Surface',
    'Sex of Casualty',
    'TimeHour',
    'Type of Vehicle',
    'Weather Conditions'
]

# Locked hard drops (do not change)
DROP_ALWAYS = [
    "Reference Number",
    "Vehicle Number",
    "Local Authority",
    "1st Road Class & No",
]

# Locked triage policy (do not change unless you explicitly decide to)
ALERT_RATE_URGENT = 0.35
ALERT_RATE_REVIEW = 0.60


def to_binary_high_risk(casualty_severity_series: pd.Series) -> np.ndarray:
    """
    Locked target mapping:
    high_risk = 1 if Casualty Severity in {"1","2"} else 0
    """
    sev = casualty_severity_series.astype(str).str.strip()
    return np.where(sev.isin(["1", "2"]), 1, 0).astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Locked preprocessing:
      Numeric -> median impute
      Categorical -> most_frequent impute + OHE(ignore unknown)
    Accident Date remains as-is (string) and will be treated as categorical if non-numeric.
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    # sklearn compatibility: sparse_output vs sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def triage_thresholds(proba: np.ndarray, alert_rate_urgent: float, alert_rate_review: float):
    """
    Quantile thresholds that achieve target flag rates on the given distribution.
    """
    t_u = float(np.quantile(proba, 1.0 - alert_rate_urgent))
    t_r = float(np.quantile(proba, 1.0 - alert_rate_review))
    return t_u, t_r


def triage_preds(proba: np.ndarray, alert_rate_urgent: float, alert_rate_review: float):
    t_u, t_r = triage_thresholds(proba, alert_rate_urgent, alert_rate_review)
    pred_urgent = (proba >= t_u).astype(int)
    pred_review_or_urgent = (proba >= t_r).astype(int)
    return pred_urgent, pred_review_or_urgent, t_u, t_r


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Simple ECE (10-bin default). Used for *comparisons*, not absolute truth.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(proba, bins) - 1

    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if mask.sum() == 0:
            continue
        conf = float(proba[mask].mean())
        acc = float(y_true[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
    return float(ece)


def get_split(df: pd.DataFrame):
    """
    Uses your project split_data EXACTLY if available; otherwise raises.
    We do NOT want silent fallback here because it changes the split semantics.
    """
    try:
        # Adjust this import if your split_data lives elsewhere.
        from src.legacy.train import split_data
    except Exception as e:
        raise ImportError(
            "Could not import split_data from src.train. "
            "Fix the import path in exp_common.get_split() so Exp004 uses the SAME split."
        ) from e


    s = split_data(
    df,
    test_size=0.20,
    valid_size=0.20,
    random_state=RANDOM_STATE,
    group_col="Reference Number",
    )
    return s.train, s.valid, s.test