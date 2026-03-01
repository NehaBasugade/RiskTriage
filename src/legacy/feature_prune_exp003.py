# src/feature_prune.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class PIRunConfig:
    """
    Permutation Importance run config.

    NOTE:
    - We compute PI at the RAW COLUMN level (X_valid columns), because
      sklearn.permutation_importance permutes the input columns *before* preprocessing
      when you pass a full Pipeline as the estimator.
    - Therefore OHE aggregation is NOT applicable here.
    """
    n_repeats: int = 10
    random_state: int = 42
    n_jobs: int = 1
    agg: str = "none"


def compute_permutation_importance_pr_auc(
    fitted_estimator,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: PIRunConfig = PIRunConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes permutation importance on VALIDATION using PR-AUC (Average Precision).

    Returns:
      - grouped_importance: feature-level PI (raw columns)
      - raw_importance: same as grouped (kept for compatibility)
    """
    result = permutation_importance(
        fitted_estimator,
        X_valid,
        y_valid,
        scoring="average_precision",
        n_repeats=cfg.n_repeats,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    feature_names = np.array(X_valid.columns, dtype=str)

    raw = pd.DataFrame({
        "feature": feature_names,
        "pi_mean": result.importances_mean.astype(float),
        "pi_std": result.importances_std.astype(float),
    }).sort_values("pi_mean", ascending=False).reset_index(drop=True)

    grouped = raw.copy()
    return grouped, raw


def suggest_prune_batch(
    grouped_importance: pd.DataFrame,
    drop_frac: float = 0.10,
    min_drop: int = 1,
    max_drop: int = 3,
    protect: List[str] | None = None,
) -> List[str]:
    """
    Suggest a prune batch from the bottom of PI.

    Safety:
      - excludes 'protect'
      - drops bottom drop_frac, capped to [min_drop, max_drop]
    """
    protect = set(protect or [])
    df = grouped_importance.copy()

    candidates = df[~df["feature"].isin(protect)].copy()
    if candidates.empty:
        return []

    k = int(np.ceil(len(candidates) * drop_frac))
    k = max(min_drop, k)
    k = min(max_drop, k)

    bottom = candidates.sort_values("pi_mean", ascending=True).head(k)
    return bottom["feature"].tolist()


def save_pi_tables(
    grouped: pd.DataFrame,
    raw: pd.DataFrame,
    out_dir: str = "artifacts/pi",
    tag: str = "exp003",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grouped.to_csv(Path(out_dir) / f"{tag}_grouped.csv", index=False)
    raw.to_csv(Path(out_dir) / f"{tag}_raw.csv", index=False)