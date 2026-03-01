

# src/decision.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score


def to_binary_high_risk(casualty_severity_series) -> np.ndarray:
    """
    Assumes typical encoding:
      1 = Fatal
      2 = Serious
      3 = Slight
    High-risk = {1,2}.
    """
    sev = casualty_severity_series.astype(str).str.strip()
    return np.where(sev.isin(["1", "2"]), 1, 0).astype(int)


@dataclass
class ThresholdResult:
    threshold: float
    recall: float
    precision: float


def select_threshold_recall_first(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.80,
    step: float = 0.01,
) -> ThresholdResult:
    """
    Picks the HIGHEST threshold that still achieves recall >= min_recall.
    (Higher threshold => fewer alerts, but lower recall.)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best: Optional[ThresholdResult] = None

    for thr in np.arange(0.0, 1.000001, step):
        y_pred = (y_prob >= thr).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= min_recall:
            p = precision_score(y_true, y_pred, zero_division=0)
            best = ThresholdResult(threshold=float(thr), recall=float(r), precision=float(p))

    if best is None:
        # If we can't meet min_recall, go to max recall (thr=0.0)
        y_pred = (y_prob >= 0.0).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        return ThresholdResult(threshold=0.0, recall=float(r), precision=float(p))

    return best


def threshold_by_alert_rate(probs: np.ndarray, alert_rate: float = 0.35) -> float:
    """
    alert_rate=0.35 => threshold at 65th percentile => top 35% are URGENT.
    """
    probs = np.asarray(probs, dtype=float)
    if not (0.0 < alert_rate < 1.0):
        raise ValueError("alert_rate must be between 0 and 1 (exclusive), e.g., 0.35")
    return float(np.quantile(probs, 1.0 - alert_rate))


def triage_level(prob: float, thr_urgent: float, thr_review: float) -> str:
    """
    Two-stage triage:
      prob >= thr_urgent -> URGENT
      prob >= thr_review -> REVIEW
      else -> LOW
    NOTE: We expect thr_urgent >= thr_review.
    """
    if prob >= thr_urgent:
        return "URGENT"
    if prob >= thr_review:
        return "REVIEW"
    return "LOW"