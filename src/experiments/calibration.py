# calibration.py
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(base_pipeline, X_cal, y_cal, method: str = "isotonic"):
    """
    Calibrate probabilities. This wraps your pipeline.
    method: "sigmoid" or "isotonic"
    """
    cal = CalibratedClassifierCV(estimator=base_pipeline, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal