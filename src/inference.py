

# src/inference.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.experiments.decision import triage_level

ARTIFACT_DIR = Path("artifacts")


def predict_one(input_json_path: str):
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    cfg = json.loads((ARTIFACT_DIR / "threshold.json").read_text())

    thr_u = float(cfg["threshold_urgent"])
    thr_r = float(cfg["threshold_review"])

    payload = json.loads(Path(input_json_path).read_text())
    X = pd.DataFrame([payload])

    prob = float(model.predict_proba(X)[:, 1][0])
    level = triage_level(prob, thr_urgent=thr_u, thr_review=thr_r)

    out = {
        "risk_score": prob,
        "triage_level": level,  # URGENT / REVIEW / LOW
        "threshold_urgent": thr_u,
        "threshold_review": thr_r,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    predict_one(args.input)