# api/app.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.experiments.decision import triage_level


MODEL_VERSION = "exp005_groupware_final"
RUN_DIR = Path("runs") / MODEL_VERSION

MODEL_PATH = RUN_DIR / "artifacts" / "model.joblib"
THRESH_PATH = RUN_DIR / "artifacts" / "threshold.json"


FEATURE_COLUMNS_UI: List[str] = [
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


BANNED_COLUMNS = {
    "Reference Number",
    "Vehicle Number",
    "Local Authority",
    "1st Road Class & No",  
    "Grid Ref: Easting",    # internal-only
    "Grid Ref: Northing",   # internal-only
}


PIPELINE_EXPECTS: List[str] = [
    "1st Road Class",
    "1st Road Class & No",
    "Accident Date",
    "Age of Casualty",
    "Casualty Class",
    "Grid Ref: Easting",
    "Grid Ref: Northing",
    "Lighting Conditions",
    "Number of Vehicles",
    "Road Surface",
    "Sex of Casualty",
    "TimeHour",
    "Type of Vehicle",
    "Weather Conditions",
]


UI_TO_PIPELINE = {
    "GridBin_E": "Grid Ref: Easting",
    "GridBin_N": "Grid Ref: Northing",
}

app = FastAPI(title="RiskTriage API", version="5.0.1")  # schema adapter fix

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]   # risktriage/
FRONTEND_DIR = BASE_DIR / "frontend"


app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if not MODEL_PATH.exists():
    raise RuntimeError(f"Missing model artifact: {MODEL_PATH}")

if not THRESH_PATH.exists():
    raise RuntimeError(f"Missing threshold artifact: {THRESH_PATH}")

model = joblib.load(MODEL_PATH)
cfg = json.loads(THRESH_PATH.read_text())

# accept common key variants safely
THR_URGENT = float(cfg.get("threshold_urgent", cfg.get("thr_urgent", cfg.get("urgent"))))
THR_REVIEW = float(cfg.get("threshold_review", cfg.get("thr_review", cfg.get("review"))))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "thresholds": {"urgent": THR_URGENT, "review": THR_REVIEW},
        "ui_schema_columns": FEATURE_COLUMNS_UI,
        "pipeline_expected_columns": PIPELINE_EXPECTS,
    }


def _validate_schema(payload: Dict[str, Any]) -> None:
    keys = set(payload.keys())

    # Banned fields must never appear
    banned_present = keys.intersection(BANNED_COLUMNS)
    if banned_present:
        raise HTTPException(
            status_code=400,
            detail=f"Banned fields present: {sorted(list(banned_present))}",
        )

    expected = set(FEATURE_COLUMNS_UI)
    missing = sorted(list(expected - keys))
    extra = sorted(list(keys - expected))

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid schema for Exp005 (UI contract).",
                "missing": missing,
                "extra": extra,
                "expected_columns": FEATURE_COLUMNS_UI,
            },
        )


def _coerce_ui_types(row_ui: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter layer for Exp005:
    - UI sends human-readable strings
    - The saved sklearn pipeline expects numeric-coded categories for some fields
    - We also normalize numeric fields + validate ranges
    """


    rc = row_ui.get("1st Road Class")
    if rc is None:
        pass
    elif isinstance(rc, (int, float)):
        row_ui["1st Road Class"] = int(rc)
    elif isinstance(rc, str):
        road_map = {"A": 1, "B": 2, "C": 3, "Unclassified": 4}

        # if UI sends numeric as string like "1"
        s = rc.strip()
        if s.isdigit():
            row_ui["1st Road Class"] = int(s)
        else:
            if rc not in road_map:
                raise HTTPException(status_code=400, detail=f"Unknown value for '1st Road Class': {rc}")
            row_ui["1st Road Class"] = road_map[rc]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid type for '1st Road Class': {type(rc).__name__}")


    cat_maps = {
        "Sex of Casualty": {"Male": 1, "Female": 2, "Unknown": 0},
        "Casualty Class": {"Driver or rider": 1, "Passenger": 2, "Pedestrian": 3, "Unknown": 0},
        "Lighting Conditions": {
            "Daylight": 1,
            "Darkness - lights lit": 2,
            "Darkness - no lighting": 3,
            "Darkness - lights unlit": 4,
            "Unknown": 0,
        },
        "Road Surface": {
            "Dry": 1,
            "Wet or damp": 2,
            "Frost or ice": 3,
            "Snow": 4,
            "Flood": 5,
            "Unknown": 0,
        },
        "Weather Conditions": {
            "Fine no high winds": 1,
            "Raining no high winds": 2,
            "Snowing": 3,
            "Fog or mist": 4,
            "Unknown": 0,
        },
        "Type of Vehicle": {
            "Car": 1,
            "Motorcycle": 2,
            "Bus or coach": 3,
            "Goods vehicle": 4,
            "Bicycle": 5,
            "Other": 6,
            "Unknown": 0,
        },
    }

    for col, mapping in cat_maps.items():
        v = row_ui.get(col)
        if v is None:
            continue
        if isinstance(v, str):
            if v not in mapping:
                raise HTTPException(status_code=400, detail=f"Unknown value for '{col}': {v}")
            row_ui[col] = mapping[v]
        else:
            # allow already-numeric codes
            row_ui[col] = int(float(v))

    # --- 3) Numeric normalization ---
    for k in ["Age of Casualty", "GridBin_E", "GridBin_N", "Number of Vehicles", "TimeHour"]:
        if row_ui.get(k) is None:
            continue
        try:
            row_ui[k] = int(float(row_ui[k]))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid numeric for '{k}': {row_ui.get(k)} ({e})")

    # --- 4) Range guards ---
    if row_ui.get("Age of Casualty") is not None and not (0 <= row_ui["Age of Casualty"] <= 120):
        raise HTTPException(status_code=400, detail="Age of Casualty out of range (0–120).")
    if row_ui.get("TimeHour") is not None and not (0 <= row_ui["TimeHour"] <= 23):
        raise HTTPException(status_code=400, detail="TimeHour out of range (0–23).")

    return row_ui


def _adapt_ui_to_pipeline(row_ui: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the strict 13-field UI row into the schema that the saved sklearn pipeline expects.
    This is the key fix for the 'columns are missing' error.
    """
    row_pipe: Dict[str, Any] = {}


    for k in FEATURE_COLUMNS_UI:
        if k in UI_TO_PIPELINE:
            row_pipe[UI_TO_PIPELINE[k]] = row_ui.get(k, None)
        else:
            row_pipe[k] = row_ui.get(k, None)


    row_pipe["1st Road Class & No"] = row_ui.get("1st Road Class")


    for col in PIPELINE_EXPECTS:
        row_pipe.setdefault(col, None)

    return row_pipe


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Exp005 deployable endpoint.
    - Hard schema check: exactly 13 UI features
    - No banned fields accepted
    - Internally adapts to pipeline-required columns (Grid Ref: *, 1st Road Class & No)
    - Model + thresholds loaded ONLY from runs/exp005_groupware_final/
    """
    try:
        _validate_schema(payload)

        # Strict UI row
        row_ui = {k: payload.get(k, None) for k in FEATURE_COLUMNS_UI}
        row_ui = _coerce_ui_types(row_ui)

        # Adapt to the saved pipeline's expected schema
        row_pipe = _adapt_ui_to_pipeline(row_ui)
        X = pd.DataFrame([row_pipe], columns=PIPELINE_EXPECTS)

        prob = float(model.predict_proba(X)[:, 1][0])
        level = triage_level(prob, thr_urgent=THR_URGENT, thr_review=THR_REVIEW)

        return {
            "model_version": MODEL_VERSION,
            "risk_score": prob,
            "triage_level": level,  # URGENT / REVIEW / LOW
            "decision_urgent": level == "URGENT",
            "decision_review": level in ("URGENT", "REVIEW"),
            "thresholds": {"urgent": THR_URGENT, "review": THR_REVIEW},
            "why": {
                "risk_score": round(prob, 4),
                "triage_level": level,
                "thresholds": {
                    "urgent": THR_URGENT,
                    "review": THR_REVIEW,
                },
                "input_snapshot": row_ui
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

