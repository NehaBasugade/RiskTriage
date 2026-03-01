# 🚦 RiskTriage  
## Operational Accident Severity Prioritization System

RiskTriage is an end-to-end machine learning decision-support system that predicts accident severity risk and converts calibrated probabilities into actionable triage levels:

**URGENT · REVIEW · LOW**

Unlike typical ML demos focused on raw accuracy, this system is engineered around operational decision-making under workload constraints.

---

## 🎯 Project Objective

Build a production-style ML system that:

- Predicts calibrated probability of high-risk accident severity
- Uses dual-threshold triage logic instead of raw classification
- Controls alert workload via alert-rate policy
- Prevents data leakage via group-aware splitting
- Separates validation and test evaluation correctly
- Deploys via API and UI

This project treats ML as a **decision system**, not just a classifier.

---

## 📊 Dataset

- Rows: **1,907**
- Target: Casualty Severity
- Binary mapping:
  - `high_risk = 1` if severity ∈ {1, 2}
  - `high_risk = 0` otherwise
- Class imbalance present → **PR-AUC** used instead of accuracy

---

## 🔒 Data Leakage Prevention

- Group-aware splitting using Reference Number
- 0% overlap between train / validation / test
- Validation used only for threshold selection
- Test set used only for final reporting

Archived leaky experiment:

🔒 Data Leakage Prevention
- To avoid optimistic bias:
- Group-aware splitting using Reference Number
- 0% overlap between train, validation, and test
- Validation used only for threshold selection
- Test set used only for final reporting
- Earlier row-wise experiments were archived under:
runs/exp004_rowwise_leaky/
- Final deployable model:
runs/exp005_groupaware_final/


---

## 🧠 Model

**RandomForestClassifier**
- n_estimators = 150  
- max_depth = 18  
- min_samples_leaf = 2  
- class_weight = "balanced_subsample"  
- random_state = 42  

### Feature Engineering

- Hour extraction from timestamp
- Spatial grid binning (Easting/Northing)
- Rare category collapse (min_freq = 20)
- Median imputation for numeric features
- OneHotEncoder(handle_unknown="ignore")

---

## 📈 Probability Calibration

Manual Platt Scaling:

- Out-of-fold predictions using GroupKFold
- Logistic regression sigmoid fit
- No leakage
- Applied before threshold selection

Held-out Test Calibration:
- Brier Score: **0.1467**
- Expected Calibration Error (ECE): **0.031**

Low ECE indicates well-calibrated probabilities.

---

## 🚦 Dual-Threshold Triage Policy

Instead of fixed 0.5 classification:

- Alert-rate constrained thresholds
- Validation-configured policy:
  - alert_rate_urgent = 0.35
  - alert_rate_review = 0.60

Decision rule:

- `p ≥ threshold_urgent → URGENT`
- `threshold_review ≤ p < threshold_urgent → REVIEW`
- else → LOW

Final thresholds:
- threshold_urgent = 0.4577
- threshold_review = 0.3232

---

## 🏆 Final Held-Out Performance

**PR-AUC: 0.3497**

URGENT stream:
- Recall: 68%
- Precision: 0.3148

REVIEW + URGENT combined:
- Recall: 88%

Only 12% of severe cases are missed.

---

## 🌐 Deployment

Two serving modes:

1. FastAPI backend
2. Streamlit app (Hugging Face Spaces)

Deployed app displays:
- Model version
- Decision thresholds
- Risk probability
- Final triage classification

---

## 📁 Repository Structure

RiskTriage/
│
├── src/                         # Core ML logic
│   ├── data.py                  # Data loading & preprocessing
│   ├── pipeline.py              # Feature pipeline definition
│   ├── model.py                 # Model configuration
│   ├── calibration.py           # Platt scaling implementation
│   ├── decision.py              # Dual-threshold triage policy
│   ├── split_check.py           # Group-aware split validation
│   ├── evaluate.py              # Evaluation metrics (PR-AUC, etc.)
│   ├── inference.py             # Single-case prediction logic
│   └── train_eval_groupaware.py # Full training pipeline
│
├── api/                         # FastAPI backend
│   └── app.py                   # REST inference endpoint
│
├── frontend/                    # HTML/CSS/JS UI
│
├── runs/                        # Experiment tracking & artifacts
│   └── exp005_groupaware_final/
│       ├── artifacts/
│       │   ├── model.joblib
│       │   └── threshold.json
│       └── metrics/
│           └── metrics_test.json
│
├── data/                        # Dataset (ignored if large)
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation

---

## 🌐 Deployment
- Two serving modes:
- FastAPI + HTML/CSS/JS (local API server)
- Streamlit deployed on Hugging Face Spaces
- The hosted version displays:
- Model version
- Decision thresholds
- Risk probability
- Final triage classification

---

## 🧪 Reproducibility

- Create environment:
- python -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- Train:
- python src/train_eval_groupaware.py
- Evaluate:
- python src/evaluate.py
- Run API:
- uvicorn api.app:app --reload

---

## 💡 What This Demonstrates

- Applied ML engineering
- Data leakage prevention
- Group-aware cross-validation
- Probability calibration
- Threshold-based operational decision systems
- Artifact versioning
- API deployment
- End-to-end ML ownership

---

## 📌 Key Insight

RiskTriage is not optimized for accuracy.

It is engineered to balance missed severe cases versus alert fatigue under real-world operational constraints.

## 🌐 Live Demo

🔗 **Streamlit App (Hugging Face Spaces)**  
[Open RiskTriage Demo]([https://huggingface.co/spaces/nehabasugade/risktriage-demo])

🔗 **API (FastAPI Endpoint)**  
[View API Docs](https://your-api-link-here/docs)

