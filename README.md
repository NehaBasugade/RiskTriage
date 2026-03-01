🚦 RiskTriage
Operational Accident Severity Prioritization System
RiskTriage is an end-to-end machine learning decision-support system that predicts accident severity risk and converts calibrated probabilities into actionable triage levels:
URGENT
REVIEW
LOW
Unlike typical ML demos focused on accuracy, this project is designed around operational decision-making under workload constraints.

🎯 Project Objective
Build a production-style ML system that:
Predicts calibrated probability of high-risk accident severity
Uses dual-threshold triage logic instead of raw classification
Controls alert workload via alert-rate policy
Prevents data leakage via group-aware splitting
Separates validation and test evaluation correctly
Deploys via API and UI
This project treats ML as a decision system, not just a classifier.

📊 Dataset
Rows: 1,907
Target: Casualty Severity
Binary mapping:
high_risk = 1 if Casualty Severity in {1, 2}
high_risk = 0 otherwise
Class imbalance exists, so PR-AUC is used instead of accuracy.

🔒 Data Leakage Prevention
To avoid optimistic bias:
Group-aware splitting using Reference Number
0% overlap between train, validation, and test
Validation used only for threshold selection
Test set used only for final reporting
Earlier row-wise experiments were archived under:
runs/exp004_rowwise_leaky/
Final deployable model:
runs/exp005_groupaware_final/

🧠 Model
RandomForestClassifier
n_estimators = 150
max_depth = 18
min_samples_leaf = 2
class_weight = "balanced_subsample"
random_state = 42
⚙ Feature Engineering
Time (24hr) → Hour extraction
Easting/Northing → Grid binning
Rare category collapse (min_freq = 20)
Median imputation for numeric features
OneHotEncoder(handle_unknown="ignore")

📈 Probability Calibration
Manual Platt Scaling:
Out-of-fold predictions using GroupKFold
Logistic regression sigmoid fit
No leakage
Applied before threshold selection
Calibration Metrics (Held-out Test):
Brier Score: 0.1467
Expected Calibration Error (ECE): 0.031
Low ECE indicates well-calibrated probabilities.

🚦 Dual-Threshold Triage Policy
Instead of using a fixed 0.5 cutoff, RiskTriage uses workload-controlled thresholds.
Validation-configured policy:
alert_rate_urgent = 0.35
alert_rate_review = 0.60
Decision Rule:
If p ≥ threshold_urgent → URGENT
If threshold_review ≤ p < threshold_urgent → REVIEW
Else → LOW
Final thresholds selected on validation:
threshold_urgent = 0.4577
threshold_review = 0.3232

🏆 Final Held-Out Test Metrics (Exp005)
Ranking Metric:
PR-AUC: 0.3497
URGENT Stream:
Precision: 0.3148
Recall: 0.68
Flag Rate: 0.4263
Interpretation:
68% of high-risk cases are captured in the immediate-action stream.
REVIEW Stream (REVIEW + URGENT combined):
Precision: 0.2716
Recall: 0.88
Flag Rate: 0.6395
Interpretation:
88% of high-risk cases are captured when including the review buffer.
Only 12% of severe cases are missed.

📁 Repository Structure
risktriage/
src/ — ML logic
api/ — FastAPI backend
frontend/ — HTML/CSS/JS interface
runs/ — Experiment tracking and artifacts
README.md
Final deployable artifacts:
runs/exp005_groupaware_final/artifacts/model.joblib
runs/exp005_groupaware_final/artifacts/threshold.json
runs/exp005_groupaware_final/metrics/metrics_test.json

🌐 Deployment
Two serving modes:
FastAPI + HTML/CSS/JS (local API server)
Streamlit deployed on Hugging Face Spaces
The hosted version displays:
Model version
Decision thresholds
Risk probability
Final triage classification

🧪 Reproducibility
Create virtual environment:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Train:
python src/train_eval_groupaware.py
Evaluate:
python src/evaluate.py
Run API:
uvicorn api.app:app --reload

💡 What This Project Demonstrates
Applied ML engineering
Data leakage prevention
Group-aware cross-validation
Probability calibration
Threshold-based operational decision systems
Artifact versioning
API deployment
End-to-end ML ownership

🚀 Future Improvements
Cost-sensitive threshold optimization
Capacity-aware decision curves
Drift detection simulation
SHAP-based model explanation
Monitoring dashboard

📌 Key Insight
RiskTriage is not built to maximize accuracy.
It is built to balance missed severe cases versus alert fatigue under real-world operational constraints.
