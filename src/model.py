

# model.py
# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )


def build_logreg(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )