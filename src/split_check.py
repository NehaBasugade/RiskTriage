import os
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = os.path.join("data", "traffic.csv")
GROUP_COL = "Reference Number"
TARGET_COL = "Casualty Severity"
RANDOM_STATE = 42

TEST_SIZE = 0.20     # trainvalid vs test
VALID_SIZE = 0.20    # train vs valid 

def to_binary_high_risk(casualty_severity_series: pd.Series) -> np.ndarray:
    sev = casualty_severity_series.astype(str).str.strip()
    return np.where(sev.isin(["1", "2"]), 1, 0).astype(int)

def assert_no_group_overlap(g_train, g_valid, g_test):
    s_tr = set(pd.Series(g_train).astype(str))
    s_va = set(pd.Series(g_valid).astype(str))
    s_te = set(pd.Series(g_test).astype(str))

    ov_tr_va = s_tr & s_va
    ov_tr_te = s_tr & s_te
    ov_va_te = s_va & s_te

    print(f"[OVERLAP] train∩valid groups: {len(ov_tr_va)}")
    print(f"[OVERLAP] train∩test  groups: {len(ov_tr_te)}")
    print(f"[OVERLAP] valid∩test  groups: {len(ov_va_te)}")

    assert len(ov_tr_va) == 0, f"Leak: train/valid share {len(ov_tr_va)} groups"
    assert len(ov_tr_te) == 0, f"Leak: train/test share {len(ov_tr_te)} groups"
    assert len(ov_va_te) == 0, f"Leak: valid/test share {len(ov_va_te)} groups"

def groups_hash16(groups) -> str:
    g_sorted = sorted(set(pd.Series(groups).astype(str).tolist()))
    raw = "|".join(g_sorted).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

assert os.path.exists(DATA_PATH), f"Missing {DATA_PATH}. Run from repo root."
df = pd.read_csv(DATA_PATH)

assert GROUP_COL in df.columns, f"Missing group col: {GROUP_COL}"
assert TARGET_COL in df.columns, f"Missing target col: {TARGET_COL}"

y = to_binary_high_risk(df[TARGET_COL])
groups = df[GROUP_COL].astype(str).values

print("Rows:", len(df))
print("Unique groups:", len(set(groups)))

# 1) TrainValid vs Test 
gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
idx_trainvalid, idx_test = next(gss1.split(df, y, groups=groups))

# 2) Train vs Valid 
df_trainvalid = df.iloc[idx_trainvalid]
y_trainvalid = y[idx_trainvalid]
g_trainvalid = groups[idx_trainvalid]

gss2 = GroupShuffleSplit(n_splits=1, test_size=VALID_SIZE, random_state=RANDOM_STATE)
rel_train, rel_valid = next(gss2.split(df_trainvalid, y_trainvalid, groups=g_trainvalid))

idx_train = idx_trainvalid[rel_train]
idx_valid = idx_trainvalid[rel_valid]

# Verify overlap = 0
assert_no_group_overlap(groups[idx_train], groups[idx_valid], groups[idx_test])

# Report sizes
def summarize(split_name, idx):
    g = groups[idx]
    print(f"\n[{split_name}] rows={len(idx)} groups={len(set(g))} pos_rate={y[idx].mean():.3f}")
    print(f"[{split_name}] groups_hash16={groups_hash16(g)}")

summarize("TRAIN", idx_train)
summarize("VALID", idx_valid)
summarize("TEST", idx_test)

# Save indices for next steps 
out = {
    "random_state": RANDOM_STATE,
    "group_col": GROUP_COL,
    "target_col": TARGET_COL,
    "test_size": TEST_SIZE,
    "valid_size": VALID_SIZE,
    "idx_train": idx_train.tolist(),
    "idx_valid": idx_valid.tolist(),
    "idx_test": idx_test.tolist(),
}
os.makedirs("exp005_groupaware", exist_ok=True)
with open("exp005_groupaware/split_indices.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n[SAVED] exp005_groupaware/split_indices.json")