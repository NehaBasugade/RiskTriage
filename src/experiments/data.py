

# data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


REQUIRED_COLUMNS = [
    "Reference Number",
    "Grid Ref: Easting",
    "Grid Ref: Northing",
    "Number of Vehicles",
    "Accident Date",
    "Time (24hr)",
    "1st Road Class",
    "1st Road Class & No",
    "Road Surface",
    "Lighting Conditions",
    "Weather Conditions",
    "Local Authority",
    "Vehicle Number",
    "Type of Vehicle",
    "Casualty Class",
    "Casualty Severity",
    "Sex of Casualty",
    "Age of Casualty",
]


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize common string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    valid_size: float = 0.20,
    random_state: int = 42,
    group_col: str = "Reference Number",
) -> SplitData:
    """
    Safer split: if group_col exists, we split by groups to reduce "same accident in train+test".
    valid_size is fraction of remaining after test split.
    """
    if group_col in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_valid_idx, test_idx = next(gss.split(df, groups=df[group_col]))
        train_valid = df.iloc[train_valid_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
        train_idx, valid_idx = next(gss2.split(train_valid, groups=train_valid[group_col]))
        train = train_valid.iloc[train_idx].reset_index(drop=True)
        valid = train_valid.iloc[valid_idx].reset_index(drop=True)

        return SplitData(train=train, valid=valid, test=test)

  
    train_valid, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
    train, valid = train_test_split(train_valid, test_size=valid_size, random_state=random_state, stratify=None)
    return SplitData(train=train.reset_index(drop=True), valid=valid.reset_index(drop=True), test=test.reset_index(drop=True))