"""
src/feature_engineering.py
----------------------------
Encodes the cleaned, merged dataset into model-ready features.

Public API
----------
engineer_features(df_merged) -> (df_encoded, target_encoder)
get_feature_sets(df_encoded) -> (deployment_features: list[str], full_features: list[str])
align_to_deployment_schema(df_raw, deployment_features) -> pd.DataFrame
"""

from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import EDUCATION_ORDER, ONEHOT_COLUMNS, UNSEEN_BASE_COLUMNS


def engineer_features(df_merged: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Apply the project's fixed encoding scheme to the cleaned, merged
    dataset:

    - EDUCATION -> ordinal (has a genuine order)
    - MARITALSTATUS / GENDER / last_prod_enq2 / first_prod_enq2 -> one-hot
      (no natural order; drop_first=True avoids the dummy-variable trap)
    - Approved_Flag -> label-encoded target column "target"

    Returns
    -------
    (df_encoded, target_encoder)
        `target_encoder.classes_` maps encoded ints back to P1..P4.
    """
    df = df_merged.copy()

    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_ORDER)

    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, drop_first=True)

    target_encoder = LabelEncoder()
    df["target"] = target_encoder.fit_transform(df["Approved_Flag"])

    return df, target_encoder


def get_feature_sets(df_encoded: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Derive the two feature sets used throughout the project:

    - deployment_features: matches unseen_dataset.xlsx's schema exactly
      (no Credit_Score) — this is the genuinely deployable feature set.
    - full_features: deployment_features + Credit_Score — included only
      to quantify and demonstrate the data-leakage finding; NOT deployable
      since Credit_Score is absent from real scoring data.
    """
    dummy_columns = [
        c for c in df_encoded.columns
        if any(c.startswith(prefix + "_") for prefix in ONEHOT_COLUMNS)
    ]
    deployment_base = [c for c in UNSEEN_BASE_COLUMNS if c in df_encoded.columns]

    deployment_features = deployment_base + dummy_columns
    full_features = deployment_features + ["Credit_Score"]

    return deployment_features, full_features


def align_to_deployment_schema(
    df_raw: pd.DataFrame, deployment_features: List[str]
) -> pd.DataFrame:
    """
    Apply the identical encoding used in training to a new/unseen raw
    dataframe (e.g. unseen_dataset.xlsx, or a single applicant's form
    input as a one-row DataFrame), then reindex to exactly match the
    trained model's expected columns.

    Any expected column absent from `df_raw` (e.g. a one-hot category
    never seen in this batch) is filled with 0, and columns not in
    `deployment_features` are dropped. This is what makes batch scoring
    of new data safe against schema drift.
    """
    df = df_raw.copy()
    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_ORDER)
    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, drop_first=True)
    df = df.reindex(columns=deployment_features, fill_value=0)
    return df
