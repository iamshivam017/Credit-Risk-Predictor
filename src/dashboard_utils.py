"""
src/dashboard_utils.py
------------------------
Streamlit-cached loaders and shared helper functions used by app.py
and every page in src/pages/. Keeping these in one place means the
model/data are only loaded once per session, not on every page switch.
"""

import json
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import align_to_deployment_schema
from src.utils import METADATA_PATH, MODEL_DEPLOYMENT_PATH, MODEL_FULL_PATH


@st.cache_resource
def load_deployment_model() -> Optional[Any]:
    """The deployment-safe model (no Credit_Score) — used for all live predictions."""
    if not MODEL_DEPLOYMENT_PATH.exists():
        return None
    return joblib.load(MODEL_DEPLOYMENT_PATH)


@st.cache_resource
def load_full_model() -> Optional[Any]:
    """The full-feature model (with Credit_Score) — kept only for the Leakage Investigation page."""
    if not MODEL_FULL_PATH.exists():
        return None
    return joblib.load(MODEL_FULL_PATH)


@st.cache_data
def load_metadata() -> Optional[Dict]:
    """Training metadata: metrics, feature sets, dataset stats — see train.py."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_merged_dataset() -> pd.DataFrame:
    """The cleaned, merged dataset (cached — see src.data_cleaning)."""
    from src.data_cleaning import load_and_clean_merged
    return load_and_clean_merged()


def predict_single(model: Any, applicant: Dict, deployment_features: list) -> Dict:
    """
    Score one applicant (as a dict of raw field values, matching
    unseen_dataset.xlsx's schema) and return the predicted class,
    confidence, and full class-probability breakdown.
    """
    df_row = pd.DataFrame([applicant])
    df_aligned = align_to_deployment_schema(df_row, deployment_features)

    pred_idx = model.predict(df_aligned)[0]
    proba = model.predict_proba(df_aligned)[0]
    classes = model.classes_

    # Map back to P1..P4 labels using the metadata's target_classes order,
    # which is the LabelEncoder's fitted class order from training.
    metadata = load_metadata()
    class_names = metadata["target_classes"] if metadata else [f"class_{c}" for c in classes]

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(proba[pred_idx]),
        "class_probabilities": {class_names[i]: float(p) for i, p in enumerate(proba)},
        "aligned_features": df_aligned,
    }


def predict_batch(model: Any, df_raw: pd.DataFrame, deployment_features: list) -> pd.DataFrame:
    """
    Score a batch of applicants (e.g. an uploaded CSV/Excel matching
    unseen_dataset.xlsx's schema) and append prediction columns.
    """
    df_aligned = align_to_deployment_schema(df_raw, deployment_features)

    preds = model.predict(df_aligned)
    probas = model.predict_proba(df_aligned)

    metadata = load_metadata()
    class_names = metadata["target_classes"] if metadata else [f"class_{c}" for c in model.classes_]

    result = df_raw.copy()
    result["predicted_class"] = [class_names[p] for p in preds]
    result["confidence"] = probas.max(axis=1).round(4)
    return result
