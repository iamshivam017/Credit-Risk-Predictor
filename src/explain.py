"""
src/explain.py
----------------
SHAP-based explainability for the deployment-safe model.

Public API
----------
compute_shap_values(model, X_sample) -> (explainer, shap_values)
global_importance(shap_values, feature_names) -> pd.Series (sorted desc)
class_importance(shap_values, feature_names, class_index) -> pd.Series (sorted desc)
explain_single_prediction(model, X_row, feature_names, class_names) -> dict
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model: Any, X_sample: pd.DataFrame):
    """
    Compute SHAP values for a tree-based model using the exact,
    efficient TreeExplainer (vs. the slower, model-agnostic
    KernelExplainer, which isn't needed since both project models —
    Random Forest and XGBoost — are tree-based).

    For a multiclass model, `shap_values` has shape
    (n_samples, n_features, n_classes).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def global_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.Series:
    """
    Mean absolute SHAP value per feature, averaged across all classes
    and all samples — a single global "how much does this feature
    matter overall" ranking.
    """
    mean_abs = np.abs(shap_values).mean(axis=2).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def class_importance(
    shap_values: np.ndarray, feature_names: List[str], class_index: int
) -> pd.Series:
    """Mean absolute SHAP value per feature for one specific class only."""
    mean_abs = np.abs(shap_values[:, :, class_index]).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def explain_single_prediction(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: List[str],
    class_names: List[str],
    top_n: int = 8,
) -> Dict[str, Any]:
    """
    Explain one applicant's prediction: which features pushed the
    prediction toward the predicted class, and by how much.

    `X_row` must be a single-row DataFrame with columns matching the
    model's expected feature set (see feature_engineering.align_to_deployment_schema).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)  # shape (1, n_features, n_classes)

    pred_idx = int(model.predict(X_row)[0])
    pred_proba = model.predict_proba(X_row)[0]

    row_shap = shap_values[0, :, pred_idx]
    contributions = (
        pd.Series(row_shap, index=feature_names)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
    )

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(pred_proba[pred_idx]),
        "class_probabilities": dict(zip(class_names, pred_proba.tolist())),
        "top_contributing_features": [
            {"feature": feat, "shap_value": float(val), "applicant_value": float(X_row.iloc[0][feat])}
            for feat, val in contributions.items()
        ],
    }
