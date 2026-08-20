"""
src/train_model.py
--------------------
Trains and evaluates Logistic Regression, Random Forest, and XGBoost
on a given feature set, and returns the fitted models plus metrics.

Public API
----------
train_and_evaluate(X_train, X_test, y_train, y_test, label) -> (results_df, fitted_models_dict)
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.utils import RANDOM_STATE

# ── Fixed hyperparameters (chosen via manual grid evaluation — see docs/) ────

XGBOOST_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    eval_metric="mlogloss",
    verbosity=0,
)

RANDOM_FOREST_PARAMS = dict(n_estimators=200, max_depth=10, n_jobs=-1)

LOGREG_PARAMS = dict(max_iter=1000)


def build_models() -> Dict[str, Any]:
    """Instantiate fresh, untrained model objects with the project's fixed hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, **LOGREG_PARAMS),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, **RANDOM_FOREST_PARAMS),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, **XGBOOST_PARAMS),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str = "Model",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Logistic Regression, Random Forest, and XGBoost on the given
    feature set and evaluate each on the held-out test set.

    Returns
    -------
    (results_df, fitted_models)
        results_df has columns [Model, Accuracy, Macro F1].
        fitted_models maps model name -> the fitted estimator, so the
        caller can pick the best one (by convention: XGBoost, per the
        project's findings) and persist it.
    """
    models = build_models()
    results = []
    fitted: Dict[str, Any] = {}

    if verbose:
        print(f"{'=' * 55}\n  {label}\n{'=' * 55}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]

        results.append({"Model": name, "Accuracy": round(acc, 4), "Macro F1": round(macro_f1, 4)})
        fitted[name] = model

        if verbose:
            print(f"  {name:22s} | Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

    return pd.DataFrame(results), fitted


def evaluate_single(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Detailed evaluation of one fitted model: accuracy, macro F1,
    per-class classification report (as a dict), and confusion matrix.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
