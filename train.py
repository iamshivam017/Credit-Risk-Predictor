#!/usr/bin/env python3
"""
train.py
---------
End-to-end training script for the Credit Risk Predictor.

Reproduces, from raw Excel files, the exact pipeline documented in the
project report: clean -> merge -> engineer features -> train both the
Full (leakage) and Deployment (honest) feature sets -> save the best
model of each, plus a metadata.json capturing all metrics for the
dashboard to display without needing to retrain.

Usage
-----
    python train.py                  # full run (a few minutes; downloads nothing)
    python train.py --skip-plots     # skip regenerating plots/ (faster)
    python train.py --no-cache       # force re-reading raw Excel files instead
                                      # of data/processed/merged_clean.pkl
"""

import argparse
import json
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed — this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from src.data_cleaning import load_and_clean_merged
from src.explain import compute_shap_values, global_importance
from src.feature_engineering import engineer_features, get_feature_sets
from src.train_model import evaluate_single, train_and_evaluate
from src.utils import (
    METADATA_PATH,
    MODEL_DEPLOYMENT_PATH,
    MODEL_FULL_PATH,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_CLASSES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Credit Risk Predictor models.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip regenerating plots/")
    parser.add_argument("--no-cache", action="store_true", help="Force re-reading raw Excel files")
    args = parser.parse_args()

    print("=" * 60)
    print("  Credit Risk Predictor — Training Pipeline")
    print("=" * 60)

    # ── 1. Load, clean, merge ────────────────────────────────────────────────
    print("\n[1/6] Loading and cleaning data ...")
    df_merged = load_and_clean_merged(use_cache=not args.no_cache)
    print(f"      Merged dataset: {df_merged.shape[0]:,} rows x {df_merged.shape[1]} columns")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/6] Engineering features ...")
    df_encoded, target_encoder = engineer_features(df_merged)
    deploy_features, full_features = get_feature_sets(df_encoded)
    print(f"      Deployment feature set: {len(deploy_features)} features")
    print(f"      Full feature set:       {len(full_features)} features (includes Credit_Score)")

    y = df_encoded["target"]

    X_deploy = df_encoded[deploy_features]
    X_full = df_encoded[full_features]

    X_deploy_train, X_deploy_test, y_train, y_test = train_test_split(
        X_deploy, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_full_train, X_full_test, _, _ = train_test_split(
        X_full, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── 3. Train both feature sets ───────────────────────────────────────────
    print("\n[3/6] Training models ...")
    results_full, models_full = train_and_evaluate(
        X_full_train, X_full_test, y_train, y_test, "Model A — Full (with Credit_Score)"
    )
    results_deploy, models_deploy = train_and_evaluate(
        X_deploy_train, X_deploy_test, y_train, y_test, "Model B — Deployment (no Credit_Score)"
    )

    best_model_full = models_full["XGBoost"]
    best_model_deploy = models_deploy["XGBoost"]

    # ── 4. Save models ───────────────────────────────────────────────────────
    print("\n[4/6] Saving models ...")
    joblib.dump(best_model_full, MODEL_FULL_PATH)
    joblib.dump(best_model_deploy, MODEL_DEPLOYMENT_PATH)
    print(f"      Saved -> {MODEL_FULL_PATH}")
    print(f"      Saved -> {MODEL_DEPLOYMENT_PATH}")

    # ── 5. Detailed evaluation + metadata ────────────────────────────────────
    print("\n[5/6] Computing detailed evaluation metrics ...")
    detail_deploy = evaluate_single(best_model_deploy, X_deploy_test, y_test)
    detail_full = evaluate_single(best_model_full, X_full_test, y_test)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "target_classes": list(target_encoder.classes_),
        "dataset": {
            "merged_rows": int(df_merged.shape[0]),
            "merged_columns": int(df_merged.shape[1]),
            "class_distribution": df_merged["Approved_Flag"].value_counts().to_dict(),
        },
        "feature_sets": {
            "deployment_features": deploy_features,
            "full_features": full_features,
        },
        "model_comparison": {
            "model_a_full": results_full.to_dict(orient="records"),
            "model_b_deployment": results_deploy.to_dict(orient="records"),
        },
        "best_model": "XGBoost",
        "model_a_detail": detail_full,
        "model_b_detail": detail_deploy,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"      Saved -> {METADATA_PATH}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    if not args.skip_plots:
        print("\n[6/6] Generating plots ...")
        _generate_plots(
            df_merged, results_full, results_deploy,
            best_model_full, best_model_deploy,
            X_full_test, X_deploy_test, y_test, deploy_features, target_encoder,
        )
        print(f"      Saved -> {PLOTS_DIR}/*.png")
    else:
        print("\n[6/6] Skipped plot generation (--skip-plots)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Model A (Full)       XGBoost: {detail_full['accuracy']:.4f} acc / {detail_full['macro_f1']:.4f} macro-F1")
    print(f"  Model B (Deployment) XGBoost: {detail_deploy['accuracy']:.4f} acc / {detail_deploy['macro_f1']:.4f} macro-F1")
    print("=" * 60)


def _generate_plots(
    df_merged, results_full, results_deploy,
    best_model_full, best_model_deploy,
    X_full_test, X_deploy_test, y_test, deploy_features, target_encoder,
) -> None:
    plt.style.use("seaborn-v0_8")

    # 1. Target distribution
    counts = df_merged["Approved_Flag"].value_counts().reindex(TARGET_CLASSES)
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=2)
    plt.title("Approved_Flag Distribution", fontsize=13, fontweight="bold")
    plt.ylabel("Number of applicants")
    for bar, n in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300, f"{n:,}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close()

    # 2. Credit_Score leakage boxplot
    plt.figure(figsize=(8, 5))
    df_merged.boxplot(column="Credit_Score", by="Approved_Flag", patch_artist=True,
                       boxprops=dict(facecolor="#3498db", alpha=0.6))
    plt.title("Credit_Score by Approved_Flag", fontweight="bold")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_credit_score_leakage.png", bbox_inches="tight")
    plt.close()

    # 3. Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, results, title in zip(
        axes, [results_full, results_deploy],
        ["Model A — Full (with Credit_Score)", "Model B — Deployment (no Credit_Score)"],
    ):
        x = np.arange(len(results))
        width = 0.35
        ax.bar(x - width / 2, results["Accuracy"], width, label="Accuracy", color="#3498db")
        ax.bar(x + width / 2, results["Macro F1"], width, label="Macro F1", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(results["Model"], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_model_comparison.png", bbox_inches="tight")
    plt.close()

    # 4. Confusion matrices — both models, both real (no placeholder)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y_pred_full = best_model_full.predict(X_full_test)
    y_pred_deploy = best_model_deploy.predict(X_deploy_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_full, display_labels=target_encoder.classes_, cmap="Blues", ax=axes[0], colorbar=False
    )
    axes[0].set_title("Model A — Full (with Credit_Score)", fontweight="bold")
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_deploy, display_labels=target_encoder.classes_, cmap="Blues", ax=axes[1], colorbar=False
    )
    axes[1].set_title("Model B — Deployment (no Credit_Score)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # 5. SHAP summary
    sample_size = min(500, len(X_deploy_test))
    X_shap = X_deploy_test.iloc[:sample_size]
    _explainer, shap_values = compute_shap_values(best_model_deploy, X_shap)
    importance = global_importance(shap_values, deploy_features).head(15)

    plt.figure(figsize=(9, 7))
    plt.barh(importance.index[::-1], importance.values[::-1], color="#3498db", edgecolor="white")
    plt.title("SHAP Feature Importance — Model B (averaged across all classes)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_shap_summary.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
