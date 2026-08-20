"""src/pages/performance.py — Model comparison and per-class metrics."""

import pandas as pd
import streamlit as st

from src.utils import PLOTS_DIR


def render(metadata: dict) -> None:
    if metadata is None:
        st.warning("No trained model found yet. Run `python train.py` first.")
        return

    st.markdown("#### Model A vs Model B — six model x feature-set combinations")
    st.caption("Both feature sets were run through the same three classifiers, so the "
               "leakage cost can be isolated cleanly.")

    tab_a, tab_b = st.tabs(["Model A — Full (with Credit_Score)", "Model B — Deployment (no Credit_Score)"])

    for tab, key in [(tab_a, "model_a_full"), (tab_b, "model_b_deployment")]:
        with tab:
            df = pd.DataFrame(metadata["model_comparison"][key])
            st.dataframe(df.style.highlight_max(subset=["Accuracy", "Macro F1"], color="#dcfce7"),
                         use_container_width=True, hide_index=True)

    img_path = PLOTS_DIR / "03_model_comparison.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model B — per-class detail (the deployable model)")

    detail = metadata["model_b_detail"]
    report = detail["classification_report"]

    rows = []
    for cls in metadata["target_classes"]:
        if cls in report:
            r = report[cls]
            rows.append({"Class": cls, "Precision": round(r["precision"], 3),
                         "Recall": round(r["recall"], 3), "F1": round(r["f1-score"], 3),
                         "Support": int(r["support"])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall accuracy", f"{detail['accuracy']:.1%}")
    c2.metric("Macro F1", f"{detail['macro_f1']:.3f}")
    weakest = min(rows, key=lambda r: r["F1"])
    c3.metric("Weakest class", weakest["Class"], help=f"F1 = {weakest['F1']}")

    img_path = PLOTS_DIR / "04_confusion_matrix.png"
    if img_path.exists():
        st.markdown("##### Confusion matrix")
        st.image(str(img_path), use_container_width=True)
        st.caption(
            "Errors cluster between **neighboring** tiers (P2↔P3, P3↔P4) rather than "
            "distant ones — a lower-cost mistake for a bank's triage workflow than it "
            "might first appear from accuracy alone."
        )
