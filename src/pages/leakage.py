"""src/pages/leakage.py — The Credit_Score data-leakage investigation, told interactively."""

import pandas as pd
import streamlit as st

from src.utils import PLOTS_DIR


def render(metadata: dict, df_merged: pd.DataFrame) -> None:
    if metadata is None:
        st.warning("No trained model found yet. Run `python train.py` first.")
        return

    st.markdown(
        "Before trusting any accuracy number, this project checked whether any "
        "feature was **leaking** the target — encoding the answer rather than "
        "predicting it. Here's exactly what was found."
    )

    detail_a = metadata["model_a_detail"]
    detail_b = metadata["model_b_detail"]
    gap = detail_a["accuracy"] - detail_b["accuracy"]

    st.markdown("#### Step 1 — a suspiciously high baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric("With Credit_Score", f"{detail_a['accuracy']:.1%}")
    c2.metric("Without Credit_Score", f"{detail_b['accuracy']:.1%}")
    c3.metric("Gap", f"{gap:.1%}", help="Entirely attributable to one column")
    st.caption(
        "A 4-class problem with imbalanced classes rarely hits 99% from bureau "
        "behavioral data alone — that gap was the first red flag."
    )

    st.markdown("#### Step 2 — grouping Credit_Score by class")
    score_summary = (
        df_merged.groupby("Approved_Flag")["Credit_Score"]
        .agg(["min", "max", "mean"])
        .reindex(["P1", "P2", "P3", "P4"])
        .round(2)
    )
    st.dataframe(score_summary, use_container_width=True)

    img_path = PLOTS_DIR / "02_credit_score_leakage.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.caption(
        "P1, P2, and P4 barely overlap — a near-threshold rule. Only P3 spans a "
        "wide range, which is exactly why P3 turns out to be the hardest class "
        "to predict later on."
    )

    st.markdown("#### Step 3 — the confirming clue")
    st.info(
        "**`unseen_dataset.xlsx`** — the file meant to simulate real production "
        "scoring — has 42 columns and **does not include `Credit_Score` at all**. "
        "That's strong evidence the intended, real task deliberately excludes it."
    )

    st.markdown("#### The decision")
    st.markdown(
        "Rather than silently picking one number, this project trains **both** "
        "models and reports both, clearly labeled:\n\n"
        "| | Model A (Full) | Model B (Deployment) |\n"
        "|---|---|---|\n"
        f"| Accuracy | {detail_a['accuracy']:.1%} | {detail_b['accuracy']:.1%} |\n"
        f"| Macro F1 | {detail_a['macro_f1']:.3f} | {detail_b['macro_f1']:.3f} |\n"
        "| Deployable? | ❌ No — column missing from real scoring data | ✅ Yes |\n"
        "| What it actually learned | The bank's own score-threshold rule | Genuine behavioral risk patterns |\n\n"
        "**Every prediction on this dashboard uses Model B.**"
    )
