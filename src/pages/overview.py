"""src/pages/overview.py — Project overview and headline metrics."""

import streamlit as st


def render(metadata: dict) -> None:
    if metadata is None:
        st.warning("No trained model found yet. Run `python train.py` first.")
        return

    st.markdown(
        "Predicts a bank's **loan approval priority tier (P1–P4)** from internal "
        "trade-line records and external CIBIL bureau data — and along the way, "
        "catches a data-leakage feature the project's own benchmark set was built to expose."
    )

    detail_b = metadata["model_b_detail"]
    detail_a = metadata["model_a_detail"]
    ds = metadata["dataset"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Deployable accuracy", f"{detail_b['accuracy']:.1%}", help="Model B, XGBoost — no Credit_Score")
    col2.metric("Deployable macro-F1", f"{detail_b['macro_f1']:.3f}")
    col3.metric("Training records", f"{ds['merged_rows']:,}")
    col4.metric("Features used", len(metadata["feature_sets"]["deployment_features"]))

    st.markdown("---")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### What P1–P4 mean")
        st.markdown(
            "| Tier | Meaning |\n|---|---|\n"
            "| **P1** | Lowest risk — auto-approve |\n"
            "| **P2** | Low–moderate risk |\n"
            "| **P3** | Moderate–high risk |\n"
            "| **P4** | Highest risk — likely reject |"
        )
        st.markdown("#### Two models, on purpose")
        st.markdown(
            f"- **Model A (Full)** — includes `Credit_Score` → **{detail_a['accuracy']:.1%}** accuracy, "
            "but can't score real applicants (the field isn't available at scoring time).\n"
            f"- **Model B (Deployment)** — excludes it → **{detail_b['accuracy']:.1%}** accuracy, "
            "genuinely deployable, and what every prediction on this dashboard uses.\n\n"
            "See the **Leakage Investigation** page for why this gap exists and matters."
        )

    with right:
        st.markdown("#### Class distribution")
        dist = ds["class_distribution"]
        for cls in ["P1", "P2", "P3", "P4"]:
            n = dist.get(cls, 0)
            pct = n / ds["merged_rows"]
            st.progress(pct, text=f"{cls} — {n:,} ({pct:.1%})")

    st.markdown("---")
    st.caption(
        "Use the sidebar to score a single applicant, upload a batch file, "
        "inspect model performance, explore SHAP explanations, or read the "
        "leakage-detection story behind this project."
    )
