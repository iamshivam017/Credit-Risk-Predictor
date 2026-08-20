"""src/pages/explainability.py — SHAP-based interpretability for Model B."""

import streamlit as st

from src.utils import PLOTS_DIR


def render(metadata: dict) -> None:
    if metadata is None:
        st.warning("No trained model found yet. Run `python train.py` first.")
        return

    st.markdown(
        "SHAP (SHapley Additive exPlanations) attributes each prediction to the "
        "features that caused it — computed here for **Model B**, the deployable "
        "model, since explaining Model A would just re-confirm `Credit_Score` "
        "dominates (already established on the Leakage Investigation page)."
    )

    img_path = PLOTS_DIR / "05_shap_summary.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run `python train.py` to generate the SHAP summary plot.")

    st.markdown("---")
    st.markdown("#### How to read this")
    st.markdown(
        "- **Bar length** = average influence on predictions across all 500 sampled "
        "applicants and all 4 risk tiers.\n"
        "- Top features are genuinely behavioral — trade-line age, recent enquiry "
        "volume, delinquency level — **not** a disguised proxy for `Credit_Score` "
        "(which is excluded from this model entirely).\n"
        "- For the highest-risk class (P4) specifically, recent enquiry behavior "
        "and delinquency severity outrank trade-line age — recent behavior matters "
        "most when identifying the worst-risk applicants."
    )

    st.markdown("#### Why this matters")
    st.markdown(
        "This is the main evidence that Model B learned genuine risk patterns "
        "instead of a hidden shortcut — every top feature has a plausible, "
        "explainable connection to credit risk, which is exactly what a bank's "
        "regulator would want to see before approving an automated decision system."
    )
