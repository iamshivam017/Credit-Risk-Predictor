"""
app.py
-------
Credit Risk Predictor — Streamlit dashboard entrypoint.

Run with:  streamlit run app.py
"""

import streamlit as st

# ── Page configuration (must run before any other st.* call) ──────────────────

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Wrap the heavier imports so a setup problem (missing model, bad install)
# surfaces as one clear message instead of a raw traceback — the single most
# common reason a dashboard looks "unavailable" with no clue why.
try:
    from src.dashboard_utils import (
        load_deployment_model,
        load_full_model,
        load_merged_dataset,
        load_metadata,
    )
    from src.pages import batch_scoring, explainability, leakage, overview, performance, predict
except Exception as exc:  # noqa: BLE001
    st.error(
        "⚠️ **Credit Risk Predictor failed to start.**\n\n"
        f"```\n{type(exc).__name__}: {exc}\n```\n\n"
        "**Most common cause:** dependencies aren't installed, or the model "
        "hasn't been trained yet. Run:\n\n"
        "```\npip install -r requirements.txt\npython train.py\n```"
    )
    st.stop()


# ── Minimalist theme polish ──────────────────────────────────────────────────

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; }
    .stMetricValue { font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px;
    }
    [data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


PAGES = {
    "📊 Overview": "overview",
    "🔮 Predict": "predict",
    "📁 Batch Scoring": "batch",
    "📈 Model Performance": "performance",
    "🔍 Explainability (SHAP)": "explain",
    "🕵️ Leakage Investigation": "leakage",
}


def main() -> None:
    with st.sidebar:
        st.markdown("### 📊 Credit Risk Predictor")
        st.caption("Bank Loan Approval Priority (P1–P4)")
        st.markdown("---")
        selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption("XGBoost, trained on real internal + CIBIL bureau data")
        st.caption("Leakage-audited — see 🕵️ Leakage Investigation")

    metadata = load_metadata()
    model_deploy = load_deployment_model()

    if metadata is None or model_deploy is None:
        st.title("📊 Credit Risk Predictor")
        st.error(
            "⚠️ **No trained model found.**\n\n"
            "Run this once, then refresh:\n\n"
            "```\npython train.py\n```"
        )
        st.stop()

    deployment_features = metadata["feature_sets"]["deployment_features"]

    page = PAGES[selection]
    st.markdown(f"## {selection}")
    st.markdown("")

    if page == "overview":
        overview.render(metadata)
    elif page == "predict":
        predict.render(model_deploy, deployment_features)
    elif page == "batch":
        batch_scoring.render(model_deploy, deployment_features)
    elif page == "performance":
        performance.render(metadata)
    elif page == "explain":
        explainability.render(metadata)
    elif page == "leakage":
        df_merged = load_merged_dataset()
        leakage.render(metadata, df_merged)


if __name__ == "__main__":
    main()
