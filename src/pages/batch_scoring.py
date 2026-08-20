"""src/pages/batch_scoring.py — Upload a CSV/Excel file and score every row."""

import pandas as pd
import streamlit as st

from src.dashboard_utils import predict_batch
from src.utils import UNSEEN_BASE_COLUMNS, UNSEEN_PATH


def render(model, deployment_features: list) -> None:
    if model is None:
        st.warning("No trained model found yet. Run `python train.py` first.")
        return

    st.caption(
        "Upload a CSV or Excel file matching the bureau-data schema "
        f"({len(UNSEEN_BASE_COLUMNS)} columns — same as `unseen_dataset.xlsx`). "
        "Missing columns are auto-filled with 0."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"], label_visibility="collapsed")
    with col2:
        use_sample = st.button("Use built-in sample (100 applicants)", use_container_width=True)

    df_input = None
    source_name = None

    if uploaded is not None:
        df_input = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        source_name = uploaded.name
    elif use_sample:
        df_input = pd.read_excel(UNSEEN_PATH)
        source_name = "unseen_dataset.xlsx (built-in sample)"

    if df_input is None:
        st.info("Upload a file or click the sample button above to get started.")
        return

    missing = [c for c in UNSEEN_BASE_COLUMNS if c not in df_input.columns]
    if missing:
        st.warning(
            f"⚠️ {len(missing)} expected column(s) not found in your file — they'll be filled with 0: "
            + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        )

    with st.spinner("Scoring..."):
        result = predict_batch(model, df_input, deployment_features)

    st.success(f"Scored {len(result):,} applicants from **{source_name}**")

    dist = result["predicted_class"].value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0).astype(int)
    cols = st.columns(4)
    for i, cls in enumerate(["P1", "P2", "P3", "P4"]):
        cols[i].metric(cls, int(dist[cls]))

    st.dataframe(
        result[["predicted_class", "confidence"] + [c for c in result.columns if c not in ("predicted_class", "confidence")]],
        use_container_width=True,
        height=380,
    )

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download scored results (CSV)",
        data=csv_bytes,
        file_name="scored_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
