"""src/pages/predict.py — Score a single applicant via a manual-entry form."""

import streamlit as st

from src.dashboard_utils import predict_single
from src.utils import TARGET_CLASS_MEANING

EDUCATION_OPTIONS = ["SSC", "12TH", "UNDER GRADUATE", "GRADUATE", "POST-GRADUATE", "OTHERS"]
PRODUCT_OPTIONS = ["PL", "CC", "HL", "AL", "ConsumerLoan", "others"]

TAG_COLOR = {"P1": "🟢", "P2": "🔵", "P3": "🟠", "P4": "🔴"}


def render(model, deployment_features: list) -> None:
    if model is None:
        st.warning("No trained model found yet. Run `python train.py` first.")
        return

    st.caption("Fill in an applicant's profile. Fields are grouped to match how a bureau report reads.")

    with st.form("predict_form"):
        st.markdown("##### Demographics & income")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["M", "F"])
        marital = c2.selectbox("Marital status", ["Married", "Single"])
        education = c3.selectbox("Education", EDUCATION_OPTIONS, index=1)
        income = c4.number_input("Net monthly income (₹)", min_value=0, value=24000, step=1000)

        c1, c2 = st.columns(2)
        emp_time = c1.number_input("Time with current employer (months)", min_value=0, value=101)
        st.markdown("")

        st.markdown("##### Trade-line summary")
        c1, c2, c3, c4 = st.columns(4)
        age_oldest = c1.number_input("Age of oldest trade-line (months)", min_value=0, value=39,
                                      help="Strongest single predictor per SHAP — see Explainability page")
        age_newest = c2.number_input("Age of newest trade-line (months)", min_value=0, value=8)
        secured_tl = c3.number_input("Secured trade-lines", min_value=0, value=1)
        unsecured_tl = c4.number_input("Unsecured trade-lines", min_value=0, value=2)

        c1, c2, c3, c4 = st.columns(4)
        cc_tl = c1.number_input("Credit card trade-lines", min_value=0, value=0)
        home_tl = c2.number_input("Home loan trade-lines", min_value=0, value=0)
        pl_tl = c3.number_input("Personal loan trade-lines", min_value=0, value=0)
        other_tl = c4.number_input("Other trade-lines", min_value=0, value=0)

        st.markdown("##### Delinquency & payment history")
        c1, c2, c3, c4 = st.columns(4)
        missed_pmnt = c1.number_input("Total missed payments", min_value=0, value=0)
        recent_deliq = c2.number_input("Recent delinquency level", min_value=0, value=0,
                                        help="Second-strongest SHAP driver, especially for highest-risk (P4) predictions")
        deliq_6_12 = c3.number_input("Delinquencies (6–12 months)", min_value=0, value=0)
        times_60p_dpd = c4.number_input("Times 60+ days past due", min_value=0, value=0)

        st.markdown("##### Recent enquiry behavior")
        c1, c2, c3 = st.columns(3)
        enq_l3m = c1.number_input("Enquiries in last 3 months", min_value=0, value=0,
                                   help="Third-strongest overall SHAP driver")
        time_since_enq = c2.number_input("Days since most recent enquiry", min_value=0, value=148)
        last_product = c3.selectbox("Most recent product enquired", PRODUCT_OPTIONS, index=0)

        c1, c2 = st.columns(2)
        first_product = c1.selectbox("First product ever enquired", PRODUCT_OPTIONS, index=0)
        num_std_12mts = c2.number_input("Standard trade-lines opened (12 mo)", min_value=0, value=0)

        with st.expander("Advanced fields (optional — sensible defaults are pre-filled)"):
            c1, c2, c3 = st.columns(3)
            cc_enq_12m = c1.number_input("Credit card enquiries (12 mo)", min_value=0, value=0)
            pl_enq_12m = c2.number_input("Personal loan enquiries (12 mo)", min_value=0, value=0)
            tot_tl_closed_12m = c3.number_input("Trade-lines closed (12 mo)", min_value=0, value=0)
            time_since_pmnt = c1.number_input("Days since most recent payment", min_value=0, value=75)
            cc_flag = c2.checkbox("Has a credit card product")
            pl_flag = c3.checkbox("Has a personal loan product")
            hl_flag = c1.checkbox("Has a home loan product")
            gl_flag = c2.checkbox("Has a gold loan product")

        submitted = st.form_submit_button("Predict risk tier", type="primary", use_container_width=True)

    if not submitted:
        return

    applicant = {
        "GENDER": gender, "MARITALSTATUS": marital, "EDUCATION": education,
        "NETMONTHLYINCOME": income, "Time_With_Curr_Empr": emp_time,
        "Age_Oldest_TL": age_oldest, "Age_Newest_TL": age_newest,
        "Secured_TL": secured_tl, "Unsecured_TL": unsecured_tl,
        "CC_TL": cc_tl, "Home_TL": home_tl, "PL_TL": pl_tl, "Other_TL": other_tl,
        "Tot_Missed_Pmnt": missed_pmnt, "recent_level_of_deliq": recent_deliq,
        "max_recent_level_of_deliq": recent_deliq, "num_deliq_6_12mts": deliq_6_12,
        "num_times_60p_dpd": times_60p_dpd,
        "enq_L3m": enq_l3m, "time_since_recent_enq": time_since_enq,
        "last_prod_enq2": last_product, "first_prod_enq2": first_product,
        "num_std_12mts": num_std_12mts,
        "CC_enq_L12m": cc_enq_12m, "PL_enq_L12m": pl_enq_12m,
        "Tot_TL_closed_L12M": tot_tl_closed_12m, "time_since_recent_payment": time_since_pmnt,
        "CC_Flag": int(cc_flag), "PL_Flag": int(pl_flag), "HL_Flag": int(hl_flag), "GL_Flag": int(gl_flag),
        # Remaining fields default to 0 — align_to_deployment_schema fills any
        # not supplied here, matching how a genuinely sparse bureau record behaves.
        "pct_tl_open_L6M": 0.0, "pct_tl_closed_L6M": 0.0, "pct_tl_closed_L12M": 0.0,
        "num_sub": 0, "num_sub_6mts": 0, "num_sub_12mts": 0, "num_dbt": 0, "num_dbt_12mts": 0,
        "num_lss": 0, "pct_PL_enq_L6m_of_ever": 0.0, "pct_CC_enq_L6m_of_ever": 0.0,
    }

    result = predict_single(model, applicant, deployment_features)
    pred = result["predicted_class"]

    st.markdown("---")
    st.markdown(f"### {TAG_COLOR.get(pred, '⚪')} Predicted tier: **{pred}**")
    st.caption(TARGET_CLASS_MEANING.get(pred, ""))
    st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.1%}")

    st.markdown("##### Full class probability breakdown")
    cols = st.columns(4)
    for i, cls in enumerate(["P1", "P2", "P3", "P4"]):
        prob = result["class_probabilities"].get(cls, 0.0)
        cols[i].metric(f"{TAG_COLOR[cls]} {cls}", f"{prob:.1%}")
