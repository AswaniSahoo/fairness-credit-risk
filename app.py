"""Streamlit demo for credit risk scoring.

Uses ``src.serving.predictor.Predictor`` for inference, the same path as the REST API.
No feature construction, model invocation or threshold logic lives here.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.serving.predictor import Predictor

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")


@st.cache_resource
def _load_predictor() -> Predictor | None:
    try:
        return Predictor()
    except FileNotFoundError as exc:
        logger.warning("predictor unavailable: %s", exc)
        return None


predictor = _load_predictor()

st.title("Credit Risk Scoring")
st.caption("Single-threshold model, no protected attributes in the decision.")

if predictor is None:
    st.error(
        "Model artifact not found. Run the pipeline to produce "
        "artifacts/tracks/german_credit_T0.joblib."
    )
    st.stop()

st.sidebar.header("Model")
st.sidebar.text(f"Track: {predictor.run['track']}")
st.sidebar.text(f"Type: {predictor.run['model']['model_type']}")
st.sidebar.text(f"Threshold: {predictor.threshold:.4f}")
st.sidebar.text(f"Dataset: {predictor.run['dataset']}")

st.subheader("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 19, 75, 35)
    duration = st.slider("Loan duration (months)", 4, 72, 24)
    amount = st.number_input("Credit amount", 250, 20000, 5951, step=250)
    employment_duration = st.selectbox(
        "Employment duration",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: [
            "Unemployed", "<1 year", "1-4 years", "4-7 years", ">7 years"
        ][x],
        index=2,
    )
    installment_rate = st.selectbox("Installment rate (% income)", [1, 2, 3, 4], index=1)
    present_residence = st.selectbox("Years at residence", [1, 2, 3, 4], index=2)

with col2:
    status = st.selectbox(
        "Checking account",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["<0 DM", "0-200 DM", ">200 DM", "No account"][x],
        index=1,
    )
    credit_history = st.selectbox(
        "Credit history",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: [
            "No credits taken", "All paid back", "Existing paid",
            "Delay in past", "Critical account"
        ][x],
        index=2,
    )
    purpose = st.selectbox(
        "Purpose",
        options=list(range(10)),
        format_func=lambda x: [
            "Car (new)", "Car (used)", "Furniture", "Radio/TV",
            "Appliances", "Repairs", "Education", "Vacation",
            "Retraining", "Business"
        ][x],
        index=3,
    )
    savings = st.selectbox(
        "Savings",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: [
            "<100 DM", "100-500 DM", "500-1000 DM", ">1000 DM", "Unknown"
        ][x],
    )
    number_credits = st.selectbox("Existing credits at this bank", [1, 2, 3, 4])

with col3:
    other_debtors = st.selectbox(
        "Other debtors",
        options=[0, 1, 2],
        format_func=lambda x: ["None", "Co-applicant", "Guarantor"][x],
    )
    property_val = st.selectbox(
        "Property",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Real estate", "Savings agreement", "Car/other", "Unknown/none"
        ][x],
    )
    other_installment_plans = st.selectbox(
        "Other installment plans",
        options=[0, 1, 2],
        format_func=lambda x: ["Bank", "Stores", "None"][x],
        index=2,
    )
    housing = st.selectbox(
        "Housing",
        options=[0, 1, 2],
        format_func=lambda x: ["Rent", "Own", "For free"][x],
        index=1,
    )
    job = st.selectbox(
        "Job",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Unskilled non-resident", "Unskilled resident", "Skilled", "Management"
        ][x],
        index=2,
    )
    people_liable = st.selectbox("People liable", [1, 2])
    telephone = st.selectbox("Telephone registered", [0, 1], format_func=lambda x: ["No", "Yes"][x])

if st.button("Score applicant"):
    applicant = {
        "duration": duration,
        "amount": amount,
        "age": age,
        "number_credits": number_credits,
        "people_liable": people_liable,
        "employment_duration": employment_duration,
        "installment_rate": installment_rate,
        "present_residence": present_residence,
        "job": job,
        "status": status,
        "credit_history": credit_history,
        "purpose": purpose,
        "savings": savings,
        "other_debtors": other_debtors,
        "property": property_val,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "telephone": telephone,
    }

    result = predictor.predict(applicant)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", result["decision"].upper())
    c2.metric("P(default)", f"{result['probability_of_default']:.3f}")
    c3.metric("Threshold", f"{result['threshold']:.4f}")

    if result["decision"] == "approve":
        st.success("Loan likely to be approved under the model's threshold.")
    else:
        st.warning("Loan likely to be declined under the model's threshold.")
