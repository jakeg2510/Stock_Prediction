"""
Loan Default Prediction — Streamlit Web App
Calls a SageMaker endpoint for predictions and displays SHAP explanations.
"""
import json
import os

import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")


# ==================== AWS / SageMaker setup ====================
@st.cache_resource
def get_sagemaker_runtime():
    """Build a boto3 SageMaker runtime client from Streamlit secrets."""
    return boto3.client(
        "sagemaker-runtime",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=st.secrets.get("AWS_SESSION_TOKEN"),  # optional for permanent keys
        region_name=st.secrets["AWS_DEFAULT_REGION"],
    )


@st.cache_resource
def load_shap_artifacts():
    """SHAP explainer and feature names load locally from the repo."""
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    explainer = joblib.load(os.path.join(base_dir, "shap_explainer.pkl"))
    selected_feature_names = joblib.load(os.path.join(base_dir, "selected_feature_names.pkl"))
    feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
    pipeline = joblib.load(os.path.join(base_dir, "final_pipeline.pkl"))
    return explainer, selected_feature_names, feature_names, pipeline


runtime = get_sagemaker_runtime()
ENDPOINT_NAME = st.secrets["SAGEMAKER_ENDPOINT_NAME"]
explainer, selected_feature_names, feature_names, local_pipeline = load_shap_artifacts()


# ==================== UI ====================
st.title("Loan Default Predictor")
st.markdown(
    "Enter applicant details below to predict the probability of loan default. "
    "Predictions are served from a live AWS SageMaker endpoint, "
    "and SHAP explanations show which factors drove each decision."
)

st.header("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", 500, 40000, 10000, 500)
    term = st.selectbox("Loan Term (months)", [36, 60])
    int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
    emp_length = st.slider("Employment Length (years)", 0, 10, 5)

with col2:
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, 5000)
    purpose = st.selectbox(
        "Loan Purpose",
        [
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "car", "medical",
            "moving", "vacation", "other",
        ],
    )
    dti = st.number_input("Debt-to-Income Ratio (%)", 0.0, 60.0, 18.0, 0.5)

with col3:
    fico_range_low = st.number_input("FICO Score (low)", 600, 850, 700, 5)
    revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 50.0, 1.0)
    open_acc = st.number_input("Open Credit Accounts", 0, 50, 10, 1)
    delinq_2yrs = st.number_input("Delinquencies (last 2 yrs)", 0, 20, 0, 1)


def build_input_row():
    """Reproduce the exact feature engineering used during training."""
    row = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose,
        "dti": dti,
        "fico_range_low": fico_range_low,
        "revol_util": revol_util,
        "open_acc": open_acc,
        "delinq_2yrs": delinq_2yrs,
        "loan_income_ratio": loan_amnt / (annual_inc + 1),
        "credit_utilization": revol_util / 100,
        "income_per_account": annual_inc / (open_acc + 1),
        "log_income": np.log1p(annual_inc),
        "log_loan_amnt": np.log1p(loan_amnt),
        "high_interest": int(int_rate > 15),
        "large_loan": int(loan_amnt > 20000),
        "rate_dti_interaction": int_rate * dti,
        "has_delinq": int(delinq_2yrs > 0),
    }

    if fico_range_low < 660:
        row["fico_bucket"] = "subprime"
    elif fico_range_low < 700:
        row["fico_bucket"] = "near_prime"
    elif fico_range_low < 740:
        row["fico_bucket"] = "prime"
    else:
        row["fico_bucket"] = "super_prime"

    row["borrower_cluster"] = 0  # default cluster for live inference

    df = pd.DataFrame([row])
    # Ensure all training columns present, in correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


# ==================== Prediction ====================
if st.button("Predict Default Risk", type="primary"):
    input_df = build_input_row()

    # Call the SageMaker endpoint
    payload = input_df.to_dict(orient="records")
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        prob = result["probability"][0]
        pred = result["prediction"][0]
    except Exception as e:
        st.error(f"Endpoint call failed: {e}")
        st.stop()

    st.header("Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Default Probability", f"{prob:.1%}")
    with c2:
        st.metric("Decision", "DECLINE" if pred == 1 else "APPROVE")
    with c3:
        if prob < 0.20:
            risk = "Low Risk"
        elif prob < 0.40:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"
        st.metric("Risk Tier", risk)

    # ==================== SHAP explanation (computed locally) ====================
    st.header("Why This Prediction?")
    st.caption(
        "SHAP values show how each feature pushed the prediction toward default (red) "
        "or away from default (blue). Computed locally from the saved explainer."
    )

    preproc = local_pipeline.named_steps["preprocess"]
    selector = local_pipeline.named_steps["select"]
    transformed = preproc.transform(input_df)
    selected = selector.transform(transformed)

    shap_values = explainer.shap_values(selected)

    contributions = pd.DataFrame(
        {"Feature": selected_feature_names, "SHAP Value": shap_values[0]}
    ).sort_values("SHAP Value", key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contributions["SHAP Value"]]
    ax.barh(contributions["Feature"][::-1], contributions["SHAP Value"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SHAP Value (impact on default probability)")
    ax.set_title("Top 10 Feature Contributions")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Force Plot")
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        selected[0, :],
        feature_names=selected_feature_names,
        matplotlib=True,
        show=False,
    )
    st.pyplot(plt.gcf())
    plt.clf()


# ==================== Sidebar ====================
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app predicts loan default risk using an XGBoost pipeline trained on the "
        "Lending Club dataset (2007–2018). Predictions are served by a live AWS SageMaker "
        "endpoint, and SHAP explanations support ECOA adverse-action notice requirements."
    )
    st.markdown("**Model:** XGBoost with SMOTE resampling")
    st.markdown("**Endpoint:** AWS SageMaker")
    st.markdown("**Primary metric:** ROC-AUC")
