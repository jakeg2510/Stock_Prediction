"""
Loan Default Prediction — Streamlit Web App

Run locally:
    streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load('final_pipeline.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    feature_names = joblib.load('feature_names.pkl')
    selected_feature_names = joblib.load('selected_feature_names.pkl')
    return pipeline, explainer, feature_names, selected_feature_names


pipeline, explainer, feature_names, selected_feature_names = load_artifacts()

st.title("Loan Default Predictor")
st.markdown(
    "Enter applicant details below to predict the probability of loan default. "
    "The SHAP plot explains which factors drove the prediction."
)

# ==================== Input form ====================
st.header("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    term = st.selectbox("Loan Term (months)", [36, 60])
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)
    emp_length = st.slider("Employment Length (years)", 0, 10, 5)

with col2:
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=60000, step=5000)
    purpose = st.selectbox("Loan Purpose", [
        'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase',
        'small_business', 'car', 'medical', 'moving', 'vacation', 'other'
    ])
    dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=60.0, value=18.0, step=0.5)

with col3:
    fico_range_low = st.number_input("FICO Score (low)", min_value=600, max_value=850, value=700, step=5)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
    open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=50, value=10, step=1)
    delinq_2yrs = st.number_input("Delinquencies (last 2 yrs)", min_value=0, max_value=20, value=0, step=1)


def build_input_row():
    """Reproduce the same feature engineering applied during training."""
    row = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'purpose': purpose,
        'dti': dti,
        'fico_range_low': fico_range_low,
        'revol_util': revol_util,
        'open_acc': open_acc,
        'delinq_2yrs': delinq_2yrs,
        'loan_income_ratio': loan_amnt / (annual_inc + 1),
        'credit_utilization': revol_util / 100,
        'income_per_account': annual_inc / (open_acc + 1),
        'log_income': np.log1p(annual_inc),
        'log_loan_amnt': np.log1p(loan_amnt),
        'high_interest': int(int_rate > 15),
        'large_loan': int(loan_amnt > 20000),
        'rate_dti_interaction': int_rate * dti,
        'has_delinq': int(delinq_2yrs > 0),
    }

    # FICO bucket — same bins as training
    if fico_range_low < 660:
        row['fico_bucket'] = 'subprime'
    elif fico_range_low < 700:
        row['fico_bucket'] = 'near_prime'
    elif fico_range_low < 740:
        row['fico_bucket'] = 'prime'
    else:
        row['fico_bucket'] = 'super_prime'

    # Borrower cluster — set a default; real app would re-run KMeans
    row['borrower_cluster'] = 0

    # Match exact training column order (drop any that aren't expected)
    return pd.DataFrame([row])[[c for c in feature_names if c in row]]


# ==================== Prediction ====================
if st.button("Predict Default Risk", type="primary"):
    input_df = build_input_row()

    # Fill any missing expected columns with sensible defaults
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    prob = pipeline.predict_proba(input_df)[0, 1]
    pred = pipeline.predict(input_df)[0]

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

    # ==================== SHAP explanation ====================
    st.header("Why This Prediction?")
    st.caption("SHAP values show how each feature pushed the prediction toward default (red) or away from default (blue).")

    # Run input through preprocessor + selector
    preproc = pipeline.named_steps['preprocess']
    selector = pipeline.named_steps['select']

    transformed = preproc.transform(input_df)
    selected = selector.transform(transformed)

    shap_values = explainer.shap_values(selected)

    # Bar plot of feature contributions for this single prediction
    contributions = pd.DataFrame({
        'Feature': selected_feature_names,
        'SHAP Value': shap_values[0]
    }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in contributions['SHAP Value']]
    ax.barh(contributions['Feature'][::-1], contributions['SHAP Value'][::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("SHAP Value (impact on default probability)")
    ax.set_title("Top 10 Feature Contributions")
    plt.tight_layout()
    st.pyplot(fig)

    # SHAP force plot rendered as static matplotlib
    st.subheader("Force Plot")
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        selected[0, :],
        feature_names=selected_feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()


# ==================== Sidebar info ====================
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app predicts loan default risk using an XGBoost pipeline trained on the "
        "Lending Club dataset (2007–2018). SHAP explanations make every prediction auditable, "
        "supporting ECOA adverse-action notice requirements."
    )
    st.markdown("**Model:** XGBoost with SMOTE resampling")
    st.markdown("**Training data:** ~80,000 historical loans")
    st.markdown("**Primary metric:** ROC-AUC")
