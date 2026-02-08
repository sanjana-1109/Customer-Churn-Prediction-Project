import streamlit as st
import pandas as pd
import requests
import numpy as np

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction System")

# Upload datasets 

st.subheader("Upload Datasets")

train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
if train_file:
    train_df = pd.read_csv(train_file)
    st.success("Training dataset uploaded")
    st.dataframe(train_df.head())

test_file = st.file_uploader("Upload Testing Dataset (CSV)", type=["csv"])
if test_file:
    test_df = pd.read_csv(test_file)
    st.success("Testing dataset uploaded")
    st.dataframe(test_df.head())

# Model performance 

st.divider()
st.subheader("Model Performance")

metrics_res = requests.get(f"{FASTAPI_URL}/metrics")

if metrics_res.status_code == 200:
    metrics = metrics_res.json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", metrics["accuracy"])
    col2.metric("Precision", metrics["precision"])
    col3.metric("Recall", metrics["recall"])
    col4.metric("F1 Score", metrics["f1_score"])

    st.subheader("Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted No", "Predicted Yes"],
        index=["Actual No", "Actual Yes"]
    )

    st.dataframe(cm_df)

else:
     st.warning("Model metrics are not available.")

# Single Customer Prediction

st.subheader("Predict Churn for a Single Customer")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure (months)", min_value=0, step=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

    predict_btn = st.form_submit_button("Predict Churn")


# Prediction result 

if predict_btn:
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "Contract": Contract,
        "MultipleLines": MultipleLines,
        "OnlineSecurity": OnlineSecurity,
        "TechSupport": TechSupport,
        "PaymentMethod": PaymentMethod,
        "PaperlessBilling": PaperlessBilling,
        "InternetService": InternetService,
        "TotalCharges": TotalCharges
    }

    response = requests.post(f"{FASTAPI_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        churn = result["churn_prediction"]
        probability = result["churn_probability"]

        st.subheader("Prediction Result")

        if churn == "Yes":
            st.error(f"❌ Customer is likely to CHURN")
        else:
            st.success(f"✅ Customer is NOT likely to churn")
    else:
        st.error("Prediction failed. Please check API server.")

