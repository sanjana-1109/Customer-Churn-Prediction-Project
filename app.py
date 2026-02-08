from fastapi import FastAPI, HTTPException
from schemas import Customer
import numpy as np
import pandas as pd
import pickle
import os

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "final_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "metrics.pkl"), "rb") as f:
    metrics = pickle.load(f)


#Fast API
app = FastAPI(title="Customer Churn Prediction API")

@app.get("/")
def home():
    return {"status": "API running"}

@app.get("/metrics")
def get_model_metrics():
    return {
        "accuracy": round(metrics["accuracy"], 3),
        "precision": round(metrics["precision"], 3),
        "recall": round(metrics["recall"], 3),
        "f1_score": round(metrics["f1_score"], 3),
        "confusion_matrix": metrics["confusion_matrix"]
    }

#  Preprocessing 

def preprocess(data: Customer):
    # tenure group
    if data.tenure <= 12:
        tenure_group_enc = 0
    elif data.tenure <= 36:
        tenure_group_enc = 1
    else:
        tenure_group_enc = 2

    avg_monthly = data.TotalCharges / (data.tenure + 1)

    row = {
        "gender_enc": 1 if data.gender == "Female" else 0,
        "SeniorCitizen": data.SeniorCitizen,
        "tenure_group_enc": tenure_group_enc,
        "PhoneService_enc": 1 if data.PhoneService == "Yes" else 0,
        "Contract_enc": {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }.get(data.Contract, 0),
        "MultipleLines_enc": 1 if data.MultipleLines == "Yes" else 0,
        "OnlineSecurity_enc": 1 if data.OnlineSecurity == "Yes" else 0,
        "TechSupport_enc": 1 if data.TechSupport == "Yes" else 0,
        "PaymentMethod_enc": {
            "Electronic check": 0,
            "Mailed check": 0,
            "Bank transfer (automatic)": 1,
            "Credit card (automatic)": 1
        }.get(data.PaymentMethod, 0),
        "PaperlessBilling_enc": 1 if data.PaperlessBilling == "Yes" else 0,
        "InternetService_enc": {
            "No": 0,
            "DSL": 1,
            "Fiber optic": 2
        }.get(data.InternetService, 0),
        "AvgMonthlySpend": avg_monthly,
        "TotalCharges": data.TotalCharges
    }

    df = pd.DataFrame([row])

    return df


#  Prediction route 

@app.post("/predict")
def predict_churn(customer: Customer):
    try:
        df = preprocess(customer)

        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)[0]   # "Yes" or "No"
        probability = model.predict_proba(df_scaled)[0][1]

        return {
            "churn_prediction": prediction,
            "churn_probability": round(float(probability), 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))