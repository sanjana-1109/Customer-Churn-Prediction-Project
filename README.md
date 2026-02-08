# Customer Churn Prediction System
An end-to-end Machine Learning application that predicts whether a customer is likely to churn (leave) or not.
This project covers the complete ML lifecycle — from data preprocessing and model training to API deployment and frontend integration.

# Project Overview

Customer churn is a critical challenge for subscription-based businesses.
This system identifies customers who are at risk of leaving, enabling companies to take proactive retention measures.

Key features include:
- Model training using historical customer data
- REST API built with FastAPI
- Interactive frontend built using Streamlit
- Performance evaluation using standard ML metrics
- 
# Tech Stack
- Python
- Pandas, NumPy (Data manipulation)
- Scikit-learn (Machine learning)
- FastAPI (Backend API)
- Streamlit (Frontend interface)
- Pickle (Model serialization)
- Uvicorn (Server for FastAPI)

# Model Details
- Algorithm: Random Forest Classifier
- Feature Engineering:
Tenure grouping
Average monthly spend
Categorical encoding
- Scaling: MinMaxScaler
- Performance on Test Dataset:
- Accuracy: ~81%
- Precision: ~70%
- Recall: ~50%
- F1 Score: ~58%
The frontend also displays the confusion matrix and other evaluation metrics.

# API Endpoints
GET /
model metrix
POST /predict
Predict churn for a single customer

# Frontend Features (Streamlit)
- Upload training & testing datasets
- Single customer prediction using a form
- Clear churn prediction results (Yes / No)
- Display of model accuracy, precision, recall, and F1-score
- Confusion matrix for the test dataset

# How to Run the Project
- Start FastAPI backend
uvicorn app:app --reload
- Start Streamlit frontend
streamlit run frontend_app.py

- Access in browser:
API: http://127.0.0.1:8000
Frontend: http://localhost:8501

# Key Learnings
- Implemented an end-to-end ML workflow
- Performed feature engineering and model evaluation
- Deployed ML models using FastAPI
- Integrated backend with Streamlit frontend
- Learned to debug real-world ML deployment issues




