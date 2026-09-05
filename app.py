import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load saved model, scaler and feature-column information
# ---------------------------------------------------------
model = joblib.load("Churn_Prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

st.divider()

# ---------------------------------------------------------
# Customer information
# ---------------------------------------------------------
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

# ---------------------------------------------------------
# Services
# ---------------------------------------------------------
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No phone service", "No", "Yes"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No internet service", "No", "Yes"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No internet service", "No", "Yes"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No internet service", "No", "Yes"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No internet service", "No", "Yes"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No internet service", "No", "Yes"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No internet service", "No", "Yes"]
)

# ---------------------------------------------------------
# Account information
# ---------------------------------------------------------
contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=1.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0,
    step=10.0
)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔮 Predict Churn", use_container_width=True):

    # Create raw input DataFrame using the same feature names
    # used during model training.
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Apply the EXACT same one-hot encoding used during training.
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod"
        ],
        drop_first=True,
        dtype=int
    )

    # Add any training columns missing from the user's input.
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Keep the exact training-column order.
    input_data = input_data[columns]

    # The saved model in your code is the baseline model and expects
    # scaled input, so apply the saved StandardScaler.
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # Probability is available for Logistic Regression.
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        probability = None

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")

        if probability is not None:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )
    else:
        st.success("✅ Customer is likely to STAY.")

        if probability is not None:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )
