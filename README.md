# 📊 Customer Churn Prediction

An ML-based web application that predicts whether a telecom customer is likely to **churn or stay** based on important customer and service-related features.

🔗 **Live Demo:**  
https://customer-churn-prediction-812638.streamlit.app/

---

## 🚀 Project Overview

Customer churn is a major challenge for telecom companies. Identifying customers who are likely to leave allows businesses to take preventive retention actions.

This project uses **Machine Learning classification** to predict customer churn from the Telco Customer Churn dataset.

The model development process includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Categorical feature encoding
- Feature scaling
- Multiple classification models
- F1-score based model comparison
- Hyperparameter tuning using GridSearchCV
- Permutation feature importance
- Streamlit deployment

---

## 🎯 Objective

The objective of this project is to:

> Predict whether a customer is likely to churn based on their account and service information.

The application provides a simple interface where users enter customer details and receive:

- Churn prediction
- Churn probability
- Basic churn-risk indication

---

## 🧠 Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Baseline Model Comparison
        ↓
Top Models Selected
        ↓
GridSearchCV Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Permutation Feature Importance
        ↓
Top 10 Features
        ↓
Streamlit Web Application
