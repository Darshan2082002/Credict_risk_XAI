# 💳 Credict Risk XAI

An end-to-end **Credit Risk Assessment & Explainable AI (XAI)** platform designed to bridge the gap between high-performance machine learning models and regulatory compliance in modern financial lending. 

This repository implements rigorous credit risk modeling (Probability of Default), traditional scorecard logic, model-agnostic interpretability (SHAP/LIME), a FastAPI backend, and an interactive Streamlit UI.

---

## 📂 Project Directory Structure

```text
Credict_risk_XAI/
│
├── .vscode/               # Workspace configurations
├── api/                   # FastAPI backend endpoints for real-time risk scoring
├── data/                  # Raw and processed credit datasets
├── feature_engineering/   # Data cleaning, missing value imputation, & Weight of Evidence (WoE)
├── governance/            # Model documentation, audit logs, and performance tracking
├── model/                 # ML model training pipelines (XGBoost, LightGBM, Logistic Regression)
├── scorecard/             # Traditional credit scorecard generation & point-scaling logic
├── tests/                 # Unit tests (pytest suite)
├── xai/                   # Explainability modules (SHAP & LIME interpretability wrappers)
├── pytest.ini             # Pytest configuration
├── setup.py               # Package setup configuration
└── streamlit_app.py       # Interactive web application for loan officers****

Key Features
Rigorous Feature Engineering: Automated preprocessing pipelines featuring Missing Value Handling, Outlier Treatment, and Weight of Evidence (WoE) / Information Value (IV) binning.

Dual Risk Modeling: Combines robust gradient boosted machine learning models with traditional risk-weighted scorecard metrics.

Explainable AI (XAI): Built-in SHAP and LIME wrappers that generate local explanations (explaining why a specific loan applicant was approved or denied) and global feature impact summaries for strict regulatory compliance.

Production-Ready API: Asynchronous FastAPI endpoints capable of serving real-time predictions and explainability breakdowns.

Interactive Dashboard: A Streamlit-powered user interface (streamlit_app.py) allowing stakeholders to input applicant variables and instantly visualize credit scores alongside key risk drivers

Installation & Setup
Clone the repository:
git clone [https://github.com/Darshan2082002/Credict_risk_XAI.git](https://github.com/Darshan2082002/Credict_risk_XAI.git)
cd Credict_risk_XAI


Install the package in editable mode:

pip install -e .

Install core dependencies:
pip install fastapi uvicorn streamlit shap pandas numpy scikit-learn xgboost joblib pytest


Running the Application
1. Launch the Streamlit Frontend Dashboard
To run the interactive UI for risk assessment:

streamlit run streamlit_app.py

Start the FastAPI Backend
To launch the REST API server:
uvicorn api.main:app --reload


Running Tests
To verify code integrity and run unit tests using pytest:

pytest
