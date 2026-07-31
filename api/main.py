import os
import joblib
import yaml
import pandas as pd
import numpy as np
from fastapi import FastAPI
from api.schemas import CreditInput
from lime.lime_tabular import LimeTabularExplainer

app = FastAPI()

# Shared configuration and loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "model")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_engineering", "feature_list.yaml")
DATA_PATH = os.path.join(BASE_DIR, "data", "oot", "oot.csv")

# Global dictionary to store loaded models
models = {}
explainer = None
features = []

try:
    with open(FEATURE_PATH, "r") as f:
        feature_config = yaml.safe_load(f)
    features = feature_config["woe_features"]
    
    # Load all available models from the model directory
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_model.pkl")]
    for model_file in model_files:
        model_name = model_file.replace("_model.pkl", "")
        model_path = os.path.join(MODEL_DIR, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"Model '{model_name}' loaded successfully.")
    
    # Initialize LIME Explainer (using OOT data as a background sample)
    background_df = pd.read_csv(DATA_PATH)
    explainer = LimeTabularExplainer(
        training_data=np.array(background_df[features]),
        feature_names=features,
        class_names=["Approve", "Reject"],
        mode="classification"
    )
    print("XAI components loaded successfully.")
except Exception as e:
    print(f"Error loading models or data: {e}")

@app.get("/")
def health_check():
    return {
        "status": "API is running", 
        "models_available": list(models.keys()), 
        "xai_loaded": explainer is not None
    }

@app.post("/predict")
def predict(data: CreditInput, threshold: float = 0.5, model_name: str = "logistic"):
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found. Available: {list(models.keys())}"}
    
    selected_model = models[model_name]
    
    input_data = [[
        data.credit_utilization_ratio_woe,
        data.debt_to_income_ratio_woe,
        data.annual_income_woe,
        data.employment_years_woe,
        data.number_of_late_payments_12m_woe
    ]]

    probability = selected_model.predict_proba(input_data)[0][1]
    decision = "Approve" if probability < threshold else "Reject"

    return {
        "PD": float(probability),
        "decision": decision,
        "threshold_used": threshold,
        "model_used": model_name
    }

@app.get("/explain/global")
def explain_global(model_name: str = "logistic"):
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found"}
    
    selected_model = models[model_name]
    
    # Check if model has coefficients (Logistic)
    if not hasattr(selected_model, "coef_"):
        return {"error": "Global importance only supported for linear models currently."}
    
    coefs = selected_model.coef_[0]
    importance = []
    for feature, coef in zip(features, coefs):
        importance.append({
            "feature": feature,
            "coefficient": float(coef),
            "abs_importance": float(abs(coef))
        })
    
    importance = sorted(importance, key=lambda x: x["abs_importance"], reverse=True)
    return importance

@app.post("/explain/local")
def explain_local(data: CreditInput, model_name: str = "logistic"):
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found"}
    
    selected_model = models[model_name]
    if not hasattr(selected_model, "coef_"):
        return {"error": "Local explanation only supported for linear models currently."}
        
    coefs = selected_model.coef_[0]
    intercept = selected_model.intercept_[0]
    
    input_values = [
        data.credit_utilization_ratio_woe,
        data.debt_to_income_ratio_woe,
        data.annual_income_woe,
        data.employment_years_woe,
        data.number_of_late_payments_12m_woe
    ]
    
    contributions = []
    for feature, value, coef in zip(features, input_values, coefs):
        contributions.append({
            "feature": feature,
            "value": float(value),
            "coefficient": float(coef),
            "contribution": float(value * coef)
        })
    
    total_log_odds = float(intercept + sum(item["contribution"] for item in contributions))
    
    return {
        "intercept": float(intercept),
        "contributions": contributions,
        "total_log_odds": total_log_odds
    }

@app.post("/explain/lime")
def explain_lime(data: CreditInput, model_name: str = "logistic"):
    if explainer is None or model_name not in models:
        return {"error": "XAI component or model not loaded"}
    
    selected_model = models[model_name]
    
    input_values = [
        data.credit_utilization_ratio_woe,
        data.debt_to_income_ratio_woe,
        data.annual_income_woe,
        data.employment_years_woe,
        data.number_of_late_payments_12m_woe
    ]
    
    instance = np.array(input_values)
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=selected_model.predict_proba,
        num_features=len(features)
    )
    
    return {
        "explanation": exp.as_list(),
        "intercept": float(exp.intercept[1])
    }
