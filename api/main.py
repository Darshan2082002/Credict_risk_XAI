from fastapi import FastAPI
import joblib

from api.schemas import CreditInput

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "API is running"}


model = joblib.load("model\logistic_model.pkl")
@app.post("/predict")
def predict(data: CreditInput):
    
    # Convert input to model format
    input_data = [[
        data.credit_utilization_ratio_woe,
        data.debt_to_income_ratio_woe,
        data.employment_years_woe,
        data.number_of_late_payments_12m_woe,
        data.annual_income_woe
    ]]

    # Load model (adjust path if needed)
    import joblib
    model = joblib.load("models/model.pkl")

    probability = model.predict_proba(input_data)[0][1]

    decision = "Approve" if probability < 0.5 else "Reject"

    return {
        "PD": float(probability),
        "decision": decision
    }