import joblib
import pandas as pd

def test_model_prediction_shape():

    model = joblib.load("model/logistic_model.pkl")

    # Get exact training feature order
    features = model.feature_names_in_

    # Create sample using dictionary
    sample_dict = {
        "credit_utilization_ratio_woe": 0.2,
        "debt_to_income_ratio_woe": 0.1,
        "employment_years_woe": 0.0,
        "number_of_late_payments_12m_woe": 0.3,
        "annual_income_woe": -0.2
    }

    # Align to training order
    sample = pd.DataFrame([[sample_dict[f] for f in features]], columns=features)

    preds = model.predict_proba(sample)

    assert preds.shape == (1, 2)