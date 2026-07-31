import joblib
import yaml
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

import warnings
warnings.filterwarnings("ignore")
def explain_with_lime():

    # Load model
    model = joblib.load("model/logistic_model.pkl")

    # Load feature config
    with open("feature_engineering/feature_list.yaml", "r") as f:
        config = yaml.safe_load(f)

    features = config["woe_features"]

    # Load dataset
    df = pd.read_csv("data/oot/oot.csv")

    X = df[features]
    y = df["default"]

    # Create explainer
    explainer = LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=features,
        class_names=["Good", "Default"],
        mode="classification"
    )

    # Select one instance
    instance = X.iloc[0]

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba
    )

    print("\nLIME Explanation:")
    print(explanation.as_list())

    # explanation.show_in_notebook(show_table=True) # Not needed for standalone script


if __name__ == "__main__":
    explain_with_lime()