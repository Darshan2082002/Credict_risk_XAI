import joblib
import yaml
import pandas as pd


def explain_instance(row):
    model = joblib.load("model/logistic_model.pkl")

    with open("feature_engineering/feature_list.yaml", "r") as f:
        config = yaml.safe_load(f)

    features = config["woe_features"]

    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    contributions = row[features].values * coefs

    explanation = pd.DataFrame({
        "feature": features,
        "woe_value": row[features].values,
        "coefficient": coefs,
        "contribution": contributions
    })

    explanation = explanation.sort_values("contribution", ascending=False)

    total_log_odds = intercept + contributions.sum()

    print("\nLocal Explanation:")
    print(explanation)
    print(f"\nTotal Log-Odds: {total_log_odds:.4f}")

    return explanation


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/oot/oot.csv")
    row = df.iloc[0]

    explain_instance(row)