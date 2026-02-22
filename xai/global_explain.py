import joblib
import yaml
import pandas as pd


def global_importance():
    model = joblib.load("model/logistic_model.pkl")

    with open("feature_engineering/feature_list.yaml", "r") as f:
        config = yaml.safe_load(f)

    features = config["woe_features"]

    coefs = model.coef_[0]

    df = pd.DataFrame({
        "feature": features,
        "coefficient": coefs,
        "abs_importance": abs(coefs)
    })

    df = df.sort_values("abs_importance", ascending=False)

    print("\nGlobal Feature Importance (Log-Odds Impact):")
    print(df[["feature", "coefficient"]])

    return df


if __name__ == "__main__":
    global_importance()