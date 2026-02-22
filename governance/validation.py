import yaml
import pandas as pd


def load_feature_contract(path="governance/feature_contract.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["features"]


def align_features(df: pd.DataFrame):
    expected_features = load_feature_contract()

    # Check missing
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Reorder
    df = df[expected_features]

    return df