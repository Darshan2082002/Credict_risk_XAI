import pandas as pd
import json

TRAIN_PATH = "data/processed/train.csv"

df = pd.read_csv(TRAIN_PATH)

woe_mappings = {}

# Identify raw variables that have bins
raw_features = [
    "credit_utilization_ratio",
    "debt_to_income_ratio",
    "annual_income",
    "employment_years",
    "number_of_late_payments_12m"
]

for feature in raw_features:
    bin_col = feature + "_bin"
    woe_col = feature + "_woe"

    mapping = (
        df[[bin_col, woe_col]]
        .drop_duplicates()
        .sort_values(by=bin_col)
        .set_index(bin_col)[woe_col]
        .to_dict()
    )

    woe_mappings[feature] = mapping

with open("feature_engineering/woe_mappings.json", "w") as f:
    json.dump(woe_mappings, f, indent=4)

print("WoE mappings extracted and frozen.")