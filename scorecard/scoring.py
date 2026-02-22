import pandas as pd
import joblib
import yaml
from scorecard.scaling import ScoreScaler

# Load model
model = joblib.load("model/logistic_model.pkl")

# Load features
with open("feature_engineering/feature_list.yaml", "r") as f:
    feature_config = yaml.safe_load(f)

features = feature_config["woe_features"]

# Load OOT sample
df = pd.read_csv("data/oot/oot.csv")

X = df[features]
pd_pred = model.predict_proba(X)[:, 1]

scaler = ScoreScaler()
scores = scaler.pd_to_score(pd_pred)

df["pd"] = pd_pred
df["score"] = scores

print(df[["pd", "score"]].head())