from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import yaml
TRAIN_PATH=r"data/processed/train.csv"
FEATURE_PATH="feature_engineering/feature_list.yaml"

df=pd.read_csv(TRAIN_PATH)
with open(FEATURE_PATH, "r") as f:
    feature_config = yaml.safe_load(f)

features = feature_config["woe_features"]
target = feature_config["target"]

x=df[features]
y=df[target]

model=LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(x,y)

print("Model trained Succesfully")

joblib.dump(model,"model/logistic_model.pkl")

print("Model Saved successfully")

