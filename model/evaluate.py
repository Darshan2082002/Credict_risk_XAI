from sklearn.metrics import roc_auc_score
import pandas as pd
import yaml
import joblib
from model.metrics import calculate_ks
FEATURE_PATH = "feature_engineering/feature_list.yaml"
MODEL_PATH = "model/logistic_model.pkl"

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")
oot = pd.read_csv("data/oot/oot.csv")

with open(FEATURE_PATH,"r") as f:
    feature_config=yaml.safe_load(f)

features=feature_config["woe_features"]
target=feature_config["target"]

model = joblib.load(MODEL_PATH)

def evaluate(df,name):
    x=df[features]
    y=df[target]
    preds = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, preds)
    print(f"{name} AUC: {auc:.4f}")
    ks = calculate_ks(y, preds)
    print(f"{name} KS: {ks:.4f}")
    return auc

auc_train = evaluate(train, "Train")
auc_val = evaluate(val, "Validation")
auc_oot = evaluate(oot, "OOT")
