from sklearn.metrics import roc_auc_score
import pandas as pd
import yaml
import joblib
import os
from model.metrics import calculate_ks

FEATURE_PATH = "feature_engineering/feature_list.yaml"
MODEL_DIR = "model"

# Load data
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")
oot_df = pd.read_csv("data/oot/oot.csv")

with open(FEATURE_PATH, "r") as f:
    feature_config = yaml.safe_load(f)

features = feature_config["woe_features"]
target = feature_config["target"]

def evaluate_model(model, df, name):
    x = df[features]
    y = df[target]
    preds = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, preds)
    ks = calculate_ks(y, preds)
    return auc, ks

# Find all models
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_model.pkl")]

results = []

for model_file in model_files:
    model_name = model_file.replace("_model.pkl", "")
    print(f"\nEvaluating {model_name} model...")
    
    model_path = os.path.join(MODEL_DIR, model_file)
    model = joblib.load(model_path)
    
    auc_train, ks_train = evaluate_model(model, train_df, "Train")
    auc_val, ks_val = evaluate_model(model, val_df, "Validation")
    auc_oot, ks_oot = evaluate_model(model, oot_df, "OOT")
    
    print(f"  Train      - AUC: {auc_train:.4f}, KS: {ks_train:.4f}")
    print(f"  Validation - AUC: {auc_val:.4f}, KS: {ks_val:.4f}")
    print(f"  OOT        - AUC: {auc_oot:.4f}, KS: {ks_oot:.4f}")
    
    results.append({
        "model": model_name,
        "auc_train": auc_train, "ks_train": ks_train,
        "auc_val": auc_val, "ks_val": ks_val,
        "auc_oot": auc_oot, "ks_oot": ks_oot
    })

print("\nEvaluation completed.")
