from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import joblib
import yaml
import os

TRAIN_PATH=r"data/processed/train.csv"
FEATURE_PATH="feature_engineering/feature_list.yaml"
MODEL_DIR = "model"

# Load data
df = pd.read_csv(TRAIN_PATH)
with open(FEATURE_PATH, "r") as f:
    feature_config = yaml.safe_load(f)

features = feature_config["woe_features"]
target = feature_config["target"]

x = df[features]
y = df[target]

# Define models to train
models = {
    "logistic": LogisticRegression(max_iter=1000, solver="lbfgs"),
    "logistic_l1": LogisticRegression(max_iter=1000, solver="liblinear", penalty="l1"),
    "logistic_l2": LogisticRegression(max_iter=1000, solver="lbfgs", penalty="l2"),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "svm": SVC(probability=True, kernel="rbf", max_iter=1000, random_state=42)
}

# Train and save each model
for name, model in models.items():
    print(f"Training {name} model...")
    try:
        model.fit(x, y)
        model_filename = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        joblib.dump(model, model_filename)
        print(f"{name} model saved successfully to {model_filename}")
    except Exception as e:
        print(f"Error training {name}: {e}")

print("\nAll models trained and saved successfully.")

