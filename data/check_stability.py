import pandas as pd
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")
oot = pd.read_csv("data/oot/oot.csv")

print("Train default rate:", train["default"].mean())
print("Validation default rate:", val["default"].mean())
print("OOT default rate:", oot["default"].mean())