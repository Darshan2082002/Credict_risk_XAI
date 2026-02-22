from sklearn.model_selection import train_test_split
import pandas as pd

File_path= r"data\raw\credit_scoring_woe_large_dataset.csv"
df=pd.read_csv(File_path)

# First split the data Train(60%) and Test(40%)
train_df,temp_df=train_test_split(
    df,
    test_size=0.4,
    stratify=df["default"],
    random_state=42
)
# Second split: Validation (20%) and OOT (20%)
val_df, oot_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["default"],
    random_state=42
)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
oot_df.to_csv("data/oot/oot.csv", index=False)

print("Splits created successfully.")