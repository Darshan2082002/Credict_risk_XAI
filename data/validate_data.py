import pandas as pd
import json
import hashlib


File_path= r"data\raw\credit_scoring_woe_large_dataset.csv"
df=pd.read_csv(File_path)
report={}
# Basic Info 
report["record_count"]=len(df)
report["column_report"]=len(df.columns)
#Target Validatioin 
report["defalut_rate"]=df["default"].mean()
report["target_unique_values"] = df["default"].unique().tolist()
#Null check

report["null_counts"] = df.isnull().sum().to_dict()
# Data hash
with open(File_path, "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()

report["dataset_hash"] = file_hash

#save report
with open("data/reports/data_validation_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Validation complete.")