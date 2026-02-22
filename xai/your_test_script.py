import pandas as pd
from xai.local_explain import explain_instance

df = pd.read_csv("data/oot/oot.csv")

row = df.iloc[0]

explain_instance(row)