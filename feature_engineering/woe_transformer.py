import json
import pandas as pd

class WoETransformer:
    def __init__(self, mapping_path):
        with open(mapping_path, "r") as f:
            self.mappings = json.load(f)

    def transform(self, df):
        df = df.copy()

        for feature, mapping in self.mappings.items():
            bin_col = feature + "_bin"
            woe_col = feature + "_woe"

            if bin_col not in df.columns:
                raise ValueError(f"Missing bin column: {bin_col}")

            df[woe_col] = df[bin_col].map(mapping)

        return df