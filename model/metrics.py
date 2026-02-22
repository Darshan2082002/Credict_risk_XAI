import numpy as np
import pandas as pd


def calculate_ks(y_true, y_pred):
    """
    Calculate Kolmogorov-Smirnov statistic
    """

    df = pd.DataFrame({
        "target": y_true,
        "score": y_pred
    })

    df = df.sort_values("score", ascending=False)

    df["good"] = (df["target"] == 0).astype(int)
    df["bad"] = (df["target"] == 1).astype(int)

    df["cum_good"] = df["good"].cumsum() / df["good"].sum()
    df["cum_bad"] = df["bad"].cumsum() / df["bad"].sum()

    df["ks"] = np.abs(df["cum_bad"] - df["cum_good"])

    ks_value = df["ks"].max()

    return ks_value