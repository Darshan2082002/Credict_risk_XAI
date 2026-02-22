import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI)
    expected: training distribution (Series)
    actual: production distribution (Series)
    """

    expected = expected.dropna()
    actual = actual.dropna()

    # Define bins based on training data
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # Avoid division by zero
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * 
                 np.log(actual_perc / expected_perc))

    return psi


def evaluate_drift(train_df, prod_df, features):

    results = {}

    for feature in features:
        psi_value = calculate_psi(train_df[feature], prod_df[feature])

        if psi_value < 0.10:
            status = "Stable"
        elif psi_value < 0.25:
            status = "Moderate Drift"
        else:
            status = "Significant Drift"

        results[feature] = {
            "PSI": round(float(psi_value), 4),
            "Status": status
        }

    return pd.DataFrame(results).T