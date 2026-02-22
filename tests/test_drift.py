import pandas as pd
import numpy as np
from governance.drift_monitor import calculate_psi, evaluate_drift


def test_psi_returns_float():
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = pd.Series(np.random.normal(0, 1, 1000))

    psi = calculate_psi(expected, actual)

    assert isinstance(psi, float)


def test_psi_zero_when_same_distribution():
    data = pd.Series(np.random.normal(0, 1, 1000))

    psi = calculate_psi(data, data)

    assert psi < 0.05  # should be near zero


def test_psi_detects_large_drift():
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = pd.Series(np.random.normal(3, 1, 1000))  # shifted mean

    psi = calculate_psi(expected, actual)

    assert psi > 0.25  # significant drift


def test_evaluate_drift_dataframe_output():
    train = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(0, 1, 1000)
    })

    prod = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(3, 1, 1000)
    })

    features = ["feature1", "feature2"]

    result = evaluate_drift(train, prod, features)

    assert isinstance(result, pd.DataFrame)
    assert "PSI" in result.columns
    assert "Status" in result.columns