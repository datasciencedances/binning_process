"""Tests for compare_methods."""
import numpy as np
import pandas as pd
import pytest

from binning_process.compare import ALL_METHODS, compare_methods


@pytest.fixture
def sample_xy():
    np.random.seed(42)
    n = 200
    x = pd.Series(np.random.exponential(2, n).cumsum() + 1)
    y = pd.Series((np.random.rand(n) > 0.5).astype(int))
    return x, y


def test_compare_methods_returns_tuple(sample_xy):
    x, y = sample_xy
    result, fitted = compare_methods(x, y, feature_name="test", verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(fitted, dict)
    assert "Method" in result.columns and "IV" in result.columns
    assert len(result) == len(fitted) or len(fitted) <= len(ALL_METHODS)


def test_compare_methods_with_subset(sample_xy):
    x, y = sample_xy
    result, fitted = compare_methods(
        x, y,
        feature_name="test",
        methods=["Isotonic", "ChiMerge"],
        verbose=False,
    )
    assert len(result) <= 2
    assert set(fitted.keys()).issubset({"Isotonic", "ChiMerge"})
