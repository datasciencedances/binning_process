"""Tests for core.utils."""
import numpy as np
import pandas as pd
import pytest

from binning_process.core.utils import (
    compute_psi,
    compute_woe_iv_table,
    detect_direction,
    iv_rating,
    woe_table_to_pct_per_bin,
)


def test_detect_direction_ascending():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0, 0, 1, 1, 1])
    assert detect_direction(x, y) == "ascending"


def test_detect_direction_descending():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1, 1, 0, 0, 0])
    assert detect_direction(x, y) == "descending"


def test_compute_woe_iv_table():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    cuts = [3.0, 6.0]
    df = compute_woe_iv_table(cuts, x, y, feature_name="test")
    assert len(df) == 3
    assert "woe" in df.columns and "iv_bin" in df.columns and "iv_total" in df.columns
    assert df["iv_total"].iloc[0] == pytest.approx(df["iv_bin"].sum(), rel=1e-6)


def test_woe_table_to_pct_per_bin():
    df = pd.DataFrame({"n_total": [10, 20, 30]})
    pct = woe_table_to_pct_per_bin(df)
    assert pct.sum() == pytest.approx(1.0)
    assert len(pct) == 3


def test_compute_psi():
    p_train = np.array([0.2, 0.3, 0.5])
    p_valid = np.array([0.25, 0.25, 0.5])
    psi_total, psi_bin = compute_psi(p_train, p_valid)
    assert psi_total >= 0
    assert len(psi_bin) == 3


def test_iv_rating():
    assert "Vô dụng" in iv_rating(0.01)
    assert "Yếu" in iv_rating(0.05)
    assert "Trung bình" in iv_rating(0.2)
    assert iv_rating(None) == ""
