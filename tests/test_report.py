"""Tests for report.generate_compare_report."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from binning_process.report import generate_compare_report


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "f1": np.random.randn(n).cumsum() + 10,
        "f2": np.random.exponential(2, n),
        "y": (np.random.rand(n) > 0.6).astype(int),
    })


def test_generate_compare_report_creates_html(sample_df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.html")
        generate_compare_report(
            df_train=sample_df,
            target_col="y",
            feature_cols=["f1", "f2"],
            output_path=path,
            verbose=False,
        )
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            html = f.read()
        assert "Compare Binning Report" in html
        assert "f1" in html and "f2" in html
        assert "Overview" in html and "Chi tiết" in html
