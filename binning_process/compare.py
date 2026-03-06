"""
compare_methods() — So sánh nhiều Binner trên một biến (numerical hoặc categorical).
Dùng is_numerical=True cho biến số, is_numerical=False cho categorical.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from binning_process.categorical import (
    CategoricalBaseBinner,
    CategoricalChiMergeBinner,
    CategoricalDecisionTreeBinner,
    CategoricalWOEAdjacentBinner,
)
from binning_process.core.utils import iv_rating
from binning_process.numerical.base import NumericalBaseBinner
from binning_process.numerical.supervised import (
    NumericalChiMergeBinner,
    NumericalDecisionTreeBinner,
    NumericalKSOptimalBinner,
    NumericalMDLPBinner,
)
from binning_process.numerical.unsupervised import NumericalKMeansBinner, NumericalQuantileBinner

# Numerical: tên -> (class, n_init_bins?, max_depth?)
NUMERICAL_METHOD_CONFIG: Dict[str, dict] = {
    "Quantile": {"cls": NumericalQuantileBinner, "n_init_bins": True, "max_depth": False},
    "ChiMerge": {"cls": NumericalChiMergeBinner, "n_init_bins": True, "max_depth": False},
    "DecisionTree": {"cls": NumericalDecisionTreeBinner, "n_init_bins": False, "max_depth": True},
    "MDLP": {"cls": NumericalMDLPBinner, "n_init_bins": False, "max_depth": False},
    "KS-Optimal": {"cls": NumericalKSOptimalBinner},
    "KMeans": {"cls": NumericalKMeansBinner, "n_init_bins": True, "random_state": 0},
}
# Categorical: tên -> class
CATEGORICAL_METHOD_CONFIG: Dict[str, dict] = {
    "WOEAdjacent": {"cls": CategoricalWOEAdjacentBinner},
    "ChiMerge": {"cls": CategoricalChiMergeBinner},
    "DecisionTree": {"cls": CategoricalDecisionTreeBinner},
}

ALL_NUMERICAL_METHODS: Dict[str, type] = {k: v["cls"] for k, v in NUMERICAL_METHOD_CONFIG.items()}
ALL_CATEGORICAL_METHODS: Dict[str, type] = {k: v["cls"] for k, v in CATEGORICAL_METHOD_CONFIG.items()}


def compare_methods(
    x: pd.Series,
    y: pd.Series,
    feature_name: str = "feature",
    is_numerical: bool = True,
    max_bins: int = 6,
    n_init_bins: int = 20,
    max_depth: int = 4,
    special_values: Optional[List] = None,
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Union[NumericalBaseBinner, CategoricalBaseBinner]]]:
    """
    So sánh nhiều Binner trên (x, y).

    Args:
        x, y: Feature và target (binary).
        feature_name: Tên hiển thị.
        is_numerical: True = chạy numerical methods (Isotonic, ChiMerge, ...),
                      False = chạy categorical methods (WOEAdjacent, ChiMerge, DecisionTree).
        max_bins: Số bins tối đa (dùng chung).
        n_init_bins: Số bins khởi tạo (numerical).
        max_depth: Độ sâu cây (numerical DecisionTree).
        special_values: Giá trị đặc biệt tách riêng.
        methods: Tên methods cần chạy; None = tất cả theo loại (numerical/categorical).
        verbose: In bảng so sánh ra stdout khi True.

    Returns:
        result: DataFrame (Method, Group, IV, n_bins, Monotonic, Direction, IV_Rating, Cuts).
        fitted: Dict[str, Binner] — model đã fit.
    """
    if is_numerical:
        config = NUMERICAL_METHOD_CONFIG
        selected = methods or list(NUMERICAL_METHOD_CONFIG.keys())
    else:
        config = CATEGORICAL_METHOD_CONFIG
        selected = methods or list(CATEGORICAL_METHOD_CONFIG.keys())

    shared = dict(
        feature_name=feature_name,
        max_bins=max_bins,
        special_values=special_values,
    )

    fitted: Dict[str, Union[NumericalBaseBinner, CategoricalBaseBinner]] = {}
    rows: List[dict] = []

    for name in selected:
        if name not in config:
            if verbose:
                print(f"Không tìm thấy method: {name}")
            continue

        cfg = config[name]
        cls = cfg["cls"]
        kwargs = dict(shared)
        if is_numerical:
            if cfg.get("n_init_bins"):
                kwargs["n_init_bins"] = n_init_bins
            if cfg.get("max_depth"):
                kwargs["max_depth"] = max_depth
        model = cls(**kwargs)
        try:
            model.fit(x, y)
            fitted[name] = model
            n_bins = (len(model.final_cuts_) + 1) if model.final_cuts_ else len(model.final_woe_table_)
            cuts_or_bins: Any = [round(c, 2) for c in model.final_cuts_] if model.final_cuts_ else []
            rows.append({
                "Method": name,
                "Group": "numerical" if is_numerical else "categorical",
                "IV": round(model.final_iv_, 4),
                "n_bins": n_bins,
                "Monotonic": "✓" if model.is_monotonic() else "✗",
                "Direction": getattr(model, "direction_", ""),
                "IV_Rating": iv_rating(model.final_iv_),
                "Cuts": cuts_or_bins,
            })
        except Exception as e:
            err_msg = str(e)[:60]
            rows.append({
                "Method": name,
                "Group": "numerical" if is_numerical else "categorical",
                "IV": None,
                "n_bins": None,
                "Monotonic": "ERROR",
                "Direction": "",
                "IV_Rating": err_msg,
                "Cuts": [],
            })

    result = (
        pd.DataFrame(rows)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        kind = "NUMERICAL" if is_numerical else "CATEGORICAL"
        print(f"\n{'='*75}")
        print(f"  SO SÁNH METHODS [{kind}] — Feature: [{feature_name}]")
        print(f"{'='*75}")
        print(result[["Method", "Group", "IV", "n_bins", "Monotonic", "Direction", "IV_Rating"]].to_string(index=False))
        print(f"{'='*75}\n")

    return result, fitted
