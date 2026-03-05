"""
compare_methods() — So sánh nhiều Binner trên một biến.
Import tất cả methods từ supervised/ và unsupervised/.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from binning_process.core.base import BaseBinner
from binning_process.core.utils import iv_rating
from binning_process.supervised import (
    ChiMergeBinner,
    DecisionTreeBinner,
    IsotonicBinner,
    MDLPBinner,
    QuantileMonotonicBinner,
)

# Cấu hình methods: tên -> (class, n_init_bins?, max_depth?)
# Thêm method mới: thêm entry và set True cho n_init_bins hoặc max_depth nếu cần.
METHOD_CONFIG: Dict[str, dict] = {
    "QuantileMonotonic": {"cls": QuantileMonotonicBinner, "n_init_bins": True, "max_depth": False},
    "Isotonic": {"cls": IsotonicBinner, "n_init_bins": True, "max_depth": False},
    "ChiMerge": {"cls": ChiMergeBinner, "n_init_bins": True, "max_depth": False},
    "DecisionTree": {"cls": DecisionTreeBinner, "n_init_bins": False, "max_depth": True},
    "MDLP": {"cls": MDLPBinner, "n_init_bins": False, "max_depth": False},
}

ALL_METHODS: Dict[str, type] = {k: v["cls"] for k, v in METHOD_CONFIG.items()}


def compare_methods(
    x: pd.Series,
    y: pd.Series,
    feature_name: str = "feature",
    max_bins: int = 6,
    n_init_bins: int = 20,
    max_depth: int = 4,
    special_values: Optional[List] = None,
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, BaseBinner]]:
    """
    Fit nhiều Binner trên (x, y) và trả về bảng so sánh + dict model đã fit.

    Args:
        x, y: Feature và target (binary).
        feature_name: Tên hiển thị.
        max_bins: Số bins tối đa (dùng chung).
        n_init_bins: Số bins khởi tạo (cho Isotonic, QuantileMonotonic, ChiMerge, ...).
        max_depth: Độ sâu cây (cho DecisionTree).
        special_values: Giá trị đặc biệt tách riêng.
        methods: Tên methods cần chạy; None = tất cả trong ALL_METHODS.
        verbose: In bảng so sánh ra stdout khi True.

    Returns:
        result: DataFrame (Method, Group, IV, n_bins, Monotonic, Direction, IV_Rating, Cuts), sort theo IV giảm dần.
        fitted: Dict[str, BaseBinner] — model đã fit, dùng .summary(), .plot(), .trace_.
    """
    shared = dict(
        feature_name=feature_name,
        max_bins=max_bins,
        special_values=special_values,
    )
    selected = methods or list(ALL_METHODS.keys())
    fitted: Dict[str, BaseBinner] = {}
    rows: List[dict] = []

    for name in selected:
        if name not in METHOD_CONFIG:
            if verbose:
                print(f"Không tìm thấy method: {name}")
            continue

        cfg = METHOD_CONFIG[name]
        cls = cfg["cls"]
        kwargs = dict(shared)
        if cfg.get("n_init_bins"):
            kwargs["n_init_bins"] = n_init_bins
        if cfg.get("max_depth"):
            kwargs["max_depth"] = max_depth

        model = cls(**kwargs)
        try:
            model.fit(x, y)
            fitted[name] = model
            rows.append({
                "Method": name,
                "Group": "unsupervised" if name in {"EqualWidth", "JenksBreaks"} else "supervised",
                "IV": round(model.final_iv_, 4),
                "n_bins": len(model.final_cuts_) + 1,
                "Monotonic": "✓" if model.is_monotonic() else "✗",
                "Direction": model.direction_,
                "IV_Rating": iv_rating(model.final_iv_),
                "Cuts": [round(c, 2) for c in model.final_cuts_],
            })
        except Exception as e:
            err_msg = str(e)[:60]
            rows.append({
                "Method": name,
                "Group": "",
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
        print(f"\n{'='*75}")
        print(f"  SO SÁNH METHODS — Feature: [{feature_name}]")
        print(f"{'='*75}")
        print(result[["Method", "Group", "IV", "n_bins", "Monotonic", "Direction", "IV_Rating"]].to_string(index=False))
        print(f"{'='*75}\n")

    return result, fitted
