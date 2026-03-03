"""
compare_methods() — So sánh tất cả Binner trên 1 biến.
Import tất cả methods từ supervised/ và unsupervised/.
"""

import pandas as pd
from typing import Optional

from binning_process.supervised import (
    IsotonicBinner, QuantileMonotonicBinner, DecisionTreeBinner,
    ChiMergeBinner, MDLPBinner, SpearmanBinner, LogOddsBinner,
    KSOptimalBinner,
)
from binning_process.unsupervised import EqualWidthBinner, JenksNaturalBreaksBinner
from binning_process.core.utils import iv_rating




# # Tất cả methods có sẵn — thêm method mới vào đây
# ALL_METHODS = {
#     # ── Supervised ────────────────────────────────────────────────
#     "Isotonic"        : IsotonicBinner,
#     "QuantileMonotonic": QuantileMonotonicBinner,
#     "DecisionTree"    : DecisionTreeBinner,
#     "ChiMerge"        : ChiMergeBinner,
#     "MDLP"            : MDLPBinner,
#     "Spearman"        : SpearmanBinner,
#     "LogOdds"         : LogOddsBinner,
#     "KSOptimal"       : KSOptimalBinner,
#     # ── Unsupervised ──────────────────────────────────────────────
#     "EqualWidth"      : EqualWidthBinner,
#     "JenksBreaks"     : JenksNaturalBreaksBinner,
# }


# Tất cả methods có sẵn — thêm method mới vào đây
ALL_METHODS = {
    # ── Supervised ────────────────────────────────────────────────
    "Isotonic"        : IsotonicBinner,
    "ChiMerge"        : ChiMergeBinner,
    "DecisionTree"    : DecisionTreeBinner,
    "MDLP"            : MDLPBinner
}
def compare_methods(
    x               : pd.Series,
    y               : pd.Series,
    feature_name    : str            = "feature",
    max_bins        : int            = 6,
    n_init_bins     : int            = 20,
    max_depth       : int            = 4,
    special_values  : Optional[list] = None,
    methods         : Optional[list] = None,   # None = dùng tất cả
) -> pd.DataFrame:
    """
    Fit nhiều methods và trả về bảng so sánh IV, n_bins, monotonic.

    Args:
        methods: List tên methods muốn so sánh. None = tất cả.
                 Ví dụ: ["Isotonic", "MDLP", "KSOptimal"]

    Returns:
        df: DataFrame sorted by IV descending
        fitted: dict {name: model} để truy cập .summary(), .plot(), .trace_
    """
    shared = dict(
        feature_name   = feature_name,
        max_bins       = max_bins,
        special_values = special_values,
    )
    print(max_bins, n_init_bins, max_depth)
    # Methods có n_init_bins
    has_init_bins = {
        "Isotonic", "QuantileMonotonic", "ChiMerge", "Spearman", "LogOdds"
    }
    # Methods có max_depth
    has_max_depth = {"DecisionTree"}

    selected = methods or list(ALL_METHODS.keys())
    fitted   = {}
    rows     = []

    for name in selected:
        if name not in ALL_METHODS:
            print(f"⚠ Không tìm thấy method: {name}")
            continue

        cls    = ALL_METHODS[name]
        kwargs = dict(shared)
        if name in has_init_bins:
            kwargs["n_init_bins"] = n_init_bins
        if name in has_max_depth:
            kwargs["max_depth"] = max_depth

        model = cls(**kwargs)
        try:
            model.fit(x, y)
            fitted[name] = model
            rows.append({
                "Method"   : name,
                "Group"    : "supervised" if name not in {"EqualWidth", "JenksBreaks"}
                             else "unsupervised",
                "IV"       : round(model.final_iv_, 4),
                "n_bins"   : len(model.final_cuts_) + 1,
                "Monotonic": "✓" if model.is_monotonic() else "✗",
                "Direction": model.direction_,
                "IV_Rating": iv_rating(model.final_iv_),
                "Cuts"     : [round(c, 2) for c in model.final_cuts_],
            })
        except Exception as e:
            rows.append({
                "Method"   : name,
                "Group"    : "",
                "IV"       : None,
                "n_bins"   : None,
                "Monotonic": "ERROR",
                "Direction": "",
                "IV_Rating": str(e)[:60],
                "Cuts"     : [],
            })

    result = (
        pd.DataFrame(rows)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n{'='*75}")
    print(f"  SO SÁNH METHODS — Feature: [{feature_name}]")
    print(f"{'='*75}")
    print(result[["Method", "Group", "IV", "n_bins",
                  "Monotonic", "Direction", "IV_Rating"]].to_string(index=False))
    print(f"{'='*75}\n")

    return result, fitted