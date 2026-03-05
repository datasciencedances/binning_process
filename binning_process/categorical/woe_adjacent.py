"""
CategoricalWOEAdjacentBinner (A): WOE encoding (1 category = 1 bin) + gộp theo constraint.
Tính WOE từng category, sort theo WOE (hoặc bad rate), rồi adjacent merge để đảm bảo
min_bin_size / min_event_count và (tuỳ chọn) monotonic theo WOE.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from binning_process.categorical.base import CategoricalBaseBinner, _chi2_two_bins
from binning_process.core.utils import compute_category_woe_table
from binning_process.core.values import EPSILON


class CategoricalWOEAdjacentBinner(CategoricalBaseBinner):
    """
    WOE (1 category = 1 bin) + sort theo WOE + adjacent merge.
    Report: category_table_ (n, bad, bad_rate, woe, iv_contrib), smoothing_warnings_.
    """

    def __init__(
        self,
        feature_name: str = "feature",
        min_bin_size: float = 0.05,
        min_event_count: int = 5,
        max_bins: int = 8,
        special_values: Optional[List] = None,
        unseen_woe: float = 0.0,
        direction: str = "auto",
        enforce_monotonic: bool = True,
        sort_by: str = "woe",
    ):
        super().__init__(
            feature_name=feature_name,
            min_bin_size=min_bin_size,
            min_event_count=min_event_count,
            max_bins=max_bins,
            special_values=special_values,
            unseen_woe=unseen_woe,
            direction=direction,
        )
        self.enforce_monotonic = enforce_monotonic
        self.sort_by = sort_by  # "woe" | "bad_rate"

    def _find_bin_groups(
        self, x_cat: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[Any, int], List[str]]:
        self.category_table_, self.smoothing_warnings_ = compute_category_woe_table(
            x_cat, y, self.feature_name, smoothing_epsilon=0.5
        )
        if self.category_table_.empty:
            return {}, []

        # Sort category theo WOE hoặc bad_rate
        key = "woe" if self.sort_by == "woe" else "bad_rate"
        cat_order = (
            self.category_table_.sort_values(key)["category"].tolist()
        )
        if self.direction == "descending":
            cat_order = cat_order[::-1]
        self.direction_ = (
            self.direction if self.direction != "auto" else "ascending"
        )

        n = len(x_cat)
        min_n = max(int(self.min_bin_size * n), self.min_event_count * 2)

        def counts_for_bin(b: Set[Any]) -> Tuple[int, int]:
            mask = np.isin(x_cat, list(b))
            return int(y[mask].sum()), int((1 - y[mask]).sum())

        bins: List[Set[Any]] = [{c} for c in cat_order]

        # Merge adjacent (chi2 nhỏ nhất) cho đến đủ ràng buộc
        while len(bins) > self.max_bins:
            best_chi2 = np.inf
            best_idx = 0
            for i in range(len(bins) - 1):
                ca, cb = counts_for_bin(bins[i]), counts_for_bin(bins[i + 1])
                chi2 = _chi2_two_bins(ca, cb)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_idx = i
            bins[best_idx] = bins[best_idx] | bins[best_idx + 1]
            bins.pop(best_idx + 1)

        while len(bins) > 1:
            sizes = [sum(counts_for_bin(b)) for b in bins]
            if all(s >= min_n for s in sizes):
                break
            idx_small = int(np.argmin(sizes))
            i = (
                max(0, idx_small - 1)
                if idx_small == len(bins) - 1
                else idx_small
            )
            j = i + 1
            bins[i] = bins[i] | bins[j]
            bins.pop(j)

        # (Tuỳ chọn) Enforce monotonic: merge 2 bin kề nhau vi phạm thứ tự WOE
        if self.enforce_monotonic and len(bins) >= 2:
            total_event = max(int(y.sum()), 1)
            total_nonevent = max(int((1 - y).sum()), 1)

            def er_for_bin(b: Set[Any]) -> float:
                mask = np.isin(x_cat, list(b))
                n_t = mask.sum()
                return (y[mask].sum() / max(n_t, 1)) if n_t else 0.0

            merged = True
            while merged and len(bins) >= 2:
                merged = False
                ers = [er_for_bin(b) for b in bins]
                for i in range(len(bins) - 1):
                    if self.direction_ == "ascending" and ers[i] > ers[i + 1] + 1e-9:
                        bins[i] = bins[i] | bins[i + 1]
                        bins.pop(i + 1)
                        merged = True
                        break
                    if self.direction_ == "descending" and ers[i] < ers[i + 1] - 1e-9:
                        bins[i] = bins[i] | bins[i + 1]
                        bins.pop(i + 1)
                        merged = True
                        break

        cat_to_bin: Dict[Any, int] = {}
        bin_labels: List[str] = []
        for bin_id, b in enumerate(bins):
            lbl = ",".join(str(c) for c in sorted(b, key=str))
            bin_labels.append(lbl)
            for c in b:
                cat_to_bin[c] = bin_id

        return cat_to_bin, bin_labels
