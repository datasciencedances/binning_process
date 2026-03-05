"""
CategoricalChiMergeBinner (B): ChiMerge trên category — merge 2 bin "gần nhau"
theo chi-square của bảng contingency (event/non-event) cho tới đạt max_bins / min size.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from binning_process.categorical.base import CategoricalBaseBinner, _chi2_two_bins
from binning_process.core.values import EPSILON


def _woe_per_category(
    x_cat: np.ndarray, y: np.ndarray
) -> Dict[Any, Tuple[float, int, int]]:
    """category -> (woe, n_event, n_nonevent)."""
    total_event = max(int(y.sum()), 1)
    total_nonevent = max(int((1 - y).sum()), 1)
    out = {}
    for cat in np.unique(x_cat):
        mask = x_cat == cat
        n_e = int(y[mask].sum())
        n_ne = int((1 - y[mask]).sum())
        pct_e = n_e / total_event
        pct_ne = n_ne / total_nonevent
        woe = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
        out[cat] = (woe, n_e, n_ne)
    return out


class CategoricalChiMergeBinner(CategoricalBaseBinner):
    """
    ChiMerge trên category: mỗi category = 1 bin ban đầu, gộp cặp kề (theo thứ tự WOE)
    có chi-square nhỏ nhất cho tới đạt max_bins hoặc min_bin_size.
    """

    def _find_bin_groups(
        self, x_cat: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[Any, int], List[str]]:
        cat_woe = _woe_per_category(x_cat, y)
        if not cat_woe:
            return {}, []

        sorted_cats = sorted(cat_woe.keys(), key=lambda c: cat_woe[c][0])
        if self.direction == "descending":
            sorted_cats = sorted_cats[::-1]
        self.direction_ = (
            self.direction if self.direction != "auto" else "ascending"
        )

        bins: List[Set[Any]] = [{c} for c in sorted_cats]
        n = len(x_cat)
        min_n = max(int(self.min_bin_size * n), self.min_event_count * 2)

        def counts_for_bin(b: Set[Any]) -> Tuple[int, int]:
            mask = np.isin(x_cat, list(b))
            return int(y[mask].sum()), int((1 - y[mask]).sum())

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

        cat_to_bin: Dict[Any, int] = {}
        bin_labels: List[str] = []
        for bin_id, b in enumerate(bins):
            lbl = ",".join(str(c) for c in sorted(b, key=str))
            bin_labels.append(lbl)
            for c in b:
                cat_to_bin[c] = bin_id

        return cat_to_bin, bin_labels
