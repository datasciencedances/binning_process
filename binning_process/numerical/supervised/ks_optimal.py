from typing import List

import numpy as np

from binning_process.numerical.base import NumericalBaseBinner


class NumericalKSOptimalBinner(NumericalBaseBinner):
    """KS-Optimal Binning — Tối đa hoá Kolmogorov-Smirnov Statistic ..."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _ks_at_cut(x: np.ndarray, y: np.ndarray, cut: float) -> float:
        left  = y[x <= cut]
        right = y[x >  cut]
        if len(left) == 0 or len(right) == 0:
            return 0.0
        total_bad  = max(y.sum(), 1)
        total_good = max((1 - y).sum(), 1)
        ks = abs(
            left.sum()       / total_bad -
            (1 - left).sum() / total_good
        )
        return float(ks)

    def _best_ks_cut(self, x: np.ndarray, y: np.ndarray,
                     candidates: np.ndarray) -> float:
        best_ks, best_cut = -np.inf, candidates[len(candidates) // 2]
        for c in candidates:
            ks = self._ks_at_cut(x, y, c)
            if ks > best_ks:
                best_ks, best_cut = ks, c
        return best_cut

    def _recursive_ks(self, x: np.ndarray, y: np.ndarray,
                      cuts: list, depth: int = 0) -> None:
        if depth >= self.max_bins - 1:
            return
        min_n = max(int(self.min_bin_size * len(x)), self.min_event_count * 2)
        if len(x) < min_n * 2:
            return

        pcts       = np.linspace(10, 90, 17)
        candidates = np.unique(np.nanpercentile(x, pcts))
        if len(candidates) == 0:
            return

        cut = self._best_ks_cut(x, y, candidates)

        left_mask  = x <= cut
        right_mask = ~left_mask
        if left_mask.sum() < min_n or right_mask.sum() < min_n:
            return

        cuts.append(float(cut))
        self._recursive_ks(x[left_mask],  y[left_mask],  cuts, depth + 1)
        self._recursive_ks(x[right_mask], y[right_mask], cuts, depth + 1)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        cuts = []
        self._recursive_ks(x, y, cuts)
        cuts = sorted(set(cuts))
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]
        return cuts, None
