from typing import List, Optional
import numpy as np
from binning_process.numerical.base import NumericalBaseBinner


class NumericalMDLPBinner(NumericalBaseBinner):
    """MDLP — Minimum Description Length Principle Binning ..."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_total = 1

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        p = float(y.mean())
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    @staticmethod
    def _info_gain(y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> float:
        n, n_l, n_r = len(y), len(y_l), len(y_r)
        if n_l == 0 or n_r == 0:
            return 0.0
        return NumericalMDLPBinner._entropy(y) - \
               (n_l / n) * NumericalMDLPBinner._entropy(y_l) - \
               (n_r / n) * NumericalMDLPBinner._entropy(y_r)

    def _mdlp_stop(self, y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> bool:
        n = len(y)
        if n < 4:
            return True
        k  = max(len(np.unique(y)), 1)
        k1 = max(len(np.unique(y_l)), 1)
        k2 = max(len(np.unique(y_r)), 1)
        gain      = self._info_gain(y, y_l, y_r)
        delta     = np.log2(3 ** k - 2) - (
            k  * self._entropy(y) -
            k1 * self._entropy(y_l) -
            k2 * self._entropy(y_r)
        )
        threshold = (np.log2(n - 1) + delta) / n
        return gain <= threshold

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        unique_vals = np.unique(x)
        if len(unique_vals) < 2:
            return None

        best_gain, best_cut = -np.inf, None
        sorted_idx = np.argsort(x)
        x_s, y_s   = x[sorted_idx], y[sorted_idx]

        for i in range(1, len(unique_vals)):
            cut   = (unique_vals[i - 1] + unique_vals[i]) / 2
            left  = x_s < cut
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            gain = self._info_gain(y_s, y_s[left], y_s[right])
            if gain > best_gain:
                best_gain, best_cut = gain, cut

        return best_cut

    def _recursive_split(self, x: np.ndarray, y: np.ndarray,
                          cuts: list, depth: int = 0) -> None:
        if depth >= self.max_bins - 1:
            return
        if len(x) < max(int(self.min_bin_size * self._n_total), self.min_event_count * 2):
            return

        cut = self._best_split(x, y)
        if cut is None:
            return

        left_mask  = x <= cut
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return

        if self._mdlp_stop(y, y[left_mask], y[right_mask]):
            return

        cuts.append(float(cut))
        self._recursive_split(x[left_mask],  y[left_mask],  cuts, depth + 1)
        self._recursive_split(x[right_mask], y[right_mask], cuts, depth + 1)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        self._n_total = len(x)
        cuts = []
        self._recursive_split(x, y, cuts)
        real_cuts = []
        for c in cuts:
            left_side = x[x <= c]
            if len(left_side) > 0:
                real_cuts.append(float(np.max(left_side)))
        cuts = sorted(set(real_cuts))
        return cuts, None
