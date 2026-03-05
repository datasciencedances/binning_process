from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from binning_process.core.utils import quantile_cuts
from binning_process.numerical.base import NumericalBaseBinner


class NumericalChiMergeBinner(NumericalBaseBinner):
    """
    ChiMerge — Chi-square Based Merging
    ...
    """

    def __init__(self, n_init_bins: int = 50, confidence_level: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins      = n_init_bins
        self.confidence_level = confidence_level

    @staticmethod
    def _chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
        table = np.array([counts_a, counts_b], dtype=float)
        if table.sum() == 0:
            return 0.0
        chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
        return float(chi2)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        init_cuts = quantile_cuts(x, self.n_init_bins)
        init_cuts = [c for c in init_cuts if c < x.max()]
        edges     = [-np.inf] + init_cuts + [np.inf]
        bin_idx   = pd.cut(x, bins=edges, labels=False, right=True, include_lowest=True)

        counts = np.column_stack([
        np.bincount(bin_idx.astype(int), weights=1 - y, minlength=len(edges) - 1),
        np.bincount(bin_idx.astype(int), weights=y,     minlength=len(edges) - 1),
    ]).astype(int)

        current_cuts  = list(init_cuts)
        chi2_threshold = stats.chi2.ppf(1 - self.confidence_level, df=1)

        for _ in range(500):
            chi2_vals = [self._chi2(counts[i], counts[i+1]) for i in range(len(counts)-1)]
            if chi2_vals and min(chi2_vals) >= chi2_threshold:
                break

            min_idx = int(np.argmin(chi2_vals))
            counts[min_idx] = counts[min_idx] + counts[min_idx + 1]
            counts = np.delete(counts, min_idx + 1, axis=0)
            if min_idx < len(current_cuts):
                current_cuts.pop(min_idx)

        return sorted(current_cuts), sorted(init_cuts)
