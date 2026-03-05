from typing import List

import numpy as np
from sklearn.isotonic import IsotonicRegression

from binning_process.core.utils import quantile_cuts
from binning_process.core.values import EPSILON
from binning_process.numerical.base import NumericalBaseBinner


class NumericalLogOddsBinner(NumericalBaseBinner):
    """Log-Odds (Scorecard) Binning ..."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        sort_idx = np.argsort(x)
        x_s, y_s = x[sort_idx], y[sort_idx]

        iso    = IsotonicRegression(
            increasing=(self.direction_ == "ascending"),
            out_of_bounds="clip"
        )
        er_smooth = iso.fit_transform(x_s, y_s.astype(float))
        er_smooth = np.clip(er_smooth, EPSILON, 1 - EPSILON)

        log_odds = np.log(er_smooth / (1 - er_smooth))

        lo_min = log_odds.min()
        lo_max = log_odds.max()
        if abs(lo_max - lo_min) < EPSILON:
            return quantile_cuts(x, self.max_bins - 1)

        lo_cuts = np.linspace(lo_min, lo_max, self.max_bins + 1)[1:-1]

        cuts = []
        for lo_cut in lo_cuts:
            idx = np.searchsorted(log_odds, lo_cut)
            idx = min(idx, len(x_s) - 1)
            cuts.append(float(x_s[idx]))

        cuts = sorted(set(cuts))
        cuts = [c for c in cuts if x.min() < c < x.max()]
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]
        return cuts, None
