from typing import List

import numpy as np

from binning_process.core.utils import quantile_cuts
from binning_process.numerical.base import NumericalBaseBinner


class NumericalQuantileBinner(NumericalBaseBinner):
    """Quantile Binning + Adjacent Merge ..."""

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        init_cuts = quantile_cuts(x, self.n_init_bins)
        init_cuts = [c for c in init_cuts if c < x.max()] # nếu dùng right=True, nếu dùng right=False thì làm ngược lại
        return sorted(init_cuts), sorted(init_cuts)
