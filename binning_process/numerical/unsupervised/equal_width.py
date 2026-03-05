import numpy as np
from typing import List
from binning_process.numerical.base import NumericalBaseBinner


class NumericalEqualWidthBinner(NumericalBaseBinner):
    """Equal-Width Binning — Chia đều khoảng giá trị (Unsupervised) ..."""

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if abs(x_max - x_min) < 1e-9:
            return []
        cuts = list(np.linspace(x_min, x_max, self.max_bins + 1)[1:-1])
        return cuts
