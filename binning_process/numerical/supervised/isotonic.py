import numpy as np
from typing import List
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from binning_process.numerical.base import NumericalBaseBinner
from binning_process.core.utils import quantile_cuts
from binning_process.core.values import EPSILON

class NumericalIsotonicBinner(NumericalBaseBinner):
    """
    Isotonic Regression Binning

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Hãy hình dung bạn có đồ thị điểm (X, bad_rate). Có thể          │
    │  bad_rate lên xuống loạn xạ vì nhiễu. Isotonic Regression        │
    │  vẽ một "đường bậc thang" đi LÊN (hoặc đi XUỐNG) mượt nhất       │
    │  có thể qua các điểm đó.                                         │
    │                                                                  │
    │  Mỗi "bậc thang ngang" = 1 bin có cùng bad_rate.                 │
    │  Điểm gãy giữa 2 bậc thang = cut-point.                          │
    │                                                                  │
    │  Các bước: Chia quantile → tính event rate từng bin → fit        │
    │  Isotonic Regression → lấy điểm gãy làm cut-points.              │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # B1: Quantile chia sơ bộ (tái sử dụng quantile_cuts từ core.utils)
        init_cuts = quantile_cuts(x, self.n_init_bins)
        edges     = [-np.inf] + init_cuts + [np.inf]
        bin_idx   = pd.cut(x, bins=edges, labels=False, right=True, include_lowest=True)

        # B2: Event rate + bin center cho từng bin
        centers, rates, weights = [], [], []
        for i in range(len(edges) - 1):
            mask = bin_idx == i
            if mask.sum() == 0:
                continue
            centers.append(float(np.median(x[mask])))
            rates.append(float(y[mask].mean()))
            weights.append(int(mask.sum()))

        centers = np.array(centers)
        rates   = np.array(rates)
        weights = np.array(weights)

        # B3: Fit Isotonic Regression
        iso = IsotonicRegression(
            increasing=(self.direction_ == "ascending"),
            out_of_bounds="clip"
        )
        fitted = iso.fit_transform(centers, rates, sample_weight=weights)

        # B4: Điểm gãy = cut-points
        cuts = []
        for i in range(len(fitted) - 1):
            if abs(fitted[i] - fitted[i + 1]) > EPSILON:
                if i < len(init_cuts):
                    cuts.append(init_cuts[i])
        cuts = sorted(set(cuts))
        return cuts, None