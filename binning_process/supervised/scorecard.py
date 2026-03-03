import numpy as np
from typing import List
from binning_process.core.base import BaseBinner
from binning_process.core.utils import quantile_cuts, EPSILON
from sklearn.isotonic import IsotonicRegression

class LogOddsBinner(BaseBinner):
    """
    Log-Odds (Scorecard) Binning

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Chia đều theo thang log(odds) thay vì theo quantile của X       │
    │  hay theo event rate.                                            │
    │                                                                  │
    │  log(odds) = ln(bad_rate / good_rate) — đây chính là thứ         │
    │  scorecard thực tế dùng để tính điểm. Bins đều trên thang        │
    │  log-odds → điểm scorecard tăng đều tuyến tính, dễ giải thích    │
    │  với business.                                                   │
    │                                                                  │
    │  Bước: Tính log-odds (Isotonic làm mượt) → chia đều thang        │
    │  log-odds → ánh xạ về X để có cut-points.                        │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:

        # B1: Fit isotonic để ước lượng event rate mượt
        sort_idx = np.argsort(x)
        x_s, y_s = x[sort_idx], y[sort_idx]

        iso    = IsotonicRegression(
            increasing=(self.direction_ == "ascending"),
            out_of_bounds="clip"
        )
        er_smooth = iso.fit_transform(x_s, y_s.astype(float))
        er_smooth = np.clip(er_smooth, EPSILON, 1 - EPSILON)

        # B2: Chuyển sang log-odds
        log_odds = np.log(er_smooth / (1 - er_smooth))

        # B3: Chia đều thang log-odds
        lo_min = log_odds.min()
        lo_max = log_odds.max()
        if abs(lo_max - lo_min) < EPSILON:
            return quantile_cuts(x, self.max_bins - 1)

        lo_cuts = np.linspace(lo_min, lo_max, self.max_bins + 1)[1:-1]

        # B4: Tìm X tương ứng với mỗi ngưỡng log-odds
        cuts = []
        for lo_cut in lo_cuts:
            # Tìm điểm X đầu tiên vượt ngưỡng log-odds
            idx = np.searchsorted(log_odds, lo_cut)
            idx = min(idx, len(x_s) - 1)
            cuts.append(float(x_s[idx]))

        cuts = sorted(set(cuts))
        # Bỏ cuts trùng với min/max
        cuts = [c for c in cuts if x.min() < c < x.max()]
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]
        return cuts, None