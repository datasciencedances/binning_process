from typing import List
import numpy as np
import pandas as pd
from scipy import stats
from binning_process.core.base import BaseBinner
from binning_process.core.utils import quantile_cuts

class ChiMergeBinner(BaseBinner):
    """
    ChiMerge — Chi-square Based Merging

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Bắt đầu với RẤT NHIỀU bins nhỏ (50 bins).                       │
    │                                                                  │
    │  Ở mỗi vòng: tìm 2 bins kề nhau "giống nhau nhất" về tỷ lệ       │
    │  bad/good → gộp chúng lại.                                       │
    │                                                                  │
    │  "Giống nhau" được đo bằng Chi-square test (kiểm định thống kê): │
    │    Chi² nhỏ = 2 bins không khác biệt đáng kể → NÊN GỘP           │
    │    Chi² lớn = 2 bins khác biệt rõ ràng      → GIỮ LẠI            │
    │                                                                  │
    │  Dừng khi đạt số bins mong muốn hoặc tất cả bins đủ khác biệt.   │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 50, confidence_level: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins      = n_init_bins
        self.confidence_level = confidence_level

    @staticmethod
    def _chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
        """
        Chi-square giữa 2 bins (bảng 2x2: bin × event/nonevent).
        Chi² nhỏ → 2 bins phân phối giống nhau → nên gộp.
        """
        observed = np.array([counts_a, counts_b], dtype=float)
        row_s    = observed.sum(axis=1, keepdims=True)
        col_s    = observed.sum(axis=0, keepdims=True)
        total    = observed.sum()
        if total == 0:
            return 0.0
        expected = (row_s * col_s) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0).sum()
        return float(chi2)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # B1: Chia nhiều bins nhỏ (tái sử dụng quantile_cuts từ core.utils)
        init_cuts = quantile_cuts(x, self.n_init_bins)
        edges     = [-np.inf] + init_cuts + [np.inf]
        bin_idx   = pd.cut(x, bins=edges, labels=False, right=False)

        n_bins = len(edges) - 1
        counts = np.zeros((n_bins, 2), dtype=int)
        for i in range(n_bins):
            mask         = bin_idx == i
            counts[i, 1] = int(y[mask].sum())
            counts[i, 0] = int((1 - y[mask]).sum())

        current_cuts  = list(init_cuts)
        chi2_threshold = stats.chi2.ppf(1 - self.confidence_level, df=1)

        # B2: Gộp dần bin giống nhau nhất
        for _ in range(500):
            if len(counts) <= self.max_bins:
                # Kiểm tra nếu tất cả bins đã đủ khác biệt thì dừng
                chi2_vals = [self._chi2(counts[i], counts[i+1])
                              for i in range(len(counts)-1)]
                if chi2_vals and min(chi2_vals) >= chi2_threshold:
                    break

            if len(counts) <= 2:
                break

            chi2_vals = [self._chi2(counts[i], counts[i+1])
                          for i in range(len(counts)-1)]
            if not chi2_vals:
                break

            min_idx  = int(np.argmin(chi2_vals))
            # Gộp bin min_idx và min_idx+1
            counts[min_idx] = counts[min_idx] + counts[min_idx + 1]
            counts          = np.delete(counts, min_idx + 1, axis=0)
            if min_idx < len(current_cuts):
                current_cuts.pop(min_idx)

        return sorted(current_cuts), sorted(init_cuts)
