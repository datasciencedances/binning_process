import numpy as np
import pandas as pd
from scipy import stats
from typing import List
from binning_process.core.base import BaseBinner
from binning_process.core.utils import quantile_cuts


class SpearmanBinner(BaseBinner):
    """
    Spearman Binning — Tối đa hoá Spearman Correlation

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Bắt đầu từ nhiều bins nhỏ. Lần lượt thử xóa từng cut-point:     │
    │  cut nào khi xóa làm độ tương quan Spearman (bin ↔ event rate)   │
    │  giảm ít nhất (hoặc tăng) → được giữ lại. Lặp đến khi còn        │
    │  đủ max_bins.                                                    │
    │                                                                  │
    │  Spearman càng gần 1 hoặc -1 = thứ tự bins càng "đơn điệu"       │
    │  với event rate. Khác với enforce_monotonic (sửa vi phạm) —      │
    │  Spearman Binner tối ưu trực tiếp mức độ đơn điệu của binning.   │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _spearman_score(self, cuts: list, x: np.ndarray,
                        y: np.ndarray) -> float:
        """Tính |Spearman| giữa bin index và event rate."""
        if not cuts:
            return 0.0
        edges   = [-np.inf] + sorted(cuts) + [np.inf]
        bin_idx = pd.cut(x, bins=edges, labels=False, right=False)
        er_list, idx_list = [], []
        for i in range(len(edges) - 1):
            mask = bin_idx == i
            if mask.sum() == 0:
                continue
            er_list.append(y[mask].mean())
            idx_list.append(i)
        if len(er_list) < 2:
            return 0.0
        corr, _ = stats.spearmanr(idx_list, er_list)
        return abs(corr) if not np.isnan(corr) else 0.0

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        init_cuts = quantile_cuts(x, self.n_init_bins)
        cuts = list(sorted(init_cuts))
        # Greedy: xóa cut nào gây mất ít Spearman nhất
        while len(cuts) >= self.max_bins:
            best_score = -np.inf
            best_idx   = 0

            for i in range(len(cuts)):
                candidate = cuts[:i] + cuts[i + 1:]
                score     = self._spearman_score(candidate, x, y)
                if score > best_score:
                    best_score = score
                    best_idx   = i

            cuts = cuts[:best_idx] + cuts[best_idx + 1:]

        return cuts, sorted(init_cuts)