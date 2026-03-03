from typing import List
import numpy as np
import pandas as pd
from binning_process.core.base import BaseBinner
from binning_process.core.utils import quantile_cuts


class QuantileMonotonicBinner(BaseBinner):
    """
    Quantile Binning + Adjacent Merge

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Bước 1: Chia khách hàng thành N nhóm CÓ SỐ LƯỢNG BẰNG NHAU      │
    │          (như chia 100 người thành 10 nhóm, mỗi nhóm 10 người)   │
    │                                                                  │
    │  Bước 2: Xem tỷ lệ bad từng nhóm. Nếu nhóm nào "lồi lõm"         │
    │          (đi ngược xu hướng chung) → gộp nó với nhóm kề bên.     │
    │                                                                  │
    │  Lặp đến khi các nhóm đều đi một chiều.                          │
    │                                                                  │
    │  Đây là phương pháp đơn giản nhất, dễ giải thích nhất.           │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # B1: Quantile (tái sử dụng quantile_cuts từ core.utils)
        init_cuts = quantile_cuts(x, self.n_init_bins)

        # Giới hạn max bins sơ bộ
        if len(init_cuts) > self.max_bins * 2:
            step = max(1, len(init_cuts) // (self.max_bins * 2))
            init_cuts = init_cuts[::step]

        # B2: Gộp bins quá nhỏ trước khi enforce monotonic
        cuts = self._merge_small_bins(init_cuts, x, y)

        # B3: Giới hạn số bins theo max_bins (n_bins = len(cuts) + 1)
        # Lấy đều các cut theo chỉ số để giữ phân bố cân bằng (tránh bin cuối quá rộng)
        if len(cuts) >= self.max_bins:
            sc = sorted(cuts)
            n_keep = self.max_bins - 1
            indices = np.linspace(0, len(sc) - 1, n_keep, dtype=int)
            cuts = [sc[i] for i in indices]
        return cuts, sorted(init_cuts)

    def _merge_small_bins(self, cuts: list, x: np.ndarray, y: np.ndarray) -> list:
        """Gộp bins < min_bin_size hoặc < min_event_count."""
        n_total = len(x)
        min_n   = max(int(self.min_bin_size * n_total), self.min_event_count)

        for _ in range(100):
            if not cuts:
                break
            edges   = [-np.inf] + sorted(cuts) + [np.inf]
            bin_idx = pd.cut(x, bins=edges, labels=False, right=False)
            found   = False

            for i in range(len(edges) - 1):
                mask    = bin_idx == i
                n       = int(mask.sum())
                n_event = int(y[mask].sum())

                if n < min_n or n_event < self.min_event_count:
                    sc = sorted(cuts)
                    if i == 0:
                        cuts = sc[1:]
                    elif i == len(edges) - 2:
                        cuts = sc[:-1]
                    elif i - 1 < len(sc):
                        cuts = sc[:i - 1] + sc[i:]
                    found = True
                    break

            if not found:
                break
        return cuts