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
        init_cuts = quantile_cuts(x, self.n_init_bins)
        cuts = list(sorted(init_cuts))
        return cuts, sorted(init_cuts)