import numpy as np
from typing import List
from binning_process.core.base import BaseBinner


class EqualWidthBinner(BaseBinner):
    """
    Equal-Width Binning — Chia đều khoảng giá trị (Unsupervised)

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Chia KHOẢNG GIÁ TRỊ (max - min) thành max_bins phần BẰNG NHAU.  │
    │  Ví dụ: income từ 5tr đến 100tr → mỗi bin rộng ~19tr.            │
    │                                                                  │
    │  Đơn giản nhất, dễ giải thích (khoảng đều nhau). Nhược điểm:     │
    │  nếu phân phối lệch (skewed), số mẫu trong từng bin có thể       │
    │  rất chênh lệch; outliers dễ chiếm hết một bin.                  │
    │                                                                  │
    │  Dùng khi: muốn bins có ý nghĩa trực quan (ví dụ thu nhập        │
    │  theo khoảng đều). Không dùng khi phân phối lệch mạnh.           │
    └──────────────────────────────────────────────────────────────────┘
    """

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if abs(x_max - x_min) < 1e-9:
            return []
        # Chia đều khoảng giá trị, bỏ đầu và cuối
        cuts = list(np.linspace(x_min, x_max, self.max_bins + 1)[1:-1])
        return cuts