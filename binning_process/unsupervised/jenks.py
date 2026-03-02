import numpy as np
from typing import List
from binning_process.core.base import BaseBinner


class JenksNaturalBreaksBinner(BaseBinner):
    """
    Jenks Natural Breaks — Điểm gãy tự nhiên (Unsupervised)

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Tìm điểm "gãy tự nhiên" trong phân phối của X: tối thiểu hóa    │
    │  độ phân tán TRONG từng bin, tối đa hóa độ khác biệt GIỮA        │
    │  các bins.                                                       │
    │                                                                  │
    │  Ví dụ: income [5, 6, 7, 20, 21, 22, 80, 81] → có 3 cụm tự       │
    │  nhiên [5-7], [20-22], [80-81]. Jenks tìm ra cut tại 7→20 và     │
    │  22→80, không phụ thuộc vào target y.                            │
    │                                                                  │
    │  Dùng khi: muốn bins phản ánh cấu trúc/nhóm tự nhiên của dữ      │
    │  liệu. Thuật toán: Fisher-Jenks (DP); với N lớn có thể sample.   │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, max_sample: int = 2000, **kwargs):
        super().__init__(**kwargs)
        self.max_sample = max_sample

    def _jenks_dp(self, x_sorted: np.ndarray, k: int) -> List[float]:
        """
        Fisher-Jenks Dynamic Programming.
        x_sorted: mảng đã sort.
        k: số bins mong muốn.
        Trả về k-1 cut-points.
        """
        n = len(x_sorted)
        if n <= k:
            return list(x_sorted[1:])

        # lower_class_limits[i][j]: điểm bắt đầu tốt nhất của bin j
        # kết thúc tại i
        INF = float("inf")

        # variance_combinations[i][j]: SSW khi chia i quan sát thành j bins
        variance = [[INF] * (k + 1) for _ in range(n + 1)]
        backtrack = [[0]  * (k + 1) for _ in range(n + 1)]

        # SSW của đoạn [l, r]
        # Dùng prefix sum để tính nhanh
        prefix_sum  = np.zeros(n + 1)
        prefix_sum2 = np.zeros(n + 1)
        for i in range(n):
            prefix_sum[i + 1]  = prefix_sum[i]  + x_sorted[i]
            prefix_sum2[i + 1] = prefix_sum2[i] + x_sorted[i] ** 2

        def ssw(l, r):
            """Sum of squared within-class deviations, đoạn [l, r) (0-indexed)."""
            count  = r - l
            if count <= 0:
                return 0.0
            s  = prefix_sum[r]  - prefix_sum[l]
            s2 = prefix_sum2[r] - prefix_sum2[l]
            return s2 - s * s / count

        # Base: 1 bin
        for i in range(1, n + 1):
            variance[i][1]  = ssw(0, i)
            backtrack[i][1] = 1

        # DP
        for j in range(2, k + 1):
            for i in range(j, n + 1):
                best_cost  = INF
                best_start = j
                for m in range(j - 1, i):
                    cost = variance[m][j - 1] + ssw(m, i)
                    if cost < best_cost:
                        best_cost  = cost
                        best_start = m + 1
                variance[i][j]  = best_cost
                backtrack[i][j] = best_start

        # Truy vết để tìm cut-points
        cuts_idx = []
        current  = n
        for j in range(k, 1, -1):
            start = backtrack[current][j]
            if start > 1:
                cuts_idx.append(start - 1)   # index trong x_sorted
            current = start - 1

        cuts_idx = sorted(set(cuts_idx))
        cuts = [float((x_sorted[i - 1] + x_sorted[i]) / 2)
                for i in cuts_idx if 0 < i < n]
        return cuts

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # Sample nếu N lớn (DP là O(n²k), chậm với n lớn)
        if len(x) > self.max_sample:
            idx      = np.random.choice(len(x), self.max_sample, replace=False)
            x_sample = x[idx]
        else:
            x_sample = x

        x_sorted = np.sort(x_sample)
        cuts     = self._jenks_dp(x_sorted, self.max_bins)

        # Bỏ cuts nằm ngoài khoảng thực tế của x
        cuts = [c for c in cuts if x.min() < c < x.max()]
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]
        return sorted(cuts)