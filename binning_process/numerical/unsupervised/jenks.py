from typing import List

import numpy as np

from binning_process.numerical.base import NumericalBaseBinner


class NumericalJenksNaturalBreaksBinner(NumericalBaseBinner):
    """Jenks Natural Breaks — Điểm gãy tự nhiên (Unsupervised) ..."""

    def __init__(self, max_sample: int = 2000, **kwargs):
        super().__init__(**kwargs)
        self.max_sample = max_sample

    def _jenks_dp(self, x_sorted: np.ndarray, k: int) -> List[float]:
        n = len(x_sorted)
        if n <= k:
            return list(x_sorted[1:])

        INF = float("inf")
        variance = [[INF] * (k + 1) for _ in range(n + 1)]
        backtrack = [[0]  * (k + 1) for _ in range(n + 1)]

        prefix_sum  = np.zeros(n + 1)
        prefix_sum2 = np.zeros(n + 1)
        for i in range(n):
            prefix_sum[i + 1]  = prefix_sum[i]  + x_sorted[i]
            prefix_sum2[i + 1] = prefix_sum2[i] + x_sorted[i] ** 2

        def ssw(l, r):
            count  = r - l
            if count <= 0:
                return 0.0
            s  = prefix_sum[r]  - prefix_sum[l]
            s2 = prefix_sum2[r] - prefix_sum2[l]
            return s2 - s * s / count

        for i in range(1, n + 1):
            variance[i][1]  = ssw(0, i)
            backtrack[i][1] = 1

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

        cuts_idx = []
        current  = n
        for j in range(k, 1, -1):
            start = backtrack[current][j]
            if start > 1:
                cuts_idx.append(start - 1)
            current = start - 1

        cuts_idx = sorted(set(cuts_idx))
        cuts = [float((x_sorted[i - 1] + x_sorted[i]) / 2)
                for i in cuts_idx if 0 < i < n]
        return cuts

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(x) > self.max_sample:
            idx      = np.random.choice(len(x), self.max_sample, replace=False)
            x_sample = x[idx]
        else:
            x_sample = x

        x_sorted = np.sort(x_sample)
        cuts     = self._jenks_dp(x_sorted, self.max_bins)

        cuts = [c for c in cuts if x.min() < c < x.max()]
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]
        return sorted(cuts)
