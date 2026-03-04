"""Decision Tree Binner: cut-points từ cây quyết định (Gini) + monotonic merge."""

from typing import List

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from binning_process.core.base import BaseBinner


class DecisionTreeBinner(BaseBinner):
    """
    Decision Tree Binning + Monotonic Merge.

    Dùng cây quyết định nông (Gini) để chọn cut-points tối ưu theo bad/good,
    sau đó enforce monotonic bằng gộp bins.

    Parameters:
        max_depth: Độ sâu tối đa của cây (mặc định 4).
        min_samples_leaf_ratio: Tỷ lệ mẫu tối thiểu mỗi lá (mặc định 0.05).
        **kwargs: Truyền xuống BaseBinner (feature_name, max_bins, ...).

    Example:
        >>> b = DecisionTreeBinner(max_bins=6, max_depth=4)
        >>> b.fit(x_series, y_series)
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf_ratio: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.min_samples_leaf_ratio = min_samples_leaf_ratio

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        min_leaf = max(
            int(self.min_samples_leaf_ratio * len(x)),
            self.min_event_count * 2,
        )
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=min_leaf,
            criterion="gini",
        )
        tree.fit(x.reshape(-1, 1), y)
        thresholds = tree.tree_.threshold
        cuts = []
        for t in thresholds:
            if t == -2.0:
                continue
            t = float(t)
            left_side = x[x <= t]
            if len(left_side) > 0:
                # Giá trị thật của x gần bên trái threshold nhất = max(x <= t)
                cuts.append(float(np.max(left_side)))
        cuts = sorted(set(cuts))
        return cuts, None
