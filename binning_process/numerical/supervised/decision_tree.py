"""Decision Tree Binner: cut-points từ cây quyết định (Gini) + monotonic merge."""

from typing import List

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from binning_process.numerical.base import NumericalBaseBinner


class NumericalDecisionTreeBinner(NumericalBaseBinner):
    """
    Decision Tree Binning + Monotonic Merge.
    ...
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf_ratio: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.min_samples_leaf_ratio = min_samples_leaf_ratio
        self.method = "lower"
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
        cuts = sorted({float(t) for t in thresholds if t != -2.0})
        return cuts, None
