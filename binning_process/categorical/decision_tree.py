"""
CategoricalDecisionTreeBinner (C): target-guided grouping bằng Decision Tree.
One-hot encode category rồi fit cây quyết định; mỗi leaf = 1 bin.
Cần regularization: max_depth, min_samples_leaf để tránh overfit.
"""

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from binning_process.categorical.base import CategoricalBaseBinner

class CategoricalDecisionTreeBinner(CategoricalBaseBinner):
    """
    Decision Tree grouping: one-hot encode categories -> fit tree -> leaf = bin.
    Parameters: max_depth, min_samples_leaf (hoặc min_samples_leaf_ratio) để hạn chế overfit.
    """

    def __init__(
        self,
        feature_name: str = "feature",
        min_bin_size: float = 0.05,
        min_event_count: int = 5,
        max_bins: int = 8,
        special_values: Optional[List] = None,
        unseen_woe: float = 0.0,
        direction: str = "auto",
        max_depth: int = 4,
        min_samples_leaf_ratio: float = 0.05,
    ):
        super().__init__(
            feature_name=feature_name,
            min_bin_size=min_bin_size,
            min_event_count=min_event_count,
            max_bins=max_bins,
            special_values=special_values,
            unseen_woe=unseen_woe,
            direction=direction,
        )
        self.max_depth = max_depth
        self.min_samples_leaf_ratio = min_samples_leaf_ratio

    def _find_bin_groups(
        self, x_cat: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[Any, int], List[str]]:
        from typing import Optional

        uniq = np.unique(x_cat)
        n_cats = len(uniq)
        if n_cats == 0:
            return {}, []
        cat_to_idx = {c: i for i, c in enumerate(uniq)}

        # One-hot: (n_samples, n_cats)
        X = np.zeros((len(x_cat), n_cats), dtype=np.float64)
        for i, c in enumerate(x_cat):
            X[i, cat_to_idx[c]] = 1.0

        min_leaf = max(
            int(self.min_samples_leaf_ratio * len(y)),
            self.min_event_count * 2,
        )
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=min_leaf,
            criterion="gini",
        )
        tree.fit(X, y)
        leaves = tree.apply(X)  # (n_samples,) leaf_id

        # category -> leaf_id (lấy từ một mẫu bất kỳ có category đó)
        cat_to_leaf: Dict[Any, int] = {}
        for c in uniq:
            idx = np.where(x_cat == c)[0][0]
            cat_to_leaf[c] = int(leaves[idx])

        unique_leaves = sorted(np.unique(leaves))
        leaf_to_bin = {L: i for i, L in enumerate(unique_leaves)}
        # Thứ tự bin theo event rate trung bình trong leaf (để direction_ nhất quán)
        er_per_leaf = []
        for L in unique_leaves:
            mask = leaves == L
            er_per_leaf.append((L, float(y[mask].mean())))
        if self.direction == "descending":
            er_per_leaf.sort(key=lambda t: -t[1])
        else:
            er_per_leaf.sort(key=lambda t: t[1])
        self.direction_ = (
            self.direction if self.direction != "auto" else "ascending"
        )
        # Remap bin_id theo thứ tự event rate
        leaf_to_bin = {L: i for i, (L, _) in enumerate(er_per_leaf)}

        cat_to_bin = {c: leaf_to_bin[cat_to_leaf[c]] for c in uniq}

        # bin_labels: với mỗi bin_id, liệt kê category
        bin_id_to_cats: Dict[int, List[Any]] = {}
        for c, bid in cat_to_bin.items():
            bin_id_to_cats.setdefault(bid, []).append(c)
        bin_labels = [
            ",".join(str(c) for c in sorted(bin_id_to_cats.get(i, []), key=str))
            for i in range(len(unique_leaves))
        ]

        return cat_to_bin, bin_labels
