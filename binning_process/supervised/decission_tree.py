from typing import List
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from binning_process.core.base import BaseBinner

class DecisionTreeBinner(BaseBinner):
    """
    Decision Tree Binning + Monotonic Merge

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                              │
    │                                                                  │
    │  Cây quyết định tự hỏi: "Tại ngưỡng nào thì chia khách hàng      │
    │  thành 2 nhóm có tỷ lệ bad KHÁC NHAU NHẤT?"                      │
    │                                                                  │
    │  Ví dụ: "Nếu income < 15tr → 38% bad. Nếu ≥ 15tr → 7% bad"       │
    │  → 15tr là cut-point tốt nhất theo tiêu chí Gini                 │
    │                                                                  │
    │  Cây tiếp tục hỏi tương tự trong từng nhánh, tạo ra nhiều        │
    │  cut-points. Sau đó ta enforce monotonic bằng gộp bins.          │ 
    │                                                                  │
    │  Ưu điểm: cut-points được chọn tối ưu theo bad/good signal,      │
    │  không phụ thuộc phân phối của X.                                │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf_ratio: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_depth               = max_depth
        self.min_samples_leaf_ratio  = min_samples_leaf_ratio

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        min_leaf = max(
            int(self.min_samples_leaf_ratio * len(x)),
            self.min_event_count * 2
        )

        tree = DecisionTreeClassifier(
            max_depth        = self.max_depth,
            min_samples_leaf = min_leaf,
            criterion        = "gini",
        )
        tree.fit(x.reshape(-1, 1), y)

        # Lấy tất cả threshold mà cây dùng để split
        thresholds = tree.tree_.threshold
        cuts = sorted({float(t) for t in thresholds if t != -2.0})

        # Giới hạn bins: giữ cuts có lv thấp (gần root = quan trọng hơn)
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]

        return cuts, None
