from typing import List
import numpy as np
from sklearn.cluster import KMeans
from binning_process.numerical.base import NumericalBaseBinner



class NumericalKMeansBinner(NumericalBaseBinner):
    """
    KMeans Binning — tìm cụm tự nhiên trong x,
    đặt cut tại midpoint giữa 2 centroids kề nhau.

    Unsupervised: không dùng y để tìm cuts,
    nhưng WOE/IV vẫn tính từ y sau khi có bins.
    """

    def __init__(self, n_init_bins: int = 20, random_state: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state
        self.n_clusters = 11

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(set(x)) < self.n_clusters:
            return sorted(set(x)), None
        km = KMeans(
            n_clusters    = self.n_clusters,
            random_state  = self.random_state,
            n_init        = "auto",
        )
        km.fit(x.reshape(-1, 1))

        centroids = sorted(km.cluster_centers_.flatten())
        cuts = [
            (centroids[i] + centroids[i + 1]) / 2
            for i in range(len(centroids) - 1)
        ]
        return sorted(cuts), None
