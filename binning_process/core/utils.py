"""
Utilities dùng chung cho tất cả Binners.
Không import từ supervised/ hay unsupervised/ để tránh circular import.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List

EPSILON = 1e-9


def detect_direction(x: np.ndarray, y: np.ndarray) -> str:
    """
    Tự động phát hiện chiều monotonic bằng Spearman Correlation.
    - corr > 0 → ascending  (X tăng → risk tăng, vd: số ngày quá hạn)
    - corr < 0 → descending (X tăng → risk giảm, vd: thu nhập)
    """
    corr, _ = stats.spearmanr(x, y)
    return "ascending" if corr >= 0 else "descending"


def compute_woe_iv_table(cuts: list, x: np.ndarray, y: np.ndarray,
                          feature_name: str = "feature") -> pd.DataFrame:
    """
    Tính bảng WOE/IV từ danh sách cut-points.

    WOE_i = ln( %Event_i / %NonEvent_i )
    IV    = Σ (%Event_i - %NonEvent_i) * WOE_i
    """
    edges   = [-np.inf] + sorted(cuts) + [np.inf]
    labels  = [f"[{edges[i]:.4g}, {edges[i+1]:.4g})" for i in range(len(edges) - 1)]
    bin_idx = pd.cut(x, bins=edges, labels=False, right=False)

    total_event    = max(int(y.sum()), 1)
    total_nonevent = max(int((1 - y).sum()), 1)

    rows = []
    for i, lbl in enumerate(labels):
        mask       = bin_idx == i
        n_event    = int(y[mask].sum())
        n_nonevent = int((1 - y[mask]).sum())
        n_total    = n_event + n_nonevent
        pct_e      = n_event    / total_event
        pct_ne     = n_nonevent / total_nonevent
        woe        = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
        iv_bin     = (pct_e - pct_ne) * woe
        rows.append({
            "feature"      : feature_name,
            "bin"          : lbl,
            "n_total"      : n_total,
            "n_event"      : n_event,
            "n_nonevent"   : n_nonevent,
            "event_rate"   : round(n_event / max(n_total, 1), 4),
            "pct_event"    : round(pct_e, 4),
            "pct_nonevent" : round(pct_ne, 4),
            "woe"          : round(woe, 4),
            "iv_bin"       : round(iv_bin, 4),
        })

    df = pd.DataFrame(rows)
    df["iv_total"] = round(df["iv_bin"].sum(), 4)
    return df


def cap_outliers(series: pd.Series, lower_pct: float = 1,
                 upper_pct: float = 99) -> pd.Series:
    """Winsorize: kéo outlier về ngưỡng [P1, P99]."""
    lo = np.nanpercentile(series.dropna(), lower_pct)
    hi = np.nanpercentile(series.dropna(), upper_pct)
    return series.clip(lower=lo, upper=hi)


def is_monotonic_series(series: pd.Series, direction: str,
                        tol: float = 1e-6) -> bool:
    diffs = series.diff().dropna()
    if direction == "ascending":
        return bool((diffs >= -tol).all())
    return bool((diffs <= tol).all())


def iv_rating(iv: float) -> str:
    if iv is None: return ""
    if iv < 0.02:  return "Vô dụng"
    if iv < 0.1:   return "Yếu"
    if iv < 0.3:   return "Trung bình"
    if iv < 0.5:   return "Tốt"
    return "Nghi ngờ overfit"


def quantile_cuts(x: np.ndarray, n_bins: int) -> List[float]:
    """Tạo cut-points từ quantile đều nhau."""
    pcts = np.linspace(0, 100, n_bins + 1)[1:-1]
    return sorted(list(np.unique(np.nanpercentile(x, pcts))))