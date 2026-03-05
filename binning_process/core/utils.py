"""
Utilities dùng chung cho tất cả Binners.
Không import từ supervised/ hay unsupervised/ để tránh circular import.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from binning_process.core.values import EPSILON


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
    labels  = [f"({edges[i]:.4g}, {edges[i+1]:.4g}]" for i in range(len(edges) - 1)]
    bin_idx = pd.cut(x, bins=edges, labels=False, right=True, include_lowest=True)

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


def compute_woe_iv_table_categorical(
    group_ids: np.ndarray,
    y: np.ndarray,
    feature_name: str = "feature",
    bin_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Tính bảng WOE/IV theo nhóm (dùng cho categorical).
    group_ids: mảng cùng độ dài với y; mỗi phần tử là id nhóm (0, 1, 2, ...).
    bin_labels: tên hiển thị từng bin; None thì dùng "Bin_0", "Bin_1", ...
    """
    group_ids = np.asarray(group_ids)
    if group_ids.dtype.kind == "f":
        group_ids = np.where(np.isnan(group_ids), -1, group_ids).astype(int)
    else:
        group_ids = group_ids.astype(int)
    uniq = sorted([u for u in np.unique(group_ids) if u >= 0])

    total_event = max(int(y.sum()), 1)
    total_nonevent = max(int((1 - y).sum()), 1)

    rows = []
    for i, g in enumerate(uniq):
        mask = group_ids == g
        n_event = int(y[mask].sum())
        n_nonevent = int((1 - y[mask]).sum())
        n_total = n_event + n_nonevent
        pct_e = n_event / total_event
        pct_ne = n_nonevent / total_nonevent
        woe = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
        iv_bin = (pct_e - pct_ne) * woe
        lbl = (
            bin_labels[i]
            if bin_labels and i < len(bin_labels)
            else f"Bin_{g}"
        )
        rows.append({
            "feature": feature_name,
            "bin": lbl,
            "n_total": n_total,
            "n_event": n_event,
            "n_nonevent": n_nonevent,
            "event_rate": round(n_event / max(n_total, 1), 4),
            "pct_event": round(pct_e, 4),
            "pct_nonevent": round(pct_ne, 4),
            "woe": round(woe, 4),
            "iv_bin": round(iv_bin, 4),
        })

    df = pd.DataFrame(rows)
    df["iv_total"] = round(df["iv_bin"].sum(), 4)
    return df


def compute_category_woe_table(
    x_cat: np.ndarray,
    y: np.ndarray,
    feature_name: str = "feature",
    smoothing_epsilon: float = 0.5,
) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Bảng WOE/IV theo từng category (1 category = 1 dòng).
    smoothing_epsilon: cộng vào n_event và n_nonevent để tránh WOE vô hạn khi bad=0 hoặc good=0.
    Returns:
        df: columns category, n_total, n_event, n_nonevent, bad_rate, woe, iv_contrib
        zero_warning: list category có n_event=0 hoặc n_nonevent=0 (trước smoothing).
    """
    total_event = max(int(y.sum()), 1)
    total_nonevent = max(int((1 - y).sum()), 1)
    rows = []
    zero_warning = []
    for cat in np.unique(x_cat):
        mask = x_cat == cat
        n_e = int(y[mask].sum())
        n_ne = int((1 - y[mask]).sum())
        n_total = n_e + n_ne
        if n_e == 0 or n_ne == 0:
            zero_warning.append(cat)
        n_e_s = n_e + smoothing_epsilon
        n_ne_s = n_ne + smoothing_epsilon
        pct_e = n_e_s / (total_event + smoothing_epsilon * 2)
        pct_ne = n_ne_s / (total_nonevent + smoothing_epsilon * 2)
        woe = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
        iv_contrib = (pct_e - pct_ne) * woe
        bad_rate = n_e / max(n_total, 1)
        rows.append({
            "feature": feature_name,
            "category": cat,
            "n_total": n_total,
            "n_event": n_e,
            "n_nonevent": n_ne,
            "bad_rate": round(bad_rate, 4),
            "woe": round(woe, 4),
            "iv_contrib": round(iv_contrib, 4),
        })
    df = pd.DataFrame(rows)
    return df, zero_warning


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
    if iv is None:
        return ""
    if iv < 0.02:
        return "Vô dụng"
    if iv < 0.1:
        return "Yếu"
    if iv < 0.3:
        return "Trung bình"
    if iv < 0.5:
        return "Tốt"
    return "Nghi ngờ overfit"


def quantile_cuts(x: np.ndarray, n_bins: int) -> List[float]:
    """Tạo cut-points từ quantile đều nhau."""
    pcts = np.linspace(0, 100, n_bins + 1)[1:-1]
    cuts = sorted(list(np.unique(np.nanpercentile(x, pcts, method='lower'))))
    return cuts


def woe_table_to_pct_per_bin(woe_table: pd.DataFrame) -> np.ndarray:
    """Tỷ lệ mẫu mỗi bin: n_total / sum(n_total). Dùng cho PSI hoặc chart."""
    n_total = woe_table["n_total"].values.astype(float)
    return n_total / max(n_total.sum(), 1.0)


def compute_psi(
    p_train: np.ndarray,
    p_valid: np.ndarray,
    epsilon: float = EPSILON,
) -> Tuple[float, np.ndarray]:
    """
    PSI (Population Stability Index) theo bin.
    PSI_i = (p_valid_i - p_train_i) * ln(p_valid_i / p_train_i).
    Returns:
        (psi_total, psi_per_bin)
    """
    p_train = np.asarray(p_train, dtype=float)
    p_valid = np.asarray(p_valid, dtype=float)
    p_train = np.clip(p_train, epsilon, 1.0)
    p_valid = np.clip(p_valid, epsilon, 1.0)
    psi_bin = (p_valid - p_train) * np.log(p_valid / p_train)
    return float(np.sum(psi_bin)), psi_bin