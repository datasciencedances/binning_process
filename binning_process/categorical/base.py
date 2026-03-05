"""
CategoricalBaseBinner: base class cho mọi Binner categorical (supervised).
Subclass implement _find_bin_groups(x_cat, y) -> (category_to_bin_id, bin_labels).
Có hỗ trợ: preprocessing (Missing/Special), transform (category -> WOE), summary, plot.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from binning_process.core.utils import compute_woe_iv_table_categorical
from binning_process.core.values import EPSILON

warnings.filterwarnings("ignore")


def _chi2_two_bins(counts_a: Tuple[int, int], counts_b: Tuple[int, int]) -> float:
    """Chi-square giữa 2 bins (event, nonevent)."""
    n_e_a, n_ne_a = counts_a
    n_e_b, n_ne_b = counts_b
    observed = np.array([[n_e_a, n_ne_a], [n_e_b, n_ne_b]], dtype=float)
    row_s = observed.sum(axis=1, keepdims=True)
    col_s = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return 0.0
    expected = (row_s * col_s) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(
            expected > 0, (observed - expected) ** 2 / expected, 0.0
        ).sum()
    return float(chi2)


class CategoricalBaseBinner(ABC, BaseEstimator, TransformerMixin):
    """
    Base class cho binning biến categorical.
    Subclass phải implement _find_bin_groups(x_cat, y) -> (Dict[category, bin_id], List[bin_label]).

    Public attributes sau fit():
        final_woe_table_: pd.DataFrame — bảng WOE/IV theo bin.
        final_iv_: float — Information Value.
        final_bin_map_: Dict[Any, int] — category -> bin_id.
        final_cuts_: None — tương thích report (numerical dùng final_cuts_).
        direction_: str — "ascending" | "descending".
        special_table_: pd.DataFrame | None — WOE cho Missing/Special.
        category_table_: pd.DataFrame | None — (tuỳ subclass) bảng category-level n, bad, bad_rate, woe, iv_contrib.
        smoothing_warnings_: List — category có bad=0 hoặc good=0 (cần smoothing).
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
    ):
        self.feature_name = feature_name
        self.min_bin_size = min_bin_size
        self.min_event_count = min_event_count
        self.max_bins = max_bins
        self.special_values = special_values or []
        self.unseen_woe = unseen_woe
        self.direction = direction

        self.final_cuts_: Optional[List[float]] = None
        self.final_woe_table_: Optional[pd.DataFrame] = None
        self.final_iv_: float = 0.0
        self.final_bin_map_: Dict[Any, int] = {}
        self._bin_id_to_woe: Dict[int, float] = {}
        self.direction_: str = direction
        self.special_table_: Optional[pd.DataFrame] = None
        self.category_table_: Optional[pd.DataFrame] = None
        self.smoothing_warnings_: List[Any] = []

    def _preprocess(
        self, x: pd.Series, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (x_cat_main, y_main). Đã set self.special_table_."""
        mask_miss = x.isna()
        mask_special = x.isin(self.special_values) if self.special_values else pd.Series(False, index=x.index)
        mask_main = ~mask_miss & ~mask_special

        x_cat = x.astype(object)
        x_main = x_cat[mask_main].values
        y_main = y[mask_main].values.astype(int)

        total_event = max(int(y.sum()), 1)
        total_nonevent = max(int((1 - y).sum()), 1)
        special_rows = []

        for label, mask in [(("Missing"), mask_miss)] + [
            (f"Special={sv}", x == sv) for sv in self.special_values
        ]:
            if mask.sum() == 0:
                continue
            y_grp = y[mask]
            n_event = int(y_grp.sum())
            n_nonevent = int((1 - y_grp).sum())
            pct_e = n_event / total_event
            pct_ne = n_nonevent / total_nonevent
            woe = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
            special_rows.append({
                "feature": self.feature_name,
                "bin": label,
                "n_total": int(mask.sum()),
                "n_event": n_event,
                "n_nonevent": n_nonevent,
                "event_rate": round(n_event / max(int(mask.sum()), 1), 4),
                "pct_event": round(pct_e, 4),
                "pct_nonevent": round(pct_ne, 4),
                "woe": round(woe, 4),
                "iv_bin": round((pct_e - pct_ne) * woe, 4),
            })

        self.special_table_ = pd.DataFrame(special_rows) if special_rows else None
        return x_main, y_main

    @abstractmethod
    def _find_bin_groups(
        self, x_cat: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[Any, int], List[str]]:
        """
        Trả về (category -> bin_id, bin_labels cho từng bin_id).
        Subclass implement theo từng phương pháp (WOE+merge, ChiMerge, DecisionTree).
        """
        pass

    def fit(self, x: pd.Series, y: pd.Series) -> "CategoricalBaseBinner":
        x_main, y_main = self._preprocess(x, y)
        if len(x_main) == 0:
            self.final_woe_table_ = pd.DataFrame()
            self.final_iv_ = 0.0
            self.final_bin_map_ = {}
            self._bin_id_to_woe = {}
            return self

        self.final_bin_map_, bin_labels = self._find_bin_groups(x_main, y_main)
        group_ids = np.array([self.final_bin_map_.get(c, -1) for c in x_main])
        valid = group_ids >= 0
        self.final_woe_table_ = compute_woe_iv_table_categorical(
            group_ids[valid], y_main[valid], self.feature_name, bin_labels
        )
        self.final_iv_ = float(self.final_woe_table_["iv_bin"].sum())
        self._bin_id_to_woe = dict(
            zip(
                range(len(self.final_woe_table_)),
                self.final_woe_table_["woe"].values,
            )
        )
        return self

    def transform(self, x: pd.Series) -> pd.Series:
        if self.final_woe_table_ is None:
            raise RuntimeError("Chưa fit. Gọi .fit() trước.")

        result = x.map(
            lambda v: self._bin_id_to_woe.get(
                self.final_bin_map_.get(v, -1), self.unseen_woe
            )
            if not pd.isna(v)
            and (not self.special_values or v not in self.special_values)
            else None
        )
        result = result.astype(float)

        if self.special_table_ is not None:
            miss_row = self.special_table_[self.special_table_["bin"] == "Missing"]
            if not miss_row.empty:
                result[x.isna()] = miss_row["woe"].values[0]
            for sv in self.special_values:
                spec_row = self.special_table_[
                    self.special_table_["bin"] == f"Special={sv}"
                ]
                if not spec_row.empty:
                    result[x == sv] = spec_row["woe"].values[0]
                else:
                    result[x == sv] = self.unseen_woe

        result = result.fillna(self.unseen_woe)
        return result

    def fit_transform(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return self.fit(x, y).transform(x)

    def summary(self) -> pd.DataFrame:
        if self.final_woe_table_ is None:
            raise RuntimeError("Chưa fit.")
        cols = [
            "feature",
            "bin",
            "n_total",
            "n_event",
            "n_nonevent",
            "event_rate",
            "woe",
            "iv_bin",
            "iv_total",
        ]
        df = self.final_woe_table_.copy()
        df["iv_total"] = round(self.final_iv_, 4)
        main = df[cols]
        if self.special_table_ is not None:
            spec_cols = [c for c in cols if c in self.special_table_.columns]
            return pd.concat(
                [main, self.special_table_[spec_cols]], ignore_index=True
            )
        return main

    def is_monotonic(self) -> bool:
        if self.final_woe_table_ is None or len(self.final_woe_table_) < 2:
            return True
        woe = self.final_woe_table_["woe"].values
        if self.direction_ == "ascending":
            return bool(np.all(np.diff(woe) >= -1e-6))
        return bool(np.all(np.diff(woe) <= 1e-6))

    def _draw_woe_axes(self, ax, df: pd.DataFrame, label: str) -> None:
        x_pos = list(range(len(df)))
        colors = ["#e74c3c" if w >= 0 else "#27ae60" for w in df["woe"]]
        ax.bar(x_pos, df["woe"], color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        iv_val = (
            df["iv_total"].values[0]
            if "iv_total" in df.columns
            else 0.0
        )
        ax.set_title(
            f"WOE theo {label} Bin | IV = {iv_val:.4f}", fontweight="bold"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("WOE")
        for i, (w, n) in enumerate(zip(df["woe"], df["n_total"])):
            offset = 0.03 if w >= 0 else -0.06
            ax.text(i, w + offset, f"{w:.3f}\nn={n}", ha="center", fontsize=7.5)

    def _draw_event_rate_axes(self, ax, df: pd.DataFrame, label: str) -> None:
        arrow = "↑" if self.direction_ == "ascending" else "↓"
        x_pos = list(range(len(df)))
        er_pct = df["event_rate"] * 100
        ax.plot(
            x_pos,
            er_pct,
            marker="o",
            color="#2980b9",
            linewidth=2,
            markersize=8,
            zorder=3,
        )
        ax.fill_between(x_pos, er_pct, alpha=0.12, color="#2980b9")
        ax.set_title(
            f"Event Rate (%) theo {label} Bin  {arrow}", fontweight="bold"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Event Rate (%)")
        for i, er in enumerate(er_pct):
            ax.text(i, er + 0.3, f"{er:.1f}%", ha="center", fontsize=8)

    def plot(self, figsize=(14, 4)):
        if self.final_woe_table_ is None:
            raise RuntimeError("Chưa fit.")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._draw_woe_axes(axes[0], self.final_woe_table_, "Final")
        self._draw_event_rate_axes(axes[1], self.final_woe_table_, "Final")
        arrow = "↑" if self.direction_ == "ascending" else "↓"
        fig.suptitle(
            f"{self.__class__.__name__}  |  {self.feature_name}  |  "
            f"IV = {self.final_iv_:.4f}  |  Monotonic: {self.is_monotonic()}  {arrow}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig
