
"""
BaseBinner: base class cho mọi Binner (supervised/unsupervised).
Preprocessing, fit, transform, summary, plot. Subclass chỉ cần implement _find_cuts().
"""

import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from binning_process.core.merge_process import MergeTrace, enforce_monotonic_traced
from binning_process.core.utils import (
    cap_outliers,
    compute_woe_iv_table,
    detect_direction,
    is_monotonic_series,
)
from binning_process.core.values import EPSILON

warnings.filterwarnings("ignore")


class BaseBinner(BaseEstimator, TransformerMixin):
    """
    Base class: preprocessing, fit/transform/summary/plot.
    Subclass chỉ cần implement _find_cuts(x, y) → (algo_cuts, init_cuts) hoặc (algo_cuts, None).

    Public attributes sau fit():
        final_cuts_: List[float] — cut-points cuối (sau monotonic merge).
        final_woe_table_: pd.DataFrame — bảng WOE/IV theo bin (final).
        final_iv_: float — Information Value cuối.
        trace_: MergeTrace | None — lịch sử merge (để plot_steps).
        direction_: str — "ascending" | "descending".
        init_cuts_, algo_cuts_: List | None — cuts ban đầu / sau algo (nếu có).
        init_woe_table_, algo_woe_table_: pd.DataFrame | None — bảng WOE tương ứng (nếu có).
        special_table_: pd.DataFrame | None — WOE cho Missing/Special (nếu có).
    """

    def __init__(
        self,
        feature_name    : str            = "feature",
        min_bin_size    : float          = 0.05,
        min_event_count : int            = 5,
        max_bins        : int            = 8,
        special_values  : Optional[List] = None,
        cap_outliers_   : bool           = True,
        lower_pct       : float          = 1.0,
        upper_pct       : float          = 99.0,
        direction       : str            = "auto",
    ):
        self.feature_name    = feature_name
        self.min_bin_size    = min_bin_size
        self.min_event_count = min_event_count
        self.max_bins        = max_bins
        self.special_values  = special_values or []
        self.cap_outliers_   = cap_outliers_
        self.lower_pct       = lower_pct
        self.upper_pct       = upper_pct
        self.direction       = direction

        # Sau fit
        self.init_cuts_     : List[float]            = []
        self.algo_cuts_     : List[float]            = []
        self.final_cuts_    : List[float]            = []
        self.trace_         : Optional[MergeTrace]   = None
        self.direction_     : str                    = direction
        self.init_woe_table_ : Optional[pd.DataFrame] = None
        self.algo_woe_table_ : Optional[pd.DataFrame] = None
        self.final_woe_table_ : Optional[pd.DataFrame] = None
        self.final_iv_      : float                  = 0.0
        self.algo_iv_       : float                  = 0.0
        self.init_iv_       : float                  = 0.0
        self.special_table_ : Optional[pd.DataFrame] = None

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, x: pd.Series, y: pd.Series):
        x_proc = cap_outliers(x, self.lower_pct, self.upper_pct) \
                 if self.cap_outliers_ else x.copy()

        mask_miss    = x.isna()
        mask_special = x_proc.isin(self.special_values) \
                       if self.special_values else pd.Series(False, index=x.index)
        mask_main    = ~mask_miss & ~mask_special

        x_main = x_proc[mask_main].values.astype(float)
        y_main = y[mask_main].values.astype(int)

        total_event    = max(int(y.sum()), 1)
        total_nonevent = max(int((1 - y).sum()), 1)
        special_rows   = []

        groups = [(("Missing"), mask_miss)] + \
                 [(f"Special={sv}", x == sv) for sv in self.special_values]

        for label, mask in groups:
            if mask.sum() == 0:
                continue
            y_grp      = y[mask]
            n_event    = int(y_grp.sum())
            n_nonevent = int((1 - y_grp).sum())
            pct_e      = n_event    / total_event
            pct_ne     = n_nonevent / total_nonevent
            woe        = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
            special_rows.append({
                "feature"      : self.feature_name,
                "bin"          : label,
                "n_total"      : int(mask.sum()),
                "n_event"      : n_event,
                "n_nonevent"   : n_nonevent,
                "event_rate"   : round(n_event / max(int(mask.sum()), 1), 4),
                "pct_event"    : round(pct_e, 4),
                "pct_nonevent" : round(pct_ne, 4),
                "woe"          : round(woe, 4),
                "iv_bin"       : round((pct_e - pct_ne) * woe, 4),
            })

        self.special_table_ = pd.DataFrame(special_rows) if special_rows else None
        return x_main, y_main

    # ── Abstract ───────────────────────────────────────────────────────────

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        raise NotImplementedError("Subclass phải implement _find_cuts()")

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, x: pd.Series, y: pd.Series):
        x_main, y_main = self._preprocess(x, y)

        if self.direction == "auto":
            self.direction_ = detect_direction(x_main, y_main)
        else:
            self.direction_ = self.direction

        self.algo_cuts_, self.init_cuts_ = self._find_cuts(x_main, y_main)
        self.trace_, self.final_cuts_  = enforce_monotonic_traced(
            self.algo_cuts_, x_main, y_main, self.direction_,
            feature_name=self.feature_name, verbose=False
        )

        if self.init_cuts_ is not None:
            self.init_woe_table_ = compute_woe_iv_table(
                self.init_cuts_, x_main, y_main, self.feature_name
            )
            self.init_iv_ = self.init_woe_table_["iv_bin"].sum()
        else:
            self.init_woe_table_ = None
            self.init_iv_ = 0.0
    
        self.algo_woe_table_ = compute_woe_iv_table(
                self.algo_cuts_, x_main, y_main, self.feature_name
            )
        self.algo_iv_ = self.algo_woe_table_["iv_bin"].sum()
    
        self.final_woe_table_ = compute_woe_iv_table(
            self.final_cuts_, x_main, y_main, self.feature_name
        )
        self.final_iv_ = self.final_woe_table_["iv_bin"].sum()
        return self

    # ── Transform ──────────────────────────────────────────────────────────

    def transform(self, x: pd.Series) -> pd.Series:
        if self.final_woe_table_ is None:
            raise RuntimeError("Chưa fit. Gọi .fit() trước.")

        x_proc  = cap_outliers(x, self.lower_pct, self.upper_pct) \
                  if self.cap_outliers_ else x.copy()
        edges   = [-np.inf] + sorted(self.final_cuts_) + [np.inf]
        bin_idx = pd.cut(x_proc, bins=edges, labels=False, right=True, include_lowest=True)
        woe_map = dict(enumerate(self.final_woe_table_["woe"].values))
        result  = bin_idx.map(woe_map)

        if self.special_table_ is not None:
            for _, row in self.special_table_.iterrows():
                if row["bin"] == "Missing":
                    result[x.isna()] = row["woe"]
                elif str(row["bin"]).startswith("Special="):
                    sv = float(str(row["bin"]).replace("Special=", ""))
                    result[x == sv] = row["woe"]
        return result

    def fit_transform(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return self.fit(x, y).transform(x)

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        if self.final_woe_table_ is None:
            raise RuntimeError("Chưa fit.")
        cols = ["feature", "bin", "n_total", "n_event", "n_nonevent",
                "event_rate", "woe", "iv_bin", "iv_total"]
        df = self.final_woe_table_.copy()
        df["iv_total"] = round(self.final_iv_, 4)
        main = df[cols]
        if self.special_table_ is not None:
            spec_cols = [c for c in cols if c in self.special_table_.columns]
            return pd.concat([main, self.special_table_[spec_cols]], ignore_index=True)
        return main

    def is_monotonic(self) -> bool:
        if self.final_woe_table_ is None:
            return False
        return is_monotonic_series(self.final_woe_table_["woe"], self.direction_)

    # ── Plot helpers (dùng nội bộ và có thể tái sử dụng từ report) ─────────

    def _draw_woe_axes(self, ax, df: pd.DataFrame, label: str) -> None:
        """Vẽ cột WOE lên axes cho một bảng WOE."""
        x_pos = list(range(len(df)))
        colors = ["#e74c3c" if w >= 0 else "#27ae60" for w in df["woe"]]
        ax.bar(x_pos, df["woe"], color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        iv_val = df["iv_total"].values[0] if "iv_total" in df.columns else 0.0
        ax.set_title(f"WOE theo {label} Bin | IV = {iv_val:.4f}", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("WOE")
        for i, (w, n) in enumerate(zip(df["woe"], df["n_total"])):
            offset = 0.03 if w >= 0 else -0.06
            ax.text(i, w + offset, f"{w:.3f}\nn={n}", ha="center", fontsize=7.5)

    def _draw_event_rate_axes(self, ax, df: pd.DataFrame, label: str) -> None:
        """Vẽ đường Event Rate (%) lên axes cho một bảng WOE."""
        arrow = "↑" if self.direction_ == "ascending" else "↓"
        x_pos = list(range(len(df)))
        er_pct = df["event_rate"] * 100
        ax.plot(x_pos, er_pct, marker="o", color="#2980b9",
                linewidth=2, markersize=8, zorder=3)
        ax.fill_between(x_pos, er_pct, alpha=0.12, color="#2980b9")
        ax.set_title(f"Event Rate (%) theo {label} Bin  {arrow}", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Event Rate (%)")
        for i, er in enumerate(er_pct):
            ax.text(i, er + 0.3, f"{er:.1f}%", ha="center", fontsize=8)

    # ── Plot ───────────────────────────────────────────────────────────────

    def plot(self, figsize=(18, 4)):
        """Vẽ 2 hàng (WOE, Event Rate) × n cột (Init / Algo / Final)."""
        if self.final_woe_table_ is None:
            raise RuntimeError("Chưa fit.")

        tables_to_plot: List[pd.DataFrame] = []
        table_labels: List[str] = []

        if hasattr(self, "init_woe_table_") and self.init_woe_table_ is not None:
            tables_to_plot.append(self.init_woe_table_)
            table_labels.append("Init")
        if hasattr(self, "algo_woe_table_") and self.algo_woe_table_ is not None:
            tables_to_plot.append(self.algo_woe_table_)
            table_labels.append("Algo")
        tables_to_plot.append(self.final_woe_table_)
        table_labels.append("Final")

        n_tables = len(tables_to_plot)
        fig, axes = plt.subplots(2, n_tables, figsize=(figsize[0], figsize[1] * 2))
        if n_tables == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        arrow = "↑" if self.direction_ == "ascending" else "↓"
        fig.suptitle(
            f"{self.__class__.__name__}  |  {self.feature_name}  |  "
            f"Final IV = {self.final_iv_:.4f}  |  Monotonic: {self.is_monotonic()}  {arrow}",
            fontsize=12, fontweight="bold",
        )

        for col, (df, lbl) in enumerate(zip(tables_to_plot, table_labels)):
            self._draw_woe_axes(axes[0, col], df, lbl)
            self._draw_event_rate_axes(axes[1, col], df, lbl)

        plt.tight_layout()
        return fig