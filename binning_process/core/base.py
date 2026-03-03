


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List

from binning_process.core.utils import EPSILON, detect_direction, compute_woe_iv_table, cap_outliers, is_monotonic_series
from binning_process.core.merge_process import enforce_monotonic_traced, MergeTrace

import warnings
warnings.filterwarnings("ignore")

class BaseBinner(BaseEstimator, TransformerMixin):
    """
    Base class: preprocessing, fit/transform/summary/plot.
    Mỗi subclass chỉ cần implement _find_cuts(x, y) → List[float].
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
        self.woe_table_     : Optional[pd.DataFrame] = None
        self.iv_            : float                  = 0.0
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

        self.woe_table_ = compute_woe_iv_table(
            self.final_cuts_, x_main, y_main, self.feature_name
        )
        self.iv_ = self.woe_table_["iv_bin"].sum()
        return self

    # ── Transform ──────────────────────────────────────────────────────────

    def transform(self, x: pd.Series) -> pd.Series:
        if self.woe_table_ is None:
            raise RuntimeError("Chưa fit. Gọi .fit() trước.")

        x_proc  = cap_outliers(x, self.lower_pct, self.upper_pct) \
                  if self.cap_outliers_ else x.copy()
        edges   = [-np.inf] + sorted(self.cuts_) + [np.inf]
        bin_idx = pd.cut(x_proc, bins=edges, labels=False, right=False)
        woe_map = dict(enumerate(self.woe_table_["woe"].values))
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
        if self.woe_table_ is None:
            raise RuntimeError("Chưa fit.")
        cols = ["feature", "bin", "n_total", "n_event", "n_nonevent",
                "event_rate", "woe", "iv_bin", "iv_total"]
        df = self.woe_table_.copy()
        df["iv_total"] = round(self.iv_, 4)
        main = df[cols]
        if self.special_table_ is not None:
            spec_cols = [c for c in cols if c in self.special_table_.columns]
            return pd.concat([main, self.special_table_[spec_cols]], ignore_index=True)
        return main

    def is_monotonic(self) -> bool:
        if self.woe_table_ is None:
            return False
        return is_monotonic_series(self.woe_table_["woe"], self.direction_)

    # ── Plot ───────────────────────────────────────────────────────────────

    def plot(self, figsize=(14, 5)):
        if self.woe_table_ is None:
            raise RuntimeError("Chưa fit.")
        df    = self.woe_table_
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        arrow = "↑" if self.direction_ == "ascending" else "↓"
        fig.suptitle(
            f"{self.__class__.__name__}  |  {self.feature_name}  |  "
            f"IV = {self.iv_:.4f}  |  Monotonic: {self.is_monotonic()}  {arrow}",
            fontsize=12, fontweight="bold"
        )
        x_pos  = list(range(len(df)))
        ax     = axes[0]
        colors = ["#e74c3c" if w >= 0 else "#27ae60" for w in df["woe"]]
        ax.bar(x_pos, df["woe"], color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("WOE theo Bin", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("WOE")
        for i, (w, n) in enumerate(zip(df["woe"], df["n_total"])):
            offset = 0.03 if w >= 0 else -0.06
            ax.text(i, w + offset, f"{w:.3f}\nn={n}", ha="center", fontsize=7.5)

        ax     = axes[1]
        er_pct = df["event_rate"] * 100
        ax.plot(x_pos, er_pct, marker="o", color="#2980b9",
                linewidth=2, markersize=8, zorder=3)
        ax.fill_between(x_pos, er_pct, alpha=0.12, color="#2980b9")
        ax.set_title(f"Event Rate (%) theo Bin  {arrow}", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Event Rate (%)")
        for i, er in enumerate(er_pct):
            ax.text(i, er + 0.3, f"{er:.1f}%", ha="center", fontsize=8)

        plt.tight_layout()
        return fig