"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MONOTONIC WOE BINNING — 5 Phương Pháp, Đóng Gói Dạng Class           ║
║                                                                              ║
║  Không dùng OptBinning. Mỗi method đều giải thích được bằng ngôn ngữ         ║
║  tự nhiên cho người không chuyên sâu thuật toán.                             ║
║                                                                              ║
║  CLASS 1: IsotonicBinner          — Isotonic Regression      (mục 4)         ║
║  CLASS 2: QuantileMonotonicBinner — Quantile + Merge         (mục 3)         ║
║  CLASS 3: DecisionTreeBinner      — Decision Tree + Merge    (mục 2)         ║
║  CLASS 4: ChiMergeBinner          — Chi-square Merge         (mục 2)         ║
║  CLASS 5: MDLPBinner              — Entropy / MDLP Split     (mục 1)         ║
║                                                                              ║
║  Cài đặt: pip install pandas numpy scikit-learn scipy matplotlib             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
EPSILON = 1e-9
from binning_process.merge_process import enforce_monotonic_traced, MergeTrace

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES DÙNG CHUNG
# ══════════════════════════════════════════════════════════════════════════════

def _detect_direction(x: np.ndarray, y: np.ndarray) -> str:
    """
    Tự động phát hiện chiều monotonic bằng Spearman Correlation.

    Giải thích đơn giản:
        Spearman đo: khi X tăng thì tỷ lệ bad có xu hướng tăng hay giảm?
        - Tương quan dương → ascending  (X cao → risk cao,  vd: số ngày quá hạn)
        - Tương quan âm    → descending (X cao → risk thấp, vd: thu nhập, tuổi)
    """
    corr, _ = stats.spearmanr(x, y)
    return "ascending" if corr >= 0 else "descending"


def _compute_woe_iv_table(cuts: list, x: np.ndarray, y: np.ndarray,
                           feature_name: str = "feature") -> pd.DataFrame:
    """
    Tính bảng WOE/IV từ danh sách cut-points.

    WOE (Weight of Evidence) từng bin:
        WOE_i = ln( %Event_i / %NonEvent_i )
        → WOE > 0: bin này có tỷ lệ bad cao hơn bình quân (rủi ro cao)
        → WOE < 0: bin này có tỷ lệ bad thấp hơn bình quân (rủi ro thấp)

    IV (Information Value) tổng:
        IV = Σ (%Event_i - %NonEvent_i) * WOE_i
        → IV < 0.02: biến vô dụng | 0.02 < IV < 0.1: yếu | 0.1 < IV < 0.3: trung bình
        → 0.3 < IV < 0.5: tốt          | IV > 0.5: nghi ngờ overfit
    """
    edges  = [-np.inf] + sorted(cuts) + [np.inf]
    labels = [f"({edges[i]:.4g}, {edges[i+1]:.4g}]" for i in range(len(edges) - 1)]
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


def _enforce_monotonic_by_merge(cuts: list, x: np.ndarray, y: np.ndarray,
                                  direction: str) -> list:
    """
    Gộp các bins vi phạm monotonicity bằng Adjacent Merge.

    Giải thích đơn giản:
        Nhìn event rate của từng bin theo thứ tự từ trái → phải.
        Nếu 2 bin kề nhau "đi ngược chiều" mong muốn → gộp lại.
        Lặp lại cho đến khi tất cả bins đi đúng chiều.

    Ví dụ (ascending): event rate = [5%, 8%, 6%, 12%]
        → Bin 2 (8%) > Bin 3 (6%) → VI PHẠM → gộp bin 2&3 → [5%, 7%, 12%] ✓
    """
    if not cuts:
        return cuts

    for _ in range(100):
        edges   = [-np.inf] + sorted(cuts) + [np.inf]
        bin_idx = pd.cut(x, bins=edges, labels=False, right=False)

        # Tính event rate từng bin
        event_rates = []
        for i in range(len(edges) - 1):
            mask = bin_idx == i
            event_rates.append(y[mask].mean() if mask.sum() > 0 else 0.0)

        # Tìm vi phạm đầu tiên
        violation_idx = None
        for i in range(len(event_rates) - 1):
            if direction == "ascending"  and event_rates[i] > event_rates[i+1] + EPSILON:
                violation_idx = i
                break
            if direction == "descending" and event_rates[i] < event_rates[i+1] - EPSILON:
                violation_idx = i
                break

        if violation_idx is None:
            break  # Đã monotonic hoàn toàn

        # Gộp bin: xóa cut-point tại vị trí vi phạm
        sorted_cuts = sorted(cuts)
        if violation_idx < len(sorted_cuts):
            cuts = sorted_cuts[:violation_idx] + sorted_cuts[violation_idx + 1:]

    return cuts


def _cap_outliers(series: pd.Series, lower_pct: float = 1, upper_pct: float = 99) -> pd.Series:
    """Winsorize: kéo outlier về ngưỡng [P1, P99]."""
    lo = np.nanpercentile(series.dropna(), lower_pct)
    hi = np.nanpercentile(series.dropna(), upper_pct)
    return series.clip(lower=lo, upper=hi)


def _is_monotonic(series: pd.Series, direction: str, tol: float = 1e-6) -> bool:
    diffs = series.diff().dropna()
    return bool((diffs >= -tol).all()) if direction == "ascending" else bool((diffs <= tol).all())


def _iv_rating(iv: float) -> str:
    if iv is None: return ""
    if iv < 0.02:  return "Vô dụng"
    if iv < 0.1:   return "Yếu"
    if iv < 0.3:   return "Trung bình"
    if iv < 0.5:   return "Tốt"
    return "Nghi ngờ overfit"


# ══════════════════════════════════════════════════════════════════════════════
#  BASE CLASS — Logic chung cho tất cả methods
# ══════════════════════════════════════════════════════════════════════════════

class _BaseBinner(BaseEstimator, TransformerMixin):
    """
    Base class: preprocessing, fit/transform/summary/plot.
    Mỗi subclass chỉ cần implement _find_cuts().
    """

    def __init__(
        self,
        feature_name    : str            = "feature",
        min_bin_size    : float          = 0.05,
        min_event_count : int            = 5,
        max_bins        : int            = 8,
        special_values  : Optional[List] = None,
        cap_outliers    : bool           = True,
        lower_pct       : float          = 1.0,
        upper_pct       : float          = 99.0,
        direction       : str            = "auto",
    ):
        self.feature_name    = feature_name
        self.min_bin_size    = min_bin_size
        self.min_event_count = min_event_count
        self.max_bins        = max_bins
        self.special_values  = special_values or []
        self.cap_outliers    = cap_outliers
        self.lower_pct       = lower_pct
        self.upper_pct       = upper_pct
        self.direction       = direction

        self.init_cuts_     : List[float]            = []
        self.cuts_          : List[float]            = []
        self.trace_         : Optional[MergeTrace]   = None
        self.direction_     : str                    = direction
        self.woe_table_     : Optional[pd.DataFrame] = None
        self.iv_            : float                  = 0.0
        self.special_table_ : Optional[pd.DataFrame] = None

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _preprocess(self, x: pd.Series, y: pd.Series):
        x_proc = _cap_outliers(x, self.lower_pct, self.upper_pct) \
                 if self.cap_outliers else x.copy()

        mask_miss    = x.isna()
        mask_special = x_proc.isin(self.special_values) \
                       if self.special_values else pd.Series(False, index=x.index)
        mask_main    = ~mask_miss & ~mask_special

        x_main = x_proc[mask_main].values.astype(float)
        y_main = y[mask_main].values.astype(int)

        # Tạo bảng special bins
        total_event    = max(int(y.sum()), 1)
        total_nonevent = max(int((1 - y).sum()), 1)
        special_rows   = []

        groups = [("Missing", mask_miss)] + \
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

    # ── Abstract ──────────────────────────────────────────────────────────

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        raise NotImplementedError

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, x: pd.Series, y: pd.Series):
        """Fit binning. x: biến số, y: target nhị phân (1=bad, 0=good)."""
        x_main, y_main = self._preprocess(x, y)

        if self.direction == "auto":
            self.direction_ = _detect_direction(x_main, y_main)
        else:
            self.direction_ = self.direction

        self.init_cuts_      = self._find_cuts(x_main, y_main)
        self.trace_, self.cuts_ = enforce_monotonic_traced(self.init_cuts_, x_main, y_main, self.direction_)
        self.woe_table_ = _compute_woe_iv_table(self.cuts_, x_main, y_main, self.feature_name)
        self.iv_        = self.woe_table_["iv_bin"].sum()
        return self

    # ── Transform ─────────────────────────────────────────────────────────

    def transform(self, x: pd.Series) -> pd.Series:
        """Map giá trị X sang WOE tương ứng."""
        if self.woe_table_ is None:
            raise RuntimeError("Chưa fit. Gọi .fit() trước.")

        x_proc  = _cap_outliers(x, self.lower_pct, self.upper_pct) \
                  if self.cap_outliers else x.copy()
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

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Bảng WOE/IV đầy đủ, bao gồm cả Special bins."""
        if self.woe_table_ is None:
            raise RuntimeError("Chưa fit.")
        cols = ["feature", "bin", "n_total", "n_event", "n_nonevent",
                "event_rate", "woe", "iv_bin", "iv_total"]
        df = self.woe_table_.copy()
        df["iv_total"] = round(self.iv_, 4)
        main = df[cols]
        if self.special_table_ is not None:
            spec_cols = [c for c in cols if c in self.special_table_.columns]
            spec      = self.special_table_[spec_cols]
            return pd.concat([main, spec], ignore_index=True)
        return main

    def is_monotonic(self) -> bool:
        if self.woe_table_ is None:
            return False
        return _is_monotonic(self.woe_table_["woe"], self.direction_)

    # ── Plot ──────────────────────────────────────────────────────────────

    def plot(self, figsize=(14, 5)):
        """Biểu đồ WOE bar và Event Rate line theo bin."""
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

        x_pos = list(range(len(df)))

        # ── WOE bar chart ──────────────────────────────────────────────
        ax = axes[0]
        colors = ["#e74c3c" if w >= 0 else "#27ae60" for w in df["woe"]]
        bars   = ax.bar(x_pos, df["woe"], color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("WOE theo Bin", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("WOE")
        for i, (w, n) in enumerate(zip(df["woe"], df["n_total"])):
            offset = 0.03 if w >= 0 else -0.06
            ax.text(i, w + offset, f"{w:.3f}\nn={n}", ha="center", fontsize=7.5, va="bottom")

        # ── Event Rate line chart ──────────────────────────────────────
        ax = axes[1]
        er_pct = df["event_rate"] * 100
        ax.plot(x_pos, er_pct, marker="o", color="#2980b9", linewidth=2, markersize=8, zorder=3)
        ax.fill_between(x_pos, er_pct, alpha=0.12, color="#2980b9")
        ax.set_title(f"Event Rate (%) theo Bin  {arrow}", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Event Rate (%)")
        for i, er in enumerate(er_pct):
            ax.text(i, er + 0.3, f"{er:.1f}%", ha="center", fontsize=8)

        plt.tight_layout()
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 — ISOTONIC REGRESSION BINNER  (Mục 4)
# ══════════════════════════════════════════════════════════════════════════════

class IsotonicBinner(_BaseBinner):
    """
    Isotonic Regression Binning

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                             │
    │                                                                  │
    │  Hãy hình dung bạn có đồ thị điểm (X, bad_rate). Có thể       │
    │  bad_rate lên xuống loạn xạ vì nhiễu. Isotonic Regression      │
    │  vẽ một "đường bậc thang" đi LÊN (hoặc đi XUỐNG) mượt nhất   │
    │  có thể qua các điểm đó.                                        │
    │                                                                  │
    │  Mỗi "bậc thang ngang" = 1 bin có cùng bad_rate.               │
    │  Điểm gãy giữa 2 bậc thang = cut-point.                        │
    └──────────────────────────────────────────────────────────────────┘

    Các bước:
      1. Chia data thành N bins bằng nhau (quantile) — N lớn để dày
      2. Tính event rate của từng bin nhỏ
      3. Fit Isotonic Regression lên event rate → ra "bậc thang"
      4. Tìm điểm gãy của bậc thang = cut-points
    """

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # B1: Quantile chia sơ bộ
        pcts      = np.linspace(0, 100, self.n_init_bins + 1)[1:-1]
        init_cuts = np.unique(np.nanpercentile(x, pcts)).tolist()
        edges     = [-np.inf] + init_cuts + [np.inf]
        bin_idx   = pd.cut(x, bins=edges, labels=False, right=False)

        # B2: Event rate + bin center cho từng bin
        centers, rates, weights = [], [], []
        for i in range(len(edges) - 1):
            mask = bin_idx == i
            if mask.sum() == 0:
                continue
            centers.append(float(np.median(x[mask])))
            rates.append(float(y[mask].mean()))
            weights.append(int(mask.sum()))

        centers = np.array(centers)
        rates   = np.array(rates)
        weights = np.array(weights)

        # B3: Fit Isotonic Regression
        iso = IsotonicRegression(
            increasing=(self.direction_ == "ascending"),
            out_of_bounds="clip"
        )
        fitted = iso.fit_transform(centers, rates, sample_weight=weights)

        # B4: Điểm gãy = cut-points
        sorted_x = np.sort(np.unique(x))
        cuts = []
        for i in range(len(fitted) - 1):
            if abs(fitted[i] - fitted[i + 1]) > EPSILON:
                boundary = (centers[i] + centers[i + 1]) / 2
                idx      = np.searchsorted(sorted_x, boundary)
                idx      = min(idx, len(sorted_x) - 1)
                cuts.append(float(sorted_x[idx]))

        # Giới hạn bins
        cuts = sorted(set(cuts))
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]

        return cuts


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 — QUANTILE + MONOTONIC MERGE BINNER  (Mục 3)
# ══════════════════════════════════════════════════════════════════════════════

class QuantileMonotonicBinner(_BaseBinner):
    """
    Quantile Binning + Adjacent Merge

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                             │
    │                                                                  │
    │  Bước 1: Chia khách hàng thành N nhóm CÓ SỐ LƯỢNG BẰNG NHAU   │
    │          (như chia 100 người thành 10 nhóm, mỗi nhóm 10 người) │
    │                                                                  │
    │  Bước 2: Xem tỷ lệ bad từng nhóm. Nếu nhóm nào "lồi lõm"      │
    │          (đi ngược xu hướng chung) → gộp nó với nhóm kề bên.   │
    │                                                                  │
    │  Lặp đến khi các nhóm đều đi một chiều.                        │
    │                                                                  │
    │  Đây là phương pháp đơn giản nhất, dễ giải thích nhất.         │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins = n_init_bins

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # B1: Quantile
        pcts  = np.linspace(0, 100, self.n_init_bins + 1)[1:-1]
        cuts  = sorted(list(np.unique(np.nanpercentile(x, pcts))))

        # Giới hạn max bins sơ bộ
        if len(cuts) > self.max_bins * 2:
            step = max(1, len(cuts) // (self.max_bins * 2))
            cuts = cuts[::step]

        # B2: Gộp bins quá nhỏ trước khi enforce monotonic
        cuts = self._merge_small_bins(cuts, x, y)
        return cuts

    def _merge_small_bins(self, cuts: list, x: np.ndarray, y: np.ndarray) -> list:
        """Gộp bins < min_bin_size hoặc < min_event_count."""
        n_total = len(x)
        min_n   = max(int(self.min_bin_size * n_total), self.min_event_count)

        for _ in range(100):
            if not cuts:
                break
            edges   = [-np.inf] + sorted(cuts) + [np.inf]
            bin_idx = pd.cut(x, bins=edges, labels=False, right=False)
            found   = False

            for i in range(len(edges) - 1):
                mask    = bin_idx == i
                n       = int(mask.sum())
                n_event = int(y[mask].sum())

                if n < min_n or n_event < self.min_event_count:
                    sc = sorted(cuts)
                    if i == 0:
                        cuts = sc[1:]
                    elif i == len(edges) - 2:
                        cuts = sc[:-1]
                    elif i - 1 < len(sc):
                        cuts = sc[:i - 1] + sc[i:]
                    found = True
                    break

            if not found:
                break
        return cuts


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 3 — DECISION TREE BINNER  (Mục 2)
# ══════════════════════════════════════════════════════════════════════════════

class DecisionTreeBinner(_BaseBinner):
    """
    Decision Tree Binning + Monotonic Merge

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                             │
    │                                                                  │
    │  Cây quyết định tự hỏi: "Tại ngưỡng nào thì chia khách hàng   │
    │  thành 2 nhóm có tỷ lệ bad KHÁC NHAU NHẤT?"                    │
    │                                                                  │
    │  Ví dụ: "Nếu income < 15tr → 38% bad. Nếu ≥ 15tr → 7% bad"   │
    │  → 15tr là cut-point tốt nhất theo tiêu chí Gini               │
    │                                                                  │
    │  Cây tiếp tục hỏi tương tự trong từng nhánh, tạo ra nhiều      │
    │  cut-points. Sau đó ta enforce monotonic bằng gộp bins.         │
    │                                                                  │
    │  Ưu điểm: cut-points được chọn tối ưu theo bad/good signal,    │
    │  không phụ thuộc phân phối của X.                               │
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

        return cuts


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 4 — CHI-MERGE BINNER  (Mục 2)
# ══════════════════════════════════════════════════════════════════════════════

class ChiMergeBinner(_BaseBinner):
    """
    ChiMerge — Chi-square Based Merging

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                             │
    │                                                                  │
    │  Bắt đầu với RẤT NHIỀU bins nhỏ (50 bins).                     │
    │                                                                  │
    │  Ở mỗi vòng: tìm 2 bins kề nhau "giống nhau nhất" về tỷ lệ    │
    │  bad/good → gộp chúng lại.                                      │
    │                                                                  │
    │  "Giống nhau" được đo bằng Chi-square test (kiểm định thống kê):│
    │    Chi² nhỏ = 2 bins không khác biệt đáng kể → NÊN GỘP         │
    │    Chi² lớn = 2 bins khác biệt rõ ràng      → GIỮ LẠI          │
    │                                                                  │
    │  Dừng khi đạt số bins mong muốn hoặc tất cả bins đủ khác biệt. │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_init_bins: int = 50, confidence_level: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_init_bins      = n_init_bins
        self.confidence_level = confidence_level

    @staticmethod
    def _chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
        """
        Chi-square giữa 2 bins (bảng 2x2: bin × event/nonevent).
        Chi² nhỏ → 2 bins phân phối giống nhau → nên gộp.
        """
        observed = np.array([counts_a, counts_b], dtype=float)
        row_s    = observed.sum(axis=1, keepdims=True)
        col_s    = observed.sum(axis=0, keepdims=True)
        total    = observed.sum()
        if total == 0:
            return 0.0
        expected = (row_s * col_s) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0).sum()
        return float(chi2)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # B1: Chia nhiều bins nhỏ
        pcts      = np.linspace(0, 100, self.n_init_bins + 1)[1:-1]
        init_cuts = sorted(list(np.unique(np.nanpercentile(x, pcts))))
        edges     = [-np.inf] + init_cuts + [np.inf]
        bin_idx   = pd.cut(x, bins=edges, labels=False, right=False)

        n_bins = len(edges) - 1
        counts = np.zeros((n_bins, 2), dtype=int)
        for i in range(n_bins):
            mask         = bin_idx == i
            counts[i, 1] = int(y[mask].sum())
            counts[i, 0] = int((1 - y[mask]).sum())

        current_cuts  = list(init_cuts)
        chi2_threshold = stats.chi2.ppf(1 - self.confidence_level, df=1)

        # B2: Gộp dần bin giống nhau nhất
        for _ in range(500):
            if len(counts) <= self.max_bins:
                # Kiểm tra nếu tất cả bins đã đủ khác biệt thì dừng
                chi2_vals = [self._chi2(counts[i], counts[i+1])
                              for i in range(len(counts)-1)]
                if chi2_vals and min(chi2_vals) >= chi2_threshold:
                    break

            if len(counts) <= 2:
                break

            chi2_vals = [self._chi2(counts[i], counts[i+1])
                          for i in range(len(counts)-1)]
            if not chi2_vals:
                break

            min_idx  = int(np.argmin(chi2_vals))
            # Gộp bin min_idx và min_idx+1
            counts[min_idx] = counts[min_idx] + counts[min_idx + 1]
            counts          = np.delete(counts, min_idx + 1, axis=0)
            if min_idx < len(current_cuts):
                current_cuts.pop(min_idx)

        return sorted(current_cuts)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 5 — MDLP BINNER  (Mục 1 — Entropy/MDLP)
# ══════════════════════════════════════════════════════════════════════════════

class MDLPBinner(_BaseBinner):
    """
    MDLP — Minimum Description Length Principle Binning

    ┌──────────────────────────────────────────────────────────────────┐
    │  Giải thích cho người không chuyên:                             │
    │                                                                  │
    │  MDLP tìm cut-points bằng cách tối đa hoá Information Gain.    │
    │                                                                  ║
    │  Entropy = mức độ "hỗn hợp" bad/good trong một nhóm:           │
    │    Entropy = 0   → nhóm thuần (toàn bad HOẶC toàn good)        │
    │    Entropy = 1   → nhóm hỗn hợp 50% bad / 50% good            │
    │                                                                  │
    │  Information Gain = Entropy trước khi chia TRỪ entropy sau chia │
    │  → IG lớn = điểm chia tốt (hai nhóm con khác biệt rõ ràng)    │
    │                                                                  │
    │  Điều kiện dừng MDL: "Chỉ chia thêm nếu thông tin thu được     │
    │  đủ lớn để bù đắp chi phí mô tả thêm 1 cut-point."             │
    │  Điều này giúp tránh chia quá nhiều bins (overfitting).         │
    └──────────────────────────────────────────────────────────────────┘

    Thuật toán: Recursive binary split — chia đôi từng đoạn đệ quy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_total = 1  # Sẽ gán trong _find_cuts

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        p = float(y.mean())
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    @staticmethod
    def _info_gain(y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> float:
        n, n_l, n_r = len(y), len(y_l), len(y_r)
        if n_l == 0 or n_r == 0:
            return 0.0
        return MDLPBinner._entropy(y) - \
               (n_l / n) * MDLPBinner._entropy(y_l) - \
               (n_r / n) * MDLPBinner._entropy(y_r)

    def _mdlp_stop(self, y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> bool:
        """
        Điều kiện dừng MDLP (Fayyad & Irani, 1993):
        Dừng nếu Information Gain < (log2(n-1) + delta) / n.

        Giải thích: "Chia thêm chỉ có ý nghĩa nếu lợi ích > chi phí."
        """
        n = len(y)
        if n < 4:
            return True
        k  = max(len(np.unique(y)), 1)
        k1 = max(len(np.unique(y_l)), 1)
        k2 = max(len(np.unique(y_r)), 1)
        gain      = self._info_gain(y, y_l, y_r)
        delta     = np.log2(3 ** k - 2) - (
            k  * self._entropy(y) -
            k1 * self._entropy(y_l) -
            k2 * self._entropy(y_r)
        )
        threshold = (np.log2(n - 1) + delta) / n
        return gain <= threshold

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Tìm cut-point cho IG lớn nhất trong đoạn này."""
        unique_vals = np.unique(x)
        if len(unique_vals) < 2:
            return None

        best_gain, best_cut = -np.inf, None
        sorted_idx = np.argsort(x)
        x_s, y_s   = x[sorted_idx], y[sorted_idx]

        for i in range(1, len(unique_vals)):
            cut   = (unique_vals[i - 1] + unique_vals[i]) / 2
            left  = x_s < cut
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            gain = self._info_gain(y_s, y_s[left], y_s[right])
            if gain > best_gain:
                best_gain, best_cut = gain, cut

        return best_cut

    def _recursive_split(self, x: np.ndarray, y: np.ndarray,
                          cuts: list, depth: int = 0) -> None:
        """Chia đôi đệ quy và thêm cut-point nếu thỏa điều kiện MDLP."""
        if depth >= self.max_bins - 1:
            return
        if len(x) < max(int(self.min_bin_size * self._n_total), self.min_event_count * 2):
            return

        cut = self._best_split(x, y)
        if cut is None:
            return

        left_mask  = x <= cut
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return

        if self._mdlp_stop(y, y[left_mask], y[right_mask]):
            return

        cuts.append(float(cut))
        self._recursive_split(x[left_mask],  y[left_mask],  cuts, depth + 1)
        self._recursive_split(x[right_mask], y[right_mask], cuts, depth + 1)

    def _find_cuts(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        self._n_total = len(x)
        cuts = []
        self._recursive_split(x, y, cuts)
        cuts = sorted(set(cuts))
        if len(cuts) >= self.max_bins:
            cuts = cuts[:self.max_bins - 1]
        return cuts


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: So sánh 5 methods trên 1 biến
# ══════════════════════════════════════════════════════════════════════════════

def compare_methods(
    x: pd.Series,
    y: pd.Series,
    feature_name    : str            = "feature",
    max_bins        : int            = 6,
    special_values  : Optional[list] = None,
) -> pd.DataFrame:
    """
    Fit tất cả 5 methods và trả về bảng so sánh.
    Dùng để chọn nhanh method phù hợp cho từng biến.
    """
    kwargs = dict(
        feature_name   = feature_name,
        max_bins       = max_bins,
        special_values = special_values,
    )
    methods = {
        "1. IsotonicBinner"    : IsotonicBinner(**kwargs),
        "2. QuantileMonotonic" : QuantileMonotonicBinner(**kwargs),
        "3. DecisionTree"      : DecisionTreeBinner(**kwargs),
        "4. ChiMerge"          : ChiMergeBinner(**kwargs),
        "5. MDLP"              : MDLPBinner(**kwargs),
    }

    rows = []
    for name, model in methods.items():
        try:
            model.fit(x, y)
            rows.append({
                "Method"    : name,
                "IV"        : round(model.iv_, 4),
                "n_bins"    : len(model.cuts_) + 1,
                "Monotonic" : "✓" if model.is_monotonic() else "✗",
                "Direction" : model.direction_,
                "IV_Rating" : _iv_rating(model.iv_),
                "Cuts"      : [round(c, 2) for c in model.cuts_],
            })
        except Exception as e:
            rows.append({
                "Method"    : name,
                "IV"        : None,
                "n_bins"    : None,
                "Monotonic" : "ERROR",
                "Direction" : "",
                "IV_Rating" : str(e)[:50],
                "Cuts"      : [],
            })

    result = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)
    print(f"\n{'='*70}")
    print(f"  SO SÁNH 5 METHODS — Feature: [{feature_name}]")
    print(f"{'='*70}")
    print(result[["Method", "IV", "n_bins", "Monotonic", "Direction", "IV_Rating"]].to_string(index=False))
    print(f"{'='*70}\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(42)
    N = 5_000

    # Giả lập dữ liệu credit: income có quan hệ nghịch với bad rate
    income   = np.random.lognormal(mean=10, sigma=0.8, size=N)
    bad_prob = 1 / (1 + np.exp((income - 22000) / 6000))
    bad      = (np.random.rand(N) < bad_prob).astype(int)

    income_s = pd.Series(income, name="income")
    income_s.iloc[:200] = np.nan   # 4% missing
    income_s.iloc[200:250] = -1    # special value
    y_s = pd.Series(bad)

    print(f"Dataset: N={N}, bad_rate={bad.mean():.2%}")
    print(f"Missing={income_s.isna().sum()}, Special(-1)={(income_s==-1).sum()}\n")

    # ── So sánh 5 methods ──────────────────────────────────────────────────
    result = compare_methods(
        x=income_s, y=y_s,
        feature_name="income",
        max_bins=6,
        special_values=[-1],
    )

    # ── Xem chi tiết 1 method ─────────────────────────────────────────────
    print("Chi tiết IsotonicBinner:")
    b1 = IsotonicBinner(feature_name="income", max_bins=6, special_values=[-1])
    b1.fit(income_s, y_s)
    print(b1.summary().to_string(index=False))
    print(f"\nCut-points: {[round(c,1) for c in b1.cuts_]}")
    print(f"Monotonic : {b1.is_monotonic()} | Direction: {b1.direction_}")

    print("\nChi tiết DecisionTreeBinner:")
    b3 = DecisionTreeBinner(feature_name="income", max_bins=6, special_values=[-1])
    b3.fit(income_s, y_s)
    print(b3.summary().to_string(index=False))

    # ── Transform sang WOE ────────────────────────────────────────────────
    woe_vals = b1.transform(income_s)
    print(f"\nWOE values (sample):\n{woe_vals.dropna().head(10).values}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig1 = b1.plot()
    plt.savefig("plot_isotonic.png", dpi=120, bbox_inches="tight")
    print("\nĐã lưu: plot_isotonic.png")
    plt.show()
