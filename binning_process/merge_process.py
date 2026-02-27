"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      _enforce_monotonic_by_merge  — Phiên bản có TRACE & VISUALIZE         ║
║                                                                              ║
║  Lưu vết từng bước merge và vẽ lại toàn bộ quá trình.                      ║
║                                                                              ║
║  Outputs:                                                                    ║
║    - MergeTrace object: log từng bước dưới dạng DataFrame                  ║
║    - plot_merge_steps(): vẽ grid các bước                                   ║
║    - plot_merge_animation(): vẽ GIF animation (nếu muốn)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import copy

EPSILON = 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE — Lưu trạng thái 1 bước merge
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MergeStep:
    """
    Lưu toàn bộ thông tin của 1 bước trong quá trình merge.

    Attributes:
        step_no        : Số thứ tự bước (0 = trạng thái ban đầu)
        cuts_before    : Danh sách cut-points TRƯỚC khi merge bước này
        cuts_after     : Danh sách cut-points SAU khi merge bước này
        event_rates    : Event rate của từng bin (theo cuts_before)
        n_bins_before  : Số bins trước merge
        n_bins_after   : Số bins sau merge
        merged_pair    : (i, i+1) = chỉ số 2 bins bị gộp. None nếu bước 0
        merged_cut     : Giá trị cut-point bị xóa. None nếu bước 0
        violation_er   : (er_left, er_right) event rate của cặp vi phạm
        direction      : "ascending" | "descending"
        is_final       : True nếu đây là bước cuối (đã monotonic)
        woe_values     : WOE của từng bin (để vẽ đồ thị)
    """
    step_no       : int
    cuts_before   : List[float]
    cuts_after    : List[float]
    event_rates   : List[float]
    n_bins_before : int
    n_bins_after  : int
    merged_pair   : Optional[Tuple[int, int]]
    merged_cut    : Optional[float]
    violation_er  : Optional[Tuple[float, float]]
    direction     : str
    is_final      : bool = False
    woe_values    : List[float] = field(default_factory=list)
    n_samples     : List[int]   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step"          : self.step_no,
            "n_bins_before" : self.n_bins_before,
            "n_bins_after"  : self.n_bins_after,
            "merged_cut"    : round(self.merged_cut, 3) if self.merged_cut else None,
            "merged_bins"   : f"Bin {self.merged_pair[0]} & {self.merged_pair[1]}"
                              if self.merged_pair else "—  (trạng thái ban đầu)",
            "er_left"       : round(self.violation_er[0], 4) if self.violation_er else None,
            "er_right"      : round(self.violation_er[1], 4) if self.violation_er else None,
            "is_violation"  : (self.violation_er[0] > self.violation_er[1]
                               if self.direction == "ascending" and self.violation_er
                               else (self.violation_er[0] < self.violation_er[1]
                                     if self.violation_er else False)),
            "is_final"      : self.is_final,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE TRACE — Container lưu toàn bộ vết
# ══════════════════════════════════════════════════════════════════════════════

class MergeTrace:
    """
    Lưu toàn bộ lịch sử merge và cung cấp các method để phân tích / vẽ.

    Usage:
        trace = enforce_monotonic_traced(cuts, x, y, direction)
        trace.summary()           # Bảng log từng bước
        trace.plot_steps()        # Vẽ grid tất cả bước
        trace.plot_step(3)        # Vẽ riêng bước 3
        trace.plot_before_after() # So sánh trước/sau
    """

    def __init__(self, direction: str, feature_name: str = "feature"):
        self.direction    = direction
        self.feature_name = feature_name
        self.steps        : List[MergeStep] = []

    def add_step(self, step: MergeStep):
        self.steps.append(step)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def initial_cuts(self) -> List[float]:
        return self.steps[0].cuts_before if self.steps else []

    @property
    def final_cuts(self) -> List[float]:
        return self.steps[-1].cuts_after if self.steps else []

    @property
    def n_merges(self) -> int:
        """Số lần gộp thực sự (không tính bước 0)."""
        return sum(1 for s in self.steps if s.merged_pair is not None)

    def summary(self) -> pd.DataFrame:
        """Bảng log tất cả bước merge."""
        rows = [s.to_dict() for s in self.steps]
        df   = pd.DataFrame(rows)
        print(f"\n{'═'*70}")
        print(f"  MERGE TRACE — {self.feature_name} | Direction: {self.direction}")
        print(f"  Tổng số bước merge: {self.n_merges}")
        print(f"  Bins ban đầu: {self.steps[0].n_bins_before}  →  "
              f"Bins cuối: {self.steps[-1].n_bins_after}")
        print(f"{'═'*70}")
        print(df.to_string(index=False))
        print(f"{'═'*70}\n")
        return df

    # ── Vẽ 1 bước ────────────────────────────────────────────────────────────

    def plot_step(self, step_no: int, ax_er=None, ax_woe=None, standalone=True):
        """
        Vẽ chi tiết 1 bước: event rate bar chart, highlight cặp bị gộp.

        Args:
            step_no   : Index trong self.steps (0 = ban đầu)
            standalone: Nếu True thì tạo figure riêng
        """
        if step_no >= len(self.steps):
            raise IndexError(f"Chỉ có {len(self.steps)} bước (0-indexed).")

        step = self.steps[step_no]
        n    = len(step.event_rates)

        if standalone:
            fig, (ax_er, ax_woe) = plt.subplots(1, 2, figsize=(12, 4))
            title = (f"Bước {step.step_no} — {self.feature_name} | "
                     f"{step.n_bins_before} bins → {step.n_bins_after} bins")
            if step.merged_pair:
                title += f"\nGộp Bin {step.merged_pair[0]} & {step.merged_pair[1]} "
                title += f"(cut = {step.merged_cut:.2f})"
            fig.suptitle(title, fontsize=11, fontweight="bold")

        # ── Event Rate bars ────────────────────────────────────────────────
        colors = []
        for i in range(n):
            if step.merged_pair and i in step.merged_pair:
                colors.append("#e74c3c")   # Đỏ: 2 bins bị gộp
            else:
                colors.append("#3498db")   # Xanh: bình thường

        bars = ax_er.bar(range(n), [er * 100 for er in step.event_rates],
                          color=colors, edgecolor="white", linewidth=0.8, zorder=3)

        # Vẽ đường nối giữa 2 bins vi phạm
        if step.merged_pair:
            i, j = step.merged_pair
            ax_er.annotate(
                "",
                xy=(j, step.event_rates[j] * 100),
                xytext=(i, step.event_rates[i] * 100),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c",
                                lw=2, connectionstyle="arc3,rad=0.3"),
            )
            # Label "VI PHẠM" hoặc "ĐÃ MONOTONIC"
            mid_y = (step.event_rates[i] + step.event_rates[j]) * 50 + 5
            mid_x = (i + j) / 2
            ax_er.text(mid_x, mid_y, "⚠ VI PHẠM\n→ GỘP LẠI",
                        ha="center", va="bottom", fontsize=8,
                        color="#c0392b", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7",
                                  edgecolor="#e17055", alpha=0.9))

        # Annotation event rate
        for i, er in enumerate(step.event_rates):
            ax_er.text(i, er * 100 + 0.5, f"{er*100:.1f}%",
                        ha="center", va="bottom", fontsize=7.5)

        ax_er.set_title("Event Rate (%) từng Bin", fontweight="bold")
        ax_er.set_xticks(range(n))
        ax_er.set_xlabel("Bin index")
        ax_er.set_ylabel("Event Rate (%)")

        # Vẽ đường monotonic mong muốn (hướng)
        er_arr = np.array(step.event_rates)
        if self.direction == "ascending":
            ref = np.linspace(er_arr.min(), er_arr.max(), n)
        else:
            ref = np.linspace(er_arr.max(), er_arr.min(), n)
        ax_er.plot(range(n), ref * 100, "--", color="gray",
                    alpha=0.5, linewidth=1.2, label=f"Xu hướng {self.direction}")
        ax_er.legend(fontsize=8)

        # ── WOE bars ──────────────────────────────────────────────────────
        if step.woe_values:
            woe_colors = []
            for i, w in enumerate(step.woe_values):
                if step.merged_pair and i in step.merged_pair:
                    woe_colors.append("#e74c3c")
                else:
                    woe_colors.append("#27ae60" if w < 0 else "#e67e22")

            ax_woe.bar(range(len(step.woe_values)), step.woe_values,
                        color=woe_colors, edgecolor="white", linewidth=0.8, zorder=3)
            ax_woe.axhline(0, color="black", linewidth=0.8, linestyle="--")
            for i, w in enumerate(step.woe_values):
                ax_woe.text(i, w + (0.05 if w >= 0 else -0.1),
                              f"{w:.2f}", ha="center", fontsize=7.5)
            ax_woe.set_title("WOE từng Bin", fontweight="bold")
            ax_woe.set_xticks(range(len(step.woe_values)))
            ax_woe.set_ylabel("WOE")
            ax_woe.set_xlabel("Bin index")

        if standalone:
            plt.tight_layout()
            return plt.gcf()

    # ── Vẽ tất cả bước (grid) ────────────────────────────────────────────────

    def plot_steps(self, max_steps: Optional[int] = None, figsize_per_row=(14, 3.8)):
        """
        Vẽ tất cả các bước merge thành 1 figure lớn (grid theo hàng).

        Args:
            max_steps: Giới hạn số bước vẽ. None = vẽ tất cả.
        """
        steps_to_plot = self.steps[:max_steps] if max_steps else self.steps
        n_rows        = len(steps_to_plot)

        if n_rows == 0:
            print("Không có bước nào để vẽ.")
            return

        fig_h = figsize_per_row[1] * n_rows
        fig   = plt.figure(figsize=(figsize_per_row[0], fig_h))
        fig.suptitle(
            f"Toàn bộ quá trình Monotonic Merge — [{self.feature_name}]\n"
            f"Direction: {self.direction} | "
            f"{self.steps[0].n_bins_before} bins → {self.steps[-1].n_bins_after} bins "
            f"({self.n_merges} lần gộp)",
            fontsize=13, fontweight="bold", y=1.01
        )

        for row_idx, step in enumerate(steps_to_plot):
            ax_er  = fig.add_subplot(n_rows, 2, row_idx * 2 + 1)
            ax_woe = fig.add_subplot(n_rows, 2, row_idx * 2 + 2)

            # Row label bên trái
            step_label = f"Bước {step.step_no}"
            if step.step_no == 0:
                step_label += "\n(Ban đầu)"
            elif step.is_final:
                step_label += "\n✓ Xong"
            ax_er.set_ylabel(step_label, fontsize=9, fontweight="bold", rotation=90,
                              labelpad=40)

            self.plot_step(step.step_no, ax_er=ax_er, ax_woe=ax_woe, standalone=False)

            # Title từng row
            n = len(step.event_rates)
            if step.merged_pair:
                row_title = (f"{step.n_bins_before} bins → {step.n_bins_after} bins | "
                             f"Gộp Bin{step.merged_pair[0]}&Bin{step.merged_pair[1]} "
                             f"(cut={step.merged_cut:.2f})")
            else:
                row_title = f"{step.n_bins_before} bins | Trạng thái ban đầu"

            if step.is_final:
                row_title += " ✓ MONOTONIC"

            ax_er.set_title(row_title, fontsize=9)

        plt.tight_layout()
        return fig

    # ── So sánh trước / sau ──────────────────────────────────────────────────

    def plot_before_after(self, figsize=(14, 5)):
        """Vẽ so sánh trạng thái ban đầu và kết quả cuối."""
        if len(self.steps) < 2:
            print("Không có gì thay đổi.")
            return

        step_init  = self.steps[0]
        step_final = self.steps[-1]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Trước & Sau Monotonic Merge — [{self.feature_name}]\n"
            f"Direction: {self.direction} | "
            f"{step_init.n_bins_before} bins → {step_final.n_bins_after} bins",
            fontsize=12, fontweight="bold"
        )

        def _draw_er(ax, step, label):
            n  = len(step.event_rates)
            c  = ["#e74c3c" if not step.is_final and step.merged_pair and i in step.merged_pair
                  else "#3498db" for i in range(n)]
            ax.bar(range(n), [e * 100 for e in step.event_rates],
                    color=c, edgecolor="white")
            for i, er in enumerate(step.event_rates):
                ax.text(i, er * 100 + 0.5, f"{er*100:.1f}%", ha="center", fontsize=8)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Bin index")
            ax.set_ylabel("Event Rate (%)")
            # Monotonic check arrows
            ers = step.event_rates
            for i in range(len(ers) - 1):
                ok = (ers[i] <= ers[i+1] + EPSILON) if self.direction == "ascending" \
                     else (ers[i] >= ers[i+1] - EPSILON)
                color = "#27ae60" if ok else "#e74c3c"
                ax.annotate("", xy=(i+1, ers[i+1]*100), xytext=(i, ers[i]*100),
                             arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        def _draw_woe(ax, step, label):
            if not step.woe_values:
                ax.text(0.5, 0.5, "Không có WOE", ha="center", transform=ax.transAxes)
                return
            colors = ["#e67e22" if w >= 0 else "#27ae60" for w in step.woe_values]
            ax.bar(range(len(step.woe_values)), step.woe_values,
                    color=colors, edgecolor="white")
            ax.axhline(0, color="black", lw=0.8, linestyle="--")
            for i, w in enumerate(step.woe_values):
                ax.text(i, w + (0.05 if w >= 0 else -0.1), f"{w:.2f}",
                         ha="center", fontsize=8)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Bin index")
            ax.set_ylabel("WOE")

        _draw_er(axes[0, 0],  step_init,
                 f"Event Rate — BAN ĐẦU ({step_init.n_bins_before} bins)")
        _draw_woe(axes[0, 1], step_init,
                  f"WOE — BAN ĐẦU ({step_init.n_bins_before} bins)")
        _draw_er(axes[1, 0],  step_final,
                 f"Event Rate — SAU MERGE ({step_final.n_bins_after} bins) ✓")
        _draw_woe(axes[1, 1], step_final,
                  f"WOE — SAU MERGE ({step_final.n_bins_after} bins) ✓")

        # Legend mũi tên
        green_p = mpatches.Patch(color="#27ae60", label="Đi đúng chiều ✓")
        red_p   = mpatches.Patch(color="#e74c3c", label="Vi phạm monotonic ✗")
        fig.legend(handles=[green_p, red_p], loc="lower center",
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HÀM CHÍNH — enforce_monotonic_traced
# ══════════════════════════════════════════════════════════════════════════════

def _get_bin_stats(cuts: list, x: np.ndarray, y: np.ndarray):
    """Tính event_rate, woe, n_samples cho từng bin."""
    edges   = [-np.inf] + sorted(cuts) + [np.inf]
    bin_idx = pd.cut(x, bins=edges, labels=False, right=True)

    total_event    = max(int(y.sum()), 1)
    total_nonevent = max(int((1 - y).sum()), 1)

    event_rates, woe_vals, n_samples = [], [], []
    for i in range(len(edges) - 1):
        mask       = bin_idx == i
        n          = int(mask.sum())
        n_event    = int(y[mask].sum())
        n_nonevent = n - n_event
        er         = n_event / max(n, 1)
        pct_e      = n_event    / total_event
        pct_ne     = n_nonevent / total_nonevent
        woe        = np.log((pct_e + EPSILON) / (pct_ne + EPSILON))
        event_rates.append(er)
        woe_vals.append(round(woe, 4))
        n_samples.append(n)

    return event_rates, woe_vals, n_samples


def enforce_monotonic_traced(
    cuts      : list,
    x         : np.ndarray,
    y         : np.ndarray,
    direction : str,
    feature_name: str = "feature",
    verbose   : bool = True,
) -> MergeTrace:
    """
    Phiên bản có TRACE của _enforce_monotonic_by_merge.

    Mỗi vòng lặp:
        1. Tính event rate từng bin
        2. Tìm cặp vi phạm đầu tiên
        3. Log bước này vào MergeTrace
        4. Xóa cut-point giữa 2 bins vi phạm
        5. Lặp lại cho đến khi monotonic

    Returns:
        trace: MergeTrace object chứa toàn bộ lịch sử
        cuts_final: List cut-points sau khi đã enforce monotonic
    """
    trace      = MergeTrace(direction=direction, feature_name=feature_name)
    cuts       = sorted(list(cuts))
    step_no    = 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  BẮT ĐẦU enforce_monotonic_traced")
        print(f"  Feature   : {feature_name}")
        print(f"  Direction : {direction}")
        print(f"  Cuts ban đầu: {[round(c, 2) for c in cuts]}")
        print(f"{'─'*60}")

    # ── Bước 0: Ghi lại trạng thái ban đầu ───────────────────────────────
    er0, woe0, n0 = _get_bin_stats(cuts, x, y)
    step0 = MergeStep(
        step_no       = 0,
        cuts_before   = list(cuts),
        cuts_after    = list(cuts),
        event_rates   = er0,
        n_bins_before = len(cuts) + 1,
        n_bins_after  = len(cuts) + 1,
        merged_pair   = None,
        merged_cut    = None,
        violation_er  = None,
        direction     = direction,
        is_final      = False,
        woe_values    = woe0,
        n_samples     = n0,
    )
    trace.add_step(step0)

    if verbose:
        print(f"  Bước 0 (ban đầu): {len(cuts)+1} bins | "
              f"Event rates: {[f'{e*100:.1f}%' for e in er0]}")

    # ── Vòng lặp merge ────────────────────────────────────────────────────
    for iteration in range(200):
        er, woe, ns = _get_bin_stats(cuts, x, y)

        # Tìm vi phạm đầu tiên
        violation_idx = None
        for i in range(len(er) - 1):
            if direction == "ascending"  and er[i] > er[i+1] + EPSILON:
                violation_idx = i
                break
            if direction == "descending" and er[i] < er[i+1] - EPSILON:
                violation_idx = i
                break

        if violation_idx is None:
            # ── Đã monotonic: ghi bước cuối ──────────────────────────────
            er_final, woe_final, n_final = _get_bin_stats(cuts, x, y)
            final_step = MergeStep(
                step_no       = step_no + 1,
                cuts_before   = list(cuts),
                cuts_after    = list(cuts),
                event_rates   = er_final,
                n_bins_before = len(cuts) + 1,
                n_bins_after  = len(cuts) + 1,
                merged_pair   = None,
                merged_cut    = None,
                violation_er  = None,
                direction     = direction,
                is_final      = True,
                woe_values    = woe_final,
                n_samples     = n_final,
            )
            trace.add_step(final_step)

            if verbose:
                print(f"\n  ✓ MONOTONIC đạt được sau {step_no} lần gộp.")
                print(f"  Cuts cuối: {[round(c, 2) for c in cuts]}")
                print(f"  Event rates cuối: {[f'{e*100:.1f}%' for e in er_final]}")
                print(f"{'─'*60}\n")
            break

        # ── Ghi lại bước merge ───────────────────────────────────────────
        step_no   += 1
        cuts_old   = list(cuts)
        sorted_cuts = sorted(cuts)
        merged_cut  = sorted_cuts[violation_idx] if violation_idx < len(sorted_cuts) else None

        # Xóa cut-point
        if violation_idx < len(sorted_cuts):
            cuts = sorted_cuts[:violation_idx] + sorted_cuts[violation_idx + 1:]
        else:
            cuts = sorted_cuts

        er_new, woe_new, n_new = _get_bin_stats(cuts, x, y)

        step = MergeStep(
            step_no       = step_no,
            cuts_before   = cuts_old,
            cuts_after    = list(cuts),
            event_rates   = er,            # event rate CỦA CUTS_BEFORE
            n_bins_before = len(cuts_old) + 1,
            n_bins_after  = len(cuts) + 1,
            merged_pair   = (violation_idx, violation_idx + 1),
            merged_cut    = merged_cut,
            violation_er  = (er[violation_idx], er[violation_idx + 1]),
            direction     = direction,
            is_final      = False,
            woe_values    = er,            # sẽ update sau
            n_samples     = ns,
        )
        # Ghi woe của CUTS_BEFORE để visualize đúng trạng thái trước merge
        step.woe_values = woe

        trace.add_step(step)

        if verbose:
            arrow = "↑ nhưng" if direction == "ascending" else "↓ nhưng"
            print(f"  Bước {step_no}: Bin{violation_idx}({er[violation_idx]*100:.1f}%) "
                  f"{arrow} Bin{violation_idx+1}({er[violation_idx+1]*100:.1f}%) "
                  f"→ GỘP (xóa cut={merged_cut:.2f})")

    return trace, trace.final_cuts


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    N = 3000

    # Tạo data: income descending với bad rate, có nhiễu
    income   = np.random.lognormal(10, 0.8, N)
    bad_prob = 1 / (1 + np.exp((income - 22000) / 6000))
    bad_prob += np.random.normal(0, 0.08, N)        # Thêm nhiễu để tạo vi phạm
    bad_prob  = np.clip(bad_prob, 0.02, 0.98)
    y         = (np.random.rand(N) < bad_prob).astype(int)
    x         = income

    # Tạo 8 bins ban đầu bằng quantile (để có vi phạm cần gộp)
    pcts  = np.linspace(0, 100, 9)[1:-1]
    cuts  = sorted(list(np.unique(np.nanpercentile(x, pcts))))
    print(f"Cuts ban đầu ({len(cuts)+1} bins): {[round(c,1) for c in cuts]}")

    # ── Chạy với trace ──────────────────────────────────────────────────
    trace = enforce_monotonic_traced(
        cuts         = cuts,
        x            = x,
        y            = y,
        direction    = "descending",
        feature_name = "income",
        verbose      = True,
    )

    # ── Xem bảng log ────────────────────────────────────────────────────
    df_log = trace.summary()

    # ── Vẽ từng bước riêng ──────────────────────────────────────────────
    print("Vẽ bước 0 (ban đầu) và bước 1 (merge đầu tiên)...")
    fig0 = trace.plot_step(0)
    fig0.savefig("step_0_initial.png", dpi=120, bbox_inches="tight")

    if trace.n_steps > 1:
        fig1 = trace.plot_step(1)
        fig1.savefig("step_1_first_merge.png", dpi=120, bbox_inches="tight")

    # ── Vẽ tất cả bước (grid) ───────────────────────────────────────────
    print("Vẽ toàn bộ quá trình (grid)...")
    fig_all = trace.plot_steps(max_steps=None)
    fig_all.savefig("all_merge_steps.png", dpi=110, bbox_inches="tight")

    # ── Vẽ Before / After ───────────────────────────────────────────────
    print("Vẽ Before vs After...")
    fig_ba = trace.plot_before_after()
    fig_ba.savefig("before_after.png", dpi=120, bbox_inches="tight")

    print("\nĐã lưu: step_0_initial.png, step_1_first_merge.png, "
          "all_merge_steps.png, before_after.png")

    plt.show()