"""
HTML Report Compare Binning — so sánh nhiều thuật toán trên nhiều feature.
Hỗ trợ:
- N feature (vd 20), mỗi feature = 1 tab.
- Overview: bảng so sánh + (nếu có) Train vs Valid (final WOE & PSI) + tab theo thuật toán cho final WOE/Event rate.
- Chi tiết: plot Init/Algo/Final và plot Merge Steps (collapse theo thuật toán).
"""

import base64
import io
import sys
from typing import Optional, List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binning_process.compare import compare_methods
from binning_process.core.utils import EPSILON, cap_outliers, compute_woe_iv_table


def fig_to_base64(fig, dpi: int = 100, **savefig_kw) -> str:
    """Chuyển matplotlib Figure thành chuỗi base64 PNG để nhúng HTML."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, **savefig_kw)
    buf.seek(0)
    out = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return out


def compute_psi(
    p_train: np.ndarray,
    p_valid: np.ndarray,
    epsilon: float = EPSILON,
) -> Tuple[float, np.ndarray]:
    """
    PSI (Population Stability Index) theo bin.
    PSI_i = (p_valid_i - p_train_i) * ln(p_valid_i / p_train_i)
    """
    p_train = np.asarray(p_train, dtype=float)
    p_valid = np.asarray(p_valid, dtype=float)
    p_train = np.clip(p_train, epsilon, 1.0)
    p_valid = np.clip(p_valid, epsilon, 1.0)
    psi_bin = (p_valid - p_train) * np.log(p_valid / p_train)
    return float(np.sum(psi_bin)), psi_bin


def _woe_table_to_pct_per_bin(woe_table: pd.DataFrame) -> np.ndarray:
    """Lấy tỷ lệ mẫu mỗi bin (n_total / sum(n_total)) từ bảng WOE."""
    n_total = woe_table["n_total"].values.astype(float)
    return n_total / max(n_total.sum(), 1.0)


def _plot_final_woe_er(model) -> Optional[plt.Figure]:
    """
    Plot final WOE, final event rate và số lượng mẫu (%) theo bin
    cho Overview (1 hàng, 3 cột).
    """
    df = getattr(model, "final_woe_table_", None)
    if df is None or df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # Đảm bảo axes là mảng 1D có 3 phần tử
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    ax_woe, ax_er, ax_n = axes

    x_pos = list(range(len(df)))
    # WOE
    colors = ["#e74c3c" if w >= 0 else "#27ae60" for w in df["woe"]]
    ax_woe.bar(x_pos, df["woe"], color=colors, edgecolor="white", linewidth=0.8)
    ax_woe.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_woe.set_title("Final WOE theo Bin", fontweight="bold")
    ax_woe.set_xticks(x_pos)
    ax_woe.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
    ax_woe.set_ylabel("WOE")
    for i, (w, n) in enumerate(zip(df["woe"], df["n_total"])):
        offset = 0.03 if w >= 0 else -0.06
        ax_woe.text(i, w + offset, f"{w:.3f}\nn={n}", ha="center", fontsize=7.5)

    # Event rate
    er_pct = df["event_rate"] * 100
    ax_er.plot(x_pos, er_pct, marker="o", color="#2980b9",
               linewidth=2, markersize=8, zorder=3)
    ax_er.fill_between(x_pos, er_pct, alpha=0.12, color="#2980b9")

    arrow = "↑" if getattr(model, "direction_", "ascending") == "ascending" else "↓"
    ax_er.set_title(f"Final Event Rate (%) theo Bin  {arrow}", fontweight="bold")
    ax_er.set_xticks(x_pos)
    ax_er.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
    ax_er.set_ylabel("Event Rate (%)")
    for i, er in enumerate(er_pct):
        ax_er.text(i, er + 0.3, f"{er:.1f}%", ha="center", fontsize=8)

    # Số lượng mẫu từng bin (% n_total) + đường min_bin_size
    total_n = df["n_total"].sum()
    n_pct = df["n_total"] / max(total_n, 1) * 100
    ax_n.bar(x_pos, n_pct, color="#9b59b6", edgecolor="white", linewidth=0.8, zorder=3)
    min_bin = getattr(model, "min_bin_size", 0.0)
    if min_bin and min_bin > 0:
        ax_n.axhline(min_bin * 100, color="red", linestyle="--", linewidth=1.0, label=f"min_bin_size = {min_bin*100:.1f}%")
        ax_n.legend(fontsize=7)
    ax_n.set_title("Tỷ lệ số mẫu (%) từng Bin", fontweight="bold")
    ax_n.set_xticks(x_pos)
    ax_n.set_xticklabels(df["bin"], rotation=35, ha="right", fontsize=8)
    ax_n.set_ylabel("Tỷ lệ mẫu (%)")
    for i, (cnt, pct) in enumerate(zip(df["n_total"], n_pct)):
        ax_n.text(i, pct + 0.5, f"{cnt}\n({pct:.1f}%)", ha="center", fontsize=7.5)

    fig.tight_layout()
    return fig


def generate_compare_report(
    df_train: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    output_path: str,
    df_valid: Optional[pd.DataFrame] = None,
    max_bins: int = 6,
    n_init_bins: int = 20,
    max_depth: int = 4,
    special_values: Optional[list] = None,
    methods: Optional[list] = None,
    dpi: int = 100,
    verbose: bool = True,
) -> None:
    """
    Chạy compare_methods cho từng feature, xuất file HTML.

    - Nếu chỉ truyền df_train: fit và báo cáo như bình thường.
    - Nếu truyền thêm df_valid: fit trên train; với mỗi thuật toán so sánh
      final_woe_table_ trên train vs valid và tính PSI.
    """
    use_valid = df_valid is not None and len(df_valid) > 0
    y_train = df_train[target_col]

    per_feature: List[Dict[str, Any]] = []

    for col in feature_cols:
        if col not in df_train.columns:
            if verbose:
                print(f"⚠ Bỏ qua feature không tồn tại: {col}")
            continue

        x_train = df_train[col]
        if verbose:
            print(f"  Feature: {col} ...")

        try:
            _stdout = sys.stdout
            if not verbose:
                sys.stdout = io.StringIO()
            try:
                result, fitted = compare_methods(
                    x=x_train,
                    y=y_train,
                    feature_name=col,
                    max_bins=max_bins,
                    n_init_bins=n_init_bins,
                    max_depth=max_depth,
                    special_values=special_values,
                    methods=methods,
                )
            finally:
                if not verbose:
                    sys.stdout = _stdout
        except Exception as e:
            if verbose:
                print(f"Lỗi feature {col}: {e}")
            per_feature.append({
                "feature": col,
                "result": pd.DataFrame(),
                "fitted": {},
                "plot_init_algo_final": {},
                "plot_steps_b64": {},
                "train_valid_tables": {},
                "psi_by_method": {},
                "overview_final_plots": {},
                "error": str(e),
            })
            continue

        # Plot Init/Algo/Final & Merge Steps
        plot_init_algo_final: Dict[str, str] = {}
        plot_steps_b64: Dict[str, str] = {}
        overview_final_plots: Dict[str, str] = {}

        for name, model in fitted.items():
            # Overview final-only plot
            try:
                fig_final = _plot_final_woe_er(model)
                overview_final_plots[name] = fig_to_base64(fig_final, dpi=dpi) if fig_final else ""
            except Exception:
                overview_final_plots[name] = ""

            # Full plot (Init/Algo/Final) — dùng chart nhỏ hơn cho phần Chi tiết
            try:
                fig = model.plot(figsize=(18, 4))
                plot_init_algo_final[name] = fig_to_base64(fig, dpi=dpi)
            except Exception:
                plot_init_algo_final[name] = ""

            # Merge steps
            trace = getattr(model, "trace_", None)
            if trace is not None and getattr(trace, "steps", None) and len(trace.steps) > 0:
                try:
                    fig = trace.plot_steps(max_steps=None)
                    if fig is not None:
                        plot_steps_b64[name] = fig_to_base64(fig, dpi=dpi)
                    else:
                        plot_steps_b64[name] = ""
                except Exception:
                    plot_steps_b64[name] = ""
            else:
                plot_steps_b64[name] = ""

        # Train vs Valid
        train_valid_tables: Dict[str, str] = {}
        psi_by_method: Dict[str, float] = {}

        if use_valid and target_col in df_valid.columns:
            y_valid = df_valid[target_col]
            for name, model in fitted.items():
                try:
                    woe_train = model.final_woe_table_
                    if woe_train is None or woe_train.empty:
                        continue
                    cuts = model.final_cuts_
                    x_valid = df_valid[col]

                    x_v = cap_outliers(x_valid, model.lower_pct, model.upper_pct) if model.cap_outliers_ else x_valid
                    mask = ~x_v.isna()
                    if model.special_values:
                        mask &= ~x_v.isin(model.special_values)
                    x_v = x_v[mask].values.astype(float)
                    y_v = y_valid[mask].values.astype(int)
                    if len(x_v) == 0:
                        continue

                    woe_valid = compute_woe_iv_table(cuts, x_v, y_v, feature_name=col)
                    p_train = _woe_table_to_pct_per_bin(woe_train)
                    p_valid = _woe_table_to_pct_per_bin(woe_valid)
                    if len(p_train) != len(p_valid):
                        psi_by_method[name] = float("nan")
                    else:
                        psi_total, _ = compute_psi(p_train, p_valid)
                        psi_by_method[name] = round(psi_total, 4)

                    df_compare = pd.DataFrame({
                        "bin": woe_train["bin"],
                        "woe_train": woe_train["woe"],
                        "n_train": woe_train["n_total"],
                        "event_rate_train": woe_train["event_rate"],
                    })
                    if len(woe_valid) == len(woe_train):
                        df_compare["woe_valid"] = woe_valid["woe"].values
                        df_compare["n_valid"] = woe_valid["n_total"].values
                        df_compare["event_rate_valid"] = woe_valid["event_rate"].values
                    else:
                        df_compare["woe_valid"] = np.nan
                        df_compare["n_valid"] = np.nan
                        df_compare["event_rate_valid"] = np.nan

                    train_valid_tables[name] = df_compare.to_html(index=False, classes="table-compare")
                except Exception:
                    train_valid_tables[name] = "<p>Lỗi tính WOE valid / PSI.</p>"
                    psi_by_method[name] = float("nan")

        per_feature.append({
            "feature": col,
            "result": result,
            "fitted": fitted,
            "plot_init_algo_final": plot_init_algo_final,
            "plot_steps_b64": plot_steps_b64,
            "train_valid_tables": train_valid_tables,
            "psi_by_method": psi_by_method,
            "overview_final_plots": overview_final_plots,
            "error": None,
        })

    html = _build_html(per_feature, use_valid=use_valid)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Đã ghi report: {output_path}")


def _build_html(per_feature: List[Dict[str, Any]], use_valid: bool) -> str:
    """Tạo HTML: tab theo feature; Overview + Chi tiết."""
    tab_buttons = []
    tab_panels = []

    for idx, data in enumerate(per_feature):
        feat = data["feature"]
        fid = f"f{idx}"
        tab_buttons.append(
            f'<button class="tab-btn" data-tab="{fid}">{feat}</button>'
        )

        sec1 = ""
        if data.get("error"):
            sec1 = f'<p class="error">Lỗi: {data["error"]}</p>'
        elif isinstance(data.get("result"), pd.DataFrame) and not data["result"].empty:
            sec1 = data["result"].to_html(index=False, classes="table-result")

        # Chi tiết: tabs theo thuật toán cho plot Init → Algo → Final
        detail_tabs_html = ""
        detail_plots = data.get("plot_init_algo_final") or {}
        if detail_plots:
            d_btns = []
            d_panels = []
            detail_group = f"{fid}-detail"
            for m_name, b64 in detail_plots.items():
                algo_id = f"{detail_group}-{m_name}"
                d_btns.append(
                    f'<button class="algo-tab-btn" data-group="{detail_group}" data-algo="{algo_id}">{m_name}</button>'
                )
                if not b64:
                    d_panels.append(
                        f'<div id="{algo_id}" class="algo-tab-panel" data-group="{detail_group}">'
                        f'<p>Không có plot (Init → Algo → Final).</p>'
                        f'</div>'
                    )
                else:
                    d_panels.append(
                        f'<div id="{algo_id}" class="algo-tab-panel" data-group="{detail_group}">'
                        f'<img src="data:image/png;base64,{b64}" alt="detail {m_name}" class="report-img"/>'
                        f'</div>'
                    )
            detail_tabs_html = (
                '<div class="algo-tabs">'
                '<div class="algo-tab-buttons">' + "".join(d_btns) + "</div>"
                + "".join(d_panels)
                + "</div>"
            )

        sec3_parts = []
        for name, b64 in data.get("plot_steps_b64", {}).items():
            if not b64:
                sec3_parts.append(
                    f'<details class="collapse-block">'
                    f'<summary>{name}</summary><p>Không có bước merge.</p></details>'
                )
            else:
                sec3_parts.append(
                    f'<details class="collapse-block">'
                    f'<summary>{name}</summary>'
                    f'<img src="data:image/png;base64,{b64}" alt="steps {name}" class="report-img"/>'
                    f'</details>'
                )

        sec4_parts = []
        if use_valid and data.get("train_valid_tables"):
            for name, table_html in data["train_valid_tables"].items():
                psi_val = data.get("psi_by_method", {}).get(name, "")
                if isinstance(psi_val, float) and np.isnan(psi_val):
                    psi_str = "—"
                else:
                    psi_str = str(psi_val)
                sec4_parts.append(
                    f'<details class="collapse-block">'
                    f'<summary>{name} — PSI = {psi_str}</summary>'
                    f'<div class="table-wrapper">{table_html}</div>'
                    f'</details>'
                )

        # Tabs theo thuật toán cho final WOE & Event rate (Overview)
        algo_tabs_html = ""
        overview_final = data.get("overview_final_plots") or {}
        if overview_final:
            btns = []
            panels = []
            for m_name, b64 in overview_final.items():
                algo_id = f"{fid}-{m_name}"
                btns.append(
                    f'<button class="algo-tab-btn" data-group="{fid}" data-algo="{algo_id}">{m_name}</button>'
                )
                if not b64:
                    panels.append(
                        f'<div id="{algo_id}" class="algo-tab-panel" data-group="{fid}">'
                        f'<p>Không có plot final.</p>'
                        f'</div>'
                    )
                else:
                    panels.append(
                        f'<div id="{algo_id}" class="algo-tab-panel" data-group="{fid}">'
                        f'<img src="data:image/png;base64,{b64}" alt="final {m_name}" class="report-img"/>'
                        f'</div>'
                    )
            algo_tabs_html = (
                '<div class="algo-tabs">'
                '<div class="algo-tab-buttons">' + "".join(btns) + "</div>"
                + "".join(panels)
                + "</div>"
            )

        overview_content = f'<h4>Bảng so sánh thuật toán</h4><div class="table-wrapper">{sec1}</div>'
        if algo_tabs_html:
            overview_content += '<h4>Final WOE & Event Rate theo thuật toán</h4>' + algo_tabs_html
        if sec4_parts:
            overview_content += '<h4>Train vs Valid — Final WOE & PSI</h4>' + "".join(sec4_parts)

        panel = f'''
        <div id="{fid}" class="tab-panel">
          <h2>Feature: {feat}</h2>
          <section>
            <h3>Overview</h3>
            {overview_content}
          </section>
          <section>
            <h3>Chi tiết</h3>
            <h4>Plot Kết quả Binning (Init → Algo → Final)</h4>
            {detail_tabs_html}
            <h4>Plot Merge Steps (Algo → Final)</h4>
            {"".join(sec3_parts)}
          </section>
        </div>
'''
        tab_panels.append(panel)

    return _html_document(
        "".join(tab_buttons),
        "".join(tab_panels),
    )


def _html_document(tab_buttons: str, tab_panels: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8"/>
  <title>Compare Binning Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #f5f5f5; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }}
    .tab-btn {{ padding: 0.5rem 1rem; cursor: pointer; border: 1px solid #ccc; background: #fff; border-radius: 4px; }}
    .tab-btn:hover {{ background: #eee; }}
    .tab-btn.active {{ background: #2980b9; color: #fff; border-color: #2980b9; }}
    .tab-panel {{ display: none; background: #fff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .tab-panel.active {{ display: block; }}
    section {{ margin-bottom: 2rem; }}
    section h3 {{ margin-top: 0; color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }}
    section h4 {{ margin: 1rem 0 0.5rem; color: #34495e; font-size: 1rem; }}
    .collapse-block {{ margin: 0.5rem 0; border: 1px solid #ddd; border-radius: 4px; }}
    .collapse-block summary {{ padding: 0.5rem 1rem; cursor: pointer; background: #f8f9fa; font-weight: bold; }}
    .collapse-block summary:hover {{ background: #eee; }}
    .report-img {{ max-width: 100%; height: auto; display: block; margin: 0.5rem 0; }}
    .table-wrapper {{ overflow-x: auto; }}
    .table-wrapper table {{ border-collapse: collapse; width: 100%; }}
    .table-wrapper th, .table-wrapper td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }}
    .table-wrapper th {{ background: #34495e; color: #fff; }}
    .error {{ color: #c0392b; }}
    .algo-tabs {{ margin-top: 0.5rem; }}
    .algo-tab-buttons {{ display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.5rem; }}
    .algo-tab-btn {{ padding: 0.25rem 0.6rem; cursor: pointer; border: 1px solid #bbb; background: #fafafa; border-radius: 3px; font-size: 0.85rem; }}
    .algo-tab-btn.active {{ background: #16a085; color: #fff; border-color: #16a085; }}
    .algo-tab-panel {{ display: none; }}
    .algo-tab-panel.active {{ display: block; }}
  </style>
</head>
<body>
  <h1>Compare Binning Report</h1>
  <div class="tabs" role="tablist">
    {tab_buttons}
  </div>
  {tab_panels}
  <script>
    (function() {{
      var buttons = document.querySelectorAll(".tab-btn");
      var panels = document.querySelectorAll(".tab-panel");
      function show(id) {{
        panels.forEach(function(p) {{ p.classList.remove("active"); }});
        buttons.forEach(function(b) {{ b.classList.remove("active"); }});
        var panel = document.getElementById(id);
        if (panel) panel.classList.add("active");
        var btn = document.querySelector('.tab-btn[data-tab="' + id + '"]');
        if (btn) btn.classList.add("active");
      }}
      buttons.forEach(function(btn) {{
        btn.addEventListener("click", function() {{ show(this.getAttribute("data-tab")); }});
      }});
      if (buttons.length) show(buttons[0].getAttribute("data-tab"));
    }})();

    // Algo tabs per feature
    (function() {{
      var groups = {{}};
      document.querySelectorAll(".algo-tab-btn").forEach(function(btn) {{
        var g = btn.getAttribute("data-group");
        if (!groups[g]) groups[g] = {{buttons: [], panels: []}};
        groups[g].buttons.push(btn);
      }});
      document.querySelectorAll(".algo-tab-panel").forEach(function(p) {{
        var g = p.getAttribute("data-group");
        if (!groups[g]) groups[g] = {{buttons: [], panels: []}};
        groups[g].panels.push(p);
      }});
      Object.keys(groups).forEach(function(g) {{
        var group = groups[g];
        function showAlgo(id) {{
          group.panels.forEach(function(p) {{ p.classList.remove("active"); }});
          group.buttons.forEach(function(b) {{ b.classList.remove("active"); }});
          var panel = document.getElementById(id);
          if (panel) panel.classList.add("active");
          var btn = document.querySelector('.algo-tab-btn[data-group="' + g + '"][data-algo="' + id + '"]');
          if (btn) btn.classList.add("active");
        }}
        group.buttons.forEach(function(btn) {{
          btn.addEventListener("click", function() {{
            showAlgo(this.getAttribute("data-algo"));
          }});
        }});
        if (group.buttons.length) {{
          showAlgo(group.buttons[0].getAttribute("data-algo"));
        }}
      }});
    }})();
  </script>
</body>
</html>
"""

