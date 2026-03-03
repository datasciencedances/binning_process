"""
HTML Report Compare Binning — So sánh nhiều thuật toán trên nhiều feature.
Hỗ trợ train/valid: fit trên train, so sánh final_woe và PSI (drift).
"""

import base64
import io
import sys
from typing import Optional, List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binning_process.compare import compare_methods
from binning_process.core.utils import cap_outliers, compute_woe_iv_table
EPSILON = 1e-9


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
    Trả về (psi_total, psi_per_bin).
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
    return n_total / max(n_total.sum(), 1)


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
    - Nếu truyền thêm df_valid: fit trên train; với mỗi thuật toán transform
      train và valid bằng final bins, so sánh bảng final_woe train vs valid
      và chỉ số PSI (drift).

    Mỗi tab = 1 feature, gồm:
      Section 1: Bảng so sánh (Method, IV, n_bins, Monotonic, ...).
      Section 2: Plot Init/Algo/Final (methods[name].plot()) — collapse theo method.
      Section 3: Plot merge steps (trace_.plot_steps()) — collapse theo method.
      Section 4 (khi có df_valid): So sánh final_woe train vs valid + PSI.
    """
    use_valid = df_valid is not None and len(df_valid) > 0
    y_train = df_train[target_col]

    # Thu thập dữ liệu từng feature
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
                print(f"  Lỗi {col}: {e}")
            per_feature.append({
                "feature": col,
                "result": pd.DataFrame(),
                "fitted": {},
                "train_valid_tables": {},
                "psi_by_method": {},
                "error": str(e),
            })
            continue

        # Base64 cho từng method: plot init/algo/final, plot_steps
        plot_init_algo_final: Dict[str, str] = {}
        plot_steps_b64: Dict[str, str] = {}

        for name, model in fitted.items():
            try:
                fig = model.plot(figsize=(14, 5))
                plot_init_algo_final[name] = fig_to_base64(fig, dpi=dpi)
            except Exception:
                plot_init_algo_final[name] = ""

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

        # Train vs Valid: final_woe và PSI
        train_valid_tables: Dict[str, Dict[str, str]] = {}
        psi_by_method: Dict[str, float] = {}

        if use_valid and target_col in df_valid.columns:
            y_valid = df_valid[target_col]
            for name, model in fitted.items():
                try:
                    # WOE train (từ model)
                    woe_train = model.final_woe_table_
                    if woe_train is None:
                        continue
                    cuts = model.final_cuts_
                    x_valid = df_valid[col]
                    # WOE valid: dùng cùng cuts, tính trên valid
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
                    # HTML table: so sánh bin, woe train, woe valid, n_total train, n_total valid
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
            "error": None,
        })

    # Build HTML
    html = _build_html(per_feature, use_valid=use_valid)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Đã ghi report: {output_path}")


def _build_html(per_feature: List[Dict[str, Any]], use_valid: bool) -> str:
    """Tạo nội dung HTML đầy đủ: tab theo feature, 3 (+1) section, collapse."""
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

        sec2_parts = []
        for name, b64 in data.get("plot_init_algo_final", {}).items():
            if not b64:
                sec2_parts.append(
                    f'<details class="collapse-block">'
                    f'<summary>{name}</summary><p>Không có ảnh.</p></details>'
                )
            else:
                sec2_parts.append(
                    f'<details class="collapse-block">'
                    f'<summary>{name}</summary>'
                    f'<img src="data:image/png;base64,{b64}" alt="{name}" class="report-img"/>'
                    f'</details>'
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

        overview_content = f'<h4>Bảng so sánh thuật toán</h4><div class="table-wrapper">{sec1}</div>'
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
            {"".join(sec2_parts)}
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
  </script>
</body>
</html>
"""
