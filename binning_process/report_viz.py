"""
Compare Binning Report (Viz) — phiên bản mở rộng, tập trung vào trực quan hóa.

- Giữ nguyên `report.py`, chỉ thêm file mới để dễ so sánh.
- Hỗ trợ:
  - Dashboard tổng quan (bảng summary + 1–2 biểu đồ overview).
  - Tabs theo feature, mỗi feature có:
    - Bảng so sánh thuật toán (IV, n_bins, Monotonic, Direction, ...).
    - Plot Final WOE / Event Rate / tỷ lệ mẫu (%) theo bin (per method).
    - Plot Init/Algo/Final + Merge Steps (giống báo cáo cũ).
    - Train vs Valid:
        * Biểu đồ phân phối raw train vs valid.
        * WOE train vs valid theo bin.
        * Số lượng mẫu train vs valid theo bin.
        * Bảng WOE train vs valid + PSI tổng.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binning_process.compare import compare_methods
from binning_process.core.utils import (
    cap_outliers,
    compute_psi as _compute_psi,
    compute_woe_iv_table,
    woe_table_to_pct_per_bin,
)
from binning_process.report import fig_to_base64


def _compute_ks_from_woe_table(woe_table: pd.DataFrame) -> float:
    """
    Tính KS từ bảng WOE (dùng n_event, n_nonevent).

    KS = max |CDF_event - CDF_nonevent| theo thứ tự bin.
    """
    if woe_table is None or woe_table.empty:
        return float("nan")

    n_event = woe_table["n_event"].astype(float).values
    n_nonevent = woe_table["n_nonevent"].astype(float).values
    total_event = max(n_event.sum(), 1.0)
    total_nonevent = max(n_nonevent.sum(), 1.0)

    cdf_event = np.cumsum(n_event) / total_event
    cdf_nonevent = np.cumsum(n_nonevent) / total_nonevent
    ks = np.max(np.abs(cdf_event - cdf_nonevent))
    return float(round(ks, 4))


def _plot_raw_distribution_train_valid(
    x_train: pd.Series,
    x_valid: Optional[pd.Series],
    feature_name: str,
) -> Optional[plt.Figure]:
    """Histogram phân phối raw (hoặc đã cap) train vs valid trên cùng 1 chart."""
    if x_valid is None or len(x_valid) == 0:
        return None

    x_tr = x_train.dropna().astype(float)
    x_va = x_valid.dropna().astype(float)
    if len(x_tr) == 0 or len(x_va) == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bins = 30
    ax.hist(x_tr, bins=bins, alpha=0.45, label="Train", color="#2980b9")
    ax.hist(x_va, bins=bins, alpha=0.45, label="Valid", color="#e67e22")
    ax.set_title(f"Phân phối Train vs Valid — {feature_name}", fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Tần suất")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_woe_train_valid(
    woe_train: pd.DataFrame,
    woe_valid: Optional[pd.DataFrame],
    feature_name: str,
    method_name: str,
) -> Optional[plt.Figure]:
    """Bar chart WOE train vs valid theo bin."""
    if woe_train is None or woe_train.empty:
        return None
    if woe_valid is None or woe_valid.empty:
        return None

    bins = woe_train["bin"].astype(str).tolist()
    w_tr = woe_train["woe"].astype(float).values
    w_va = woe_valid["woe"].astype(float).values
    if len(w_tr) != len(w_va):
        return None

    x_pos = np.arange(len(bins))
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.bar(x_pos - width / 2, w_tr, width=width, label="WOE Train", color="#3498db")
    ax.bar(x_pos + width / 2, w_va, width=width, label="WOE Valid", color="#e67e22")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("WOE")
    ax.set_title(f"WOE Train vs Valid — {feature_name} — {method_name}", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_count_train_valid(
    woe_train: pd.DataFrame,
    woe_valid: Optional[pd.DataFrame],
    feature_name: str,
    method_name: str,
) -> Optional[plt.Figure]:
    """Bar chart số lượng mẫu train vs valid theo bin."""
    if woe_train is None or woe_train.empty:
        return None
    if woe_valid is None or woe_valid.empty:
        return None

    bins = woe_train["bin"].astype(str).tolist()
    n_tr = woe_train["n_total"].astype(float).values
    n_va = woe_valid["n_total"].astype(float).values
    if len(n_tr) != len(n_va):
        return None

    x_pos = np.arange(len(bins))
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.bar(x_pos - width / 2, n_tr, width=width, label="Train", color="#1abc9c")
    ax.bar(x_pos + width / 2, n_va, width=width, label="Valid", color="#9b59b6")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Số mẫu")
    ax.set_title(f"Số mẫu Train vs Valid — {feature_name} — {method_name}", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_psi_by_bin(
    p_train: np.ndarray,
    p_valid: np.ndarray,
    psi_per_bin: np.ndarray,
    bins: List[str],
    feature_name: str,
    method_name: str,
) -> Optional[plt.Figure]:
    """Bar chart PSI theo bin (tùy chọn)."""
    if len(psi_per_bin) == 0 or len(psi_per_bin) != len(bins):
        return None

    x_pos = np.arange(len(bins))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.bar(x_pos, psi_per_bin, color="#c0392b")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("PSI per bin")
    ax.set_title(f"PSI theo Bin — {feature_name} — {method_name}", fontweight="bold")
    fig.tight_layout()
    return fig


def generate_compare_report_viz(
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
    Phiên bản mở rộng của generate_compare_report với visualize phong phú hơn.

    - Giữ logic compare_methods tương tự report.py.
    - Thêm Dashboard + các plot train vs valid per feature.
    """
    use_valid = df_valid is not None and len(df_valid) > 0
    y_train = df_train[target_col]

    per_feature: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for col in feature_cols:
        if col not in df_train.columns:
            if verbose:
                print(f"⚠ Bỏ qua feature không tồn tại: {col}")
            continue

        x_train = df_train[col]
        if verbose:
            print(f"  [Viz] Feature: {col} ...")

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
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"Lỗi feature {col}: {e}")
            per_feature.append(
                {
                    "feature": col,
                    "result": pd.DataFrame(),
                    "fitted": {},
                    "plot_init_algo_final": {},
                    "plot_steps_b64": {},
                    "train_valid_tables": {},
                    "psi_by_method": {},
                    "overview_final_plots": {},
                    "raw_dist_b64": "",
                    "woe_tv_plots": {},
                    "count_tv_plots": {},
                    "psi_bin_plots": {},
                    "ks_train": {},
                    "ks_valid": {},
                    "error": str(e),
                }
            )
            continue

        # Plot Init/Algo/Final & Merge Steps (giống báo cáo cũ)
        plot_init_algo_final: Dict[str, str] = {}
        plot_steps_b64: Dict[str, str] = {}
        overview_final_plots: Dict[str, str] = {}

        for name, model in fitted.items():
            # Overview final-only plot
            try:
                # Tránh import vòng: dùng model._draw_* trong report.py via fig_to_base64
                # Ở đây giả định model có final_woe_table_ giống BaseBinner.
                from binning_process.report import _plot_final_woe_er  # type: ignore

                fig_final = _plot_final_woe_er(model)
                overview_final_plots[name] = fig_to_base64(fig_final, dpi=dpi) if fig_final else ""
            except Exception:
                overview_final_plots[name] = ""

            # Full plot (Init/Algo/Final)
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

        # Train vs Valid + các plot bổ sung
        train_valid_tables: Dict[str, str] = {}
        psi_by_method: Dict[str, float] = {}
        ks_train: Dict[str, float] = {}
        ks_valid: Dict[str, float] = {}
        woe_tv_plots: Dict[str, str] = {}
        count_tv_plots: Dict[str, str] = {}
        psi_bin_plots: Dict[str, str] = {}

        x_valid_series: Optional[pd.Series] = None
        y_valid_series: Optional[pd.Series] = None

        if use_valid and target_col in df_valid.columns:
            x_valid_series = df_valid[col] if col in df_valid.columns else None
            y_valid_series = df_valid[target_col]

        # Raw distribution plot (per feature, không phụ thuộc method)
        raw_dist_b64 = ""
        if use_valid and x_valid_series is not None:
            try:
                fig_raw = _plot_raw_distribution_train_valid(
                    x_train=x_train,
                    x_valid=x_valid_series,
                    feature_name=col,
                )
                raw_dist_b64 = fig_to_base64(fig_raw, dpi=dpi) if fig_raw else ""
            except Exception:
                raw_dist_b64 = ""

        if use_valid and x_valid_series is not None and y_valid_series is not None:
            for name, model in fitted.items():
                try:
                    woe_train = model.final_woe_table_
                    if woe_train is None or woe_train.empty:
                        continue
                    cuts = model.final_cuts_
                    x_valid = x_valid_series

                    x_v = (
                        cap_outliers(x_valid, model.lower_pct, model.upper_pct)
                        if model.cap_outliers_
                        else x_valid
                    )
                    mask = ~x_v.isna()
                    if model.special_values:
                        mask &= ~x_v.isin(model.special_values)
                    x_v = x_v[mask].values.astype(float)
                    y_v = y_valid_series[mask].values.astype(int)
                    if len(x_v) == 0:
                        continue

                    # WOE valid + PSI
                    woe_valid = compute_woe_iv_table(cuts, x_v, y_v, feature_name=col)
                    p_train = woe_table_to_pct_per_bin(woe_train)
                    p_valid = woe_table_to_pct_per_bin(woe_valid)
                    if len(p_train) != len(p_valid):
                        psi_by_method[name] = float("nan")
                        psi_per_bin = np.array([])
                    else:
                        psi_total, psi_per_bin = _compute_psi(p_train, p_valid)
                        psi_by_method[name] = float(round(psi_total, 4))

                    # KS train/valid
                    ks_train[name] = _compute_ks_from_woe_table(woe_train)
                    if woe_valid is not None and not woe_valid.empty and len(woe_valid) == len(woe_train):
                        ks_valid[name] = _compute_ks_from_woe_table(woe_valid)
                    else:
                        ks_valid[name] = float("nan")

                    # Bảng so sánh WOE train vs valid
                    df_compare = pd.DataFrame(
                        {
                            "bin": woe_train["bin"],
                            "woe_train": woe_train["woe"],
                            "n_train": woe_train["n_total"],
                            "event_rate_train": woe_train["event_rate"],
                        }
                    )
                    if len(woe_valid) == len(woe_train):
                        df_compare["woe_valid"] = woe_valid["woe"].values
                        df_compare["n_valid"] = woe_valid["n_total"].values
                        df_compare["event_rate_valid"] = woe_valid["event_rate"].values
                    else:
                        df_compare["woe_valid"] = np.nan
                        df_compare["n_valid"] = np.nan
                        df_compare["event_rate_valid"] = np.nan

                    train_valid_tables[name] = df_compare.to_html(index=False, classes="table-compare")

                    # Tạo 1 figure duy nhất: subplots 1 hàng 3 cột
                    try:
                        bins = woe_train["bin"].astype(str).tolist()
                        x_pos = np.arange(len(bins))
                        w_tr = woe_train["woe"].astype(float).values
                        w_va = woe_valid["woe"].astype(float).values
                        n_tr = woe_train["n_total"].astype(float).values
                        n_va = woe_valid["n_total"].astype(float).values

                        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                        # 1 - WOE Train vs Valid
                        width = 0.38
                        axes[0].bar(x_pos - width / 2, w_tr, width=width, label="WOE Train", color="#3498db")
                        axes[0].bar(x_pos + width / 2, w_va, width=width, label="WOE Valid", color="#e67e22")
                        axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
                        axes[0].set_xticks(x_pos)
                        axes[0].set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
                        axes[0].set_ylabel("WOE")
                        axes[0].set_title("WOE", fontweight="bold")
                        axes[0].legend(fontsize=8)

                        # 2 - Số mẫu Train vs Valid
                        axes[1].bar(x_pos - width / 2, n_tr, width=width, label="Train", color="#1abc9c")
                        axes[1].bar(x_pos + width / 2, n_va, width=width, label="Valid", color="#9b59b6")
                        axes[1].set_xticks(x_pos)
                        axes[1].set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
                        axes[1].set_ylabel("Số mẫu")
                        axes[1].set_title("Số mẫu", fontweight="bold")
                        axes[1].legend(fontsize=8)

                        # 3 - PSI theo Bin (nếu có)
                        if len(psi_per_bin) and len(psi_per_bin) == len(bins):
                            axes[2].bar(x_pos, psi_per_bin, color="#c0392b")
                            axes[2].set_xticks(x_pos)
                            axes[2].set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
                            axes[2].set_ylabel("PSI per bin")
                            axes[2].set_title("PSI", fontweight="bold")
                        else:
                            axes[2].set_title("PSI", fontweight="bold")
                            axes[2].axis("off")

                        fig.tight_layout()
                        woe_tv_plots[name] = fig_to_base64(fig, dpi=dpi)
                        count_tv_plots[name] = ""
                        psi_bin_plots[name] = ""
                        plt.close(fig)
                    except Exception:
                        woe_tv_plots[name] = ""
                        count_tv_plots[name] = ""
                        psi_bin_plots[name] = ""

                except Exception:
                    train_valid_tables[name] = "<p>Lỗi tính WOE valid / PSI / KS.</p>"
                    psi_by_method[name] = float("nan")
                    ks_train[name] = float("nan")
                    ks_valid[name] = float("nan")
                    woe_tv_plots[name] = ""
                    count_tv_plots[name] = ""
                    psi_bin_plots[name] = ""
        else:
            # Không có df_valid: vẫn có thể tính KS train
            for name, model in fitted.items():
                try:
                    woe_train = model.final_woe_table_
                    if woe_train is None or woe_train.empty:
                        continue
                    ks_train[name] = _compute_ks_from_woe_table(woe_train)
                    ks_valid[name] = float("nan")
                except Exception:
                    ks_train[name] = float("nan")
                    ks_valid[name] = float("nan")

        # Tổng hợp vào summary_rows (per feature / method)
        if isinstance(result, pd.DataFrame) and not result.empty:
            for _, row in result.iterrows():
                m_name = str(row["Method"])
                summary_rows.append(
                    {
                        "feature": col,
                        "method": m_name,
                        "Group": row.get("Group", ""),
                        "IV": row.get("IV", None),
                        "n_bins": row.get("n_bins", None),
                        "Monotonic": row.get("Monotonic", ""),
                        "Direction": row.get("Direction", ""),
                        "IV_Rating": row.get("IV_Rating", ""),
                        "KS_train": ks_train.get(m_name, float("nan")),
                        "KS_valid": ks_valid.get(m_name, float("nan")),
                        "PSI": psi_by_method.get(m_name, float("nan")) if use_valid else float("nan"),
                    }
                )

        per_feature.append(
            {
                "feature": col,
                "result": result,
                "fitted": fitted,
                "plot_init_algo_final": plot_init_algo_final,
                "plot_steps_b64": plot_steps_b64,
                "train_valid_tables": train_valid_tables,
                "psi_by_method": psi_by_method,
                "overview_final_plots": overview_final_plots,
                "raw_dist_b64": raw_dist_b64,
                "woe_tv_plots": woe_tv_plots,
                "count_tv_plots": count_tv_plots,
                "psi_bin_plots": psi_bin_plots,
                "ks_train": ks_train,
                "ks_valid": ks_valid,
                "error": None,
            }
        )

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    html = _build_html_viz(summary_df, per_feature, use_valid=use_valid)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Đã ghi report (viz): {output_path}")


def _build_html_viz(
    summary_df: pd.DataFrame,
    per_feature: List[Dict[str, Any]],
    use_valid: bool,
) -> str:
    """Tạo HTML cho report viz: Dashboard + tabs theo feature."""
    dashboard_tables = ""
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        dashboard_tables = summary_df.to_html(index=False, classes="table-summary")

    # TODO: có thể bổ sung biểu đồ overview cấp dataset sau (PSI/KS per feature).
    # Để đơn giản, hiện tại chỉ hiển thị bảng summary.

    tab_buttons = []
    tab_panels = []

    for idx, data in enumerate(per_feature):
        feat = data["feature"]
        fid = f"f{idx}"
        tab_buttons.append(f'<button class="tab-btn" data-tab="{fid}">{feat}</button>')

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
                        f"<p>Không có plot (Init → Algo → Final).</p>"
                        f"</div>"
                    )
                else:
                    d_panels.append(
                        f'<div id="{algo_id}" class="algo-tab-panel" data-group="{detail_group}">'
                        f'<img src="data:image/png;base64,{b64}" alt="detail {m_name}" class="report-img"/>'
                        f"</div>"
                    )
            detail_tabs_html = (
                '<div class="algo-tabs">'
                '<div class="algo-tab-buttons">' + "".join(d_btns) + "</div>"
                + "".join(d_panels)
                + "</div>"
            )

        # Merge steps
        sec3_parts = []
        for name, b64 in (data.get("plot_steps_b64") or {}).items():
            if not b64:
                sec3_parts.append(
                    '<details class="collapse-block">'
                    f"<summary>{name}</summary><p>Không có bước merge.</p></details>"
                )
            else:
                sec3_parts.append(
                    '<details class="collapse-block">'
                    f"<summary>{name}</summary>"
                    f'<img src="data:image/png;base64,{b64}" alt="steps {name}" class="report-img"/>'
                    "</details>"
                )

        # Train vs Valid: bảng + PSI + các plot bổ sung
        sec4_parts = []
        if use_valid and data.get("train_valid_tables"):
            for name, table_html in data["train_valid_tables"].items():
                psi_val = data.get("psi_by_method", {}).get(name, "")
                if isinstance(psi_val, float) and np.isnan(psi_val):
                    psi_str = "—"
                else:
                    psi_str = str(psi_val)
                ks_tr = data.get("ks_train", {}).get(name, float("nan"))
                ks_va = data.get("ks_valid", {}).get(name, float("nan"))
                ks_tr_str = "—" if isinstance(ks_tr, float) and np.isnan(ks_tr) else str(ks_tr)
                ks_va_str = "—" if isinstance(ks_va, float) and np.isnan(ks_va) else str(ks_va)

                woe_plot_b64 = (data.get("woe_tv_plots") or {}).get(name, "")
                count_plot_b64 = (data.get("count_tv_plots") or {}).get(name, "")
                psi_bin_b64 = (data.get("psi_bin_plots") or {}).get(name, "")

                img_parts = []
                if woe_plot_b64:
                    img_parts.append(
                        f'<img src="data:image/png;base64,{woe_plot_b64}" alt="WOE train vs valid {name}" class="report-img"/>'
                    )
                if count_plot_b64:
                    img_parts.append(
                        f'<img src="data:image/png;base64,{count_plot_b64}" alt="Count train vs valid {name}" class="report-img"/>'
                    )
                if psi_bin_b64:
                    img_parts.append(
                        f'<img src="data:image/png;base64,{psi_bin_b64}" alt="PSI by bin {name}" class="report-img"/>'
                    )

                sec4_parts.append(
                    '<details class="collapse-block">'
                    f"<summary>{name} — PSI = {psi_str} | KS(train) = {ks_tr_str} | KS(valid) = {ks_va_str}</summary>"
                    + "".join(img_parts)
                    + f'<div class="table-wrapper">{table_html}</div>'
                    "</details>"
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
                        f"<p>Không có plot final.</p>"
                        f"</div>"
                    )
                else:
                    panels.append(
                        f'<div id="{algo_id}" class="algo-tab-panel" data-group="{fid}">'
                        f'<img src="data:image/png;base64,{b64}" alt="final {m_name}" class="report-img"/>'
                        f"</div>"
                    )
            algo_tabs_html = (
                '<div class="algo-tabs">'
                '<div class="algo-tab-buttons">' + "".join(btns) + "</div>"
                + "".join(panels)
                + "</div>"
            )

        # Raw distribution plot (per feature)
        raw_dist_b64 = data.get("raw_dist_b64", "")
        raw_dist_html = ""
        if raw_dist_b64:
            raw_dist_html = (
                "<h4>Phân phối Train vs Valid (Raw)</h4>"
                f'<img src="data:image/png;base64,{raw_dist_b64}" alt="raw dist {feat}" class="report-img"/>'
            )

        overview_content = (
            "<h4>Bảng so sánh thuật toán</h4><div class=\"table-wrapper\">"
            f"{sec1}</div>"
        )
        if algo_tabs_html:
            overview_content += "<h4>Final WOE & Event Rate theo thuật toán</h4>" + algo_tabs_html
        if raw_dist_html:
            overview_content += raw_dist_html
        if sec4_parts:
            overview_content += "<h4>Train vs Valid — Final WOE, KS & PSI</h4>" + "".join(sec4_parts)

        panel = f"""
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
"""
        tab_panels.append(panel)

    return _html_document_viz(
        dashboard_tables=dashboard_tables,
        tab_buttons="".join(tab_buttons),
        tab_panels="".join(tab_panels),
    )


def _html_document_viz(
    dashboard_tables: str,
    tab_buttons: str,
    tab_panels: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8"/>
  <title>Compare Binning Report (Viz)</title>
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
    .dashboard-section {{ background: #fff; padding: 1rem 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }}
    .dashboard-section h2 {{ margin-top: 0; color: #2c3e50; }}
    .table-summary {{ margin-top: 0.5rem; }}
  </style>
</head>
<body>
  <h1>Compare Binning Report (Viz)</h1>
  <!-- section dashboard-section đã bị ẩn theo yêu cầu -->
  <section class="dashboard-section" style="display:none;">
    <h2>Dashboard tổng quan</h2>
    <p>Bảng summary theo Feature / Method (IV, n_bins, Monotonic, Direction, KS, PSI).</p>
    <div class="table-wrapper">
      {dashboard_tables}
    </div>
  </section>
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
          var btn = document.querySelector('.algo-tab-btn[data-group=\"' + g + '\"][data-algo=\"' + id + '\"]');
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

