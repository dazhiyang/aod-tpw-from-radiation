"""
12.test_analysis: OE test error diagnostics (extensible plotting scaffold).

First plot implemented:
- Violin comparison of **bias error** and **fractional error** for OE-only
- Variables: ``beta`` and ``alpha``
- Dataset:
  - Test OE prediction table: ``Data/<STATION>_<YEAR>_pred_oe<suffix>.txt``
  - Train OE retrieval table: ``Data/<STATION>_<YEAR>_train_oe<suffix>.txt`` (for FGE % diagnostics)

Error definition (reference = **AERONET**):
- fractional gross error: ``2*abs(pred - ref)/(pred + ref)``  (NaN where ``pred + ref <= 0``)

Plot styling matches ``11.train_analysis.R`` (9 pt, Times New Roman, 160 mm width).

This script is structured to allow adding more plot builders, similar to step 11.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from plotnine import (
    aes,
    coord_fixed,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_grid,
    facet_wrap,
    geom_boxplot,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    geom_violin,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_bw,
)

PROJECT = Path(__file__).resolve().parent.parent
LAM550_UM = 0.55
LAMBDA_REF_UM = float(os.environ.get("ANGSTROM_BETA_REF_UM", "1.0"))

STATION = os.environ.get("STATION", "PAL")
YEAR = int(os.environ.get("YEAR", "2024"))
LHS_N = int(os.environ.get("LHS_N", "500"))
_n = LHS_N
_k_suffix = "_0.5k" if _n == 500 else f"_{_n / 1000:g}k" if _n >= 1000 else f"_{_n}"

TEST_COMBINED = Path(
    os.environ.get("TEST_COMBINED", str(PROJECT / "Data" / f"{STATION}_{YEAR}_pred_oe{_k_suffix}.txt"))
)
TRAIN_OE = Path(os.environ.get("TRAIN_OE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train_oe{_k_suffix}.txt")))
SHAP_ALPHA = Path(os.environ.get("SHAP_ALPHA", str(PROJECT / "Data" / f"{STATION}_{YEAR}_shap_oe_alpha{_k_suffix}.txt")))
SHAP_BETA = Path(os.environ.get("SHAP_BETA", str(PROJECT / "Data" / f"{STATION}_{YEAR}_shap_oe_beta{_k_suffix}.txt")))
IRRADIANCE_IN = Path(
    os.environ.get("IRRADIANCE_IN", str(PROJECT / "Data" / f"{STATION}_{YEAR}_test_irradiance{_k_suffix}.txt"))
)

OUTPUT_FIG = Path(
    os.environ.get(
        "OUTPUT_FIG",
        str(PROJECT / "tex" / "figures" / f"{STATION}_{YEAR}_oe_train_test_error_violin{_k_suffix}.pdf"),
    )
)

FIG_W_MM = float(os.environ.get("FIG_W_MM", "160"))
FIG_H_MM = float(os.environ.get("FIG_H_MM", "95"))

# Match ``11.train_analysis.R`` (plot.size <- 9).
_PT = 9
_LW = 0.3
ANNOT_Y = 1.5
# AOD SHAP summary plot x-axis limits (matches 12.test_analysis.R).
SHAP_AOD_XLIM = (-0.52, 0.52)
WONG_ORANGE = "#E69F00"  # MERRA-2
WONG_BLUE = "#56B4E9"    # TabPFN OE (test)
WONG_GREEN = "#009E73"   # DHI
WONG_FEATURE_CMAP = LinearSegmentedColormap.from_list(
    "wong_blue_light_orange",
    [WONG_BLUE, "#F2F2F2", WONG_ORANGE],
)

DATASET_ORDER = [
    "MERRA-2",
    "TabPFN",
]
DATASET_COLORS = [WONG_ORANGE, WONG_BLUE]

XAI_FEATURE_LABELS: dict[str, str] = {
    "ghi": r"$G_h$",
    "dhi": r"$D_h$",
    "bni": r"$B_n$",
    "zenith": r"$\theta_z$",
    "merra_ALPHA": r"$\alpha_{\mathrm{merra2}}$",
    "merra_BETA": r"$\beta_{\mathrm{merra2}}$",
    "merra_ALBEDO": r"$\rho$",
    "merra_TO3": r"$u_o$",
    "merra_PS": r"$p_s$",
    "merra_TQV": r"$w$",
}


def _xai_feature_display_names(columns: list[str]) -> list[str]:
    return [XAI_FEATURE_LABELS.get(c, c) for c in columns]

def _validate_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns in {path}: {miss}")


def _aod550_from_beta_alpha(beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return beta * (LAM550_UM / LAMBDA_REF_UM) ** (-alpha)


def _load_long_error_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TEST_COMBINED.is_file():
        raise SystemExit(f"Missing TEST_COMBINED: {TEST_COMBINED}")
    if not TRAIN_OE.is_file():
        raise SystemExit(f"Missing TRAIN_OE: {TRAIN_OE}")

    te = pd.read_csv(TEST_COMBINED, sep="\t", parse_dates=["time_utc"])
    tr = pd.read_csv(TRAIN_OE, sep="\t", parse_dates=["time_utc"])

    _validate_columns(
        te,
        ["beta_pred_oe", "alpha_pred_oe", "merra_BETA", "merra_ALPHA", "aeronet_aod550", "aeronet_alpha"],
        TEST_COMBINED,
    )
    _validate_columns(
        tr,
        ["beta_oe", "alpha_oe", "merra_BETA", "merra_ALPHA", "aeronet_aod550", "aeronet_alpha"],
        TRAIN_OE,
    )

    # AOD_550 + alpha representations
    te_aod_oe = _aod550_from_beta_alpha(
        te["beta_pred_oe"].to_numpy(dtype=float), te["alpha_pred_oe"].to_numpy(dtype=float)
    )
    te_alpha_oe = te["alpha_pred_oe"].to_numpy(dtype=float)
    te_aod_merra = _aod550_from_beta_alpha(te["merra_BETA"].to_numpy(dtype=float), te["merra_ALPHA"].to_numpy(dtype=float))
    te_alpha_merra = te["merra_ALPHA"].to_numpy(dtype=float)
    te_aod_ae = te["aeronet_aod550"].to_numpy(dtype=float)
    te_alpha_ae = te["aeronet_alpha"].to_numpy(dtype=float)
    tr_aod_oe = _aod550_from_beta_alpha(tr["beta_oe"].to_numpy(dtype=float), tr["alpha_oe"].to_numpy(dtype=float))
    tr_alpha_oe = tr["alpha_oe"].to_numpy(dtype=float)
    tr_aod_merra = _aod550_from_beta_alpha(tr["merra_BETA"].to_numpy(dtype=float), tr["merra_ALPHA"].to_numpy(dtype=float))
    tr_alpha_merra = tr["merra_ALPHA"].to_numpy(dtype=float)
    tr_aod_ae = tr["aeronet_aod550"].to_numpy(dtype=float)
    tr_alpha_ae = tr["aeronet_alpha"].to_numpy(dtype=float)

    def _fge(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
        out = np.full_like(pred, np.nan, dtype=float)
        den = pred + ref
        m = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(den) & (den > 0.0)
        out[m] = 2.0 * np.abs(pred[m] - ref[m]) / den[m]
        return out

    def _summary_metrics(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float, float, int]:
        """MBE/RMSE/FB/FGE on finite pairs (FGE is mean row-wise FGE)."""
        m = np.isfinite(pred) & np.isfinite(ref)
        n = int(np.sum(m))
        if n == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        p = pred[m]
        r = ref[m]
        mbe = float(np.mean(p - r))
        rmse = float(np.sqrt(np.mean((p - r) ** 2)))
        den_fb = float(np.mean(p) + np.mean(r))
        fb = float(2.0 * (np.mean(r) - np.mean(p)) / den_fb) if den_fb > 0 else float("nan")
        fge_vec = _fge(p, r)
        m_fge = np.isfinite(fge_vec)
        fge = float(np.mean(fge_vec[m_fge])) if int(np.sum(m_fge)) > 0 else float("nan")
        return mbe, rmse, fb, fge, n

    def _pct_improve(fge_model: float, fge_baseline: float) -> float:
        """Positive value means model improves over baseline."""
        if not np.isfinite(fge_model) or not np.isfinite(fge_baseline) or fge_baseline == 0:
            return float("nan")
        return 100.0 * (fge_baseline - fge_model) / fge_baseline

    # -------- Cross-split diagnostics: does TabPFN add error? --------
    _, _, _, fge_test_tabpfn_aod, n_test_aod = _summary_metrics(te_aod_oe, te_aod_ae)
    _, _, _, fge_test_merra_aod, _ = _summary_metrics(te_aod_merra, te_aod_ae)
    _, _, _, fge_test_tabpfn_alpha, n_test_alpha = _summary_metrics(te_alpha_oe, te_alpha_ae)
    _, _, _, fge_test_merra_alpha, _ = _summary_metrics(te_alpha_merra, te_alpha_ae)
    _, _, _, fge_train_oe_aod, n_train_aod = _summary_metrics(tr_aod_oe, tr_aod_ae)
    _, _, _, fge_train_merra_aod, _ = _summary_metrics(tr_aod_merra, tr_aod_ae)
    _, _, _, fge_train_oe_alpha, n_train_alpha = _summary_metrics(tr_alpha_oe, tr_alpha_ae)
    _, _, _, fge_train_merra_alpha, _ = _summary_metrics(tr_alpha_merra, tr_alpha_ae)

    imp_test_tabpfn_vs_merra_aod = _pct_improve(fge_test_tabpfn_aod, fge_test_merra_aod)
    imp_test_tabpfn_vs_merra_alpha = _pct_improve(fge_test_tabpfn_alpha, fge_test_merra_alpha)
    imp_train_oe_vs_merra_aod = _pct_improve(fge_train_oe_aod, fge_train_merra_aod)
    imp_train_oe_vs_merra_alpha = _pct_improve(fge_train_oe_alpha, fge_train_merra_alpha)

    print("\n=== FGE % improvement diagnostics (positive = lower FGE) ===")
    print(
        f"TEST  AOD_550: TabPFN vs MERRA-2 = {imp_test_tabpfn_vs_merra_aod:+.2f}%  "
        f"(n={n_test_aod}, FGE_tabpfn={fge_test_tabpfn_aod:.6f}, FGE_merra={fge_test_merra_aod:.6f})"
    )
    print(
        f"TEST  alpha  : TabPFN vs MERRA-2 = {imp_test_tabpfn_vs_merra_alpha:+.2f}%  "
        f"(n={n_test_alpha}, FGE_tabpfn={fge_test_tabpfn_alpha:.6f}, FGE_merra={fge_test_merra_alpha:.6f})"
    )
    print(
        f"TRAIN AOD_550: OE retrieval vs MERRA-2 = {imp_train_oe_vs_merra_aod:+.2f}%  "
        f"(n={n_train_aod}, FGE_oe={fge_train_oe_aod:.6f}, FGE_merra={fge_train_merra_aod:.6f})"
    )
    print(
        f"TRAIN alpha  : OE retrieval vs MERRA-2 = {imp_train_oe_vs_merra_alpha:+.2f}%  "
        f"(n={n_train_alpha}, FGE_oe={fge_train_oe_alpha:.6f}, FGE_merra={fge_train_merra_alpha:.6f})"
    )

    blocks: list[pd.DataFrame] = []
    stat_rows: list[dict] = []
    for dataset, pred_aod, pred_alpha, ref_aod, ref_alpha in (
        ("MERRA-2", te_aod_merra, te_alpha_merra, te_aod_ae, te_alpha_ae),
        ("TabPFN", te_aod_oe, te_alpha_oe, te_aod_ae, te_alpha_ae),
    ):
        mbe_b, rmse_b, fb_b, fge_b, n_b = _summary_metrics(pred_aod, ref_aod)
        mbe_a, rmse_a, fb_a, fge_a, n_a = _summary_metrics(pred_alpha, ref_alpha)
        print(f"{dataset} AOD_550: n={n_b}, MBE={mbe_b:.6f}, RMSE={rmse_b:.6f}, FB={fb_b:.6f}, FGE={fge_b:.6f}")
        print(
            f"{dataset} Angstrom alpha: n={n_a}, MBE={mbe_a:.6f}, RMSE={rmse_a:.6f}, "
            f"FB={fb_a:.6f}, FGE={fge_a:.6f}"
        )
        stat_rows.extend(
            [
                {
                    "dataset": dataset,
                    "variable": r"Ångström $\alpha$",
                    "metric": "Fractional gross error",
                    "label": f"n={n_a}\nMBE={mbe_a:.4f}\nRMSE={rmse_a:.4f}\nFB={fb_a:.4f}\nFGE={fge_a:.4f}",
                },
                {
                    "dataset": dataset,
                    "variable": r"AOD$_{550}$",
                    "metric": "Fractional gross error",
                    "label": f"n={n_b}\nMBE={mbe_b:.4f}\nRMSE={rmse_b:.4f}\nFB={fb_b:.4f}\nFGE={fge_b:.4f}",
                },
            ]
        )

        b_fge = _fge(pred_aod, ref_aod)
        a_fge = _fge(pred_alpha, ref_alpha)
        blocks.extend(
            [
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "variable": r"AOD$_{550}$",
                        "metric": "Fractional gross error",
                        "value": b_fge,
                    }
                ),
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "variable": r"Ångström $\alpha$",
                        "metric": "Fractional gross error",
                        "value": a_fge,
                    }
                ),
            ]
        )

    long_df = pd.concat(blocks, ignore_index=True)
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"]).copy()
    metric_fge = "Fractional gross error"
    long_df["dataset"] = pd.Categorical(long_df["dataset"], categories=DATASET_ORDER, ordered=True)
    long_df["variable"] = pd.Categorical(
        long_df["variable"],
        categories=[r"AOD$_{550}$", r"Ångström $\alpha$"],
        ordered=True,
    )
    long_df["metric"] = pd.Categorical(
        long_df["metric"],
        categories=[metric_fge],
        ordered=True,
    )
    ann_df = pd.DataFrame(stat_rows)
    # Fixed y-position (not relative to violin/data spread).
    ann_df["y"] = ANNOT_Y
    ann_df["dataset"] = pd.Categorical(ann_df["dataset"], categories=DATASET_ORDER, ordered=True)
    ann_df["variable"] = pd.Categorical(
        ann_df["variable"],
        categories=[r"AOD$_{550}$", r"Ångström $\alpha$"],
        ordered=True,
    )
    ann_df["metric"] = pd.Categorical(ann_df["metric"], categories=[metric_fge], ordered=True)
    return long_df, ann_df


def _theme_common():
    return (
        theme_bw(base_size=_PT, base_family="Times New Roman")
        + theme(
            text=element_text(size=_PT, family="Times New Roman"),
            axis_title=element_text(size=_PT, family="Times New Roman"),
            axis_text=element_text(size=_PT, family="Times New Roman"),
            strip_text=element_text(size=_PT, family="Times New Roman"),
            plot_title=element_text(size=_PT, family="Times New Roman"),
            legend_title=element_blank(),
            legend_position="bottom",
            panel_grid_major=element_line(color="#d9d9d9", size=_LW),
            panel_grid_minor=element_blank(),
            panel_border=element_rect(fill=None, color="black", size=_LW),
            axis_ticks=element_line(color="black", size=_LW),
        )
    )


def _load_irradiance_scatter_table() -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    if not IRRADIANCE_IN.is_file():
        raise SystemExit(f"Missing IRRADIANCE_IN: {IRRADIANCE_IN}")
    df = pd.read_csv(IRRADIANCE_IN, sep="\t", parse_dates=["time_utc"])
    need = [
        "ghi", "bni", "dhi",
        "ghi_merra", "bni_merra", "dhi_merra",
        "ghi_oe", "bni_oe", "dhi_oe",
        "ghi_aeronet", "bni_aeronet", "dhi_aeronet",
    ]
    _validate_columns(df, need, IRRADIANCE_IN)
    sub = df.dropna(subset=need).copy()
    if len(sub) == 0:
        raise SystemExit(f"No complete rows for irradiance scatter in {IRRADIANCE_IN}")

    def _long_overlay(
        frame: pd.DataFrame, panel: str, pairs: tuple[tuple[str, str, str], ...]
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for xcol, ycol, comp in pairs:
            t = frame[[xcol, ycol]].rename(columns={xcol: "measured", ycol: "forward"})
            t["component"] = comp
            t["panel"] = panel
            parts.append(t)
        return pd.concat(parts, ignore_index=True)

    panel_merra = "MERRA-2"
    panel_oe = "TabPFN"
    panel_ae = "AERONET"
    pairs_merra = (("ghi", "ghi_merra", "GHI"), ("bni", "bni_merra", "BNI"), ("dhi", "dhi_merra", "DHI"))
    pairs_oe = (("ghi", "ghi_oe", "GHI"), ("bni", "bni_oe", "BNI"), ("dhi", "dhi_oe", "DHI"))
    pairs_ae = (("ghi", "ghi_aeronet", "GHI"), ("bni", "bni_aeronet", "BNI"), ("dhi", "dhi_aeronet", "DHI"))
    long_df = pd.concat(
        [
            _long_overlay(sub, panel_merra, pairs_merra),
            _long_overlay(sub, panel_oe, pairs_oe),
            _long_overlay(sub, panel_ae, pairs_ae),
        ],
        ignore_index=True,
    )
    panel_order = [panel_merra, panel_oe, panel_ae]
    long_df["panel"] = pd.Categorical(long_df["panel"], categories=panel_order, ordered=True)
    long_df["component"] = pd.Categorical(long_df["component"], categories=["GHI", "BNI", "DHI"], ordered=True)

    # Print component-wise irradiance errors per panel.
    print("\n=== Irradiance error diagnostics (forward - measured) ===")
    for panel in panel_order:
        for comp in ("GHI", "BNI", "DHI"):
            s = long_df[(long_df["panel"] == panel) & (long_df["component"] == comp)]
            err = (s["forward"] - s["measured"]).to_numpy(dtype=float)
            m = np.isfinite(err)
            n = int(np.sum(m))
            if n == 0:
                mbe = float("nan")
                rmse = float("nan")
            else:
                e = err[m]
                mbe = float(np.mean(e))
                rmse = float(np.sqrt(np.mean(e**2)))
            print(f"{panel} {comp}: n={n}, MBE={mbe:.3f} W m^-2, RMSE={rmse:.3f} W m^-2")

    lo = float(long_df["measured"].min())
    hi = float(long_df["measured"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    line_df = pd.concat(
        [pd.DataFrame({"measured": [lo, hi], "forward": [lo, hi], "grp": 0, "panel": p}) for p in panel_order],
        ignore_index=True,
    )
    line_df["panel"] = pd.Categorical(line_df["panel"], categories=panel_order, ordered=True)
    return long_df, line_df, lo, hi


def _load_xai_long(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing SHAP file: {path}")
    df = pd.read_csv(path, sep="\t")
    _validate_columns(df, ["sample_index", "feature", "shap_value", "feature_value"], path)
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_index", "feature", "shap_value", "feature_value"])


def _xai_to_matrices(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    shap_wide = df.pivot(index="sample_index", columns="feature", values="shap_value").sort_index().astype(float)
    feat_wide = df.pivot(index="sample_index", columns="feature", values="feature_value").sort_index().astype(float)
    cols = [c for c in shap_wide.columns if c in feat_wide.columns]
    return shap_wide[cols].to_numpy(dtype=float), feat_wide[cols]


def _reorder_xai_by_aod_columns(
    beta_vals: np.ndarray,
    beta_feat: pd.DataFrame,
    alpha_vals: np.ndarray,
    alpha_feat: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Reorder beta and alpha SHAP matrices so feature order matches mean |SHAP| on AOD (beta)."""
    order = np.argsort(-np.nanmean(np.abs(beta_vals), axis=0))
    beta_cols = [beta_feat.columns[i] for i in order]
    beta_vals = beta_vals[:, order]
    beta_feat = beta_feat[beta_cols].copy()
    extra = [c for c in alpha_feat.columns if c not in beta_cols]
    alpha_cols = [c for c in beta_cols if c in alpha_feat.columns] + extra
    idx = [alpha_feat.columns.get_loc(c) for c in alpha_cols]
    alpha_vals = alpha_vals[:, idx]
    alpha_feat = alpha_feat[alpha_cols].copy()
    return beta_vals, beta_feat, alpha_vals, alpha_feat


def plot_xai_summary_figure() -> plt.Figure:
    import shap

    alpha = _load_xai_long(SHAP_ALPHA)
    alpha = alpha[alpha["sample_index"] != 64].copy()
    beta = _load_xai_long(SHAP_BETA)
    beta_vals, beta_feat = _xai_to_matrices(beta)
    alpha_vals, alpha_feat = _xai_to_matrices(alpha)
    beta_vals, beta_feat, alpha_vals, alpha_feat = _reorder_xai_by_aod_columns(
        beta_vals, beta_feat, alpha_vals, alpha_feat
    )

    fig = plt.figure(figsize=(6.0, 3.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.rcParams.update({"font.size": _PT, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

    plt.sca(ax1)
    shap.summary_plot(
        beta_vals,
        features=beta_feat,
        feature_names=_xai_feature_display_names(list(beta_feat.columns)),
        plot_type="dot",
        alpha=0.75,
        max_display=len(beta_feat.columns),
        show=False,
        plot_size=None,
        color_bar=False,
        cmap=WONG_FEATURE_CMAP,
    )
    ax1.set_title(r"AOD$_{550}$", fontsize=_PT)
    ax1.set_xlim(*SHAP_AOD_XLIM)
    ax1.tick_params(axis="both", labelsize=_PT)

    plt.sca(ax2)
    shap.summary_plot(
        alpha_vals,
        features=alpha_feat,
        feature_names=_xai_feature_display_names(list(alpha_feat.columns)),
        plot_type="dot",
        alpha=0.75,
        max_display=len(alpha_feat.columns),
        show=False,
        plot_size=None,
        color_bar=True,
        cmap=WONG_FEATURE_CMAP,
    )
    ax2.set_title(r"Ångström $\alpha$", fontsize=_PT)
    ax2.tick_params(axis="both", labelsize=_PT)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_oe_train_test_violin(df_long: pd.DataFrame, ann_df: pd.DataFrame):
    cmap = dict(zip(DATASET_ORDER, DATASET_COLORS))
    return (
        ggplot(df_long, aes(x="dataset", y="value", fill="dataset", color="dataset"))
        + geom_hline(yintercept=0, linetype="dashed", size=0.35, color="#4d4d4d")
        + geom_violin(alpha=0.30, trim=False, size=0.25)
        + geom_boxplot(width=0.12, outlier_alpha=0.15, fill="white", size=0.25)
        + geom_text(
            data=ann_df,
            mapping=aes(x="dataset", y="y", label="label"),
            inherit_aes=False,
            size=_PT,
            ha="left",
            va="top",
            lineheight=0.9,
            color="black",
            family="Times New Roman",
        )
        + facet_wrap("~ variable", nrow=2, ncol=1, scales="free_y")
        + scale_fill_manual(values=cmap)
        + scale_color_manual(values=cmap)
        + labs(
            title="(a) Test errors vs AERONET: fractional gross error",
            x="Dataset",
            y="FGE",
        )
        + _theme_common()
    )


def plot_irradiance_scatter():
    long_df, line_df, lo, hi = _load_irradiance_scatter_table()
    colors = {"GHI": WONG_ORANGE, "BNI": WONG_BLUE, "DHI": WONG_GREEN}
    return (
        ggplot(long_df, aes(x="measured", y="forward", color="component"))
        + geom_point(size=0.9, alpha=0.55, stroke=0)
        + geom_line(
            data=line_df,
            mapping=aes(x="measured", y="forward", group="grp"),
            inherit_aes=False,
            linetype="dashed",
            size=0.3,
            color="black",
            alpha=0.65,
        )
        + facet_wrap("~ panel", nrow=1, scales="fixed")
        + scale_color_manual(values=colors, breaks=["GHI", "BNI", "DHI"], name="Irrad. comp.")
        + coord_fixed(ratio=1, xlim=(lo, hi), ylim=(lo, hi), expand=False)
        + labs(
            title="(b) Test irradiance scatter: measured vs forward",
            x=r"Measured (W m$^{-2}$)",
            y=r"libRadtran forward (W m$^{-2}$)",
        )
        + _theme_common()
    )


def _figure_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    # Robust extraction across backends/layout engines.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    if rgba.ndim == 3 and rgba.shape[2] >= 3:
        return rgba[:, :, :3].copy()
    raise RuntimeError("Unexpected canvas buffer shape while rasterizing figure.")


def build_composite_figure(df_long: pd.DataFrame, ann_df: pd.DataFrame) -> plt.Figure:
    violin_fig = (plot_oe_train_test_violin(df_long, ann_df) + theme(figure_size=(5.6, 3.6))).draw()
    shap_fig = plot_xai_summary_figure()
    scatter_fig = (plot_irradiance_scatter() + theme(figure_size=(10.5, 3.2))).draw()

    violin_img = _figure_to_array(violin_fig)
    shap_img = _figure_to_array(shap_fig)
    scatter_img = _figure_to_array(scatter_fig)
    plt.close(violin_fig)
    plt.close(shap_fig)
    plt.close(scatter_fig)

    out = plt.figure(figsize=(FIG_W_MM / 25.4, (FIG_H_MM * 1.9) / 25.4))
    gs = GridSpec(2, 2, figure=out, height_ratios=[1.05, 1.0], width_ratios=[1.0, 2.0], hspace=0.08, wspace=0.05)
    ax_tl = out.add_subplot(gs[0, 0])
    ax_tr = out.add_subplot(gs[0, 1])
    ax_b = out.add_subplot(gs[1, :])
    for ax, img in ((ax_tl, violin_img), (ax_tr, shap_img), (ax_b, scatter_img)):
        ax.imshow(img)
        ax.axis("off")
    return out


def save_composite_pdf(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_path)) as pdf:
        fig.patch.set_facecolor("white")
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="none")
    plt.close(fig)


def main() -> None:
    df_long, ann_df = _load_long_error_table()
    fig = build_composite_figure(df_long, ann_df)
    save_composite_pdf(fig, OUTPUT_FIG)

    # Export tidy table too (useful for custom R plotting)
    out_txt = OUTPUT_FIG.with_suffix(".txt")
    df_long.to_csv(out_txt, sep="\t", index=False, float_format="%.8g")
    print(f"Wrote: {OUTPUT_FIG}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()

