"""
12.test_analysis: OE train-vs-test error diagnostics (extensible plotting scaffold).

First plot implemented:
- Violin comparison of **bias error** and **fractional error** for OE-only
- Variables: ``beta`` and ``alpha``
- Datasets:
  - Train OE table: ``Data/<STATION>_<YEAR>_train_oe<suffix>.txt``
  - Test OE prediction table: ``Data/<STATION>_<YEAR>_pred_oe<suffix>.txt``

Error definitions (reference = **AERONET** for all panels; train/test tables must include
``aeronet_aod550`` and ``aeronet_alpha``):
- squared error: ``(pred - ref)^2``
- fractional gross error: ``2*abs(pred - ref)/(pred + ref)``  (NaN where ``pred + ref <= 0``)
Plotting transform:
- squared error is displayed as ``log10(squared error + EPS_LOG)`` for readability.
- fractional gross error is displayed as ``log10(fractional gross error + EPS_LOG)``.

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
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_grid,
    geom_boxplot,
    geom_hline,
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

TRAIN_OE = Path(os.environ.get("TRAIN_OE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train_oe{_k_suffix}.txt")))
TEST_COMBINED = Path(
    os.environ.get("TEST_COMBINED", str(PROJECT / "Data" / f"{STATION}_{YEAR}_pred_oe{_k_suffix}.txt"))
)

OUTPUT_FIG = Path(
    os.environ.get(
        "OUTPUT_FIG",
        str(PROJECT / "tex" / "figures" / f"{STATION}_{YEAR}_oe_train_test_error_violin{_k_suffix}.pdf"),
    )
)

FIG_W_MM = float(os.environ.get("FIG_W_MM", "160"))
FIG_H_MM = float(os.environ.get("FIG_H_MM", "95"))

_PT = 8
_LW = 0.3
EPS_LOG = float(os.environ.get("EPS_LOG", "1e-8"))

WONG_ORANGE = "#E69F00"  # MERRA-2
WONG_BLUE = "#56B4E9"    # TabPFN OE (test)
WONG_GREEN = "#009E73"   # OE train

DATASET_ORDER = [
    "Train (OE retrieval vs AERONET)",
    "Test (TabPFN OE vs AERONET)",
    "Test (MERRA-2 vs AERONET)",
]
DATASET_COLORS = [WONG_GREEN, WONG_BLUE, WONG_ORANGE]


def _validate_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns in {path}: {miss}")


def _aod550_from_beta_alpha(beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return beta * (LAM550_UM / LAMBDA_REF_UM) ** (-alpha)


def _load_long_error_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_OE.is_file():
        raise SystemExit(f"Missing TRAIN_OE: {TRAIN_OE}")
    if not TEST_COMBINED.is_file():
        raise SystemExit(f"Missing TEST_COMBINED: {TEST_COMBINED}")

    tr = pd.read_csv(TRAIN_OE, sep="\t", parse_dates=["time_utc"])
    te = pd.read_csv(TEST_COMBINED, sep="\t", parse_dates=["time_utc"])

    _validate_columns(
        tr,
        [
            "beta_oe",
            "alpha_oe",
            "merra_BETA",
            "merra_ALPHA",
            "aeronet_aod550",
            "aeronet_alpha",
        ],
        TRAIN_OE,
    )
    _validate_columns(
        te,
        ["beta_pred_oe", "alpha_pred_oe", "merra_BETA", "merra_ALPHA", "aeronet_aod550", "aeronet_alpha"],
        TEST_COMBINED,
    )

    # AOD_550 + alpha representations
    tr_aod_p = _aod550_from_beta_alpha(tr["beta_oe"].to_numpy(dtype=float), tr["alpha_oe"].to_numpy(dtype=float))
    tr_alpha_p = tr["alpha_oe"].to_numpy(dtype=float)
    # Reference: AERONET (same convention as test block: τ₅₅₀ and Ångström α)
    tr_aod_r = tr["aeronet_aod550"].to_numpy(dtype=float)
    tr_alpha_r = tr["aeronet_alpha"].to_numpy(dtype=float)

    te_aod_oe = _aod550_from_beta_alpha(
        te["beta_pred_oe"].to_numpy(dtype=float), te["alpha_pred_oe"].to_numpy(dtype=float)
    )
    te_alpha_oe = te["alpha_pred_oe"].to_numpy(dtype=float)
    te_aod_merra = _aod550_from_beta_alpha(te["merra_BETA"].to_numpy(dtype=float), te["merra_ALPHA"].to_numpy(dtype=float))
    te_alpha_merra = te["merra_ALPHA"].to_numpy(dtype=float)
    te_aod_ae = te["aeronet_aod550"].to_numpy(dtype=float)
    te_alpha_ae = te["aeronet_alpha"].to_numpy(dtype=float)

    def _fge(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
        out = np.full_like(pred, np.nan, dtype=float)
        den = pred + ref
        m = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(den) & (den > 0.0)
        out[m] = 2.0 * np.abs(pred[m] - ref[m]) / den[m]
        return out

    def _rmse_fge(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, int]:
        """Match step-11 metrics: RMSE on finite pairs; FGE is mean row-wise FGE."""
        m = np.isfinite(pred) & np.isfinite(ref)
        n = int(np.sum(m))
        if n == 0:
            return float("nan"), float("nan"), 0
        p = pred[m]
        r = ref[m]
        rmse = float(np.sqrt(np.mean((p - r) ** 2)))
        fge_vec = _fge(p, r)
        m_fge = np.isfinite(fge_vec)
        fge = float(np.mean(fge_vec[m_fge])) if int(np.sum(m_fge)) > 0 else float("nan")
        return rmse, fge, n

    blocks: list[pd.DataFrame] = []
    stat_rows: list[dict] = []
    for dataset, pred_aod, pred_alpha, ref_aod, ref_alpha in (
        (DATASET_ORDER[0], tr_aod_p, tr_alpha_p, tr_aod_r, tr_alpha_r),
        (DATASET_ORDER[1], te_aod_oe, te_alpha_oe, te_aod_ae, te_alpha_ae),
        (DATASET_ORDER[2], te_aod_merra, te_alpha_merra, te_aod_ae, te_alpha_ae),
    ):
        rmse_b, fge_b, n_b = _rmse_fge(pred_aod, ref_aod)
        rmse_a, fge_a, n_a = _rmse_fge(pred_alpha, ref_alpha)
        print(f"{dataset} AOD_550: n={n_b}, RMSE={rmse_b:.6f}, FGE={fge_b:.6f}")
        print(f"{dataset} Angstrom alpha: n={n_a}, RMSE={rmse_a:.6f}, FGE={fge_a:.6f}")
        stat_rows.extend(
            [
                {
                    "dataset": dataset,
                    "variable": r"AOD$_{550}$",
                    "metric": "Squared error",
                    "label": f"n={n_b}\nRMSE={rmse_b:.4f}\nFGE={fge_b:.4f}",
                },
                {
                    "dataset": dataset,
                    "variable": r"Ångström $\alpha$",
                    "metric": "Squared error",
                    "label": f"n={n_a}\nRMSE={rmse_a:.4f}\nFGE={fge_a:.4f}",
                },
                {
                    "dataset": dataset,
                    "variable": r"AOD$_{550}$",
                    "metric": "Fractional gross error",
                    "label": f"n={n_b}\nRMSE={rmse_b:.4f}\nFGE={fge_b:.4f}",
                },
                {
                    "dataset": dataset,
                    "variable": r"Ångström $\alpha$",
                    "metric": "Fractional gross error",
                    "label": f"n={n_a}\nRMSE={rmse_a:.4f}\nFGE={fge_a:.4f}",
                },
            ]
        )

        b_se = (pred_aod - ref_aod) ** 2
        a_se = (pred_alpha - ref_alpha) ** 2
        b_fge = _fge(pred_aod, ref_aod)
        a_fge = _fge(pred_alpha, ref_alpha)
        blocks.extend(
            [
                pd.DataFrame(
                    {"dataset": dataset, "variable": r"AOD$_{550}$", "metric": "Squared error", "value": b_se}
                ),
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "variable": r"Ångström $\alpha$",
                        "metric": "Squared error",
                        "value": a_se,
                    }
                ),
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
    m_se = long_df["metric"] == "Squared error"
    m_fge = long_df["metric"] == "Fractional gross error"
    long_df.loc[m_se, "value"] = np.log10(long_df.loc[m_se, "value"].to_numpy(dtype=float) + EPS_LOG)
    long_df.loc[m_fge, "value"] = np.log10(long_df.loc[m_fge, "value"].to_numpy(dtype=float) + EPS_LOG)
    long_df.loc[m_se, "metric"] = f"log10(Squared error + {EPS_LOG:g})"
    long_df.loc[m_fge, "metric"] = f"log10(Fractional gross error + {EPS_LOG:g})"
    metric_sq = f"log10(Squared error + {EPS_LOG:g})"
    metric_fge = f"log10(Fractional gross error + {EPS_LOG:g})"
    long_df["dataset"] = pd.Categorical(long_df["dataset"], categories=DATASET_ORDER, ordered=True)
    long_df["variable"] = pd.Categorical(
        long_df["variable"],
        categories=[r"AOD$_{550}$", r"Ångström $\alpha$"],
        ordered=True,
    )
    long_df["metric"] = pd.Categorical(
        long_df["metric"],
        categories=[metric_sq, metric_fge],
        ordered=True,
    )
    ann_df = pd.DataFrame(stat_rows)
    ann_df["metric"] = ann_df["metric"].replace(
        {"Squared error": metric_sq, "Fractional gross error": metric_fge}
    )
    # Place annotations near top of each facet.
    y_pos = (
        long_df.groupby(["metric", "variable"], observed=True)["value"]
        .agg(["min", "max"])
        .reset_index()
    )
    y_pos["y"] = y_pos["max"] - 0.06 * (y_pos["max"] - y_pos["min"]).replace(0, 1.0)
    ann_df = ann_df.merge(y_pos[["metric", "variable", "y"]], on=["metric", "variable"], how="left")
    ann_df["dataset"] = pd.Categorical(ann_df["dataset"], categories=DATASET_ORDER, ordered=True)
    ann_df["variable"] = pd.Categorical(
        ann_df["variable"],
        categories=[r"AOD$_{550}$", r"Ångström $\alpha$"],
        ordered=True,
    )
    ann_df["metric"] = pd.Categorical(ann_df["metric"], categories=[metric_sq, metric_fge], ordered=True)
    return long_df, ann_df


def _theme_common():
    return (
        theme_bw(base_size=_PT, base_family="serif")
        + theme(
            text=element_text(size=_PT, family="serif"),
            axis_title=element_text(size=_PT),
            axis_text=element_text(size=_PT),
            strip_text=element_text(size=_PT),
            legend_title=element_blank(),
            legend_position="top",
            panel_grid_major=element_line(color="#d9d9d9", size=_LW),
            panel_grid_minor=element_blank(),
            panel_border=element_rect(fill=None, color="black", size=_LW),
            axis_ticks=element_line(color="black", size=_LW),
        )
    )


def plot_oe_train_test_violin(df_long: pd.DataFrame, ann_df: pd.DataFrame):
    cmap = dict(zip(DATASET_ORDER, DATASET_COLORS))
    return (
        ggplot(df_long, aes(x="dataset", y="value", fill="dataset", color="dataset"))
        + geom_hline(yintercept=0, linetype="dashed", size=0.35, color="#4d4d4d")
        + geom_violin(alpha=0.30, trim=False, size=0.25)
        + geom_boxplot(width=0.12, outlier_alpha=0.15, fill="white", size=0.25)
        + geom_text(
            data=ann_df,
            mapping=aes(x="dataset", y="y", label="label", color="dataset"),
            inherit_aes=False,
            size=6,
            ha="left",
            va="top",
            lineheight=0.9,
        )
        + facet_grid("metric ~ variable", scales="free_y")
        + scale_fill_manual(values=cmap)
        + scale_color_manual(values=cmap)
        + labs(
            title="(a) OE train vs test errors vs AERONET: squared and fractional gross",
            x="Dataset",
            y=f"log10(metric + {EPS_LOG:g})",
        )
        + _theme_common()
    )


def build_plot_collection(df_long: pd.DataFrame, ann_df: pd.DataFrame):
    """Extensible registry for adding more plots later."""
    return [
        ("oe_train_test_violin", plot_oe_train_test_violin(df_long, ann_df), FIG_H_MM),
        # Future additions can append tuples: (name, ggplot_obj, height_mm)
    ]


def save_plots_pdf(plot_specs: list[tuple[str, object, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_in = FIG_W_MM / 25.4
    with PdfPages(str(out_path)) as pdf:
        for _, p, h_mm in plot_specs:
            fig = (p + theme(figure_size=(w_in, h_mm / 25.4))).draw()
            fig.patch.set_facecolor("white")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="none")
            plt.close(fig)


def main() -> None:
    df_long, ann_df = _load_long_error_table()
    plots = build_plot_collection(df_long, ann_df)
    save_plots_pdf(plots, OUTPUT_FIG)

    # Export tidy table too (useful for custom R plotting)
    out_txt = OUTPUT_FIG.with_suffix(".txt")
    df_long.to_csv(out_txt, sep="\t", index=False, float_format="%.8g")
    print(f"Wrote: {OUTPUT_FIG}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()

