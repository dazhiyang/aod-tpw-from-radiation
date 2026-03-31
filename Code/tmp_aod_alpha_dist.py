r"""Temporary diagnostic: density + scatter plots of AOD₅₅₀ and Ångström α.

Four sources per panel: AERONET, MERRA-2, LS-retrieved, OE-retrieved.

Row 1: density of AOD₅₅₀ and α.  Row 2: 1:1 scatters (AERONET x vs model y) for
AOD₅₅₀ and α with MBE / RMSE / FB / FGE annotations.

Usage:
    /opt/anaconda3/bin/python Code/tmp_aod_alpha_dist.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from plotnine import (
    aes,
    after_stat,
    as_labeller,
    coord_fixed,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_abline,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    stat_density,
    theme,
)
from plotnine.themes.elements.margin import margin

from libRadtran import DEFAULT_CLEARSKY_CONFIG

PROJECT = Path(__file__).resolve().parent.parent

_PT = 8
_LW = 0.3
_DENSITY_LW = 0.12

COLOR_MERRA = "#E69F00"
COLOR_AERONET = "#56B4E9"
COLOR_LS = "#009E73"
COLOR_OE = "#CC79A7"

SRC_MERRA = "MERRA-2"
SRC_AERONET = "AERONET"
SRC_LS = "LS retrieval"
SRC_OE = "OE retrieval"
_ORDER = [SRC_MERRA, SRC_AERONET, SRC_LS, SRC_OE]
_COLORS = [COLOR_MERRA, COLOR_AERONET, COLOR_LS, COLOR_OE]

LAM550_UM = 0.55
LAMBDA_REF_UM = 1.0
DENSITY_N = 512
RIBBON_ALPHA = 0.18
AOD_X_MAX = 0.6

STATION = os.environ.get("STATION", "PAL")
YEAR = int(os.environ.get("YEAR", "2024"))
LHS_N = int(os.environ.get("LHS_N", "100"))
_n = LHS_N
_k_suffix = "_0.5k" if _n == 500 else f"_{_n / 1000:g}k" if _n >= 1000 else f"_{_n}"

LHS_TABLE = Path(
    os.environ.get("LHS_TABLE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train{_k_suffix}.txt"))
)
RET_LS = Path(
    os.environ.get("RET_LS", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train_ls{_k_suffix}.txt"))
)
RET_OE = Path(
    os.environ.get("RET_OE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train_oe{_k_suffix}.txt"))
)
OUTPUT_PNG = Path(
    os.environ.get(
        "OUTPUT_PNG",
        str(PROJECT / "tex" / "figures" / "tmp_aod_alpha_distributions.png"),
    )
)


FIG_H_SCATTER_MM = 85


def aod550_angstrom(beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """τ(550 nm) from Ångström law; λ_ref in µm."""
    return beta * (LAM550_UM / LAMBDA_REF_UM) ** (-alpha)


def metrics_vs_ref(o: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Compare model P to reference O (AERONET). All finite pairs."""
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if len(o) == 0:
        return {"n": 0.0, "mbe": np.nan, "rmse": np.nan, "fb": np.nan, "fge": np.nan}
    mo, mp = float(np.mean(o)), float(np.mean(p))
    denom = mo + mp
    mbe = float(np.mean(p - o))
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    fb = float(2.0 * (mo - mp) / denom) if denom > 0 else float("nan")
    fge = float(2.0 * np.mean(np.abs(o - p)) / denom) if denom > 0 else float("nan")
    return {"n": float(len(o)), "mbe": mbe, "rmse": rmse, "fb": fb, "fge": fge}


def _metrics_text(m: dict[str, float]) -> str:
    return (
        f"$n$ = {int(m['n'])}\n"
        f"MBE = {m['mbe']:.4f}\n"
        f"RMSE = {m['rmse']:.4f}\n"
        f"FB = {m['fb']:.4f}\n"
        f"FGE = {m['fge']:.4f}"
    )


def main() -> None:
    for pth, label in ((LHS_TABLE, "LHS_TABLE"), (RET_LS, "RET_LS"), (RET_OE, "RET_OE")):
        if not pth.is_file():
            print(f"ERROR: Missing {label}: {pth}", file=sys.stderr)
            sys.exit(1)

    lhs = pd.read_csv(LHS_TABLE, sep="\t", comment="#", parse_dates=["time_utc"])
    ls = pd.read_csv(RET_LS, sep="\t", comment="#", parse_dates=["time_utc"])
    oe = pd.read_csv(RET_OE, sep="\t", comment="#", parse_dates=["time_utc"])

    need_lhs = ["time_utc", "merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha"]
    need_ls = ["time_utc", "beta_ls", "alpha_ls"]
    need_oe = ["time_utc", "beta_oe", "alpha_oe"]
    for c in need_lhs:
        if c not in lhs.columns:
            print(f"ERROR: Missing {c!r} in {LHS_TABLE}", file=sys.stderr)
            sys.exit(1)
    for c in need_ls:
        if c not in ls.columns:
            print(f"ERROR: Missing {c!r} in {RET_LS}", file=sys.stderr)
            sys.exit(1)
    for c in need_oe:
        if c not in oe.columns:
            print(f"ERROR: Missing {c!r} in {RET_OE}", file=sys.stderr)
            sys.exit(1)

    df = (lhs[need_lhs]
          .merge(ls[need_ls], on="time_utc", how="inner")
          .merge(oe[need_oe], on="time_utc", how="inner"))
    df = df.dropna(subset=["merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha",
                            "beta_ls", "alpha_ls", "beta_oe", "alpha_oe"])

    if len(df) == 0:
        print("ERROR: No rows after merge and dropna.", file=sys.stderr)
        sys.exit(1)
    print(f"Merged: {len(df)} rows with all four sources.")

    # --- Compute AOD550 ---
    aod_merra = aod550_angstrom(df["merra_BETA"].values, df["merra_ALPHA"].values)
    aod_aeronet = df["aeronet_aod550"].values.astype(float)
    aod_ls = aod550_angstrom(df["beta_ls"].values, df["alpha_ls"].values)
    aod_oe = aod550_angstrom(df["beta_oe"].values, df["alpha_oe"].values)

    # --- Alpha ---
    alpha_merra = df["merra_ALPHA"].values.astype(float)
    alpha_aeronet = df["aeronet_alpha"].values.astype(float)
    alpha_ls_vals = df["alpha_ls"].values.astype(float)
    alpha_oe_vals = df["alpha_oe"].values.astype(float)

    # --- Build long-form dataframe ---
    PANEL_AOD = "aod"
    PANEL_ALPHA = "alpha"
    _PANEL_ORDER = [PANEL_AOD, PANEL_ALPHA]

    def _aod_keep(a: np.ndarray) -> np.ndarray:
        m = np.isfinite(a) & (a <= AOD_X_MAX)
        return a[m]

    parts: list[pd.DataFrame] = []
    for src, vals in ((SRC_MERRA, aod_merra), (SRC_AERONET, aod_aeronet),
                      (SRC_LS, aod_ls), (SRC_OE, aod_oe)):
        parts.append(pd.DataFrame({"panel": PANEL_AOD, "source": src, "value": _aod_keep(vals)}))
    for src, vals in ((SRC_MERRA, alpha_merra), (SRC_AERONET, alpha_aeronet),
                      (SRC_LS, alpha_ls_vals), (SRC_OE, alpha_oe_vals)):
        finite = vals[np.isfinite(vals)]
        parts.append(pd.DataFrame({"panel": PANEL_ALPHA, "source": src, "value": finite}))

    long_df = pd.concat(parts, ignore_index=True)
    long_df["panel"] = pd.Categorical(long_df["panel"], categories=_PANEL_ORDER, ordered=True)
    long_df["source"] = pd.Categorical(long_df["source"], categories=_ORDER, ordered=True)

    _facet_lbl = as_labeller({
        PANEL_AOD: r"AOD$_{550}$",
        PANEL_ALPHA: r"Ångström $\alpha$",
    })

    _theme = theme(
        text=element_text(family="Times New Roman", size=_PT),
        axis_text=element_text(size=_PT),
        axis_title=element_text(size=_PT),
        plot_title=element_text(size=_PT),
        plot_subtitle=element_text(size=_PT),
        legend_text=element_text(
            size=_PT, margin=margin(t=0, r=0, b=0, l=0, unit="pt")
        ),
        legend_title=element_blank(),
        strip_text=element_text(
            size=_PT, margin=margin(t=0.3, r=1, b=0.3, l=1, unit="pt")
        ),
        panel_spacing_x=0.02,
        panel_grid_major=element_line(color="black", size=_LW, alpha=0.35),
        panel_grid_minor=element_blank(),
        plot_background=element_rect(fill="white", color="none"),
        panel_background=element_rect(fill="white", color="none"),
        panel_border=element_rect(fill="none", color="black", size=_LW),
        axis_line=element_blank(),
        axis_ticks=element_line(color="black", size=_LW),
        legend_position="bottom",
        legend_direction="horizontal",
        legend_margin=0,
        legend_box_margin=0,
        legend_box_spacing=0,
        legend_spacing=4,
        legend_key_spacing_x=4,
        legend_key_spacing_y=0,
        legend_key_size=8 * 1.2,
        legend_background=element_blank(),
        legend_key=element_rect(fill="none", color="none", size=0),
        legend_frame=element_blank(),
        strip_background=element_rect(fill="white", color="black", size=_LW),
    )

    _dens = {"position": "identity", "n": DENSITY_N}
    FIG_W_MM = 160
    FIG_H_MM = 80

    p_density = (
        ggplot(long_df, aes(x="value", fill="source", color="source"))
        + stat_density(
            aes(ymin=0, ymax=after_stat("density")),
            geom="ribbon",
            alpha=RIBBON_ALPHA,
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(y=after_stat("density")),
            geom="line",
            size=_DENSITY_LW,
            **_dens,
        )
        + facet_wrap("~ panel", nrow=1, scales="free", labeller=_facet_lbl)
        + scale_fill_manual(values=dict(zip(_ORDER, _COLORS)), breaks=_ORDER, name="")
        + scale_color_manual(values=dict(zip(_ORDER, _COLORS)), breaks=_ORDER, name="")
        + labs(x="", y="Density", title=f"{STATION} {YEAR} — $n = {len(df)}$ (LS + OE retrieval)")
        + _theme
    )

    # --- Metrics ---
    met_aod_m = metrics_vs_ref(aod_aeronet, aod_merra)
    met_aod_ls = metrics_vs_ref(aod_aeronet, aod_ls)
    met_aod_oe = metrics_vs_ref(aod_aeronet, aod_oe)
    met_alpha_m = metrics_vs_ref(alpha_aeronet, alpha_merra)
    met_alpha_ls = metrics_vs_ref(alpha_aeronet, alpha_ls_vals)
    met_alpha_oe = metrics_vs_ref(alpha_aeronet, alpha_oe_vals)
    for tag, met in (("AOD MERRA", met_aod_m), ("AOD LS", met_aod_ls), ("AOD OE", met_aod_oe),
                     ("Alpha MERRA", met_alpha_m), ("Alpha LS", met_alpha_ls),
                     ("Alpha OE", met_alpha_oe)):
        print(
            f"{tag} vs AERONET: n={int(met['n'])}, MBE={met['mbe']:.6f}, "
            f"RMSE={met['rmse']:.6f}, FB={met['fb']:.6f}, FGE={met['fge']:.6f}"
        )

    # --- Scatter data ---
    _scatter_order = [SRC_MERRA, SRC_LS, SRC_OE]
    _scatter_colors = [COLOR_MERRA, COLOR_LS, COLOR_OE]
    n_row = len(df)
    n_src = len(_scatter_order)

    # AOD scatter
    hi_aod = max(float(np.nanmax(np.concatenate(
        [aod_aeronet, aod_merra, aod_ls, aod_oe])) * 1.05), 0.05)
    aod_scatter = pd.DataFrame({
        "aeronet": np.tile(aod_aeronet, n_src),
        "model": np.concatenate([aod_merra, aod_ls, aod_oe]),
        "source": np.repeat(_scatter_order, n_row),
    })
    aod_scatter["source"] = pd.Categorical(aod_scatter["source"], categories=_scatter_order,
                                           ordered=True)
    aod_label = pd.DataFrame({
        "source": _scatter_order,
        "tx": [hi_aod * 0.04] * n_src,
        "ty": [hi_aod * 0.95] * n_src,
        "txt": [_metrics_text(met_aod_m), _metrics_text(met_aod_ls), _metrics_text(met_aod_oe)],
    })
    aod_label["source"] = pd.Categorical(aod_label["source"], categories=_scatter_order,
                                         ordered=True)

    # Alpha scatter
    all_alpha = np.concatenate([alpha_aeronet, alpha_merra, alpha_ls_vals, alpha_oe_vals])
    hi_alpha = max(float(np.nanmax(all_alpha) * 1.05), 0.5)
    alpha_scatter = pd.DataFrame({
        "aeronet": np.tile(alpha_aeronet, n_src),
        "model": np.concatenate([alpha_merra, alpha_ls_vals, alpha_oe_vals]),
        "source": np.repeat(_scatter_order, n_row),
    })
    alpha_scatter["source"] = pd.Categorical(alpha_scatter["source"], categories=_scatter_order,
                                             ordered=True)
    alpha_label = pd.DataFrame({
        "source": _scatter_order,
        "tx": [hi_alpha * 0.04] * n_src,
        "ty": [hi_alpha * 0.95] * n_src,
        "txt": [_metrics_text(met_alpha_m), _metrics_text(met_alpha_ls),
                _metrics_text(met_alpha_oe)],
    })
    alpha_label["source"] = pd.Categorical(alpha_label["source"], categories=_scatter_order,
                                           ordered=True)

    # Combine AOD + Alpha scatters into one 6-facet plot
    aod_scatter["var"] = r"AOD$_{550}$"
    alpha_scatter["var"] = r"Ångström $\alpha$"
    aod_label["var"] = r"AOD$_{550}$"
    alpha_label["var"] = r"Ångström $\alpha$"
    scatter_all = pd.concat([aod_scatter, alpha_scatter], ignore_index=True)
    label_all = pd.concat([aod_label, alpha_label], ignore_index=True)
    _var_order = [r"AOD$_{550}$", r"Ångström $\alpha$"]
    scatter_all["var"] = pd.Categorical(scatter_all["var"], categories=_var_order, ordered=True)
    label_all["var"] = pd.Categorical(label_all["var"], categories=_var_order, ordered=True)

    scatter_all["facet"] = scatter_all["var"].astype(str) + " — " + scatter_all["source"].astype(str)
    label_all["facet"] = label_all["var"].astype(str) + " — " + label_all["source"].astype(str)
    facet_order = []
    for v in _var_order:
        for s in _scatter_order:
            facet_order.append(f"{v} — {s}")
    scatter_all["facet"] = pd.Categorical(scatter_all["facet"], categories=facet_order, ordered=True)
    label_all["facet"] = pd.Categorical(label_all["facet"], categories=facet_order, ordered=True)

    p_scatter = (
        ggplot(scatter_all, aes(x="aeronet", y="model", color="source"))
        + geom_point(alpha=0.45, size=1.5)
        + geom_abline(intercept=0, slope=1, color="#737373", linetype="dashed", size=_LW)
        + geom_text(
            aes(x="tx", y="ty", label="txt"),
            data=label_all, ha="left", va="top", size=8, color="black",
        )
        + facet_wrap("~ facet", nrow=1, ncol=6, scales="free")
        + scale_color_manual(values=dict(zip(_scatter_order, _scatter_colors)))
        + labs(x="AERONET (reference)", y="Model", title="")
        + _theme
        + theme(legend_position="none", panel_spacing_x=0.02)
    )

    # --- Composite: density on top, scatter on bottom ---
    buf_d = io.BytesIO()
    p_density.save(buf_d, format="png", dpi=300, width=FIG_W_MM / 25.4, height=FIG_H_MM / 25.4,
                   units="in", verbose=False, facecolor="white", edgecolor="none")
    buf_d.seek(0)
    img_d = plt.imread(buf_d)

    buf_s = io.BytesIO()
    p_scatter.save(buf_s, format="png", dpi=300, width=FIG_W_MM / 25.4,
                   height=FIG_H_SCATTER_MM / 25.4, units="in", verbose=False,
                   facecolor="white", edgecolor="none")
    buf_s.seek(0)
    img_s = plt.imread(buf_s)

    w_in = FIG_W_MM / 25.4
    h_top = FIG_H_MM / 25.4
    h_bot = FIG_H_SCATTER_MM / 25.4
    fig = plt.figure(figsize=(w_in, h_top + h_bot + 0.25), dpi=300, facecolor="white")
    gs = GridSpec(2, 1, figure=fig, height_ratios=[FIG_H_MM, FIG_H_SCATTER_MM], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_top.imshow(img_d, aspect="auto")
    ax_top.axis("off")
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.imshow(img_s, aspect="auto")
    ax_bot.axis("off")

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PNG), dpi=300, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {OUTPUT_PNG}")

    if os.environ.get("PLOT_OPEN", "").strip().lower() in {"1", "true", "yes"}:
        if sys.platform == "darwin":
            subprocess.run(["open", str(OUTPUT_PNG)], check=False)


if __name__ == "__main__":
    main()
