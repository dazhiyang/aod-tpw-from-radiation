r"""
Distributions of **AOD at 550 nm** and **column water w** (mm): MERRA-2 prior vs TabPFN
predictions trained on LS vs OE retrievals.

AOD at 550 nm uses the Ångström power law with exponent ``merra_ALPHA`` and coefficient
``β`` at reference wavelength ``λ_ref`` (μm):

    τ(λ) = β · (λ / λ_ref)^(-α)   →   τ_550nm = β · (0.55 / λ_ref)^(-α)

LS/OE curves use ``beta_pred_ls`` / ``w_pred_ls`` and ``beta_pred_oe`` / ``w_pred_oe`` on the
same test instants (inner merge on ``time_utc``). MERRA prior uses ``merra_BETA``,
``merra_ALPHA``, and ``merra_TQV`` (scaled to mm PW via ``ClearskyConfig.tqv_bsrn_to_mm_pw``).

For the AOD panel, samples with τ\ :sub:`550` **>** ``AOD_X_MAX`` are **dropped** (not clipped).
KDE uses ``bounds=(0, AOD_X_MAX)`` so the x-axis ends at that cap. The **w** panel still uses
all merged rows.

Env:
    TEST_LS — default ``Data/pred_ls_0.5k.txt``
    TEST_OE — default ``Data/pred_oe_0.5k.txt``
    OUTPUT_PNG — default ``tex/figures/aod_w_distributions.png``
    ANGSTROM_BETA_REF_UM — β reference wavelength in μm (default **1.0**; use **0.55** if β is
        already τ at 550 nm in your pipeline)
    DENSITY_N — KDE evaluation grid size (default **512**)
    RIBBON_ALPHA — fill alpha for density ribbons (default **0.18**)
    AOD_X_MAX — drop AOD at 550 nm **>** this value; KDE x-range ``[0, AOD_X_MAX]`` (default **0.6**)
    DENSITY_LINE_WIDTH — KDE line and ribbon-edge width in pt (``size`` aesthetic; default **0.12**)
    PLOT_WIDTH_MM / PLOT_HEIGHT_MM — figure size (default **160** × **72**)
    PLOT_SHOW, PLOT_OPEN — same as other plot scripts

Styling: ``.agents/SKILL.md`` (Times New Roman, 8 pt, 0.3 pt lines, Wong palette, 160 mm width).

Requires: pandas, numpy, plotnine, matplotlib (optional for ``PLOT_SHOW``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    after_stat,
    as_labeller,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    stat_density,
    theme,
)
from plotnine.themes.elements.margin import margin

from libRadtran import ClearskyConfig

PROJECT = Path(__file__).resolve().parent.parent

_PT = 8
_LW = 0.3
_DENSITY_LW = float(os.environ.get("DENSITY_LINE_WIDTH", "0.12"))

TEST_LS = Path(os.environ.get("TEST_LS", str(PROJECT / "Data" / "pred_ls_0.5k.txt")))
TEST_OE = Path(os.environ.get("TEST_OE", str(PROJECT / "Data" / "pred_oe_0.5k.txt")))
OUTPUT_PNG = Path(
    os.environ.get(
        "OUTPUT_PNG",
        str(PROJECT / "tex" / "figures" / "aod_w_distributions.png"),
    )
)
LAMBDA_REF_UM = float(os.environ.get("ANGSTROM_BETA_REF_UM", "1.0"))
DENSITY_N = int(os.environ.get("DENSITY_N", "512"))
RIBBON_ALPHA = float(os.environ.get("RIBBON_ALPHA", "0.18"))
AOD_X_MAX = float(os.environ.get("AOD_X_MAX", "0.6"))
FIG_W_MM = float(os.environ.get("PLOT_WIDTH_MM", "160"))
FIG_H_MM = float(os.environ.get("PLOT_HEIGHT_MM", "72"))

# Wong order: MERRA, LS, OE
COLOR_MERRA = "#E69F00"
COLOR_LS = "#56B4E9"
COLOR_OE = "#009E73"

SRC_MERRA = "MERRA-2 prior"
SRC_LS = "TabPFN (LS train)"
SRC_OE = "TabPFN (OE train)"
_SOURCE_ORDER = [SRC_MERRA, SRC_LS, SRC_OE]
_FILL_COLORS = [COLOR_MERRA, COLOR_LS, COLOR_OE]

PANEL_AOD = "aod"
PANEL_W = "w"
_PANEL_ORDER = [PANEL_AOD, PANEL_W]

LAM550_UM = 0.55
_PW_SCALE = ClearskyConfig.tqv_bsrn_to_mm_pw


def aod550_angstrom(beta: np.ndarray, alpha: np.ndarray, lambda_ref_um: float) -> np.ndarray:
    """τ(550 nm) from Ångström law; λ in μm."""
    return beta * (LAM550_UM / lambda_ref_um) ** (-alpha)


def main() -> None:
    for p in (TEST_LS, TEST_OE):
        if not p.is_file():
            print(f"ERROR: Missing {p}", file=sys.stderr)
            sys.exit(1)

    need_ls = [
        "time_utc",
        "merra_ALPHA",
        "merra_BETA",
        "merra_TQV",
        "beta_pred_ls",
        "w_pred_ls",
    ]
    need_oe = ["time_utc", "beta_pred_oe", "w_pred_oe"]

    ls = pd.read_csv(TEST_LS, sep="\t", comment="#", parse_dates=["time_utc"])
    oe = pd.read_csv(TEST_OE, sep="\t", comment="#", parse_dates=["time_utc"])

    for c in need_ls:
        if c not in ls.columns:
            print(f"ERROR: Missing {c!r} in {TEST_LS}", file=sys.stderr)
            sys.exit(1)
    for c in need_oe:
        if c not in oe.columns:
            print(f"ERROR: Missing {c!r} in {TEST_OE}", file=sys.stderr)
            sys.exit(1)

    df = ls[need_ls].merge(oe[need_oe], on="time_utc", how="inner")
    cols = [
        "merra_ALPHA",
        "merra_BETA",
        "merra_TQV",
        "beta_pred_ls",
        "w_pred_ls",
        "beta_pred_oe",
        "w_pred_oe",
    ]
    df = df.dropna(subset=cols)
    if len(df) == 0:
        print("ERROR: No rows after merge and dropna.", file=sys.stderr)
        sys.exit(1)

    alpha = df["merra_ALPHA"].to_numpy(dtype=float)
    beta_m = df["merra_BETA"].to_numpy(dtype=float)
    beta_ls = df["beta_pred_ls"].to_numpy(dtype=float)
    beta_oe = df["beta_pred_oe"].to_numpy(dtype=float)

    aod_m = aod550_angstrom(beta_m, alpha, LAMBDA_REF_UM)
    aod_ls = aod550_angstrom(beta_ls, alpha, LAMBDA_REF_UM)
    aod_oe = aod550_angstrom(beta_oe, alpha, LAMBDA_REF_UM)

    w_m = df["merra_TQV"].to_numpy(dtype=float) * _PW_SCALE
    w_ls = df["w_pred_ls"].to_numpy(dtype=float)
    w_oe = df["w_pred_oe"].to_numpy(dtype=float)

    def _aod_keep(a: np.ndarray) -> np.ndarray:
        m = np.isfinite(a) & (a <= AOD_X_MAX)
        return a[m]

    parts: list[pd.DataFrame] = []
    for src, aod_v, w_v in (
        (SRC_MERRA, aod_m, w_m),
        (SRC_LS, aod_ls, w_ls),
        (SRC_OE, aod_oe, w_oe),
    ):
        parts.append(
            pd.DataFrame(
                {
                    "panel": PANEL_AOD,
                    "source": src,
                    "value": _aod_keep(aod_v),
                }
            )
        )
        parts.append(
            pd.DataFrame(
                {
                    "panel": PANEL_W,
                    "source": src,
                    "value": w_v,
                }
            )
        )

    long_df = pd.concat(parts, ignore_index=True)
    long_df["source"] = pd.Categorical(long_df["source"], categories=_SOURCE_ORDER, ordered=True)
    long_df["panel"] = pd.Categorical(long_df["panel"], categories=_PANEL_ORDER, ordered=True)

    aod_df = long_df.loc[long_df["panel"] == PANEL_AOD].copy()
    w_df = long_df.loc[long_df["panel"] == PANEL_W].copy()
    if len(aod_df) == 0:
        print("ERROR: No AOD samples after τ ≤ AOD_X_MAX filter.", file=sys.stderr)
        sys.exit(1)

    _facet_lbl = as_labeller(
        {
            PANEL_AOD: rf"AOD$_{{550}}$ ($\lambda_{{\mathrm{{ref}}}}$ = {LAMBDA_REF_UM:g} $\mu$m)",
            PANEL_W: r"Column water $w$ (mm)",
        }
    )

    _theme = theme(
        text=element_text(family="Times New Roman", size=_PT),
        axis_text=element_text(size=_PT),
        axis_title=element_text(size=_PT),
        plot_title=element_text(size=_PT),
        plot_subtitle=element_text(size=_PT),
        legend_text=element_text(size=_PT, margin=margin(t=0, r=0, b=0, l=0, unit="pt")),
        legend_title=element_blank(),
        strip_text=element_text(
            size=_PT,
            margin=margin(t=0.3, r=1, b=0.3, l=1, unit="pt"),
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
    p = (
        ggplot(long_df, aes(x="value", fill="source", color="source"))
        + stat_density(
            aes(ymin=0, ymax=after_stat("density")),
            data=aod_df,
            geom="ribbon",
            bounds=(0.0, AOD_X_MAX),
            alpha=RIBBON_ALPHA,
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(y=after_stat("density")),
            data=aod_df,
            geom="line",
            bounds=(0.0, AOD_X_MAX),
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(ymin=0, ymax=after_stat("density")),
            data=w_df,
            geom="ribbon",
            alpha=RIBBON_ALPHA,
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(y=after_stat("density")),
            data=w_df,
            geom="line",
            size=_DENSITY_LW,
            **_dens,
        )
        + facet_wrap("~ panel", nrow=1, scales="free", labeller=_facet_lbl)
        + scale_fill_manual(
            values=_FILL_COLORS,
            breaks=_SOURCE_ORDER,
            name="",
        )
        + scale_color_manual(
            values=_FILL_COLORS,
            breaks=_SOURCE_ORDER,
            name="",
        )
        + labs(
            x="",
            y="Density",
            title=(
                f"$n = {len(df)}$ test rows (merged LS/OE); "
                r"$\tau_{550}=\beta\,(0.55/\lambda_{\mathrm{ref}})^{-\alpha}$"
            ),
        )
        + _theme
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    p.save(
        str(OUTPUT_PNG),
        width=FIG_W_MM / 25.4,
        height=FIG_H_MM / 25.4,
        dpi=300,
        units="in",
        verbose=False,
        facecolor="white",
        edgecolor="none",
    )
    print(
        f"Wrote: {OUTPUT_PNG} (merged n={len(df)}, AOD τ≤{AOD_X_MAX}, "
        f"x bounds [0,{AOD_X_MAX}] on AOD panel, ANGSTROM_BETA_REF_UM={LAMBDA_REF_UM}, "
        f"RIBBON_ALPHA={RIBBON_ALPHA})"
    )

    _env_truthy = {"1", "true", "yes"}
    if os.environ.get("PLOT_SHOW", "").strip().lower() in _env_truthy:
        print(p)

    if os.environ.get("PLOT_OPEN", "").strip().lower() in _env_truthy and sys.platform == "darwin":
        subprocess.run(["open", str(OUTPUT_PNG)], check=False)


if __name__ == "__main__":
    main()
