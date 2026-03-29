"""
Scatter plots: one row, three panels — **MERRA-2**, **TabPFN 0.5k**, **TabPFN 2k**.

Each panel overlays **GHI, BNI, DHI** (Wong colors) as measured vs libRadtran forward for that
source. MERRA-2 and LS fluxes come from ``6.evaluation.py`` outputs; the 0.5k and 2k panels use
``*_ls`` columns from the respective ``test_ls_*.txt`` files (same test instants).

Pooled statistics (GHI + BNI + DHI, all rows) per panel: MBE [W m⁻²], RMSE%, R².

Env: ``PLOT_INPUT_0_5K`` (default ``Data/test_ls_0.5k.txt``), ``PLOT_INPUT_2K`` (default
``Data/test_ls_2k.txt``). If ``PLOT_INPUT`` is set and ``PLOT_INPUT_0_5K`` is not, ``PLOT_INPUT``
is used as the 0.5k path. ``PLOT_OUTPUT``, ``PLOT_WIDTH_MM`` (default **160**), ``PLOT_HEIGHT_MM``
(default **56**), ``PLOT_SHOW``, ``PLOT_OPEN``.

Requires: pandas, numpy, plotnine, matplotlib.
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
    coord_fixed,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    theme,
)
from plotnine.themes.elements.margin import margin

PROJECT = Path(__file__).resolve().parent.parent
INPUT_LS = Path(
    os.environ.get(
        "PLOT_INPUT_LS",
        os.environ.get("PLOT_INPUT", str(PROJECT / "Data" / "test_ls_0.5k.txt")),
    ),
)
INPUT_OE = Path(os.environ.get("PLOT_INPUT_OE", str(PROJECT / "Data" / "test_oe_0.5k.txt")))
OUTPUT_PNG = Path(
    os.environ.get(
        "PLOT_OUTPUT",
        str(PROJECT / "tex" / "figures" / "retrieval_scatter_ls_oe.png"),
    )
)

FIG_W_MM = float(os.environ.get("PLOT_WIDTH_MM", "160"))
FIG_H_MM = float(os.environ.get("PLOT_HEIGHT_MM", "70"))

WONG_GHI_BNI_DHI = ("#E69F00", "#56B4E9", "#009E73")

need_ls = [
    "ghi", "bni", "dhi", "ghi_merra", "bni_merra", "dhi_merra",
    "ghi_ls", "bni_ls", "dhi_ls",
]
need_oe = [
    "ghi", "bni", "dhi", "ghi_oe", "bni_oe", "dhi_oe",
]

PANEL_MERRA = "MERRA-2"
PANEL_05 = "TabPFN 0.5k trained"
PANEL_2K = "TabPFN 2k trained"


def _load_sub(path: Path, columns: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path, sep="\t", comment="#", parse_dates=["time_utc"])
    for c in columns:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c!r} in {path}")
    sub = df.dropna(subset=columns).copy()
    if len(sub) == 0:
        return None
    return sub


def _long_overlay(
    pairs: tuple[tuple[str, str, str], ...],
    frame: pd.DataFrame,
    panel: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for xcol, ycol, comp in pairs:
        t = frame[[xcol, ycol]].rename(columns={xcol: "measured", ycol: "forward"})
        t["component"] = comp
        t["panel"] = panel
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def pooled_measured_forward(
    pairs: tuple[tuple[str, str, str], ...], frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xcol, ycol, _ in pairs:
        xs.append(frame[xcol].to_numpy(dtype=float))
        ys.append(frame[ycol].to_numpy(dtype=float))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def mbe_rmsepct_r2(x_meas: np.ndarray, y_fwd: np.ndarray) -> tuple[float, float, float]:
    err = y_fwd - x_meas
    mbe = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    xbar = float(np.mean(x_meas))
    rmse_pct = float(100.0 * rmse / xbar) if np.isfinite(xbar) and xbar != 0.0 else float("nan")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((x_meas - np.mean(x_meas)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return mbe, rmse_pct, r2


sub_ls = _load_sub(INPUT_LS, need_ls)
sub_oe = _load_sub(INPUT_OE, need_oe)

if sub_ls is None:
    raise SystemExit(f"Missing LS table: {INPUT_LS}")
if sub_oe is None:
    raise SystemExit(f"Missing OE table: {INPUT_OE}")

# Verify parity
if not sub_ls["time_utc"].equals(sub_oe["time_utc"]):
    print("WARNING: time_utc differs between LS and OE. Filtering to intersection...")
    common = sub_ls["time_utc"].reset_index().merge(sub_oe["time_utc"].reset_index(), on="time_utc")["time_utc"]
    sub_ls = sub_ls[sub_ls["time_utc"].isin(common)].copy()
    sub_oe = sub_oe[sub_oe["time_utc"].isin(common)].copy()

base_meas = ["ghi", "bni", "dhi"]
for c in base_meas:
    a = sub_ls[c].to_numpy(dtype=float)
    b = sub_oe[c].to_numpy(dtype=float)
    if not np.allclose(a, b, rtol=0, atol=1e-5, equal_nan=True):
         print(f"WARNING: Measured {c!r} differs between inputs.")

PANEL_MERRA = "MERRA-2"
PANEL_LS = "TabPFN (LS)"
PANEL_OE = "TabPFN (OE)"

pairs_merra = (
    ("ghi", "ghi_merra", "GHI"),
    ("bni", "bni_merra", "BNI"),
    ("dhi", "dhi_merra", "DHI"),
)
pairs_ls = (
    ("ghi", "ghi_ls", "GHI"),
    ("bni", "bni_ls", "BNI"),
    ("dhi", "dhi_ls", "DHI"),
)
pairs_oe = (
    ("ghi", "ghi_oe", "GHI"),
    ("bni", "bni_oe", "BNI"),
    ("dhi", "dhi_oe", "DHI"),
)

panels_to_plot = []
panels_to_plot.append(_long_overlay(pairs_merra, sub_ls, PANEL_MERRA))
panels_to_plot.append(_long_overlay(pairs_ls, sub_ls, PANEL_LS))
panels_to_plot.append(_long_overlay(pairs_oe, sub_oe, PANEL_OE))

long_df = pd.concat(panels_to_plot, ignore_index=True)

_comp_cat = pd.CategoricalDtype(categories=["GHI", "BNI", "DHI"], ordered=True)
_panel_cat_list = [PANEL_MERRA, PANEL_LS, PANEL_OE]
_panel_cat = pd.CategoricalDtype(categories=_panel_cat_list, ordered=True)

long_df["component"] = long_df["component"].astype(_comp_cat)
long_df["panel"] = long_df["panel"].astype(_panel_cat)

lo = float(long_df["measured"].min())
hi = float(long_df["measured"].max())
pad = (hi - lo) * 0.03
if pad <= 0:
    pad = 1.0

_line_seg = pd.DataFrame({"measured": [lo, hi], "forward": [lo, hi], "grp": 0})
line_df = pd.concat(
    [_line_seg.assign(panel=p) for p in _panel_cat_list],
    ignore_index=True,
)
line_df["panel"] = line_df["panel"].astype(_panel_cat)

stat_runs = [
    (PANEL_MERRA, sub_ls, pairs_merra),
    (PANEL_LS, sub_ls, pairs_ls),
    (PANEL_OE, sub_oe, pairs_oe),
]

stat_rows: list[dict[str, str | float]] = []
for panel_name, frame, pairs in stat_runs:
    xm, yf = pooled_measured_forward(pairs, frame)
    mbe_v, rmse_pct_v, r2_v = mbe_rmsepct_r2(xm, yf)
    stat_rows.append(
        {
            "panel": panel_name,
            "measured": hi - pad,
            "forward": lo + pad,
            "label": (
                f"MBE = {mbe_v:.2f} W m$^{{-2}}$\n"
                f"RMSE% = {rmse_pct_v:.2f} %\n"
                f"$R^2$ = {r2_v:.3f}"
            ),
        },
    )
stat_df = pd.DataFrame(stat_rows)
stat_df["panel"] = stat_df["panel"].astype(_panel_cat)

_pt_size = 0.9
_theme = theme(
    text=element_text(family="Times New Roman", size=8),
    axis_text=element_text(size=8),
    axis_title=element_text(size=8),
    plot_title=element_blank(),
    legend_text=element_text(size=8, margin=margin(t=0, r=0, b=0, l=0, unit="pt")),
    legend_title=element_text(
        size=8,
        margin=margin(t=1, r=2, b=0, l=2, unit="pt"),
    ),
    strip_text=element_text(
        size=8,
        margin=margin(t=0.3, r=1, b=0.3, l=1, unit="pt"),
    ),
    panel_spacing_x=0.006,
    panel_grid_major=element_line(color="#A8A8A8", size=0.3, alpha=0.35),
    panel_grid_minor=element_blank(),
    axis_line=element_line(color="black", size=0.3),
    axis_ticks=element_line(color="black", size=0.3),
    legend_position="bottom",
    legend_direction="horizontal",
    legend_margin=0,
    legend_box_margin=0,
    legend_box_spacing=0,
    legend_spacing=2,
    legend_key_spacing_x=3,
    legend_key_spacing_y=0,
    legend_key_size=8 * 1.2,
    legend_background=element_blank(),
    legend_key=element_rect(fill="none", color="none", size=0),
    legend_frame=element_blank(),
    strip_background=element_rect(fill="white", color="#BFBFBF", size=0.3),
)

p = (
    ggplot(long_df, aes(x="measured", y="forward", color="component"))
    + geom_point(size=_pt_size, alpha=0.55, stroke=0)
    + geom_line(
        aes(x="measured", y="forward", group="grp"),
        data=line_df,
        color="black",
        linetype="dashed",
        size=0.3,
        alpha=0.65,
        inherit_aes=False,
    )
    + geom_text(
        aes(x="measured", y="forward", label="label"),
        data=stat_df,
        ha="right",
        va="bottom",
        size=8,
        color="black",
        inherit_aes=False,
    )
    + facet_wrap("~ panel", nrow=1, ncol=len(_panel_cat_list), scales="fixed")
    + scale_color_manual(
        values=list(WONG_GHI_BNI_DHI),
        breaks=["GHI", "BNI", "DHI"],
        name="Irrad. comp.",
    )
    + coord_fixed(ratio=1, xlim=(lo, hi), ylim=(lo, hi), expand=False)
    + labs(
        x=r"Measured (W m$^{-2}$)",
        y=r"libRadtran forward (W m$^{-2}$)",
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

print(f"Wrote: {OUTPUT_PNG}")

_env_truthy = {"1", "true", "yes"}
if os.environ.get("PLOT_SHOW", "").strip().lower() in _env_truthy:
    import matplotlib.pyplot as plt

    plt.show()

if os.environ.get("PLOT_OPEN", "").strip().lower() in _env_truthy and sys.platform == "darwin":
    subprocess.run(["open", str(OUTPUT_PNG)], check=False)
