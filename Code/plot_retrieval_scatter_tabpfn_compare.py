"""
2×2 scatter comparison: same test rows, TabPFN 0.5k vs TabPFN 2k.

Columns (left → right): **MERRA-2** | **Retrieval**. Rows: **TabPFN 0.5k** | **TabPFN 2k**.
Shared axis limits, Wong palette, SKILL styling (8 pt, 0.3 pt, Times New Roman).

Env: ``PLOT_INPUT_0_5K``, ``PLOT_INPUT_2K`` (paths to ``test_ls_*.txt``),
``PLOT_OUTPUT``, ``PLOT_WIDTH_MM`` (default **160**), ``PLOT_HEIGHT_MM`` (default **100**),
``PLOT_SHOW``, ``PLOT_OPEN`` (same semantics as ``plot_retrieval_scatter.py``).
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
    facet_grid,
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
INPUT_0_5K = Path(
    os.environ.get("PLOT_INPUT_0_5K", str(PROJECT / "Data" / "test_ls_0.5k.txt")),
)
INPUT_2K = Path(os.environ.get("PLOT_INPUT_2K", str(PROJECT / "Data" / "test_ls_2k.txt")))
OUTPUT_PNG = Path(
    os.environ.get(
        "PLOT_OUTPUT",
        str(PROJECT / "tex" / "figures" / "retrieval_scatter_tabpfn_compare.png"),
    ),
)

FIG_W_MM = float(os.environ.get("PLOT_WIDTH_MM", "160"))
FIG_H_MM = float(os.environ.get("PLOT_HEIGHT_MM", "70"))

WONG_GHI_BNI_DHI = ("#E69F00", "#56B4E9", "#009E73")

need = [
    "ghi",
    "bni",
    "dhi",
    "ghi_merra",
    "bni_merra",
    "dhi_merra",
    "ghi_ls",
    "bni_ls",
    "dhi_ls",
]
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

panel_left = "MERRA-2"
panel_right = "Retrieval"
row_0_5k = "TabPFN 0.5k"
row_2k = "TabPFN 2k"


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


def _long_points(
    pairs: tuple[tuple[str, str, str], ...],
    frame: pd.DataFrame,
    panel: str,
    tab_row: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for xcol, ycol, comp in pairs:
        t = frame[[xcol, ycol]].rename(columns={xcol: "measured", ycol: "forward"})
        t["component"] = comp
        t["panel"] = panel
        t["tab_row"] = tab_row
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def _load_sub(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing table: {path}")
    df = pd.read_csv(path, sep="\t", comment="#", parse_dates=["time_utc"])
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c!r} in {path}")
    sub = df.dropna(subset=need).copy()
    if len(sub) == 0:
        raise SystemExit(f"No complete rows in {path}")
    return sub


sub_05 = _load_sub(INPUT_0_5K)
sub_2k = _load_sub(INPUT_2K)

lo = float(
    min(
        sub_05[list(need)].min().min(),
        sub_2k[list(need)].min().min(),
        0.0,
    ),
)
hi = float(
    max(
        sub_05[list(need)].max().max(),
        sub_2k[list(need)].max().max(),
        1.0,
    ),
)
pad = (hi - lo) * 0.03
if pad <= 0:
    pad = 1.0

long_df = pd.concat(
    [
        _long_points(pairs_merra, sub_05, panel_left, row_0_5k),
        _long_points(pairs_ls, sub_05, panel_right, row_0_5k),
        _long_points(pairs_merra, sub_2k, panel_left, row_2k),
        _long_points(pairs_ls, sub_2k, panel_right, row_2k),
    ],
    ignore_index=True,
)

_panel_cat = pd.CategoricalDtype(categories=[panel_left, panel_right], ordered=True)
_tab_cat = pd.CategoricalDtype(categories=[row_0_5k, row_2k], ordered=True)
long_df["panel"] = long_df["panel"].astype(_panel_cat)
long_df["tab_row"] = long_df["tab_row"].astype(_tab_cat)

_line_seg = pd.DataFrame({"measured": [lo, hi], "forward": [lo, hi], "grp": 0})
line_parts: list[pd.DataFrame] = []
for tr in (row_0_5k, row_2k):
    for pan in (panel_left, panel_right):
        line_parts.append(_line_seg.assign(panel=pan, tab_row=tr))
line_df = pd.concat(line_parts, ignore_index=True)
line_df["panel"] = line_df["panel"].astype(_panel_cat)
line_df["tab_row"] = line_df["tab_row"].astype(_tab_cat)

stat_rows: list[dict[str, str | float]] = []
for tab_label, frame in ((row_0_5k, sub_05), (row_2k, sub_2k)):
    for panel_name, pairs in ((panel_left, pairs_merra), (panel_right, pairs_ls)):
        xm, yf = pooled_measured_forward(pairs, frame)
        mbe_v, rmse_pct_v, r2_v = mbe_rmsepct_r2(xm, yf)
        stat_rows.append(
            {
                "tab_row": tab_label,
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
stat_df["tab_row"] = stat_df["tab_row"].astype(_tab_cat)

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
    panel_spacing_y=0.008,
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

_pt_size = 0.9
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
    + facet_grid("tab_row ~ panel", scales="fixed")
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
