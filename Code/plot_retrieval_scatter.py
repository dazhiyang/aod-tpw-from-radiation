"""
Scatter plots: MERRA-2 forward vs measured, and OE forward vs measured (side by side).

Reads tab-separated output from ``3.retrieval.py`` / ``oe_retrieve_beta_h2o.py`` (e.g.
``Data/train_oe_beta_h2o.txt``). Each panel overlays GHI, BNI, and DHI. 
Draws a 1:1 line; axis limits match across panels for comparability.

Pooled statistics (GHI + BNI + DHI, all rows): **MBE** = mean(model − measured)
[W m⁻²], **RMSE%** = RMSE / mean(measured) × 100, **R²** = 1 − Σ(y−x)² / Σ(x−x̄)² with
x = measured, y = forward.

Env: ``PLOT_INPUT`` (default ``Data/train_ls_0.5k.txt``), ``PLOT_OUTPUT`` (default
``tex/figures/retrieval_scatter_merra_ls.png``).

Requires: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
INPUT_TXT = Path(os.environ.get("PLOT_INPUT", str(PROJECT / "Data" / "train_ls_0.5k.txt")))
OUTPUT_PNG = Path(
    os.environ.get(
        "PLOT_OUTPUT",
        str(PROJECT / "tex" / "figures" / "retrieval_scatter_merra_ls.png"),
    )
)

if not INPUT_TXT.is_file():
    raise SystemExit(f"Missing table: {INPUT_TXT}  (run 3.retrieval.py first)")

df = pd.read_csv(INPUT_TXT, sep="\t", comment="#", parse_dates=["time_utc"])

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
for c in need:
    if c not in df.columns:
        raise SystemExit(f"Missing column {c!r} in {INPUT_TXT}")

# Right panel: training retrieval vs TabPFN evaluation use the same flux column names.
_right_title = (
    "TabPFN forward vs measured"
    if "beta_pred" in df.columns and "beta_retrieved" not in df.columns
    else "LS forward vs measured"
)

sub = df.dropna(subset=need).copy()
if len(sub) == 0:
    raise SystemExit("No rows with finite measured and forward fluxes.")

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

colors = {"GHI": "C0", "BNI": "C1", "DHI": "C2"}


def pooled_measured_forward(pairs: tuple[tuple[str, str, str], ...], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stack GHI, beam, DHI pairs into 1-D arrays (measured, forward)."""
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
    """
    x_meas = measured, y_fwd = model. Returns (MBE [W m⁻²], RMSE [%], R²).
    RMSE% = RMSE / mean(|x|) × 100 if mean(|x|) > 0 (use mean of measured flux magnitude).
    """
    err = y_fwd - x_meas
    mbe = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    xbar = float(np.mean(x_meas))
    rmse_pct = float(100.0 * rmse / xbar) if np.isfinite(xbar) and xbar != 0.0 else float("nan")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((x_meas - np.mean(x_meas)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return mbe, rmse_pct, r2


fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")

for ax, pairs, title in (
    (axes[0], pairs_merra, "MERRA-2 forward vs measured"),
    (axes[1], pairs_ls, _right_title),
):
    for xcol, ycol, label in pairs:
        x = sub[xcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[m],
            y[m],
            s=14,
            alpha=0.55,
            c=colors[label],
            edgecolors="none",
            label=label,
        )

    xm, yf = pooled_measured_forward(pairs, sub)
    mbe_v, rmse_pct_v, r2_v = mbe_rmsepct_r2(xm, yf)
    stat_lines = [
        f"MBE = {mbe_v:.2f} W m$^{{-2}}$",
        f"RMSE% = {rmse_pct_v:.2f} %",
        f"$R^2$ = {r2_v:.3f}",
    ]
    ax.text(
        0.98,
        0.02,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )

    lo = min(sub[[c for c in need]].min().min(), 0.0)
    hi = max(sub[[c for c in need]].max().max(), 1.0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.65, label="1:1")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Measured (W m$^{-2}$)")
    ax.set_ylabel(r"libRadtran forward (W m$^{-2}$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9)

fig.suptitle(f"{INPUT_TXT.name}  (n={len(sub)})", fontsize=10, y=1.02)

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=160)
plt.close(fig)

print(f"Wrote: {OUTPUT_PNG}")
