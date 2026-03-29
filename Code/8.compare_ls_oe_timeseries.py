"""
Compare LS and OE retrieval results on a single axes for 2024-01-21.
Measured GHI, BNI, DHI from ``qiq_1min_merra_qc.txt`` as lines.
Retrieved BNI points (clearsky == 1) from ``test_ls_0.5k.txt`` and ``test_oe_0.5k.txt``.

Env (optional):
    TEST_LS — path to ``test_ls_*.txt`` (default ``Data/test_ls_0.5k.txt``).
    TEST_OE — path to ``test_oe_*.txt`` (default ``Data/test_oe_0.5k.txt``).
    QIQ_MASTER — path to ``qiq_1min_merra_qc.txt``.
    OUTPUT_PNG — output figure path (default ``tex/figures/compare_ls_oe_timeseries.png``).
    UTC_START_HOUR — inclusive start hour (default ``0``).
    UTC_END_HOUR — exclusive end hour (default ``8``).

Requires: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

# SKILL.md: 8 pt text, 0.3 pt lines, Times New Roman
_PT = 8
_LW = 0.3

TEST_LS = Path(os.environ.get("TEST_LS", str(PROJECT / "Data" / "test_ls_0.5k.txt")))
TEST_OE = Path(os.environ.get("TEST_OE", str(PROJECT / "Data" / "test_oe_0.5k.txt")))
QIQ_MASTER = Path(os.environ.get("QIQ_MASTER", str(PROJECT / "Data" / "qiq_1min_merra_qc.txt")))
OUTPUT_PNG = Path(
    os.environ.get(
        "OUTPUT_PNG",
        str(PROJECT / "tex" / "figures" / "compare_ls_oe_timeseries.png"),
    )
)
TARGET_DATE = "2024-01-21"
UTC_START_HOUR = int(os.environ.get("UTC_START_HOUR", "0"))
UTC_END_HOUR = int(os.environ.get("UTC_END_HOUR", "8"))

WONG_ORANGE = "#E69F00"
WONG_SKY = "#56B4E9"
WONG_BLU_GR = "#009E73"
COLOR_LS = "#D55E00"  # Vermillion
COLOR_OE = "#CC79A7"  # Reddish purple


def _load_qiq_day(qiq_path: Path, day_str: str) -> pd.DataFrame:
    """Load only rows whose UTC calendar date equals ``day_str`` (chunked read)."""
    day = pd.to_datetime(day_str).date()
    usecols = ["time_utc", "ghi", "bni", "dhi", "clearsky"]
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        qiq_path,
        sep="\t",
        comment="#",
        parse_dates=["time_utc"],
        usecols=usecols,
        chunksize=200_000,
    )
    for chunk in reader:
        # Avoid warnings by making sure time_utc is datetime
        if not pd.api.types.is_datetime64_any_dtype(chunk["time_utc"]):
             chunk["time_utc"] = pd.to_datetime(chunk["time_utc"], utc=True)
        # Use .dt.date for comparison
        chunk = chunk[chunk["time_utc"].dt.date == day]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values("time_utc")


def _filter_utc_hours(df: pd.DataFrame, day_str: str, h0: int, h1: int) -> pd.DataFrame:
    """Keep rows on ``day`` with time in ``[h0:00, h1:00)`` UTC."""
    day = pd.to_datetime(day_str).date()
    start = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h0)
    end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h1)
    tt = pd.to_datetime(df["time_utc"], utc=True)
    m = (tt >= start) & (tt < end)
    out = df.loc[m].copy()
    out["time_utc"] = tt[m]
    return out


def _combined_ylim(*arrays: np.ndarray) -> tuple[float, float]:
    stacked = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if stacked.size == 0:
        return 0.0, 1.0
    y_lo, y_hi = float(np.min(stacked)), float(np.max(stacked))
    pad = max((y_hi - y_lo) * 0.05, 5.0)
    return y_lo - pad, y_hi + pad


def main() -> None:
    for p in (TEST_LS, TEST_OE, QIQ_MASTER):
        if not p.is_file():
            print(f"ERROR: Missing {p}", file=sys.stderr)
            sys.exit(1)

    # 1. Load Measured data
    print(f"Loading measurements for {TARGET_DATE} from {QIQ_MASTER.name}...")
    qiq_day = _load_qiq_day(QIQ_MASTER, TARGET_DATE)
    if qiq_day.empty:
        print(f"ERROR: No qiq rows for {TARGET_DATE}", file=sys.stderr)
        sys.exit(1)
    qiq_win = _filter_utc_hours(qiq_day, TARGET_DATE, UTC_START_HOUR, UTC_END_HOUR)

    # 2. Load LS and OE retrieval points
    print(f"Loading LS/OE results for {TARGET_DATE}...")
    ls_df = pd.read_csv(TEST_LS, sep="\t", parse_dates=["time_utc"])
    oe_df = pd.read_csv(TEST_OE, sep="\t", parse_dates=["time_utc"])
    
    # Filter for target day and hours
    ls_day = _filter_utc_hours(ls_df, TARGET_DATE, UTC_START_HOUR, UTC_END_HOUR)
    oe_day = _filter_utc_hours(oe_df, TARGET_DATE, UTC_START_HOUR, UTC_END_HOUR)

    # Styling setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": _PT,
        "axes.linewidth": _LW,
        "axes.titlesize": _PT,
        "axes.labelsize": _PT,
        "xtick.labelsize": _PT,
        "ytick.labelsize": _PT,
        "legend.fontsize": _PT,
        "lines.linewidth": _LW,
        "xtick.major.width": _LW,
        "ytick.major.width": _LW,
        "grid.linewidth": _LW,
    })

    fig, ax = plt.subplots(figsize=(160 / 25.4, 70 / 25.4), layout="constrained")
    
    # Plot measured components
    t_meas = qiq_win["time_utc"]
    ax.plot(t_meas, qiq_win["ghi"], color=WONG_ORANGE, label="Measured GHI", alpha=0.9)
    ax.plot(t_meas, qiq_win["dhi"], color=WONG_BLU_GR, label="Measured DHI", alpha=0.9)
    ax.plot(t_meas, qiq_win["bni"], color=WONG_SKY, label="Measured BNI", alpha=1.0, zorder=3)

    # Plot LS points
    if not ls_day.empty:
        ax.scatter(
            ls_day["time_utc"], ls_day["bni_ls"],
            s=8, marker="o", color=COLOR_LS, label="TabPFN LS BNI",
            edgecolors="black", linewidths=0.2, zorder=5, alpha=0.8
        )
    
    # Plot OE points
    if not oe_day.empty:
        ax.scatter(
            oe_day["time_utc"], oe_day["bni_oe"],
            s=8, marker="^", color=COLOR_OE, label="TabPFN OE BNI",
            edgecolors="black", linewidths=0.2, zorder=6, alpha=0.8
        )
    
    # Plot MERRA-2 forward as points
    if not ls_day.empty:
        ax.scatter(
            ls_day["time_utc"], ls_day["bni_merra"],
            s=4, marker="x", color="#000000", label="MERRA-2 Forward BNI",
            alpha=0.4, zorder=4
        )

    y_lo, y_hi = _combined_ylim(
        qiq_win["ghi"], qiq_win["bni"], qiq_win["dhi"],
        ls_day["bni_merra"], ls_day["bni_ls"], oe_day["bni_oe"]
    )
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("UTC", fontsize=_PT)
    ax.set_ylabel(r"Irradiance (W m$^{-2}$)", fontsize=_PT)
    ax.set_title(f"Measured vs. Retrieval BNI — {TARGET_DATE} (UTC)", fontsize=_PT)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="both", labelsize=_PT, width=_LW, length=2.5)
    ax.grid(True, alpha=0.35, linewidth=_LW)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
