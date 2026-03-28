"""
Pick the UTC calendar day in a test evaluation table with the largest mean |bni_merra − bni|,
then plot GHI, BNI, and DHI from ``qiq_1min_merra_qc.txt`` on **one** axes for **00:00–08:00 UTC**
that day. GHI and DHI are lines only. **Clearsky** scatter markers apply **only to BNI**
(``clearsky == 1``); no clearsky overlay on GHI or DHI.

Env (optional):
    TEST_LS — path to ``test_ls_*.txt`` (default ``Data/test_ls_0.5k.txt``).
    QIQ_MASTER — path to ``qiq_1min_merra_qc.txt``.
    OUTPUT_PNG — output figure path (default ``tex/figures/bni_worst_day_timeseries.png``).
    FORCE_DATE — ``YYYY-MM-DD`` to skip worst-day search and plot that UTC date instead.
    MIN_TEST_ROWS_PER_DAY — minimum test rows on a UTC day to be eligible for ``worst day``
    (default ``3``). If no day qualifies, falls back to any day with the largest mean error.
    UTC_START_HOUR — inclusive start hour (default ``0``).
    UTC_END_HOUR — exclusive end hour (default ``8`` → window [00:00, 08:00) UTC).

Requires: pandas, numpy, matplotlib.

Styling matches ``.agents/SKILL.md`` (Times New Roman, **8 pt** text, **0.3** pt lines, Wong
discrete colors, **160 mm** figure width).
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
QIQ_MASTER = Path(os.environ.get("QIQ_MASTER", str(PROJECT / "Data" / "qiq_1min_merra_qc.txt")))
OUTPUT_PNG = Path(
    os.environ.get(
        "OUTPUT_PNG",
        str(PROJECT / "tex" / "figures" / "bni_worst_day_timeseries.png"),
    )
)
FORCE_DATE = os.environ.get("FORCE_DATE", "").strip()
MIN_TEST_ROWS_PER_DAY = int(os.environ.get("MIN_TEST_ROWS_PER_DAY", "3"))
UTC_START_HOUR = int(os.environ.get("UTC_START_HOUR", "0"))
UTC_END_HOUR = int(os.environ.get("UTC_END_HOUR", "8"))

WONG_ORANGE = "#E69F00"
WONG_SKY = "#56B4E9"
WONG_BLU_GR = "#009E73"
WONG_MARK = "#CC79A7"


def _pick_worst_day_utc(
    ts: pd.DataFrame, min_rows: int,
) -> tuple[object, pd.DataFrame]:
    """
    Return (calendar date, subset of ts for that date) using mean |bni_merra − bni| per UTC day.

    Prefer days with at least ``min_rows`` test samples; if none qualify, use all days.
    """
    work = ts.dropna(subset=["bni", "bni_merra"]).copy()
    if work.empty:
        raise ValueError("No rows with finite bni and bni_merra.")
    work["date_utc"] = work["time_utc"].dt.date
    work["abs_bni_err"] = (work["bni_merra"] - work["bni"]).abs()
    g = work.groupby("date_utc", sort=False).agg(
        mean_abs=("abs_bni_err", "mean"),
        n=("abs_bni_err", "count"),
        max_abs=("abs_bni_err", "max"),
    )
    eligible = g[g["n"] >= min_rows] if min_rows > 0 else g
    if eligible.empty:
        eligible = g
    eligible = eligible.sort_values(["mean_abs", "max_abs", "n"], ascending=[False, False, False])
    worst = eligible.index[0]
    return worst, work.loc[work["date_utc"] == worst]


def _load_qiq_day(qiq_path: Path, day: object) -> pd.DataFrame:
    """Load only rows whose UTC calendar date equals ``day`` (chunked read)."""
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
        chunk = chunk[chunk["time_utc"].dt.date == day]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values("time_utc")


def _filter_utc_hours(df: pd.DataFrame, day: object, h0: int, h1: int) -> pd.DataFrame:
    """Keep rows on ``day`` with time in ``[h0:00, h1:00)`` UTC."""
    if h1 <= h0:
        raise ValueError("UTC_END_HOUR must be greater than UTC_START_HOUR.")
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
    if not TEST_LS.is_file():
        print(f"ERROR: Missing {TEST_LS}", file=sys.stderr)
        sys.exit(1)
    if not QIQ_MASTER.is_file():
        print(f"ERROR: Missing {QIQ_MASTER}", file=sys.stderr)
        sys.exit(1)

    ts = pd.read_csv(TEST_LS, sep="\t", parse_dates=["time_utc"])
    for c in ("bni", "bni_merra", "time_utc"):
        if c not in ts.columns:
            print(f"ERROR: Missing column {c!r} in {TEST_LS}", file=sys.stderr)
            sys.exit(1)

    if FORCE_DATE:
        worst_day = pd.to_datetime(FORCE_DATE).date()
        sub_eval = ts.loc[ts["time_utc"].dt.date == worst_day]
        mean_abs = (
            (sub_eval["bni_merra"] - sub_eval["bni"]).abs().mean()
            if len(sub_eval) and sub_eval[["bni", "bni_merra"]].notna().all(axis=1).any()
            else float("nan")
        )
        print(f"Using FORCE_DATE={worst_day} (mean |bni_merra−bni| on test rows: {mean_abs:.2f})")
    else:
        worst_day, day_eval = _pick_worst_day_utc(ts, MIN_TEST_ROWS_PER_DAY)
        mean_abs = day_eval["abs_bni_err"].mean()
        n_eval = len(day_eval)
        print(
            f"Worst UTC day by mean |bni_merra−bni|: {worst_day} "
            f"(mean_abs={mean_abs:.2f} W m⁻², n_test_rows={n_eval}, "
            f"min_rows_filter={MIN_TEST_ROWS_PER_DAY})"
        )

    qiq_day = _load_qiq_day(QIQ_MASTER, worst_day)
    if qiq_day.empty:
        print(f"ERROR: No qiq rows for UTC date {worst_day}", file=sys.stderr)
        sys.exit(1)

    for c in ("ghi", "bni", "dhi", "clearsky"):
        if c not in qiq_day.columns:
            print(f"ERROR: Missing column {c!r} in qiq extract", file=sys.stderr)
            sys.exit(1)

    qiq_win = _filter_utc_hours(qiq_day, worst_day, UTC_START_HOUR, UTC_END_HOUR)
    if qiq_win.empty:
        print(
            f"ERROR: No qiq rows for {worst_day} in "
            f"[{UTC_START_HOUR:02d}:00, {UTC_END_HOUR:02d}:00) UTC",
            file=sys.stderr,
        )
        sys.exit(1)

    cs = qiq_win["clearsky"].fillna(0).astype(int).eq(1).to_numpy()
    t = qiq_win["time_utc"]
    ghi = qiq_win["ghi"].to_numpy(dtype=float)
    bni = qiq_win["bni"].to_numpy(dtype=float)
    dhi = qiq_win["dhi"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
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
        }
    )

    y_lo, y_hi = _combined_ylim(ghi, bni, dhi)
    fig, ax = plt.subplots(figsize=(160 / 25.4, 60 / 25.4), layout="constrained")

    ax.plot(t, ghi, color=WONG_ORANGE, linewidth=_LW, label="GHI", zorder=2)
    ax.plot(t, dhi, color=WONG_BLU_GR, linewidth=_LW, label="DHI", zorder=2)
    ax.plot(t, bni, color=WONG_SKY, linewidth=_LW, label="BNI", zorder=3)

    idx = np.where(cs)[0]
    if idx.size:
        ax.scatter(
            t.iloc[idx],
            bni[idx],
            s=6,
            c=WONG_MARK,
            alpha=0.9,
            linewidths=0,
            label="Clear-sky points",
            zorder=5,
        )

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("UTC", fontsize=_PT)
    ax.set_ylabel(r"Irradiance (W m$^{-2}$)", fontsize=_PT)
    ax.set_title(
        f"Worst-day SW — {worst_day}  "
        f"({UTC_START_HOUR:02d}:00–{UTC_END_HOUR:02d}:00 UTC, n={len(qiq_win)})",
        fontsize=_PT,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="both", labelsize=_PT, width=_LW, length=2.5)
    ax.grid(True, alpha=0.35, linewidth=_LW)
    ax.legend(
        loc="upper left",
        frameon=False,
        borderpad=0.2,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
