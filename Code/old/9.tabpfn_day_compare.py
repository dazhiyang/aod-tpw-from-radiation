r"""
9.tabpfn_day_compare: One **Beijing** calendar day — **MERRA-2 prior forward** plus **two TabPFN**
pipelines (different training targets), each with **one** ``run_clearsky`` forward per row.

* **TabPFN (LS train):** fit on ``beta_ls``, ``w_ls`` from ``train_ls_*.txt``; predict β, w; forward.
* **TabPFN (OE train):** fit on ``beta_oe``, ``w_oe`` from ``train_oe_*.txt``; predict β, w; forward.

Same input features for both; targets differ (LS vs OE retrieval labels in the training tables). The
figure has **three rows** (GHI, BNI, DHI): measured, MERRA-2 forward, TabPFN-LS forward, TabPFN-OE
forward. **No CSV** is written; only ``OUTPUT_PNG``.

The Beijing day maps to a **single continuous UTC interval** (typically spanning two UTC calendar
dates). Daytime rows: ``zenith < 87``.

Env:
    BEIJING_DATE — overrides the in-file ``BEIJING_DATE`` default (``YYYY-MM-DD``, Asia/Shanghai day).
    K_SUFFIX — training file suffix (default ``_0.5k``).
    TRAIN_LS — default ``Data/train_ls{K_SUFFIX}.txt``.
    TRAIN_OE — default ``Data/train_oe{K_SUFFIX}.txt``.
    QIQ_MASTER — default ``Data/qiq_1min_merra_qc.txt``.
    OUTPUT_PNG — default ``tex/figures/tabpfn_day_{BEIJING_DATE}.png``.
    MAX_ROWS — optional subsample for debugging.

Requires: pandas, numpy, torch, tabpfn, tqdm, matplotlib, libRadtran (**three** ``uvspec`` calls per row).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from tqdm.auto import tqdm

from libRadtran import (
    BETA_MAX,
    BETA_MIN,
    CLEARSKY_CONFIG,
    LIBRADTRANDIR,
    W_MAX,
    W_MIN,
    forward_merra_explicit,
    merra_explicit_physics,
    run_clearsky,
)

PROJECT = Path(__file__).resolve().parent.parent

# Beijing calendar day to process (Asia/Shanghai). Override: ``BEIJING_DATE`` env or ``argv[1]``.
BEIJING_DATE = "2024-09-06"
K_SUFFIX = os.environ.get("K_SUFFIX", "_0.5k")

_PT = 8
_LW = 0.3

_CLEAR = ("ghi_clear", "bni_clear", "dhi_clear")
_TRANS = ("ghi_trans", "bni_trans", "dhi_trans")
_MEAS = ("ghi", "bni", "dhi")

# Same TabPFN inputs as ``5.tabpfn``. ``*_trans`` are computed below (not in ``FEATURES`` unless you switch ``5``).
FEATURES = [
    *_MEAS,
    "zenith",
    "merra_ALPHA",
    "merra_ALBEDO",
    "merra_TQV",
    "merra_TO3",
    "merra_PS",
]

_QIQ_USECOLS = list(_MEAS) + list(_CLEAR) + [
    "zenith",
    "merra_ALPHA",
    "merra_ALBEDO",
    "merra_TQV",
    "merra_TO3",
    "merra_PS",
    "time_utc",
    "clearsky",
    "merra_BETA",
]

TZ_BEIJING = ZoneInfo("Asia/Shanghai")


def _add_transmittance(df: pd.DataFrame) -> pd.DataFrame:
    """Measured / REST2 clear-sky; same as ``5.tabpfn._add_transmittance`` (optional future inputs)."""
    out = df.copy()
    out["ghi_trans"] = out["ghi"] / out["ghi_clear"]
    out["bni_trans"] = out["bni"] / out["bni_clear"]
    out["dhi_trans"] = out["dhi"] / out["dhi_clear"]
    return out

COLOR_MEAS = "#333333"
COLOR_MERRA = "#E69F00"
COLOR_TABPFN_LS = "#56B4E9"
COLOR_TABPFN_OE = "#009E73"
CLEAR_CURTAIN_COLOR = "#B3B3B3"
CLEAR_CURTAIN_ALPHA = 0.15


def _beijing_day_utc_bounds(day_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``[start_utc, end_utc)`` for the Beijing calendar date ``day_str``."""
    d = pd.Timestamp(day_str).normalize().date()
    start_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, tzinfo=TZ_BEIJING)
    end_local = start_local + pd.Timedelta(days=1)
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def _load_qiq_window(
    qiq_path: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Chunked read; keep rows with ``start_utc <= time_utc < end_utc`` (UTC)."""
    usecols = _QIQ_USECOLS
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
        tt = pd.to_datetime(chunk["time_utc"], utc=True)
        m = (tt >= start_utc) & (tt < end_utc)
        if not m.any():
            continue
        chunk = chunk.loc[m].copy()
        chunk["time_utc"] = tt[m]
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=_QIQ_USECOLS)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values("time_utc")


def _forward_merra_row(row: pd.Series) -> pd.Series:
    """One clear-sky forward using MERRA-2 ``merra_BETA`` and scaled ``merra_TQV`` (prior only)."""
    sim = forward_merra_explicit(row, LIBRADTRANDIR, CLEARSKY_CONFIG, quiet=True)
    return pd.Series(
        {
            "ghi_merra": float(sim.get("ghi_sim", np.nan)),
            "bni_merra": float(sim.get("bni_sim", np.nan)),
            "dhi_merra": float(sim.get("dhi_sim", np.nan)),
        }
    )


def _forward_tabpfn_row(row: pd.Series, beta_p: float, w_p: float, tag: str) -> pd.Series:
    """One clear-sky forward; ``tag`` is ``ls`` or ``oe`` (output column suffix)."""
    alpha_m, o3_du_m, _, _ = merra_explicit_physics(row, CLEARSKY_CONFIG)
    sim = run_clearsky(
        row,
        LIBRADTRANDIR,
        CLEARSKY_CONFIG,
        angstrom_alpha=alpha_m,
        o3_du=o3_du_m,
        angstrom_beta=beta_p,
        w=w_p,
        quiet=True,
    )
    return pd.Series(
        {
            f"ghi_tabpfn_{tag}": float(sim.get("ghi_sim", np.nan)),
            f"bni_tabpfn_{tag}": float(sim.get("bni_sim", np.nan)),
            f"dhi_tabpfn_{tag}": float(sim.get("dhi_sim", np.nan)),
        }
    )


def _tabpfn_fit_predict_clip(
    train_df: pd.DataFrame,
    targets: tuple[str, str],
    X_day: pd.DataFrame,
    device: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit TabPFN per target (same as ``5.tabpfn``), predict on ``X_day``, return clipped β and w."""
    beta_t, w_t = targets
    tr = train_df.dropna(subset=FEATURES + [beta_t, w_t])
    print(f"TabPFN [{label}] training rows: {len(tr)}")
    for tgt in (beta_t, w_t):
        print(f"  fit/predict: {tgt}")
        model = TabPFNRegressor(device=device)
        model.fit(tr[FEATURES], tr[tgt])
        if tgt == beta_t:
            beta = np.clip(model.predict(X_day), BETA_MIN, BETA_MAX)
        else:
            w = np.clip(model.predict(X_day), W_MIN, W_MAX)
    # Force plain arrays so callers can explicitly reindex by timestamp.
    return np.asarray(beta, dtype=float), np.asarray(w, dtype=float)


def _contiguous_true_spans(
    t_local: pd.Series,
    mask: np.ndarray,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous ``True`` spans as ``[(start, end), ...]`` for ``axvspan`` curtains."""
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_i: int | None = None
    n = len(mask)
    for i in range(n):
        if mask[i] and start_i is None:
            start_i = i
        if start_i is not None and (i == n - 1 or not mask[i + 1]):
            left = t_local.iloc[start_i]
            if i + 1 < n:
                right = t_local.iloc[i + 1]
            elif i > start_i:
                right = t_local.iloc[i]
            else:
                right = t_local.iloc[i] + pd.Timedelta(minutes=1)
            spans.append((left, right))
            start_i = None
    return spans


def main() -> None:
    beijing = BEIJING_DATE.strip()
    env_bj = os.environ.get("BEIJING_DATE", "").strip()
    if env_bj:
        beijing = env_bj
    if len(sys.argv) > 1:
        beijing = sys.argv[1].strip()
    if not beijing:
        print("ERROR: Set BEIJING_DATE in this file, or env, or pass YYYY-MM-DD as argv.", file=sys.stderr)
        sys.exit(1)

    train_ls = Path(
        os.environ.get("TRAIN_LS", str(PROJECT / "Data" / f"train_ls{K_SUFFIX}.txt"))
    )
    train_oe = Path(
        os.environ.get("TRAIN_OE", str(PROJECT / "Data" / f"train_oe{K_SUFFIX}.txt"))
    )
    qiq_master = Path(os.environ.get("QIQ_MASTER", str(PROJECT / "Data" / "qiq_1min_merra_qc.txt")))
    out_png = Path(
        os.environ.get(
            "OUTPUT_PNG",
            str(PROJECT / "tex" / "figures" / f"tabpfn_day_{beijing}.png"),
        )
    )
    max_rows = os.environ.get("MAX_ROWS", "").strip()
    n_max = int(max_rows) if max_rows.isdigit() else None

    for p in (train_ls, train_oe, qiq_master):
        if not p.is_file():
            print(f"ERROR: Missing {p}", file=sys.stderr)
            sys.exit(1)

    start_utc, end_utc = _beijing_day_utc_bounds(beijing)
    print(f"Beijing day {beijing} -> UTC [{start_utc}, {end_utc})")

    print(f"Loading QIQ window from {qiq_master.name}...")
    raw = _load_qiq_window(qiq_master, start_utc, end_utc)
    if raw.empty:
        print("ERROR: No rows in UTC window.", file=sys.stderr)
        sys.exit(1)

    raw["clearsky"] = pd.to_numeric(raw["clearsky"], errors="coerce").fillna(0).astype(int)
    day = raw.copy()
    day = day[np.isfinite(day["zenith"].to_numpy(dtype=float))]
    day = day[day["zenith"].to_numpy(dtype=float) < 87.0]
    day = _add_transmittance(day)
    day = day.replace([np.inf, -np.inf], np.nan)
    day = day.dropna(subset=FEATURES)
    day["clearsky"] = pd.to_numeric(day["clearsky"], errors="coerce").fillna(0).astype(int)
    if day.empty:
        print("ERROR: No daytime rows after QC.", file=sys.stderr)
        sys.exit(1)

    if n_max is not None and len(day) > n_max:
        day = day.iloc[:n_max].copy()
        print(f"MAX_ROWS: using first {n_max} rows.")

    day = day.set_index("time_utc").sort_index()
    day_clear = day[day["clearsky"].eq(1)].copy()
    if day_clear.empty:
        print("ERROR: No clear-sky daytime rows for retrieval on this day.", file=sys.stderr)
        sys.exit(1)

    X_day = day_clear[FEATURES]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"TabPFN device: {device}")

    print(f"Loading: {train_ls.name}")
    df_ls = pd.read_csv(train_ls, sep="\t")
    df_ls = _add_transmittance(df_ls).replace([np.inf, -np.inf], np.nan)
    beta_ls, w_ls = _tabpfn_fit_predict_clip(
        df_ls, ("beta_ls", "w_ls"), X_day, device, "LS targets",
    )

    print(f"Loading: {train_oe.name}")
    df_oe = pd.read_csv(train_oe, sep="\t")
    df_oe = _add_transmittance(df_oe).replace([np.inf, -np.inf], np.nan)
    beta_oe, w_oe = _tabpfn_fit_predict_clip(
        df_oe, ("beta_oe", "w_oe"), X_day, device, "OE targets",
    )

    # Bind predictions to timestamps explicitly to avoid any positional drift.
    beta_ls_s = pd.Series(beta_ls, index=day_clear.index, dtype=float)
    w_ls_s = pd.Series(w_ls, index=day_clear.index, dtype=float)
    beta_oe_s = pd.Series(beta_oe, index=day_clear.index, dtype=float)
    w_oe_s = pd.Series(w_oe, index=day_clear.index, dtype=float)

    print("Forward models on clear-sky rows only (MERRA + TabPFN-LS + TabPFN-OE)...")
    m_rows: list[pd.Series] = []
    tls_rows: list[pd.Series] = []
    toe_rows: list[pd.Series] = []
    for ts, row in tqdm(
        day_clear.iterrows(),
        total=len(day_clear),
        desc="libRadtran forward",
    ):
        m_rows.append(_forward_merra_row(row))
        tls_rows.append(_forward_tabpfn_row(row, float(beta_ls_s.loc[ts]), float(w_ls_s.loc[ts]), "ls"))
        toe_rows.append(_forward_tabpfn_row(row, float(beta_oe_s.loc[ts]), float(w_oe_s.loc[ts]), "oe"))
    day = pd.concat(
        [
            day,
            pd.DataFrame(m_rows, index=day_clear.index),
            pd.DataFrame(tls_rows, index=day_clear.index),
            pd.DataFrame(toe_rows, index=day_clear.index),
        ],
        axis=1,
    )

    out = day.reset_index()

    # --- Figure: three rows (GHI, BNI, DHI), x = Beijing local time
    t_utc = pd.to_datetime(out["time_utc"], utc=True)
    t_bj = t_utc.dt.tz_convert(TZ_BEIJING)
    clear_mask = out["clearsky"].fillna(0).astype(int).to_numpy() == 1
    clear_spans = _contiguous_true_spans(t_bj.reset_index(drop=True), clear_mask)

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
            "grid.color": "black",
            "grid.alpha": 0.35,
            "grid.linewidth": _LW,
        }
    )

    flux_panels: list[tuple[str, str]] = [
        ("ghi", r"GHI (W m$^{-2}$)"),
        ("bni", r"BNI (W m$^{-2}$)"),
        ("dhi", r"DHI (W m$^{-2}$)"),
    ]
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(160 / 25.4, 195 / 25.4),
        sharex=True,
        layout="constrained",
    )
    fmt = mdates.DateFormatter("%H:%M")

    for ax, (comp, ylab) in zip(axes, flux_panels):
        for t0, t1 in clear_spans:
            ax.axvspan(
                t0,
                t1,
                color=CLEAR_CURTAIN_COLOR,
                alpha=CLEAR_CURTAIN_ALPHA,
                linewidth=0,
                zorder=0,
            )
        ax.plot(t_bj, out[comp], color=COLOR_MEAS, label="Measured", zorder=5)
        ax.plot(t_bj, out[f"{comp}_merra"], color=COLOR_MERRA, label="MERRA-2 forward", zorder=2)
        ax.plot(
            t_bj,
            out[f"{comp}_tabpfn_ls"],
            color=COLOR_TABPFN_LS,
            label="TabPFN (LS train) forward",
            zorder=3,
        )
        ax.plot(
            t_bj,
            out[f"{comp}_tabpfn_oe"],
            color=COLOR_TABPFN_OE,
            label="TabPFN (OE train) forward",
            zorder=4,
        )
        ax.set_ylabel(ylab, fontsize=_PT)
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="both", labelsize=_PT, width=_LW, length=2.5)
        ax.grid(True, color="black", alpha=0.35, linewidth=_LW)

    axes[0].set_title(
        f"Irradiance — Beijing {beijing} (CST); shaded = clear-sky retrieval windows",
        fontsize=_PT,
    )
    axes[-1].set_xlabel("Beijing time (CST)", fontsize=_PT)
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        fontsize=_PT - 1,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
