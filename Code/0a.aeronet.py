"""Read two fixed AERONET Level 2.0 all-points files for BSRN colocation (PAL, TAT).

``19930101_20260328_Palaiseau.lev20`` → **PAL**; ``19930101_20260328_TGF_Tsukuba.lev20`` → **TAT**.
Comma-separated, **6** lines skipped before the header (R ``read.table(..., skip=6)``). Missing
values **-999** → NaN; Ångström QC and **AOD550** from ``log τ = a₀ + a₁ log λ + a₂ (log λ)²`` using
the first four valid columns among 440, 500, 675, 870, then 380, 1020 (same column order as R
``tmp[, 3:8]``). Raw ``*.lev20`` and ``processed_{PAL|TAT}.txt`` live under ``Data/AERONET/PAL/`` and
``Data/AERONET/TAT/``. On first run, older flat files in ``Data/AERONET/`` are moved into those folders
if the destination path is still missing.
Complete instantaneous rows are averaged to **1-minute** UTC means (bucket = ``Time`` floored to
the minute; stamp written as ``YYYY-MM-DD HH:MM:00``). Header lines summarize
1-min row counts by year and raw instantaneous counts before averaging.

**R reference (same order as this script)**

- ``read.table(..., sep=',', header=TRUE, skip=6)`` → ``pd.read_csv(..., skiprows=6)``.
- Build ``time`` from date + time columns (R ``as.Date`` + ``ymd_hms``); prefer names
  ``Date.dd.mm.yyyy.`` and ``Time.hh.mm.ss.`` when present.
- ``length.alpha.min/max`` on raw exponent **before** ``replace_with_na(-999)``.
- ``tibble(time, alpha=X440.870..., AOD*)`` then ``replace_with_na`` / clip alpha to (-0.25, 2.5).
- ``wl <- log(c(440,500,675,870,380,1020))``; ``tau <- log(tmp[,3:8])``; ``tf_mat``;
  ``unique(tf_mat)`` loop; ``.lm.fit(X,y)`` with multivariate ``y`` → ``aod550_quadratic_batch``
  (``np.linalg.lstsq``).
- Count AOD550 <0 / >5, set those to NA; ``meas <- complete.cases`` → written TSV.

Env: ``AERONET_ROOT`` or ``AERONET_DIR`` (default ``<project>/Data/AERONET``); each station uses
``<root>/<STATION>/``.

Runs flat on load (no ``if __name__`` guard), like other numbered scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
AERONET_ROOT = Path(
    os.environ.get("AERONET_ROOT", os.environ.get("AERONET_DIR", str(PROJECT / "Data" / "AERONET")))
)

# BSRN code, filename inside ``AERONET_ROOT/<STATION>/``.
AERONET_FILES: tuple[tuple[str, str], ...] = (
    ("PAL", "19930101_20260328_Palaiseau.lev20"),
    ("TAT", "19930101_20260328_TGF_Tsukuba.lev20"),
)


def _migrate_flat_into_station_dirs() -> None:
    """Move ``root/*.lev20`` and ``root/processed_*.txt`` into ``root/PAL/`` / ``root/TAT/`` if needed."""
    for station, fname in AERONET_FILES:
        sub = AERONET_ROOT / station
        sub.mkdir(parents=True, exist_ok=True)
        flat_raw = AERONET_ROOT / fname
        flat_proc = AERONET_ROOT / f"processed_{station}.txt"
        tgt_raw = sub / fname
        tgt_proc = sub / f"processed_{station}.txt"
        if flat_raw.is_file() and not tgt_raw.is_file():
            flat_raw.replace(tgt_raw)
        if flat_proc.is_file() and not tgt_proc.is_file():
            flat_proc.replace(tgt_proc)

MISSING = -999.0

# R ``tmp[, 3:8]`` order: 440, 500, 675, 870, then fallbacks 380, 1020.
_AOD_BANDS_NM = (440, 500, 675, 870, 380, 1020)
_LOG_WL = np.log(np.asarray(_AOD_BANDS_NM, dtype=float))
_LOG_550 = np.log(550.0)
_AOD_FILE = tuple(f"AOD_{w}nm" for w in _AOD_BANDS_NM)
_AOD_KEY = tuple(f"AOD{w}" for w in _AOD_BANDS_NM)

# Preferred column names when the file uses this AERONET spelling.
AERONET_DATE_COL = "Date.dd.mm.yyyy."
AERONET_TIME_COL = "Time.hh.mm.ss."
AERONET_ANGSTROM_COL = "X440.870_Angstrom_Exponent"


def aod550_quadratic_batch(tau_log: np.ndarray, tf_mat: np.ndarray) -> np.ndarray:
    """R ``unique(tf_mat)`` + ``coef(.lm.fit(X, y))`` with ``y`` shape (4, n).

    ``tau_log`` is (n, 6) ``log(AOD)``; ``tf_mat`` is (n, 6) 1 where ``tau`` is finite. Returns (n,)
    AOD at 550 nm via ``exp(β₀ + β₁ log550 + β₂ log550²)``.
    """
    n = tau_log.shape[0]
    aod550 = np.full(n, np.nan, dtype=float)
    patterns, inv = np.unique(tf_mat, axis=0, return_inverse=True)
    for j in range(patterns.shape[0]):
        pattern = patterns[j]
        valid_idx = np.flatnonzero(pattern == 1)
        if valid_idx.size < 4:
            continue
        valid_idx = valid_idx[:4]
        row_sel = np.flatnonzero(inv == j)
        if row_sel.size == 0:
            continue
        x = _LOG_WL[valid_idx]
        X = np.column_stack([np.ones(4), x, x * x])
        y = tau_log[np.ix_(row_sel, valid_idx)].T
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < 3:
            continue
        pred_log = coef[0, :] + coef[1, :] * _LOG_550 + coef[2, :] * (_LOG_550**2)
        aod550[row_sel] = np.exp(pred_log)
    return aod550


_migrate_flat_into_station_dirs()

for station, fname in AERONET_FILES:
    station_dir = AERONET_ROOT / station
    station_dir.mkdir(parents=True, exist_ok=True)
    path = station_dir / fname
    if not path.is_file():
        print(f"missing {path}")
        continue

    raw = pd.read_csv(path, sep=",", header=0, skiprows=6, low_memory=False, encoding="utf-8")

    date_col = AERONET_DATE_COL if AERONET_DATE_COL in raw.columns else None
    if date_col is None:
        date_col = next(
            (c for c in raw.columns if "date" in str(c).lower() and "yyyy" in str(c).lower()),
            None,
        )
    time_col = AERONET_TIME_COL if AERONET_TIME_COL in raw.columns else None
    if time_col is None:
        time_col = next(
            (
                c
                for c in raw.columns
                if "time" in str(c).lower() and ("hh" in str(c).lower() or "ss" in str(c).lower())
            ),
            None,
        )
    if date_col is None or time_col is None:
        print(f"{fname}: no Date*/Time* column in {list(raw.columns)[:12]}...")
        continue

    # R: ``as.Date(..., format='%d:%m:%Y')`` then ``ymd_hms(paste(date, Time...))``.
    d_raw = raw[date_col].astype(str).str.strip()
    t_raw = raw[time_col].astype(str).str.strip()
    date_parsed = pd.to_datetime(d_raw, format="%d:%m:%Y", errors="coerce")
    date_parsed = date_parsed.combine_first(pd.to_datetime(d_raw, format="%d.%m.%Y", errors="coerce"))
    date_parsed = date_parsed.combine_first(pd.to_datetime(d_raw, dayfirst=True, errors="coerce"))
    time = pd.to_datetime(date_parsed.dt.strftime("%Y-%m-%d") + " " + t_raw, errors="coerce")

    alpha_col = AERONET_ANGSTROM_COL if AERONET_ANGSTROM_COL in raw.columns else None
    if alpha_col is None:
        alpha_col = next(
            (
                c
                for c in raw.columns
                if "angstrom" in str(c).lower()
                and "440" in str(c)
                and "870" in str(c).replace("-", "")
            ),
            None,
        )
    if alpha_col is None:
        for c in raw.columns:
            if "angstrom" in str(c).lower():
                alpha_col = c
                break
    if alpha_col is None:
        print(f"{fname}: no Ångström column")
        continue

    miss = [c for c in _AOD_FILE if c not in raw.columns]
    if miss:
        print(f"{fname}: missing columns {miss}")
        continue

    alpha_num = pd.to_numeric(raw[alpha_col], errors="coerce")
    # R: ``length.alpha.*`` on raw exponent before ``replace_with_na(-999)``.
    n_ab = int(((alpha_num > MISSING) & (alpha_num < -0.25)).sum())
    n_aa = int((alpha_num > 2.5).sum())

    # R: ``tibble`` then ``naniar::replace_with_na`` / ``ifelse`` on alpha.
    work = pd.DataFrame(
        {
            "time": time,
            "alpha": alpha_num,
            **{
                k: pd.to_numeric(raw[c], errors="coerce")
                for k, c in zip(_AOD_KEY, _AOD_FILE, strict=True)
            },
        }
    )
    for c in ("alpha", *_AOD_KEY):
        work[c] = work[c].replace(MISSING, np.nan).astype(float)
    work.loc[work["alpha"] < -0.25, "alpha"] = np.nan
    work.loc[work["alpha"] > 2.5, "alpha"] = np.nan

    tau = np.log(work[list(_AOD_KEY)].to_numpy(dtype=float))
    with np.errstate(invalid="ignore"):
        tf_mat = np.isfinite(tau).astype(np.int8)
    work["AOD550"] = aod550_quadratic_batch(tau, tf_mat)

    # R: ``length.aod550.min/max`` then ``ifelse(AOD550 < 0 | > 5, NA, AOD550)``.
    n550_lo = int((work["AOD550"] < 0).sum())
    n550_hi = int((work["AOD550"] > 5).sum())
    work.loc[(work["AOD550"] < 0) | (work["AOD550"] > 5), "AOD550"] = np.nan

    # R: ``meas %>% filter(complete.cases)``.
    inst = work.rename(columns={"time": "Time"})[["Time", "alpha", "AOD550"]].dropna(how="any")
    # explicitly localize to UTC since AERONET source is UTC
    inst["Time"] = pd.to_datetime(inst["Time"], utc=True)
    n_inst = len(inst)

    if n_inst:
        bucket = inst["Time"].dt.floor("1min")
        out = (
            inst.assign(_b=bucket)
            .groupby("_b", sort=True)[["alpha", "AOD550"]]
            .mean()
            .reset_index()
            .rename(columns={"_b": "Time"})
        )
        by_year = out["Time"].dt.year.value_counts().sort_index()
        year_part = " ".join(f"{int(y)}={int(by_year.loc[y])}" for y in by_year.index)
        y0, y1 = int(by_year.index.min()), int(by_year.index.max())
        year_summary = (
            f"1min_rows={len(out)} year_span={y0}-{y1} by_year: {year_part}  "
            f"raw_instantaneous_rows={n_inst}"
        )
        out["Time"] = out["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        out = pd.DataFrame(columns=["Time", "alpha", "AOD550"])
        year_summary = "1min_rows=0 by_year: (none)  raw_instantaneous_rows=0"

    n_valid = len(out)
    out_txt = station_dir / f"processed_{station}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            f"# AERONET {station} from {fname}; source fill {MISSING:g} -> NaN; "
            "1-min mean of complete rows; UTC; Time = floor minute as YYYY-MM-DD HH:MM:00.\n"
        )
        f.write(f"# {year_summary}\n")
    out.to_csv(out_txt, mode="a", sep="\t", index=False, float_format="%.6f", na_rep="")

    print(
        f"{station} {fname}: 1min_rows={n_valid} raw={n_inst} -> {out_txt.name}  "
        f"alpha<-0.25={n_ab}  alpha>2.5={n_aa}  AOD550<0={n550_lo}  AOD550>5={n550_hi}"
    )
    print(f"  {year_summary}")
