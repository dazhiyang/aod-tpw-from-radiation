"""BSRN LR0100 + MERRA + REST2 clear-sky + Lefèvre CSD, then left-merge 1-min AERONET (AOD550, alpha).

AERONET is treated as **instantaneous** (1-min aggregated) photometer data: we **reindex** to BSRN UTC minutes
with **no interpolation** and no forward/back fill—only exact timestamp matches get values; elsewhere NaN
(sparse columns in the output table).

Same chain as ``Code/old/1.arrange.py`` (QIQ), for any BSRN code in ``bsrn.constants.BSRN_STATIONS`` that has
LR0100 monthlies and a matching ``Data/AERONET/<STATION>/processed_<STATION>.txt``.

LR0100 monthlies live under ``BSRN_DIR``; AERONET table at ``AERONET_PROCESSED``. Output columns match
``qiq_1min_merra_qc.txt`` plus ``aeronet_aod550`` and ``aeronet_alpha``.

QIQ-only pipeline: ``Code/old/1.arrange.py``. All run-specific choices live in **CONFIG** below.
"""

from __future__ import annotations

import math
from io import StringIO
from pathlib import Path

import bsrn
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIG — edit paths and station here (e.g. PAL + ``pal*24.dat.gz`` under ``Data/BSRN/PAL``).
# =============================================================================
STATION = "PAL"
YEAR = 2024
BSRN_DIR = PROJECT / "Data" / "BSRN" / STATION
AERONET_PROCESSED = PROJECT / "Data" / "AERONET" / STATION / f"processed_{STATION}.txt"
OUTPUT_TXT = PROJECT / "Data" / f"{STATION}_{YEAR}_all.txt"
# ``None`` → all ``{prefix}{MM}{yy}.dat.gz`` present under ``BSRN_DIR``; else a single filename.
ONE_MONTHLY_FILE: str | None = None
# =============================================================================

PREFIX = STATION.lower()
YY = YEAR % 100

BSRN_COLS = (
    "ghi",
    "bni",
    "dhi",
    "ghi_clear",
    "bni_clear",
    "dhi_clear",
    "zenith",
    "merra_ALPHA",
    "merra_ALBEDO",
    "merra_TQV",
    "merra_TO3",
    "merra_PS",
    "merra_BETA",
    "clearsky",
)
OUTPUT_COLUMNS = BSRN_COLS + ("aeronet_aod550", "aeronet_alpha")
_meta = bsrn.constants.BSRN_STATIONS[STATION]


def _fmt_csv_float(x: object) -> str:
    if pd.isna(x):
        return ""
    v = float(x)
    if math.isfinite(v) and math.isclose(v, round(v), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(v)))
    return format(v, ".12g")


def _read_aeronet_processed(path: Path) -> pd.DataFrame:
    """Load tab file, skipping ``#`` header lines; return UTC-indexed ``aeronet_*`` columns.

    ``Time`` in the file is treated as **UTC** (see ``0a.aeronet.py``). Values are aligned by **position**
    to the index (``.to_numpy()``), not by pandas Series index labels, so rows are not dropped to NaN.
    """
    if not path.is_file():
        raise SystemExit(f"Missing AERONET file: {path}")
    with open(path, encoding="utf-8") as f:
        body = "".join(ln for ln in f if ln.strip() and not ln.lstrip().startswith("#"))
    ae = pd.read_csv(StringIO(body), sep="\t")
    if not {"Time", "alpha", "AOD550"}.issubset(ae.columns):
        raise SystemExit(f"AERONET columns expected Time, alpha, AOD550; got {list(ae.columns)}")
    idx = pd.DatetimeIndex(pd.to_datetime(ae["Time"], utc=True), tz="UTC")
    out = pd.DataFrame(
        {
            "aeronet_aod550": ae["AOD550"].astype(float).to_numpy(),
            "aeronet_alpha": ae["alpha"].astype(float).to_numpy(),
        },
        index=idx,
    )
    return out[~out.index.duplicated(keep="first")]


if ONE_MONTHLY_FILE:
    paths = [BSRN_DIR / ONE_MONTHLY_FILE]
else:
    paths = [BSRN_DIR / f"{PREFIX}{m:02d}{YY:02d}.dat.gz" for m in range(1, 13)]
    paths = [p for p in paths if p.is_file()]
if not paths:
    raise SystemExit(f"No monthly {PREFIX}*{YY:02d}.dat.gz under {BSRN_DIR}")

print(f"{STATION} {YEAR}: {len(paths)} monthly file(s) from {BSRN_DIR}")

aeronet = _read_aeronet_processed(AERONET_PROCESSED)
aeronet = aeronet[(aeronet.index.year == YEAR)]

chunks: list[pd.DataFrame] = []
for path in paths:
    df = bsrn.io.reader.read_lr0100(str(path))
    if df is None or df.empty:
        continue

    # Explicitly ensure index is UTC-aware early on.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = bsrn.physics.geometry.add_solpos_columns(df, station_code=STATION)
    df = bsrn.qc.wrapper.run_qc(df, station_code=STATION)
    merra = bsrn.io.merra2.fetch_rest2(df.index, STATION)
    merged = df.join(merra.add_prefix("merra_"), how="left")

    merged = bsrn.modeling.clear_sky.add_clearsky_columns(
        merged, station_code=STATION, model="rest2"
    )

    # Lefèvre et al. (2013) CSD: GHI, DHI, extraterrestrial GHI, zenith (``bsrn.utils.lefevre_csd``).
    csd = bsrn.utils.lefevre_csd(
        merged["ghi"].to_numpy(dtype=float),
        merged["dhi"].to_numpy(dtype=float),
        merged["ghi_extra"].to_numpy(dtype=float),
        merged["zenith"].to_numpy(dtype=float),
        times=merged.index,
        return_diagnostics=False,
    )

    merged["clearsky"] = (
        pd.Series(csd["is_clearsky"], index=merged.index).fillna(False).astype(bool).astype(int)
    )

    flags = [c for c in merged.columns if c.startswith("flag")]
    if flags:
        bad = (merged[flags] == 1).any(axis=1)
        merged.loc[bad, ["ghi", "bni", "dhi"]] = np.nan

    # Align AERONET data exactly on UTC minutes using label indexing.
    # Note: reindex(merged.index) ensures we only get rows matching the BSRN minutes.
    ae_aligned = aeronet.reindex(merged.index)
    merged["aeronet_aod550"] = ae_aligned["aeronet_aod550"]
    merged["aeronet_alpha"] = ae_aligned["aeronet_alpha"]
    chunks.append(merged.loc[:, OUTPUT_COLUMNS])

if not chunks:
    raise SystemExit("No data (empty or missing LR0100).")

combined = pd.concat(chunks, axis=0).sort_index()
# Save as naive UTC to avoid timezone-string overhead in CSV, while being explicitly UTC.
combined.index = combined.index.tz_convert("UTC").tz_localize(None)
combined.index.name = "time_utc"

OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="ascii") as f:
    f.write(
        f"# {STATION} {YEAR}: BSRN + REST2 clear-sky + zenith + MERRA_* + clearsky (Lefevre CSD); "
        f"aeronet_* = AERONET only at exact UTC minutes (no interpolation); else blank. UTC.\n"
        f"# lat_deg={_meta['lat']:.6f} lon_deg={_meta['lon']:.6f} elev_m={_meta['elev']:.3f}\n"
        f"# aeronet_source={AERONET_PROCESSED.name}\n"
    )
combined.to_csv(OUTPUT_TXT, mode="a", sep="\t", float_format=_fmt_csv_float, na_rep="")
n_ae = combined["aeronet_aod550"].notna().sum()
print(f"{len(combined)} rows -> {OUTPUT_TXT}  ({n_ae} rows with AERONET)")
