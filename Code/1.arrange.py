"""QIQ LR0100 → solpos + QC + MERRA + BrightSun clearsky flag; NaN ghi/bni/dhi if any ``flag*`` == 1.

Output columns (``time_utc`` index + tab-separated): ``ghi``, ``bni``, ``dhi``, ``zenith``,
``merra_*`` (ALPHA, ALBEDO, TQV, TO3, PS, BETA), ``clearsky``. Nothing else is written.

**Default:** process **all** ``qiq*.dat.gz`` in ``Data/QIQ/`` (e.g. one year = 12 monthly files).

Env: ``ARRANGE_FIRST_MONTH_ONLY=1`` — only the first file (sorted name), for a quick test.
``ARRANGE_ONE_FILE=qiq0124.dat.gz`` — a single month. ``ARRANGE_OUTPUT`` — output path.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import bsrn
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
QIQ_DIR = PROJECT / "Data" / "QIQ"
STATION = "QIQ"
OUTPUT_TXT = Path(os.environ.get("ARRANGE_OUTPUT", str(PROJECT / "Data" / "qiq_1min_merra_qc.txt")))

_first_only = os.environ.get("ARRANGE_FIRST_MONTH_ONLY", "").strip().lower() in ("1", "true", "yes")
_one = os.environ.get("ARRANGE_ONE_FILE", "").strip()
paths = sorted(QIQ_DIR.glob("qiq*.dat.gz"))
if _one:
    paths = [QIQ_DIR / _one]
elif paths and _first_only:
    paths = [paths[0]]
    print(f"ARRANGE_FIRST_MONTH_ONLY: single file -> {paths[0].name}")

if not paths:
    raise SystemExit(f"No monthly files in {QIQ_DIR}")

print(f"Processing {len(paths)} monthly file(s)")

OUTPUT_COLUMNS = (
    "ghi",
    "bni",
    "dhi",
    "zenith",
    "merra_ALPHA",
    "merra_ALBEDO",
    "merra_TQV",
    "merra_TO3",
    "merra_PS",
    "merra_BETA",
    "clearsky",
)
_meta = bsrn.constants.BSRN_STATIONS[STATION]


def _fmt_csv_float(x: object) -> str:
    if pd.isna(x):
        return ""
    v = float(x)
    if math.isfinite(v) and math.isclose(v, round(v), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(v)))
    return format(v, ".12g")


chunks: list[pd.DataFrame] = []
for path in paths:
    df = bsrn.io.reader.read_lr0100(str(path))
    if df is None or df.empty:
        continue

    df = bsrn.physics.geometry.add_solpos_columns(df, station_code=STATION)
    df = bsrn.qc.wrapper.run_qc(df, station_code=STATION)
    merra = bsrn.io.merra2.fetch_rest2(df.index, STATION)
    merged = df.join(merra.add_prefix("merra_"), how="left")

    merged = bsrn.modeling.clear_sky.add_clearsky_columns(
        merged, station_code=STATION, model="rest2"
    )

    csd = bsrn.utils.brightsun_csd(
        merged["zenith"].to_numpy(dtype=float),
        merged["ghi"].to_numpy(dtype=float),
        merged["ghi_clear"].to_numpy(dtype=float),
        merged["dhi"].to_numpy(dtype=float),
        merged["dhi_clear"].to_numpy(dtype=float),
        times=merged.index,
        return_diagnostics=False,
    )

    # Clear-sky flag: 1 = BrightSun clear, 0 = not (missing -> 0)
    merged["clearsky"] = (
        pd.Series(csd["is_clearsky"], index=merged.index).fillna(False).astype(bool).astype(int)
    )

    flags = [c for c in merged.columns if c.startswith("flag")]
    if flags:
        bad = (merged[flags] == 1).any(axis=1)
        merged.loc[bad, ["ghi", "bni", "dhi"]] = np.nan

    chunks.append(merged.loc[:, OUTPUT_COLUMNS])

if not chunks:
    raise SystemExit("No data (empty or missing LR0100).")

combined = pd.concat(chunks, axis=0).sort_index()
combined.index.name = "time_utc"

OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="ascii") as f:
    f.write(
        "# QIQ: BSRN + zenith + MERRA_* + clearsky BrightSun (0/1); no lwd/temp/rh/pressure; "
        "SW NaN if QC fail. UTC.\n"
        f"# lat_deg={_meta['lat']:.6f} lon_deg={_meta['lon']:.6f} elev_m={_meta['elev']:.3f}\n"
    )
combined.to_csv(OUTPUT_TXT, mode="a", sep="\t", float_format=_fmt_csv_float, na_rep="")
print(f"{len(combined)} rows -> {OUTPUT_TXT}")
