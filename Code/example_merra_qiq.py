"""
Extract one UTC day (Jan 4) of 1-min QIQ BSRN from a monthly .dat.gz, add solar geometry with
``add_solpos_columns``, then align MERRA-2 (``fetch_rest2``), and save the full table as
tab-separated text under ``Data/qiq_YYYYMMDD_all_1min.txt``.

``add_solpos_columns`` adds: zenith, apparent_zenith, azimuth, bni_extra, ghi_extra.

bsrn.fetch_rest2 reindexes MERRA-2 to your timestamps and time-interpolates to 1-min.

Requires: bsrn, huggingface_hub, pandas.
MERRA-2 from Hugging Face (dazhiyang/bsrn-merra2) — needs network.

After this, ``Code/run_uvspec_forward.py`` can run libRadtran ``uvspec`` on the saved table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsrn.constants import BSRN_STATIONS
from bsrn.io.merra2 import fetch_rest2
from bsrn.io.reader import read_lr0100
from bsrn.physics.geometry import add_solpos_columns

PROJECT = Path(__file__).resolve().parent.parent
BSRN_FILE = PROJECT / "Data" / "QIQ" / "qiq0124.dat.gz"
STATION = "QIQ"

TARGET_MONTH = 1
TARGET_DAY = 7

df = read_lr0100(str(BSRN_FILE))
if df is None or df.empty:
    raise SystemExit(f"Could not read LR0100 from {BSRN_FILE}")

m = (df.index.month == TARGET_MONTH) & (df.index.day == TARGET_DAY)
day = df.loc[m].copy()
if day.empty:
    raise SystemExit(
        f"No 1-min rows for month={TARGET_MONTH} day={TARGET_DAY} in {BSRN_FILE}"
    )

day = add_solpos_columns(day, station_code=STATION)

target_index = pd.DatetimeIndex(day.index)
merra = fetch_rest2(target_index, STATION)
merra = merra.add_prefix("merra_")

merged = day.join(merra, how="left")

y = int(day.index[0].year)
tag = f"qiq_{y}{TARGET_MONTH:02d}{TARGET_DAY:02d}"
meta = BSRN_STATIONS[STATION]
out_txt = PROJECT / "Data" / f"{tag}_all_1min.txt"

out_txt.parent.mkdir(parents=True, exist_ok=True)

merged_out = merged.copy()
merged_out.index.name = "time_utc"
with open(out_txt, "w", encoding="ascii") as f:
    f.write(
        "# QIQ: BSRN + add_solpos_columns + MERRA-2 (fetch_rest2). UTC.\n"
        f"# lat_deg={meta['lat']:.6f} lon_deg={meta['lon']:.6f} elev_m={meta['elev']:.3f}\n"
    )
merged_out.to_csv(out_txt, mode="a", sep="\t", float_format="%.6f")

print("BSRN file:", BSRN_FILE)
print("UTC day:", target_index[0].strftime("%Y-%m-%d"), "  1-min rows:", len(merged))
print("Columns:", list(merged.columns))
print("\nFirst 3 rows:")
print(merged.head(3))
print("\nWrote:", out_txt)
