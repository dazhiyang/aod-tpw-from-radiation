"""2.create_holdout: Daytime pool (complete FEATURES) → 50 % train / 50 % test split (before LHS).

Edit **CONFIG** below for **PAL**, **TAT**, or another station/year (must match ``1.arrange.py`` output
``Data/<STATION>_<year>_all.txt``).

Rows are kept only if **all** ``FEATURES`` are finite, physical GHI/GHI\\_clear cuts pass,
``clearsky == 1``, and **both** ``aeronet_aod550`` and ``aeronet_alpha`` are present.

Writes ``Data/<STATION>_<year>_trainpool.txt`` and ``Data/<STATION>_<year>_testpool.txt``.

**QIQ** master file: ``Code/old/2.create_holdout.py`` → ``trainpool.txt`` / ``testpool.txt``.

Set ``LHS_INPUT`` / ``TEST_POOL`` for station-specific pool names when running steps 3 and 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIG — edit station and year (same convention as ``1.arrange.py``).
# =============================================================================
STATION = "PAL"
YEAR = 2024
INPUT_TXT = PROJECT / "Data" / f"{STATION}_{YEAR}_all.txt"
OUTPUT_TESTPOOL = PROJECT / "Data" / f"{STATION}_{YEAR}_testpool.txt"
OUTPUT_TRAINPOOL = PROJECT / "Data" / f"{STATION}_{YEAR}_trainpool.txt"
# =============================================================================

ZENITH_MAX = 87.0
FRACTION = 0.50
SEED = 42
GHI_CLEAR_MIN = 10.0

FEATURES = [
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
]

OPTIONAL_COLS = ("clearsky", "aeronet_aod550", "aeronet_alpha")
REQUIRED_EXTRA = ("clearsky", "aeronet_aod550", "aeronet_alpha")


def _coerce_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


if not INPUT_TXT.is_file():
    print(f"ERROR: Missing input file: {INPUT_TXT}")
    sys.exit(1)

print(f"{STATION} {YEAR}: loading {INPUT_TXT.name}...")
df = pd.read_csv(INPUT_TXT, sep="\t", comment="#", parse_dates=["time_utc"], index_col="time_utc")
df = df.sort_index()

day = df["zenith"].astype(float) <= ZENITH_MAX
day_df = df.loc[day].copy()

missing = [c for c in FEATURES if c not in day_df.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    sys.exit(1)
for c in REQUIRED_EXTRA:
    if c not in day_df.columns:
        print(f"ERROR: Missing column '{c}'; input must be from 1.arrange with AERONET merge.")
        sys.exit(1)

coerce_cols = list(FEATURES) + ["aeronet_aod550", "aeronet_alpha"]
day_df = _coerce_numeric(day_df, coerce_cols)

optional_avail = [c for c in OPTIONAL_COLS if c in day_df.columns]
out_cols = list(FEATURES) + optional_avail

complete = day_df[FEATURES].notna().all(axis=1)
phys = (day_df["ghi_clear"] > GHI_CLEAR_MIN) & (day_df["ghi"] >= 0)
clear = day_df["clearsky"].astype(float) == 1
aeronet_ok = day_df["aeronet_aod550"].notna() & day_df["aeronet_alpha"].notna()
pool = day_df.loc[complete & phys & clear & aeronet_ok, out_cols].copy()
pool.index.name = "time_utc"

total_valid = len(pool)
print(
    f"Daytime (zenith<={ZENITH_MAX}), FEATURES complete, ghi_clear>{GHI_CLEAR_MIN}, ghi>=0, "
    f"clearsky==1, AERONET AOD+alpha present: {total_valid} rows."
)

if total_valid == 0:
    print("ERROR: Empty pool.")
    sys.exit(1)

holdout = pool.sample(frac=FRACTION, random_state=SEED).copy()
trainpool = pool.drop(holdout.index).copy()

print(f"Split results ({FRACTION * 100:.0f}% test / {100 - FRACTION * 100:.0f}% train):")
print(f" -> Test pool:   {len(holdout)} rows")
print(f" -> Train pool:  {len(trainpool)} rows")

common_meta = (
    f"# Station={STATION} Year={YEAR} | Source: {INPUT_TXT.name}\n"
    f"# Filters: Zenith<={ZENITH_MAX}; FEATURES complete; ghi_clear>{GHI_CLEAR_MIN}; ghi>=0; "
    f"clearsky==1; aeronet_aod550+aeronet_alpha present\n"
    f"# Total={total_valid} | Test fraction={FRACTION} | Seed={SEED}\n"
)

print("Saving isolated datasets...")
with open(OUTPUT_TESTPOOL, "w", encoding="ascii") as f:
    f.write(common_meta + f"# TEST POOL ({len(holdout)} rows)\n")
holdout.to_csv(OUTPUT_TESTPOOL, mode="a", sep="\t", float_format="%.12g", na_rep="")

with open(OUTPUT_TRAINPOOL, "w", encoding="ascii") as f:
    f.write(common_meta + f"# TRAIN POOL ({len(trainpool)} rows)\n")
trainpool.to_csv(OUTPUT_TRAINPOOL, mode="a", sep="\t", float_format="%.12g", na_rep="")

print(f"Wrote {OUTPUT_TESTPOOL}")
print(f"Wrote {OUTPUT_TRAINPOOL}")
