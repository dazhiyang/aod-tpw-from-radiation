"""
2.create_holdout: Extracts a strict 30% global holdout test set before LHS sampling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

INPUT_TXT = PROJECT / "Data" / "qiq_1min_merra_qc.txt"
OUTPUT_TESTPOOL = PROJECT / "Data" / "testpool.txt"
OUTPUT_TRAINPOOL = PROJECT / "Data" / "trainpool.txt"

ZENITH_MAX = 85.0
FRACTION = 0.30
SEED = 42

FEATURES = [
    "ghi", "bni", "dhi", "zenith",
    "merra_ALPHA", "merra_ALBEDO", "merra_TQV", 
    "merra_TO3", "merra_PS", "merra_BETA"
]

def main():
    if not INPUT_TXT.is_file():
        print(f"ERROR: Missing input file: {INPUT_TXT}")
        sys.exit(1)

    print(f"Loading master pool: {INPUT_TXT.name}...")
    df = pd.read_csv(INPUT_TXT, sep="\t", comment="#", parse_dates=["time_utc"], index_col="time_utc")
    df = df.sort_index()

    # Apply physical filters
    day = df["zenith"].astype(float) <= ZENITH_MAX
    clear = df["clearsky"].astype(int) == 1
    
    print("Filtering for clear-sky daytime conditions and dropping NaNs...")
    pool = df.loc[day & clear, FEATURES].dropna()
    
    total_valid = len(pool)
    print(f"Total valid clear-sky rows available: {total_valid}")
    
    # 30% random sample for the strict holdout test set
    holdout = pool.sample(frac=FRACTION, random_state=SEED).copy()
    holdout.index.name = "time_utc"
    
    # The remainder forms the new training pool (for Latin Hypercube sampling)
    trainpool = pool.drop(holdout.index).copy()
    trainpool.index.name = "time_utc"
    
    print(f"Split results ({FRACTION*100}% holdout):")
    print(f" -> Holdout Test Set: {len(holdout)} rows")
    print(f" -> LHS Train Pool:   {len(trainpool)} rows")
    
    # Write exact metadata
    common_meta = (
        f"# Source: {INPUT_TXT.name} | Filters: Zenith<={ZENITH_MAX}, Clearsky=1\n"
        f"# Total={total_valid} | Split Fraction={FRACTION} | Seed={SEED}\n"
    )
    
    print("Saving isolated datasets...")
    with open(OUTPUT_TESTPOOL, "w", encoding="ascii") as f:
        f.write(common_meta + f"# HOLDOUT TEST SET ({len(holdout)} rows)\n")
    holdout.to_csv(OUTPUT_TESTPOOL, mode="a", sep="\t", float_format="%.12g")
    
    with open(OUTPUT_TRAINPOOL, "w", encoding="ascii") as f:
        f.write(common_meta + f"# LHS TRAINING POOL ({len(trainpool)} rows)\n")
    trainpool.to_csv(OUTPUT_TRAINPOOL, mode="a", sep="\t", float_format="%.12g")

    print(f"Successfully wrote {OUTPUT_TESTPOOL.name}")
    print(f"Successfully wrote {OUTPUT_TRAINPOOL.name}")

if __name__ == "__main__":
    main()
