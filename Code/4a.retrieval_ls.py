"""
4.retrieval: Minimalist driver for libRadtran clear-sky simulation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from libRadtran import (
    LIBRADTRANDIR, CLEARSKY_CONFIG, process_row_ls
)

PROJECT = Path(__file__).resolve().parent.parent

# --- Suffix Handling ---
INPUT_DATA = Path(os.environ.get("INPUT_DATA", str(PROJECT / "Data" / "train_0.5k.txt")))
_out_name = INPUT_DATA.name.replace("train", "train_ls") if "train" in INPUT_DATA.name else f"{INPUT_DATA.stem}_ls.txt"
OUTPUT_DATA = Path(os.environ.get("OUTPUT_DATA", str(PROJECT / "Data" / _out_name)))

# --- Execution Logic ---
if not INPUT_DATA.is_file():
    print(f"ERROR: Missing input file: {INPUT_DATA}")
    sys.exit(1)

print(f"Loading dataset: {INPUT_DATA.name}")
df = pd.read_csv(INPUT_DATA, sep="\t", comment="#", parse_dates=["time_utc"])
df = df.set_index("time_utc").sort_index()

print(f"Starting retrieval for {len(df)} rows using libRadtran...")

_ls_fn = lambda r: process_row_ls(r, LIBRADTRANDIR, CLEARSKY_CONFIG)
if _tqdm is not None:
    _tqdm.pandas(desc="LS Beta + H2O", leave=True)
    results = df.progress_apply(_ls_fn, axis=1)
else:
    results = df.apply(_ls_fn, axis=1)

# Drop input columns that are recomputed in process_row_ls (LHS copies of fluxes and
# repeated MERRA scalars) so we do not duplicate column names after concat.
_overlap = [c for c in results.columns if c in df.columns]
df_base = df.drop(columns=_overlap, errors="ignore")
out = pd.concat([df_base, results], axis=1)

cols = [
    "ghi", "bni", "dhi", "ghi_merra", "bni_merra", "dhi_merra",
    "ghi_ls", "bni_ls", "dhi_ls", "beta_ls", "w_ls",
    "merra_ALPHA", "merra_BETA", "merra_TO3", "merra_TQV", "merra_ALBEDO", "merra_PS", "zenith",
]
final_cols = [c for c in cols if c in out.columns]
out.reset_index()[["time_utc"] + final_cols].to_csv(
    OUTPUT_DATA, sep="\t", index=False, float_format="%.8f"
)

print(f"Successfully wrote: {OUTPUT_DATA.name}")
