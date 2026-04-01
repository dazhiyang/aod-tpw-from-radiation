"""
4b.retrieval_oe: libRadtran optimal-estimation retrieval (beta, alpha).

Edit **CONFIG** to match ``1.arrange.py`` / ``2.create_holdout.py`` / ``3.latin_hypercube.py``.
Default input is ``Data/<STATION>_<YEAR>_train_<suffix>.txt`` (same naming as step 3).

**Overrides:** ``INPUT_DATA``, ``OUTPUT_DATA``, ``STATION``, ``YEAR``, ``LHS_N`` (env).

**Atmosphere:** By default sets ``LRT_SEASONAL_ATMOSPHERE=1`` before importing ``libRadtran`` so each
row uses month-based AFGL ``afglms``/``afglmw``. To use a fixed profile or disable seasonal selection,
set ``LRT_ATMOSPHERE`` / ``LRT_SEASONAL_ATMOSPHERE=0`` in the environment (see ``libRadtran.ClearskyConfig``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Month-based AFGL profile; must run before ``from libRadtran`` (``CLEARSKY_CONFIG`` reads env at import).
if "LRT_SEASONAL_ATMOSPHERE" not in os.environ:
    os.environ["LRT_SEASONAL_ATMOSPHERE"] = "1"

import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from libRadtran import (
    LIBRADTRANDIR, CLEARSKY_CONFIG, process_row_oe
)

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIG — same STATION/YEAR/LHS_N as step 3 for default paths.
# =============================================================================
STATION = os.environ.get("STATION", "PAL")
YEAR = int(os.environ.get("YEAR", "2024"))
LHS_N = int(os.environ.get("LHS_N", "500"))
_n = LHS_N
if _n == 500:
    _k_suffix = "_0.5k"
elif _n >= 1000:
    _k_suffix = f"_{_n / 1000:g}k"
else:
    _k_suffix = f"_{_n}"
_DEFAULT_INPUT = PROJECT / "Data" / f"{STATION}_{YEAR}_train{_k_suffix}.txt"
INPUT_DATA = Path(os.environ.get("INPUT_DATA", str(_DEFAULT_INPUT)))
_out_name = INPUT_DATA.name.replace("train", "train_oe") if "train" in INPUT_DATA.name else f"{INPUT_DATA.stem}_oe.txt"
OUTPUT_DATA = Path(os.environ.get("OUTPUT_DATA", str(PROJECT / "Data" / _out_name)))
# =============================================================================

# --- Execution Logic ---
if not INPUT_DATA.is_file():
    print(f"ERROR: Missing input file: {INPUT_DATA}")
    sys.exit(1)

if os.environ.get("INPUT_DATA"):
    print(f"Loading dataset: {INPUT_DATA.name}")
else:
    print(f"Loading dataset: {INPUT_DATA.name}  (STATION={STATION}, YEAR={YEAR}, LHS_N={LHS_N})")
df = pd.read_csv(INPUT_DATA, sep="\t", comment="#", parse_dates=["time_utc"])
df = df.set_index("time_utc").sort_index()

print(f"Starting retrieval for {len(df)} rows using libRadtran...")

_oe_fn = lambda r: process_row_oe(r, LIBRADTRANDIR, CLEARSKY_CONFIG)
if _tqdm is not None:
    _tqdm.pandas(desc="OE Beta + Alpha", leave=True)
    results = df.progress_apply(_oe_fn, axis=1)
else:
    results = df.apply(_oe_fn, axis=1)

# Keep every column from the input table; overwrite overlapping MERRA/flux fields with
# retrieval outputs and append new columns (e.g. beta_oe, alpha_oe, ghi_oe).
_original = list(df.columns)
out = df.copy()
for col in results.columns:
    out[col] = results[col]
_extra = [c for c in results.columns if c not in df.columns]
write_cols = ["time_utc"] + _original + _extra
out.reset_index()[write_cols].to_csv(
    OUTPUT_DATA, sep="\t", index=False, float_format="%.8f"
)

print(f"Successfully wrote: {OUTPUT_DATA.name}")
