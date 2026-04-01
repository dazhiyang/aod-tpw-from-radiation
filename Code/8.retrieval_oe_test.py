"""
8.retrieval_oe_test: libRadtran optimal-estimation retrieval on the **test pool** (beta, alpha).

Same OE stack as ``4b.retrieval_oe.py`` (``process_row_oe``), but defaults to the holdout table from
step **2** — ``Data/<STATION>_<YEAR>_testpool.txt`` — and writes ``test_oe.txt`` (no LHS row-count
suffix; the holdout is not an LHS subsample).

**Overrides:** ``INPUT_DATA`` or ``TEST_POOL`` (same env as ``5.tabpfn.py``), ``OUTPUT_DATA``,
``STATION``, ``YEAR``.

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
# CONFIG — STATION/YEAR match ``1.arrange`` / ``2.create_holdout`` / ``5.tabpfn``.
# =============================================================================
STATION = os.environ.get("STATION", "PAL")
YEAR = int(os.environ.get("YEAR", "2024"))
_DEFAULT_TESTPOOL = PROJECT / "Data" / f"{STATION}_{YEAR}_testpool.txt"
_DEFAULT_INPUT = _DEFAULT_TESTPOOL
INPUT_DATA = Path(
    os.environ.get("INPUT_DATA", os.environ.get("TEST_POOL", str(_DEFAULT_INPUT)))
)
_DEFAULT_OUTPUT = PROJECT / "Data" / f"{STATION}_{YEAR}_test_oe.txt"
OUTPUT_DATA = Path(os.environ.get("OUTPUT_DATA", str(_DEFAULT_OUTPUT)))
# =============================================================================

# --- Execution Logic ---
if not INPUT_DATA.is_file():
    print(f"ERROR: Missing input file: {INPUT_DATA}")
    sys.exit(1)

if os.environ.get("INPUT_DATA") or os.environ.get("TEST_POOL"):
    print(f"Loading dataset: {INPUT_DATA.name}")
else:
    print(f"Loading dataset: {INPUT_DATA.name}  (STATION={STATION}, YEAR={YEAR})")
df = pd.read_csv(INPUT_DATA, sep="\t", comment="#", parse_dates=["time_utc"])
df = df.set_index("time_utc").sort_index()

print(f"Starting OE retrieval for {len(df)} test rows using libRadtran...")

_oe_fn = lambda r: process_row_oe(r, LIBRADTRANDIR, CLEARSKY_CONFIG)
if _tqdm is not None:
    _tqdm.pandas(desc="OE Beta + Alpha (test)", leave=True)
    results = df.progress_apply(_oe_fn, axis=1)
else:
    results = df.apply(_oe_fn, axis=1)

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
