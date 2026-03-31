"""
6.evaluation: Validate TabPFN predictions by forward-model parity (libRadtran ``uvspec``).

For each row in the prediction file, runs **MERRA-2 explicit** forward and a second forward using
``beta_pred_<MODE>`` and ``w_pred_<MODE>`` from TabPFN. Writes the input table plus simulated fluxes.

**MODE** — ``ls`` (default, matches ``MODE=ls`` / 4a labels) or ``oe`` (matches ``MODE=oe`` / 4b).

Defaults align with ``5.tabpfn.py``:

- ``PRED_IN`` — ``Data/<STATION>_<YEAR>_pred_<MODE><suffix>.txt``
- ``VAL_OUT`` — ``Data/<STATION>_<YEAR>_test_<MODE><suffix>.txt`` (enriched predictions + sims)

**Overrides:** ``PRED_IN``, ``VAL_OUT``, ``STATION``, ``YEAR``, ``LHS_N``, ``MODE`` (env).

Example (OE): ``MODE=oe /opt/anaconda3/bin/python Code/6.evaluation.py``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from libRadtran import (
    LIBRADTRANDIR,
    CLEARSKY_CONFIG,
    forward_merra_explicit,
    run_clearsky,
)

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIG — same STATION/YEAR/LHS_N/MODE as ``5.tabpfn.py``.
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

MODE = os.environ.get("MODE", "ls").lower()
if MODE not in ("ls", "oe"):
    print(f"ERROR: MODE must be ls or oe, got {MODE!r}", file=sys.stderr)
    sys.exit(1)

_DEFAULT_PRED = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_{MODE}{_k_suffix}.txt"
_DEFAULT_VAL = PROJECT / "Data" / f"{STATION}_{YEAR}_test_{MODE}{_k_suffix}.txt"
PRED_IN = Path(os.environ.get("PRED_IN", str(_DEFAULT_PRED)))
VAL_OUT = Path(os.environ.get("VAL_OUT", str(_DEFAULT_VAL)))
# =============================================================================


def run_val(row: pd.Series) -> pd.Series:
    """MERRA forward + TabPFN-parameter forward for one row."""
    merra_sim = forward_merra_explicit(row, LIBRADTRANDIR, CLEARSKY_CONFIG)
    sim_out = run_clearsky(
        row,
        LIBRADTRANDIR,
        CLEARSKY_CONFIG,
        angstrom_beta=row[f"beta_pred_{MODE}"],
        w=row[f"w_pred_{MODE}"],
    )
    return pd.Series(
        {
            "ghi_merra": merra_sim.get("ghi_sim", np.nan),
            "bni_merra": merra_sim.get("bni_sim", np.nan),
            "dhi_merra": merra_sim.get("dhi_sim", np.nan),
            f"ghi_{MODE}": sim_out.get("ghi_sim", np.nan),
            f"bni_{MODE}": sim_out.get("bni_sim", np.nan),
            f"dhi_{MODE}": sim_out.get("dhi_sim", np.nan),
        }
    )


if not PRED_IN.is_file():
    print(f"ERROR: Missing predictions: {PRED_IN}")
    sys.exit(1)

if os.environ.get("PRED_IN"):
    print(f"Loading predictions: {PRED_IN}")
else:
    print(
        f"Loading predictions: {PRED_IN.name}  "
        f"(STATION={STATION}, YEAR={YEAR}, MODE={MODE}, LHS_N={LHS_N})"
    )
df = pd.read_csv(PRED_IN, sep="\t")

cols = ["ghi_merra", "bni_merra", "dhi_merra", f"ghi_{MODE}", f"bni_{MODE}", f"dhi_{MODE}"]

if VAL_OUT.is_file():
    print(f"Loading existing validation results to resume: {VAL_OUT.name}")
    existing = pd.read_csv(VAL_OUT, sep="\t")
    for c in cols:
        if c not in df.columns and c in existing.columns:
            df[c] = existing[c]

for col in cols:
    if col not in df.columns:
        df[col] = np.nan

mask = df[f"ghi_{MODE}"].isna() | df["ghi_merra"].isna()
sub = df[mask].copy()

if len(sub) == 0:
    print("All rows already processed. Nothing to do.")
    sys.exit(0)

print(f"Running validation forward models for {len(sub)} rows...")

tqdm.pandas(desc="Validation Forward Models")
val_results = sub.progress_apply(run_val, axis=1)

df.loc[mask, cols] = val_results

print(f"Saving validation results to {VAL_OUT.name}...")
df.to_csv(VAL_OUT, sep="\t", index=False, float_format="%.8f")

print("Evaluation complete.")
