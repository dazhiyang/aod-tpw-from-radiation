"""
6.evaluation: Final validation of TabPFN predictions via forward model parity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from libRadtran import (
    LIBRADTRANDIR, CLEARSKY_CONFIG, run_clearsky, forward_merra_explicit
)

PROJECT = Path(__file__).resolve().parent.parent

# --- Suffix Handling ---
K_SUFFIX = os.environ.get("K_SUFFIX", "_0.5k")
MODE = os.environ.get("MODE", "ls").lower()
PRED_IN = PROJECT / "Data" / f"pred_{MODE}{K_SUFFIX}.txt"
VAL_OUT = PROJECT / "Data" / f"test_{MODE}{K_SUFFIX}.txt"

# --- Execution Logic ---
if not PRED_IN.is_file():
    print(f"ERROR: Missing predictions: {PRED_IN}")
    sys.exit(1)

print(f"Loading predictions: {PRED_IN.name}")
df = pd.read_csv(PRED_IN, sep="\t")

def run_val(row):
    """Run forward models for both MERRA-2 (prior) and TabPFN (predicted) parameters."""
    # 1. MERRA-2 Forward
    merra_sim = forward_merra_explicit(row, LIBRADTRANDIR, CLEARSKY_CONFIG)
    
    # 2. Predicted Forward
    sim_out = run_clearsky(
        row, 
        LIBRADTRANDIR, 
        CLEARSKY_CONFIG, 
        angstrom_beta=row[f"beta_pred_{MODE}"], 
        w=row[f"w_pred_{MODE}"]
    )
    
    return pd.Series({
        "ghi_merra": merra_sim.get("ghi_sim", np.nan),
        "bni_merra": merra_sim.get("bni_sim", np.nan),
        "dhi_merra": merra_sim.get("dhi_sim", np.nan),
        f"ghi_{MODE}": sim_out.get("ghi_sim", np.nan),
        f"bni_{MODE}": sim_out.get("bni_sim", np.nan),
        f"dhi_{MODE}": sim_out.get("dhi_sim", np.nan)
    })

# Identify columns to add
cols = ["ghi_merra", "bni_merra", "dhi_merra", f"ghi_{MODE}", f"bni_{MODE}", f"dhi_{MODE}"]

# Define output path early to check for existence
if VAL_OUT.is_file():
    print(f"Loading existing validation results to resume: {VAL_OUT.name}")
    existing = pd.read_csv(VAL_OUT, sep="\t")
    # For simplicity, we just check if the columns exist and merge if possible
    # But if we are running the whole thing, we just overwrite later
    # The user said "do not overwrite" - likely meaning don't re-run points already in VAL_OUT
    for c in cols:
        if c not in df.columns and c in existing.columns:
            df[c] = existing[c]

for col in cols:
    if col not in df.columns:
        df[col] = np.nan

# We only process rows where simulation is missing (checked by ghi_{MODE} as proxy)
mask = df[f"ghi_{MODE}"].isna() | df["ghi_merra"].isna()
sub = df[mask].copy()

if len(sub) == 0:
    print("All rows already processed. Nothing to do.")
    sys.exit(0)

print(f"Running validation forward models for {len(sub)} rows...")

tqdm.pandas(desc="Validation Forward Models")
val_results = sub.progress_apply(run_val, axis=1)

# Update the main dataframe
df.loc[mask, cols] = val_results

print(f"Saving validation results to {VAL_OUT.name}...")
df.to_csv(VAL_OUT, sep="\t", index=False, float_format="%.8f")

print("Evaluation complete.")

