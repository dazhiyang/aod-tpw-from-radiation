"""
6.evaluation: Final validation of TabPFN predictions via forward model parity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from libRadtran import (
    LIBRADTRANDIR, CLEARSKY_CONFIG, build_uvspec_input, run_clearsky
)

PROJECT = Path(__file__).resolve().parent.parent

# --- Suffix Handling ---
K_SUFFIX = os.environ.get("K_SUFFIX", "_0.5k")
PRED_IN = PROJECT / "Data" / f"testpool_tabpfn{K_SUFFIX}.txt"
PLOT_OUT = PROJECT / "tex" / "figures" / f"evaluation_scatter_testpool{K_SUFFIX}.png"

def main():
    if not PRED_IN.is_file():
        print(f"ERROR: Missing predictions: {PRED_IN}")
        return

    print(f"Loading predictions: {PRED_IN.name}")
    df = pd.read_csv(PRED_IN, sep="\t")
    
    # We evaluate everything provided in the prediction file
    sub = df.copy()
    print(f"Running validation forward models for {len(sub)} rows...")
    
    # Function to run forward model with predicted values
    def run_val(row):
        row_pred = row.copy()
        row_pred["beta_retrieved"] = row["beta_pred"]
        row_pred["h2o_mm_retrieved"] = row["h2o_mm_pred"]
        fwd = run_clearsky(build_uvspec_input(row_pred, CLEARSKY_CONFIG), LIBRADTRANDIR)
        return pd.Series({
            "ghi_pred": fwd.get("ghi", np.nan),
            "dni_pred": fwd.get("bni", np.nan),
            "dhi_pred": fwd.get("dhi", np.nan)
        })

    from tqdm import tqdm
    tqdm.pandas(desc="Validation Forward Models")
    val_fluxes = sub.progress_apply(run_val, axis=1)
    
    final = pd.concat([sub, val_fluxes], axis=1)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # GHI Parity
    ax = axes[0]
    ax.scatter(final["ghi"], final["ghi_pred"], s=5, alpha=0.5, color="blue")
    ax.plot([0, 1100], [0, 1100], "r--")
    ax.set_title(f"GHI Parity (TabPFN {K_SUFFIX})")
    ax.set_xlabel("Measured GHI [W/m²]")
    ax.set_ylabel("Predicted GHI [W/m²]")
    
    # DNI Parity
    ax = axes[1]
    ax.scatter(final["bni"], final["dni_pred"], s=5, alpha=0.5, color="green")
    ax.plot([0, 1100], [0, 1100], "r--")
    ax.set_title(f"DNI Parity (TabPFN {K_SUFFIX})")
    ax.set_xlabel("Measured BNI [W/m²]")
    ax.set_ylabel("Predicted DNI [W/m²]")
    
    plt.tight_layout()
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=200)
    print(f"Saved parity plot: {PLOT_OUT}")

if __name__ == "__main__":
    main()
