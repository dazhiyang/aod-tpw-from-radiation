"""
5.tabpfn: Trains and runs the TabPFN machine-learning model for AOD/TPW retrieval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tabpfn import TabPFNRegressor

PROJECT = Path(__file__).resolve().parent.parent

# --- Suffix & Sampling Handling ---
K_SUFFIX = os.environ.get("K_SUFFIX", "_0.5k")
N_TEST = int(os.environ.get("N_TEST", "5000"))

# Input: The output from 4.retrieval.py
TRAIN_LS = PROJECT / "Data" / f"train_ls{K_SUFFIX}.txt"
TEST_POOL = PROJECT / "Data" / "testpool.txt"
PRED_OUT = PROJECT / "Data" / f"testpool_tabpfn{K_SUFFIX}.txt"

FEATURES = [
    "ghi", "bni", "dhi", "zenith",
    "merra_ALPHA", "merra_ALBEDO", "merra_TQV", 
    "merra_TO3", "merra_PS"
]
TARGETS = ["beta_retrieved", "h2o_mm_retrieved"]

def main():
    if not TRAIN_LS.is_file():
        print(f"ERROR: Missing training data: {TRAIN_LS}")
        sys.exit(1)
    
    print(f"Loading training data: {TRAIN_LS.name}")
    train_df = pd.read_csv(TRAIN_LS, sep="\t")
    
    print(f"Loading test pool: {TEST_POOL.name}")
    test_df = pd.read_csv(TEST_POOL, sep="\t", comment="#")

    # Sample for standardized evaluation (configurable via N_TEST)
    if len(test_df) > N_TEST:
        print(f"Sub-sampling {N_TEST} rows from test pool for standardized evaluation...")
        test_df = test_df.sample(n=N_TEST, random_state=42).copy()

    # Filter for valid rows
    train_df = train_df.dropna(subset=FEATURES + TARGETS)
    test_df = test_df.dropna(subset=FEATURES)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGETS]
    X_test = test_df[FEATURES]

    print(f"Training on {len(X_train)} samples...")
    print(f"Predicting for {len(X_test)} samples...")

    # Initialize TabPFN
    # TabPFN is a foundation model; 'fitting' is basically just providing the context.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # We use a smaller subset or batching if the test pool is too large
    # TabPFN has a max context size (usually 1024 or 2048 samples)
    # For large-scale inference, we iterate.
    
    model = TabPFNRegressor(device=device)
    model.fit(X_train, y_train)
    
    # Batch inference
    batch_size = 512
    preds_list = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(X_test), batch_size), desc="TabPFN Inference"):
        batch_X = X_test.iloc[i : i + batch_size]
        preds_list.append(model.predict(batch_X))
    
    preds = np.vstack(preds_list)
    
    # Save predictions
    test_df["beta_pred"] = preds[:, 0]
    test_df["h2o_mm_pred"] = preds[:, 1]
    
    test_df.to_csv(PRED_OUT, sep="\t", index=False)
    print(f"Successfully saved predictions to {PRED_OUT.name}")

if __name__ == "__main__":
    main()
