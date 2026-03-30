"""
5.tabpfn: Trains and runs the TabPFN machine-learning model for AOD/TPW retrieval.

We always compute ``ghi_trans``, ``bni_trans``, ``dhi_trans`` (measured / REST2 clear-sky) so they
are present in memory and written to ``pred_*.txt``; TabPFN currently uses ``FEATURES`` below
(measured flux + zenith + MERRA). To switch back to transmittance-only inputs, set
``FEATURES = [*_TRANS, "zenith", ...]`` (and optionally drop ``*_MEAS`` from that list).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tabpfn import TabPFNRegressor

from libRadtran import BETA_MAX, BETA_MIN, W_MAX, W_MIN

PROJECT = Path(__file__).resolve().parent.parent

# --- Suffix & Sampling Handling ---
K_SUFFIX = os.environ.get("K_SUFFIX", "_0.5k")
N_TEST = int(os.environ.get("N_TEST", "5000"))

MODE = os.environ.get("MODE", "ls").lower()

# Input: The output from retrieval step
TRAIN_IN = PROJECT / "Data" / f"train_{MODE}{K_SUFFIX}.txt"
TEST_POOL = PROJECT / "Data" / "testpool.txt"
PRED_OUT = PROJECT / "Data" / f"pred_{MODE}{K_SUFFIX}.txt"

_CLEAR = ("ghi_clear", "bni_clear", "dhi_clear")
_TRANS = ("ghi_trans", "bni_trans", "dhi_trans")
_MEAS = ("ghi", "bni", "dhi")

# TabPFN input columns (change to ``*_TRANS`` instead of ``*_MEAS`` to use transmittance as features).
FEATURES = [
    *_MEAS,
    "zenith",
    "merra_ALPHA", "merra_ALBEDO", "merra_TQV",
    "merra_TO3", "merra_PS",
]
TARGETS = [f"beta_{MODE}", f"w_{MODE}"]


def _add_transmittance(df: pd.DataFrame) -> pd.DataFrame:
    """Transmittance proxy: measured / REST2 clear-sky (e.g. ``ghi_trans`` = ghi/ghi_clear). Zenith ≤ 87°; no epsilon on the denominator."""
    out = df.copy()
    out["ghi_trans"] = out["ghi"] / out["ghi_clear"]
    out["bni_trans"] = out["bni"] / out["bni_clear"]
    out["dhi_trans"] = out["dhi"] / out["dhi_clear"]
    return out


# --- Execution Logic ---
if not TRAIN_IN.is_file():
    print(f"ERROR: Missing training data: {TRAIN_IN}")
    sys.exit(1)

print(f"Loading training data: {TRAIN_IN.name}")
train_df = pd.read_csv(TRAIN_IN, sep="\t")
for c in _CLEAR:
    if c not in train_df.columns:
        print(f"ERROR: Missing column {c!r} in {TRAIN_IN}", file=sys.stderr)
        sys.exit(1)
train_df = _add_transmittance(train_df)

print(f"Loading test pool: {TEST_POOL.name}")
test_df = pd.read_csv(TEST_POOL, sep="\t", comment="#")
for c in _CLEAR:
    if c not in test_df.columns:
        print(f"ERROR: Missing column {c!r} in {TEST_POOL}", file=sys.stderr)
        sys.exit(1)
test_df = _add_transmittance(test_df)
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

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
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = TabPFNRegressor(device=device)

# Batch inference: We loop over targets since TabPFN is single-output
all_preds = {}
for target in TARGETS:
    print(f"Training and predicting for target: {target}...")
    y_train_single = train_df[target]
    model.fit(X_train, y_train_single)
    
    batch_size = 512
    preds_list = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(X_test), batch_size), desc=f"Inference [{target}]"):
        batch_X = X_test.iloc[i : i + batch_size]
        preds_list.append(model.predict(batch_X))
    all_preds[target] = np.concatenate(preds_list)

# Save predictions (clip to forward-model bounds; matches libRadtran.retrieve_one_row_ls)
test_df[f"beta_pred_{MODE}"] = np.clip(
    all_preds[f"beta_{MODE}"], BETA_MIN, BETA_MAX,
)
test_df[f"w_pred_{MODE}"] = np.clip(
    all_preds[f"w_{MODE}"], W_MIN, W_MAX,
)

test_df.to_csv(PRED_OUT, sep="\t", index=False)
print(f"Successfully saved predictions to {PRED_OUT.name}")
