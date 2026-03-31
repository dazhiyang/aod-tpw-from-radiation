"""
5.tabpfn: Trains and runs TabPFN to predict Ångström ``beta`` and ``alpha`` (same state as steps 4a/4b).

**Training labels** come from step **4a** (``MODE=ls``) or **4b** (``MODE=oe``): default input is
``Data/<STATION>_<YEAR>_train_<MODE><suffix>.txt`` (same naming as ``4a`` / ``4b`` outputs from
``3.latin_hypercube`` LHS). **Test pool** defaults to ``Data/<STATION>_<YEAR>_testpool.txt`` (step 2).

We always compute ``ghi_trans``, ``bni_trans``, ``dhi_trans`` (measured / REST2 clear-sky) so they
are present in memory and written to the prediction file; TabPFN uses ``FEATURES`` below (measured
flux + zenith + MERRA). To use transmittance-only inputs, set
``FEATURES = [*_TRANS, "zenith", ...]``.

**Overrides:** ``TRAIN_IN``, ``TEST_POOL``, ``PRED_OUT``, ``STATION``, ``YEAR``, ``LHS_N``, ``MODE``,
``N_TEST`` (env). For a legacy flat name like ``Data/train_ls_0.5k.txt``, set ``TRAIN_IN`` explicitly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tabpfn import TabPFNRegressor

from libRadtran import ALPHA_MAX, ALPHA_MIN, BETA_MAX, BETA_MIN

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIG — align with steps 2–4 (STATION, YEAR, LHS_N); MODE = ls (4a) or oe (4b).
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

N_TEST = int(os.environ.get("N_TEST", "5000"))

# Outputs of 4a / 4b: e.g. Data/PAL_2024_train_ls_0.5k.txt, PAL_2024_train_oe_0.5k.txt
_DEFAULT_TRAIN = PROJECT / "Data" / f"{STATION}_{YEAR}_train_{MODE}{_k_suffix}.txt"
TRAIN_IN = Path(os.environ.get("TRAIN_IN", str(_DEFAULT_TRAIN)))

_DEFAULT_TEST = PROJECT / "Data" / f"{STATION}_{YEAR}_testpool.txt"
TEST_POOL = Path(os.environ.get("TEST_POOL", str(_DEFAULT_TEST)))

_DEFAULT_PRED = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_{MODE}{_k_suffix}.txt"
PRED_OUT = Path(os.environ.get("PRED_OUT", str(_DEFAULT_PRED)))
# =============================================================================

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
TARGETS = [f"beta_{MODE}", f"alpha_{MODE}"]


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

if os.environ.get("TRAIN_IN"):
    print(f"Loading training data: {TRAIN_IN}")
else:
    print(
        f"Loading training data: {TRAIN_IN.name}  "
        f"(STATION={STATION}, YEAR={YEAR}, MODE={MODE}, LHS_N={LHS_N})"
    )
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

# Save predictions (clip to forward-model bounds; matches libRadtran retrieval)
test_df[f"beta_pred_{MODE}"] = np.clip(
    all_preds[f"beta_{MODE}"], BETA_MIN, BETA_MAX,
)
test_df[f"alpha_pred_{MODE}"] = np.clip(
    all_preds[f"alpha_{MODE}"], ALPHA_MIN, ALPHA_MAX,
)

test_df.to_csv(PRED_OUT, sep="\t", index=False)
print(f"Successfully saved predictions to {PRED_OUT.name}")
