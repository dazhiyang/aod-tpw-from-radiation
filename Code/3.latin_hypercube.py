"""
3.latin_hypercube: Generates optimized training samples from the trainpool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import qmc

PROJECT = Path(__file__).resolve().parent.parent

# Input: standardized training pool from 2.create_holdout.py
INPUT_TXT = Path(os.environ.get("LHS_INPUT", str(PROJECT / "Data" / "trainpool.txt")))

# Sampling count (N) and filename suffix (e.g., _0.5k, _2k)
LHS_N = int(os.environ.get("LHS_N", "500"))
k_suffix = f"_{LHS_N/1000:g}k" if LHS_N >= 1000 else f"_{LHS_N/1000:.13g}k".replace("0.", "0.5")[:4]

# Ensure precise _0.5k naming logic
if LHS_N == 500: k_suffix = "_0.5k"

TRAIN_TXT = Path(os.environ.get("LHS_TRAIN", str(PROJECT / "Data" / f"train{k_suffix}.txt")))
SEED = int(os.environ.get("LHS_SEED", "42"))
ZENITH_MAX = float(os.environ.get("LHS_ZENITH_MAX", "87"))

# LHS stratification dimensions (measured irradiance, not REST2 clear-sky).
FEATURES = [
    "ghi", "bni", "dhi",
    "zenith",
    "merra_ALPHA", "merra_ALBEDO", "merra_TQV",
    "merra_TO3", "merra_PS", "merra_BETA",
]

# --- Execution Logic ---
if not INPUT_TXT.is_file():
    print(f"ERROR: Missing input table: {INPUT_TXT}")
    sys.exit(1)

print(f"Loading master pool: {INPUT_TXT.name}...")
df = pd.read_csv(INPUT_TXT, sep="\t", comment="#", parse_dates=["time_utc"], index_col="time_utc")
df = df.sort_index()

# Column presence check
for c in FEATURES + ["zenith"]:
    if c not in df.columns:
        raise SystemExit(f"Missing column {c!r} in {INPUT_TXT}")

day = df["zenith"].astype(float) <= ZENITH_MAX
clear = pd.Series(True, index=df.index) # Already pre-filtered by 2.create_holdout.py

pool_all = df.loc[day & clear].dropna(subset=FEATURES)
pool = pool_all[FEATURES]
print(f"Pool: {len(pool)} clear-sky daytime rows (LHS uses measured flux + MERRA + zenith)")

# Latin Hypercube Sampling (Over-sample to guarantee uniqueness)
print(f"Generating {LHS_N*2} LHS candidates to ensure {LHS_N} unique physical rows...")
sampler = qmc.LatinHypercube(d=len(FEATURES), seed=SEED)
sample = sampler.random(n=LHS_N * 2)

# Scale LHS to pool data ranges (min/max)
l_bounds = pool.min().values
u_bounds = pool.max().values
lhs_scaled = qmc.scale(sample, l_bounds, u_bounds)

# Nearest Neighbor mapping back to pool
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=100).fit(pool.values)  
distances, indices = nn.kneighbors(lhs_scaled)

# Collect unique indices
unique_idx = []
used = set()
for neighborhood in indices:
    if len(unique_idx) >= LHS_N:
        break
    for idx in neighborhood:
        if idx not in used:
            unique_idx.append(idx)
            used.add(idx)
            break

if len(unique_idx) < LHS_N:
    print(f"WARNING: Only found {len(unique_idx)} unique rows. Target was {LHS_N}.")
else:
    print(f"Successfully found exactly {LHS_N} unique rows.")

train = pool_all.iloc[unique_idx].copy()
train.index.name = "time_utc"

# Save
TRAIN_TXT.parent.mkdir(parents=True, exist_ok=True)
common_meta = f"# Input: {INPUT_TXT.name} | LHS_N={LHS_N} | Seed={SEED}\n"
with open(TRAIN_TXT, "w", encoding="ascii") as f:
    f.write(common_meta + f"# train: all {len(train)} LHS samples\n")
train.to_csv(TRAIN_TXT, mode="a", sep="\t", float_format="%.12g")

print(f"Generated {len(train)} samples -> {TRAIN_TXT.name}")
