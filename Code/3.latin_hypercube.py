"""3.latin_hypercube: LHS sample (default 500) from station trainpool (PAL/TAT/…).

Edit **CONFIG** to match ``1.arrange.py`` / ``2.create_holdout.py``. Reads
``Data/<STATION>_<year>_trainpool.txt``, writes ``Data/<STATION>_<year>_train_0.5k.txt`` when
``LHS_N == 500``.

**QIQ / generic paths:** ``Code/old/3.latin_hypercube.py`` (``trainpool.txt`` → ``train_0.5k.txt``).

Step **4a** default is ``Data/train_0.5k.txt``; set ``INPUT_DATA=Data/PAL_2024_train_0.5k.txt`` (or your
pool) when using this naming scheme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIG — same STATION/YEAR as steps 1–2.
# =============================================================================
STATION = "PAL"
YEAR = 2024
INPUT_TXT = PROJECT / "Data" / f"{STATION}_{YEAR}_trainpool.txt"
LHS_N = 100
SEED = 42
ZENITH_MAX = 87.0

_k_suffix = "_0.5k" if LHS_N == 500 else f"_{LHS_N / 1000:g}k" if LHS_N >= 1000 else f"_{LHS_N}"
TRAIN_TXT = PROJECT / "Data" / f"{STATION}_{YEAR}_train{_k_suffix}.txt"
# =============================================================================

# LHS stratification dimensions (measured irradiance + MERRA + zenith; not REST2 clears).
FEATURES = [
    "ghi",
    "bni",
    "dhi",
    "zenith",
    "merra_ALPHA",
    "merra_ALBEDO",
    "merra_TQV",
    "merra_TO3",
    "merra_PS",
    "merra_BETA",
]

if not INPUT_TXT.is_file():
    print(f"ERROR: Missing input table: {INPUT_TXT}")
    sys.exit(1)

print(f"{STATION} {YEAR}: loading {INPUT_TXT.name}...")
df = pd.read_csv(INPUT_TXT, sep="\t", comment="#", parse_dates=["time_utc"], index_col="time_utc")
df = df.sort_index()

for c in FEATURES:
    if c not in df.columns:
        raise SystemExit(f"Missing column {c!r} in {INPUT_TXT}")

day = df["zenith"].astype(float) <= ZENITH_MAX
pool_all = df.loc[day].dropna(subset=FEATURES)
pool = pool_all[FEATURES]
print(f"Pool: {len(pool)} rows (LHS stratification on {len(FEATURES)} features)")

if len(pool) < LHS_N:
    print(f"ERROR: Pool size {len(pool)} < LHS_N={LHS_N}")
    sys.exit(1)

print(f"Generating {LHS_N * 2} LHS candidates to ensure {LHS_N} unique physical rows...")
sampler = qmc.LatinHypercube(d=len(FEATURES), seed=SEED)
sample = sampler.random(n=LHS_N * 2)

l_bounds = pool.min().values
u_bounds = pool.max().values
lhs_scaled = qmc.scale(sample, l_bounds, u_bounds)

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=min(100, len(pool))).fit(pool.values)
_, indices = nn.kneighbors(lhs_scaled)

unique_idx: list[int] = []
used: set[int] = set()
for neighborhood in indices:
    if len(unique_idx) >= LHS_N:
        break
    for idx in neighborhood:
        if idx not in used:
            unique_idx.append(int(idx))
            used.add(int(idx))
            break

if len(unique_idx) < LHS_N:
    print(f"WARNING: Only found {len(unique_idx)} unique rows. Target was {LHS_N}.")
else:
    print(f"Successfully found exactly {LHS_N} unique rows.")

train = pool_all.iloc[unique_idx].copy()
train.index.name = "time_utc"

TRAIN_TXT.parent.mkdir(parents=True, exist_ok=True)
common_meta = (
    f"# Station={STATION} Year={YEAR} | Input: {INPUT_TXT.name} | LHS_N={LHS_N} | Seed={SEED}\n"
)
with open(TRAIN_TXT, "w", encoding="ascii") as f:
    f.write(common_meta + f"# train: {len(train)} LHS samples (all columns from input)\n")
train.to_csv(TRAIN_TXT, mode="a", sep="\t", float_format="%.12g", na_rep="")

print(f"Generated {len(train)} samples -> {TRAIN_TXT}")
