"""
6b.evaluation_aeronet: Forward-model validation using **AERONET** aerosol at the photometer.

For each row, runs **MERRA-2 explicit** forward (prior) and a second forward using Ångström
parameters from AERONET: ``aeronet_aod550`` (:math:`\\tau_{550}`) and ``aeronet_alpha``. **Precipitable
water** is **MERRA-2 only**: ``libRadtran.merra_explicit_physics`` → ``w`` (mm) from ``merra_TQV``
and ``ClearskyConfig.tqv_bsrn_to_mm_pw``, passed explicitly to ``run_clearsky`` (same as retrieval).

Ångström :math:`\\beta` at ``ANGSTROM_BETA_REF_UM`` (default **1.0** µm) follows the same inverse
relation as ``7.aod_w_distributions.py``:

    \\tau_{550} = \\beta\\,(0.55/\\lambda_{\\mathrm{ref}})^{-\\alpha}
    \\quad\\Rightarrow\\quad
    \\beta = \\tau_{550}\\,(0.55/\\lambda_{\\mathrm{ref}})^{\\alpha}

**Input** — any table with BSRN/MERRA + AERONET columns (e.g. TabPFN ``pred_*.txt`` or test pool).
Default matches ``5`` / ``6`` naming:

- ``PRED_IN`` — ``Data/<STATION>_<YEAR>_pred_ls<suffix>.txt`` (override if you use another base).
- ``VAL_OUT`` — ``Data/<STATION>_<YEAR>_test_aeronet<suffix>.txt``

**Overrides:** ``PRED_IN``, ``VAL_OUT``, ``STATION``, ``YEAR``, ``LHS_N``, ``ANGSTROM_BETA_REF_UM``.

Requires: pandas, numpy, tqdm, libRadtran.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from libRadtran import (
    CLEARSKY_CONFIG,
    LIBRADTRANDIR,
    forward_merra_explicit,
    merra_explicit_physics,
    run_clearsky,
)

PROJECT = Path(__file__).resolve().parent.parent

LAM550_UM = 0.55

# =============================================================================
# CONFIG — paths align with ``5`` / ``6``; no TabPFN MODE (AERONET-only forward).
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

LAMBDA_REF_UM = float(os.environ.get("ANGSTROM_BETA_REF_UM", "1.0"))

_DEFAULT_PRED = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_ls{_k_suffix}.txt"
_DEFAULT_VAL = PROJECT / "Data" / f"{STATION}_{YEAR}_test_aeronet{_k_suffix}.txt"
PRED_IN = Path(os.environ.get("PRED_IN", str(_DEFAULT_PRED)))
VAL_OUT = Path(os.environ.get("VAL_OUT", str(_DEFAULT_VAL)))
# =============================================================================


def tau550_to_beta_angstrom(tau550: float, alpha: float, lambda_ref_um: float) -> float:
    """Inverse of :math:`\\tau_{550} = \\beta\\,(\\lambda_{550}/\\lambda_{\\mathrm{ref}})^{-\\alpha}`."""
    return tau550 * (LAM550_UM / lambda_ref_um) ** alpha


def run_val(row: pd.Series) -> pd.Series:
    """MERRA prior forward + AERONET-parameter forward (MERRA-2 :math:`w` only)."""
    merra_sim = forward_merra_explicit(row, LIBRADTRANDIR, CLEARSKY_CONFIG)
    tau = row.get("aeronet_aod550")
    alpha_ae = row.get("aeronet_alpha")
    if pd.isna(tau) or pd.isna(alpha_ae) or not np.isfinite(float(tau)) or not np.isfinite(
        float(alpha_ae)
    ):
        return pd.Series(
            {
                "ghi_merra": merra_sim.get("ghi_sim", np.nan),
                "bni_merra": merra_sim.get("bni_sim", np.nan),
                "dhi_merra": merra_sim.get("dhi_sim", np.nan),
                "ghi_aeronet": np.nan,
                "bni_aeronet": np.nan,
                "dhi_aeronet": np.nan,
            }
        )
    _a_m, _o3_du, _beta_m, w_merra_mm = merra_explicit_physics(row, CLEARSKY_CONFIG)
    beta_ae = tau550_to_beta_angstrom(float(tau), float(alpha_ae), LAMBDA_REF_UM)
    sim_out = run_clearsky(
        row,
        LIBRADTRANDIR,
        CLEARSKY_CONFIG,
        angstrom_alpha=float(alpha_ae),
        angstrom_beta=beta_ae,
        w=w_merra_mm,
    )
    return pd.Series(
        {
            "ghi_merra": merra_sim.get("ghi_sim", np.nan),
            "bni_merra": merra_sim.get("bni_sim", np.nan),
            "dhi_merra": merra_sim.get("dhi_sim", np.nan),
            "ghi_aeronet": sim_out.get("ghi_sim", np.nan),
            "bni_aeronet": sim_out.get("bni_sim", np.nan),
            "dhi_aeronet": sim_out.get("dhi_sim", np.nan),
        }
    )


if not PRED_IN.is_file():
    print(f"ERROR: Missing input table: {PRED_IN}")
    sys.exit(1)

if os.environ.get("PRED_IN"):
    print(f"Loading input: {PRED_IN}")
else:
    print(
        f"Loading input: {PRED_IN.name}  "
        f"(STATION={STATION}, YEAR={YEAR}, LHS_N={LHS_N})"
    )
df = pd.read_csv(PRED_IN, sep="\t")

need_cols = ("aeronet_aod550", "aeronet_alpha", "merra_TQV")
for c in need_cols:
    if c not in df.columns:
        print(f"ERROR: Missing column {c!r} in {PRED_IN}", file=sys.stderr)
        sys.exit(1)

cols = [
    "ghi_merra", "bni_merra", "dhi_merra",
    "ghi_aeronet", "bni_aeronet", "dhi_aeronet",
]

if VAL_OUT.is_file():
    print(f"Loading existing validation results to resume: {VAL_OUT.name}")
    existing = pd.read_csv(VAL_OUT, sep="\t")
    for c in cols:
        if c not in df.columns and c in existing.columns:
            df[c] = existing[c]

for col in cols:
    if col not in df.columns:
        df[col] = np.nan

mask = df["ghi_aeronet"].isna() | df["ghi_merra"].isna()
sub = df[mask].copy()

if len(sub) == 0:
    print("All rows already processed. Nothing to do.")
    sys.exit(0)

print(
    f"Running MERRA + AERONET forward models for {len(sub)} rows "
    f"(ANGSTROM_BETA_REF_UM={LAMBDA_REF_UM})..."
)

tqdm.pandas(desc="AERONET forward (uvspec)")
val_results = sub.progress_apply(run_val, axis=1)

df.loc[mask, cols] = val_results

print(f"Saving validation results to {VAL_OUT.name}...")
VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(VAL_OUT, sep="\t", index=False, float_format="%.8f")

print("AERONET evaluation complete.")
