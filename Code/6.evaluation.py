"""
6.evaluation: Forward-model validation — **MERRA-2 explicit**, **TabPFN (LS)**, **TabPFN (OE)**,
and **AERONET** aerosol (same ``uvspec`` stack as ``4a``/``4b``).

**Per row, one pass:** ``merra_explicit_physics`` runs **once**; then **four** ``uvspec`` clearsky
runs — MERRA explicit (MERRA β/α), TabPFN LS, TabPFN OE, AERONET — all sharing the same explicit
O₃ and H₂O (mm). MERRA and AERONET do **not** depend on LS vs OE predictions, so they are not
recomputed twice.

**Input**

- **Default:** load ``PRED_LS`` (full row: BSRN, MERRA, AERONET, ``beta_pred_ls``/``alpha_pred_ls``)
  and merge **only** ``beta_pred_oe``/``alpha_pred_oe`` from ``PRED_OE`` on ``time_utc`` — both
  prediction files carry the same scene/MERRA/AERONET; only TabPFN labels differ.
- **Single table:** ``PRED_IN`` = one file with all columns (skips merge).

**Output**

- ``VAL_OUT`` — default ``Data/<STATION>_<YEAR>_test_combined<suffix>.txt`` with columns
  ``ghi_merra``, ``*_ls``, ``*_oe``, ``*_aeronet`` (GHI/BNI/DHI each).

**Overrides:** ``PRED_IN``, ``PRED_LS``, ``PRED_OE``, ``VAL_OUT``, ``STATION``, ``YEAR``, ``LHS_N``,
``ANGSTROM_BETA_REF_UM``.

**Atmosphere:** ``LRT_SEASONAL_ATMOSPHERE=1`` by default before importing ``libRadtran`` (same as
``4a``/``4b``). Override with ``LRT_SEASONAL_ATMOSPHERE=0`` or ``LRT_ATMOSPHERE`` if needed.

Example: ``/opt/anaconda3/bin/python Code/6.evaluation.py``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Month-based AFGL profile; must run before ``from libRadtran`` (``CLEARSKY_CONFIG`` reads env at import).
if "LRT_SEASONAL_ATMOSPHERE" not in os.environ:
    os.environ["LRT_SEASONAL_ATMOSPHERE"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

from libRadtran import (
    LIBRADTRANDIR,
    CLEARSKY_CONFIG,
    merra_explicit_physics,
    run_clearsky,
)

PROJECT = Path(__file__).resolve().parent.parent

LAM550_UM = 0.55

# =============================================================================
# CONFIG
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

_DEFAULT_PRED_LS = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_ls{_k_suffix}.txt"
_DEFAULT_PRED_OE = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_oe{_k_suffix}.txt"
_DEFAULT_VAL = PROJECT / "Data" / f"{STATION}_{YEAR}_test_combined{_k_suffix}.txt"

PRED_IN = os.environ.get("PRED_IN", "").strip()
PRED_LS = Path(os.environ.get("PRED_LS", str(_DEFAULT_PRED_LS)))
PRED_OE = Path(os.environ.get("PRED_OE", str(_DEFAULT_PRED_OE)))
VAL_OUT = Path(os.environ.get("VAL_OUT", str(_DEFAULT_VAL)))

SIM_COLS = [
    "ghi_merra", "bni_merra", "dhi_merra",
    "ghi_ls", "bni_ls", "dhi_ls",
    "ghi_oe", "bni_oe", "dhi_oe",
    "ghi_aeronet", "bni_aeronet", "dhi_aeronet",
]
REQUIRED_BASE = (
    "beta_pred_ls", "alpha_pred_ls", "beta_pred_oe", "alpha_pred_oe",
    "aeronet_aod550", "aeronet_alpha", "merra_TQV",
)
# =============================================================================


def tau550_to_beta_angstrom(tau550: float, alpha: float, lambda_ref_um: float) -> float:
    """β at ref λ from τ₅₅₀ and Ångström α (same convention as ``old/7.aod_w_distributions.py``)."""
    return tau550 * (LAM550_UM / lambda_ref_um) ** alpha


def run_val(row: pd.Series) -> pd.Series:
    """Four clearsky runs; shared ``merra_explicit_physics`` (same as ``forward_merra_explicit``)."""
    alpha_m, o3_du_m, beta_m, w_m = merra_explicit_physics(row, CLEARSKY_CONFIG)

    merra_sim = run_clearsky(
        row,
        LIBRADTRANDIR,
        CLEARSKY_CONFIG,
        angstrom_alpha=alpha_m,
        angstrom_beta=beta_m,
        o3_du=o3_du_m,
        w=w_m,
        quiet=True,
    )
    sim_ls = run_clearsky(
        row,
        LIBRADTRANDIR,
        CLEARSKY_CONFIG,
        angstrom_beta=row["beta_pred_ls"],
        angstrom_alpha=row["alpha_pred_ls"],
        o3_du=o3_du_m,
        w=w_m,
        quiet=True,
    )
    sim_oe = run_clearsky(
        row,
        LIBRADTRANDIR,
        CLEARSKY_CONFIG,
        angstrom_beta=row["beta_pred_oe"],
        angstrom_alpha=row["alpha_pred_oe"],
        o3_du=o3_du_m,
        w=w_m,
        quiet=True,
    )

    tau = row.get("aeronet_aod550")
    alpha_ae = row.get("aeronet_alpha")
    if pd.isna(tau) or pd.isna(alpha_ae) or not np.isfinite(float(tau)) or not np.isfinite(
        float(alpha_ae)
    ):
        ghi_ae = bni_ae = dhi_ae = np.nan
    else:
        beta_ae = tau550_to_beta_angstrom(float(tau), float(alpha_ae), LAMBDA_REF_UM)
        sim_ae = run_clearsky(
            row,
            LIBRADTRANDIR,
            CLEARSKY_CONFIG,
            angstrom_alpha=float(alpha_ae),
            angstrom_beta=beta_ae,
            o3_du=o3_du_m,
            w=w_m,
            quiet=True,
        )
        ghi_ae = sim_ae.get("ghi_sim", np.nan)
        bni_ae = sim_ae.get("bni_sim", np.nan)
        dhi_ae = sim_ae.get("dhi_sim", np.nan)

    return pd.Series(
        {
            "ghi_merra": merra_sim.get("ghi_sim", np.nan),
            "bni_merra": merra_sim.get("bni_sim", np.nan),
            "dhi_merra": merra_sim.get("dhi_sim", np.nan),
            "ghi_ls": sim_ls.get("ghi_sim", np.nan),
            "bni_ls": sim_ls.get("bni_sim", np.nan),
            "dhi_ls": sim_ls.get("dhi_sim", np.nan),
            "ghi_oe": sim_oe.get("ghi_sim", np.nan),
            "bni_oe": sim_oe.get("bni_sim", np.nan),
            "dhi_oe": sim_oe.get("dhi_sim", np.nan),
            "ghi_aeronet": ghi_ae,
            "bni_aeronet": bni_ae,
            "dhi_aeronet": dhi_ae,
        }
    )


def _load_merged_frame() -> pd.DataFrame:
    if PRED_IN:
        path = Path(PRED_IN)
        if not path.is_file():
            print(f"ERROR: PRED_IN not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading single table: {path}")
        return pd.read_csv(path, sep="\t")

    if not PRED_LS.is_file():
        print(f"ERROR: Missing predictions (LS): {PRED_LS}", file=sys.stderr)
        sys.exit(1)
    if not PRED_OE.is_file():
        print(f"ERROR: Missing predictions (OE): {PRED_OE}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Merging TabPFN OE columns from {PRED_OE.name} onto {PRED_LS.name} (same MERRA/AERONET row)…"
    )
    pls = pd.read_csv(PRED_LS, sep="\t")
    poe = pd.read_csv(PRED_OE, sep="\t")
    if "time_utc" not in pls.columns or "time_utc" not in poe.columns:
        print("ERROR: Both PRED_LS and PRED_OE must contain column 'time_utc'.", file=sys.stderr)
        sys.exit(1)
    oe_cols = ["time_utc", "beta_pred_oe", "alpha_pred_oe"]
    missing_oe = [c for c in oe_cols if c not in poe.columns]
    if missing_oe:
        print(f"ERROR: PRED_OE missing columns: {missing_oe}", file=sys.stderr)
        sys.exit(1)
    for c in ("beta_pred_oe", "alpha_pred_oe"):
        if c in pls.columns:
            pls = pls.drop(columns=[c])
    merged = pls.merge(poe[oe_cols], on="time_utc", how="inner")
    if len(merged) == 0:
        print("ERROR: Inner merge on time_utc produced 0 rows.", file=sys.stderr)
        sys.exit(1)
    return merged


df = _load_merged_frame()

for c in REQUIRED_BASE:
    if c not in df.columns:
        print(f"ERROR: Missing column {c!r} in input table(s).", file=sys.stderr)
        sys.exit(1)

if VAL_OUT.is_file():
    print(f"Loading existing validation results to resume: {VAL_OUT.name}")
    existing = pd.read_csv(VAL_OUT, sep="\t")
    for c in SIM_COLS:
        if c not in df.columns and c in existing.columns:
            df[c] = existing[c]

for col in SIM_COLS:
    if col not in df.columns:
        df[col] = np.nan

mask = df[SIM_COLS].isna().any(axis=1)
sub = df[mask].copy()

if len(sub) == 0:
    print("All rows already processed. Nothing to do.")
    sys.exit(0)

print(
    f"Running forward models (MERRA + TabPFN LS + TabPFN OE + AERONET) for {len(sub)} rows "
    f"(ANGSTROM_BETA_REF_UM={LAMBDA_REF_UM})..."
)

tqdm.pandas(desc="Forward (uvspec)")
val_results = sub.progress_apply(run_val, axis=1)

df.loc[mask, SIM_COLS] = val_results

print(f"Saving validation results to {VAL_OUT.name}...")
VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(VAL_OUT, sep="\t", index=False, float_format="%.8f")

print("Evaluation complete.")
