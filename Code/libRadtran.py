"""
libRadtran: Core library for Radiative Transfer retrieval project.
Contains physical configurations, uvspec handlers, and OE logic.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

LIBRADTRANDIR = os.environ.get("LIBRADTRANDIR", "/Users/seryangd/libRadtran-2.0.6")

# Global Clearsky Physics Configuration
CLEARSKY_CONFIG = {
    "rte_solver": "disort",
    "number_of_streams": 6,
    "source": "solar",
    "wavelength": "280 2500",
    "mol_abs_param": "lowran",
    "aerosol_default": "on",
    "aerosol_species_library": "petzold",
    "aerosol_angstrom": "1.0 0.05",  # alpha beta (beta is retrieved)
    "albedo_method": "constant",
    "albedo": "0.2",
    "pressure_method": "constant",
    "pressure": "1013.25",
}

# =============================================================================
# CORE PHYSICS WRAPPERS
# =============================================================================

def build_uvspec_input(row: pd.Series, config: dict) -> str:
    """Constructs the uvspec input string for a single measurement row."""
    lines = []
    lines.append(f"rte_solver {config['rte_solver']}")
    lines.append(f"number_of_streams {config['number_of_streams']}")
    lines.append(f"source {config['source']}")
    lines.append(f"wavelength {config['wavelength']}")
    lines.append(f"mol_abs_param {config['mol_abs_param']}")
    lines.append(f"sza {row['zenith']}")
    
    # Overrides from MERRA-2 or OE
    h2o = row.get("h2o_mm_retrieved", row.get("merra_TQV", 20.0))
    lines.append(f"mol_modify H2O {h2o} MM")
    
    o3 = row.get("merra_TO3", 300.0)
    lines.append(f"mol_modify O3 {o3} DU")
    
    beta = row.get("beta_retrieved", row.get("merra_BETA", 0.05))
    alpha = row.get("merra_ALPHA", 1.0)
    lines.append(f"aerosol_default on")
    lines.append(f"aerosol_species_library petzold")
    lines.append(f"aerosol_angstrom {alpha} {beta}")
    
    albedo = row.get("merra_ALBEDO", config["albedo"])
    lines.append(f"albedo {albedo}")
    
    pressure = row.get("merra_PS", config["pressure"])
    lines.append(f"pressure {pressure}")
    
    lines.append("output_quantity direct diffuse global")
    return "\n".join(lines)

def run_clearsky(input_str: str, libradtran_dir: str) -> dict:
    """Runs uvspec binary and parses GHI, BNI, DHI output."""
    uvspec_bin = os.path.join(libradtran_dir, "bin", "uvspec")
    try:
        process = subprocess.Popen(
            [uvspec_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_str)
        if process.returncode != 0:
            return {"error": stderr.strip()}
            
        vals = [float(x) for x in stdout.split()]
        return {"ghi": vals[2], "bni": vals[0], "dhi": vals[1]}
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# OPTIMAL ESTIMATION LOGIC
# =============================================================================

def residuals_beta_h2o(params: np.ndarray, row: pd.Series, libradtran_dir: str, config: dict) -> np.ndarray:
    """Objective function for least_squares: (Model - Observed) / Error."""
    beta, h2o = params
    row_copy = row.copy()
    row_copy["beta_retrieved"] = beta
    row_copy["h2o_mm_retrieved"] = h2o
    
    inp = build_uvspec_input(row_copy, config)
    fwd = run_clearsky(inp, libradtran_dir)
    
    if "error" in fwd:
        return np.array([1e6, 1e6])
        
    # Weights for GHI and BNI
    res = [
        (fwd["ghi"] - row["ghi"]) / max(row["ghi"] * 0.05, 1.0),
        (fwd["bni"] - row["bni"]) / max(row["bni"] * 0.05, 1.0)
    ]
    return np.array(res)

def process_row_oe(row: pd.Series, libradtran_dir: str, config: dict) -> pd.Series:
    """Performs the OE retrieval of Beta and H2O for a single row."""
    x0 = [row.get("merra_BETA", 0.05), row.get("merra_TQV", 20.0)]
    bounds = ([0.0, 0.05], [2.0, 80.0])
    
    res = least_squares(
        residuals_beta_h2o, x0, bounds=bounds,
        args=(row, libradtran_dir, config),
        ftol=1e-3, xtol=1e-3, gtol=1e-3, max_nfev=20
    )
    
    beta_ret, h2o_ret = res.x
    
    # Final forward model with retrieved values
    row_final = row.copy()
    row_final["beta_retrieved"] = beta_ret
    row_final["h2o_mm_retrieved"] = h2o_ret
    fwd = run_clearsky(build_uvspec_input(row_final, config), libradtran_dir)
    
    # MERRA-2 Comparison run
    fwd_merra = run_clearsky(build_uvspec_input(row, config), libradtran_dir)
    
    return pd.Series({
        "ghi_merra": fwd_merra.get("ghi", np.nan),
        "dni_merra": fwd_merra.get("bni", np.nan),
        "dhi_merra": fwd_merra.get("dhi", np.nan),
        "ghi_ls": fwd.get("ghi", np.nan),
        "dni_ls": fwd.get("bni", np.nan),
        "dhi_ls": fwd.get("dhi", np.nan),
        "beta_retrieved": beta_ret,
        "h2o_mm_retrieved": h2o_ret,
        "ls_nfev": res.nfev,
        "ls_success": res.success
    })
