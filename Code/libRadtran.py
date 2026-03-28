"""
libRadtran + Optimal Estimation: Forward modeling and parameter retrieval.
Combined from clearsky_forward.py and oe_retrieve_beta_h2o.py.
"""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

# =============================================================================
# CONFIGURATION
# =============================================================================

LIBRADTRANDIR = os.environ.get("LIBRADTRANDIR", "/Users/seryangd/libRadtran-2.0.6").strip()

PROJECT = Path(__file__).resolve().parent.parent

# =============================================================================
# LIBRARY CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ClearskyConfig:
    """Defaults for ``build_uvspec_input`` / ``run_clearsky``."""
    wavelength_nm_min: float = 240.12
    wavelength_nm_max: float = 3000.0
    apply_wavelength_limits: bool = True
    source_solar: str = "solar_flux/atlas_plus_modtran"
    mol_abs_param: str = "kato2"
    atmosphere_file: str = "atmmod/afglmw.dat"
    sza_column: str = "zenith"
    use_pseudospherical: bool = True
    number_of_streams: int = 12
    rte_solver: str = "disort"
    disort_intcor: str | None = "moments"
    to3_bsrn_to_du: float = 1000.0
    tqv_bsrn_to_mm_pw: float = 10.0
    aerosol_setup: str = "vulcan"
    aerosol_vulcan: int = 1
    aerosol_haze: int = 6
    aerosol_season: int = 2
    aerosol_visibility_km: float = 20.0
    uvspec_timeout_s: int = 120

DEFAULT_CLEARSKY_CONFIG = ClearskyConfig()
CLEARSKY_CONFIG = DEFAULT_CLEARSKY_CONFIG

# OE Parameters
BETA_MIN = 1e-6
BETA_MAX = 2.0
H2O_MM_MIN = 0.05
H2O_MM_MAX = 80.0
DIFF_STEP_BETA = 0.002
DIFF_STEP_H2O_MM = 0.05
MAX_NFEV = 40
JAC_MODE = "2-point"
FAILURE_PENALTY = 1e6


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _resolve_physics(
    row: pd.Series,
    config: ClearskyConfig,
    *,
    o3_du: float | None = None,
    h2o_mm: float | None = None,
    albedo: float | None = None,
    pressure_hpa: float | None = None,
    angstrom_alpha: float | None = None,
    angstrom_beta: float | None = None,
    sza_deg: float | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Acts as a parameter arbitrator that prepares the physical inputs required for a libRadtran simulation. 
    It resolves geometry, albedo, pressure, and gas/aerosol optics, prioritizing explicit overrides 
    (typically from Optimal Estimation) over the background MERRA-2 or BSRN data columns.

    Parameters:
    ------------
    row : pd.Series
        Data row containing base measurements (merra_* and zenith).
    config : ClearskyConfig
        Reference for default headers and unit scaling factors.
    o3_du : float, optional
        Override for total column ozone (in DU).
    h2o_mm : float, optional
        Override for precipitable water (in mm).
    albedo : float, optional
        Override for surface albedo (0 to 1).
    pressure_hpa : float, optional
        Override for surface pressure (in hPa).
    angstrom_alpha : float, optional
        Override for Angstrom alpha (exponent).
    angstrom_beta : float, optional
        Override for Angstrom beta (aerosol load).
    sza_deg : float, optional
        Override for solar zenith angle (degrees).

    Returns:
    ------------
    Return (sza, albedo, p_hpa, o3_du, h2o_mm, alpha, beta). Overrides beat ``row``.
    """
    sza = float(row[config.sza_column]) if sza_deg is None else float(sza_deg)
    alb = float(row["merra_ALBEDO"]) if albedo is None else float(albedo)
    p = float(row["merra_PS"]) if pressure_hpa is None else float(pressure_hpa)
    o3 = float(row["merra_TO3"]) * config.to3_bsrn_to_du if o3_du is None else float(o3_du)
    h2o = float(row["merra_TQV"]) * config.tqv_bsrn_to_mm_pw if h2o_mm is None else float(h2o_mm)
    a = float(row["merra_ALPHA"]) if angstrom_alpha is None else float(angstrom_alpha)
    b = float(row["merra_BETA"]) if angstrom_beta is None else float(angstrom_beta)
    return sza, alb, p, o3, h2o, a, b

def merra_explicit_physics(row: pd.Series, config: ClearskyConfig) -> tuple[float, float, float, float]:
    """
    Extracts explicit physics (alpha, O3, beta, H2O) using MERRA-2 columns and internal scaling.

    Parameters:
    ------------
    row : pd.Series
        Data row containing 'merra_ALPHA', 'merra_TO3', 'merra_BETA', and 'merra_TQV'.
    config : ClearskyConfig
        Reference for unit conversion factors (e.g., to3_bsrn_to_du).

    Returns:
    ------------
    (alpha, o3_du, beta, h2o_mm) : tuple of 4 floats.
    """
    alpha_m = float(row["merra_ALPHA"])
    o3_du_m = float(row["merra_TO3"]) * config.to3_bsrn_to_du
    beta_m = float(row["merra_BETA"])
    h2o_mm_m = float(row["merra_TQV"]) * config.tqv_bsrn_to_mm_pw
    return alpha_m, o3_du_m, beta_m, h2o_mm_m

def row_skip_oe(row: pd.Series) -> bool:
    """
    Determines whether a row should be skipped due to solar zenith angle or missing data.
    
    Parameters:
    ------------
    row : pd.Series
        Data row containing 'zenith' and required measurement/MERRA columns.

    Returns:
    ------------
    skip : bool
        True if zenith >= 87 or if essential parameters are NaN.
    """
    if float(row["zenith"]) >= 87.0:
        return True
    for c in ("ghi", "bni", "dhi", "merra_ALPHA", "merra_BETA", "merra_TO3", "merra_TQV"):
        if pd.isna(row.get(c, np.nan)):
            return True
    return False

# =============================================================================
# ACTUAL FUNCTIONS
# =============================================================================

def build_uvspec_input(
    row: pd.Series,
    libradtran_dir: str,
    config: ClearskyConfig = DEFAULT_CLEARSKY_CONFIG,
    *,
    o3_du: float | None = None,
    h2o_mm: float | None = None,
    albedo: float | None = None,
    pressure_hpa: float | None = None,
    angstrom_alpha: float | None = None,
    angstrom_beta: float | None = None,
    sza_deg: float | None = None,
) -> str:
    """
    Constructs the full ``uvspec`` standard input string for a single row.

    Parameters:
    ------------
    row : pd.Series
        Data row containing base measurements.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig, optional
        Simulation configuration defaults.
    o3_du : float, optional
        Override for total column ozone (DU).
    h2o_mm : float, optional
        Override for precipitable water (mm).
    albedo : float, optional
        Override for surface albedo (0-1).
    pressure_hpa : float, optional
        Override for surface pressure (hPa).
    angstrom_alpha : float, optional
        Override for Angstrom alpha.
    angstrom_beta : float, optional
        Override for Angstrom beta.
    sza_deg : float, optional
        Override for solar zenith angle (degrees).

    Returns:
    ------------
    stdin : str
        The multiline string for ``uvspec`` standard input.
    """
    data_dir = os.path.join(libradtran_dir, "data")
    atmo_path = f"{data_dir}/{config.atmosphere_file}"
    solar_path = f"{data_dir}/{config.source_solar}"

    sza, albedo_v, pressure_hpa_v, o3_v, h2o_mm_v, a, b = _resolve_physics(
        row, config, o3_du=o3_du, h2o_mm=h2o_mm, albedo=albedo,
        pressure_hpa=pressure_hpa, angstrom_alpha=angstrom_alpha,
        angstrom_beta=angstrom_beta, sza_deg=sza_deg,
    )

    lines: list[str] = [
        f"data_files_path {data_dir}",
        f"atmosphere_file {atmo_path}",
        f"source solar {solar_path}",
    ]

    if config.apply_wavelength_limits:
        lines.append(f"wavelength {config.wavelength_nm_min:.2f} {config.wavelength_nm_max:.2f}")

    if np.isfinite(o3_v) and o3_v > 0.0:
        lines.append(f"mol_modify O3 {o3_v:.6f} DU")
    if np.isfinite(h2o_mm_v) and h2o_mm_v > 0.0:
        lines.append(f"mol_modify h2o {h2o_mm_v:.6f} MM")

    lines.extend([
        f"pressure {pressure_hpa_v:.6f}",
        f"sza {sza:.6f}",
        f"albedo {albedo_v:.6f}",
        f"rte_solver {config.rte_solver}",
        f"number_of_streams {config.number_of_streams}",
    ])
    if config.disort_intcor:
        lines.append(f"disort_intcor {config.disort_intcor}")
    if config.use_pseudospherical:
        lines.append("pseudospherical")

    lines.append(f"mol_abs_param {config.mol_abs_param}")
    lines.append("output_process sum")
    lines.append("aerosol_default")
    
    if config.aerosol_setup == "vulcan":
        lines.extend([
            f"aerosol_vulcan {config.aerosol_vulcan}",
            f"aerosol_haze {config.aerosol_haze}",
            f"aerosol_season {config.aerosol_season}",
            f"aerosol_visibility {config.aerosol_visibility_km}",
        ])
    elif config.aerosol_setup == "opac":
        lines.extend(["aerosol_species_library OPAC", "aerosol_species_file continental_clean"])
    else:
        raise ValueError(f"Unknown aerosol_setup: {config.aerosol_setup}")

    lines.append(f"aerosol_angstrom {a:.6f} {b:.6f}")
    lines.append("output_user edir edn eglo")
    lines.append("quiet")

    return "\n".join(lines) + "\n"

def run_clearsky(
    row: pd.Series,
    libradtran_dir: str,
    config: ClearskyConfig = DEFAULT_CLEARSKY_CONFIG,
    *,
    o3_du: float | None = None,
    h2o_mm: float | None = None,
    albedo: float | None = None,
    pressure_hpa: float | None = None,
    angstrom_alpha: float | None = None,
    angstrom_beta: float | None = None,
    sza_deg: float | None = None,
    quiet: bool = False,
) -> pd.Series:
    """
    Runs a single clear-sky uvspec simulation and returns simulated fluxes.

    Parameters:
    ------------
    row : pd.Series
        Data row containing base measurements.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig, optional
        Simulation configuration defaults.
    o3_du : float, optional
        Override for total column ozone (DU).
    h2o_mm : float, optional
        Override for precipitable water (mm).
    albedo : float, optional
        Override for surface albedo (0-1).
    pressure_hpa : float, optional
        Override for surface pressure (hPa).
    angstrom_alpha : float, optional
        Override for Angstrom alpha.
    angstrom_beta : float, optional
        Override for Angstrom beta.
    sza_deg : float, optional
        Override for solar zenith angle (degrees).
    quiet : bool, optional
        If True, suppresses error messages to stdout.

    Returns:
    ------------
    sim : pd.Series
        Series with 'ghi_sim', 'bni_sim', and 'dhi_sim' [W m-2].
    """
    sza, _, _, _, _, a, b = _resolve_physics(
        row, config, o3_du=o3_du, h2o_mm=h2o_mm, albedo=albedo,
        pressure_hpa=pressure_hpa, angstrom_alpha=angstrom_alpha,
        angstrom_beta=angstrom_beta, sza_deg=sza_deg,
    )

    need = [sza, a, b]
    albedo_r = float(row["merra_ALBEDO"]) if albedo is None else float(albedo)
    p_r = float(row["merra_PS"]) if pressure_hpa is None else float(pressure_hpa)
    need.extend([albedo_r, p_r])

    if not np.isfinite(np.asarray(need, dtype=float)).all() or sza >= 90.0:
        return pd.Series({"ghi_sim": np.nan, "bni_sim": np.nan, "dhi_sim": np.nan}, dtype=float)

    uvspec_exe = os.path.join(libradtran_dir, "bin", "uvspec")
    inp_content = build_uvspec_input(
        row, libradtran_dir, config, o3_du=o3_du, h2o_mm=h2o_mm, albedo=albedo,
        pressure_hpa=pressure_hpa, angstrom_alpha=angstrom_alpha,
        angstrom_beta=angstrom_beta, sza_deg=sza_deg,
    )

    try:
        process = subprocess.run(
            [uvspec_exe], input=inp_content, text=True, capture_output=True,
            check=True, timeout=config.uvspec_timeout_s,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        if not quiet:
            print(f"libRadtran/Execution error at {row.name}: {e}")
        return pd.Series({"ghi_sim": np.nan, "bni_sim": np.nan, "dhi_sim": np.nan}, dtype=float)

    raw = process.stdout.strip()
    if not raw:
        return pd.Series({"ghi_sim": np.nan, "bni_sim": np.nan, "dhi_sim": np.nan}, dtype=float)

    out = pd.read_csv(StringIO(raw), sep=r"\s+", header=None, names=["edir_horiz", "dhi_sim", "ghi_sim"], engine="python")
    edir_h = float(out["edir_horiz"].iloc[0])
    dhi_sim = float(out["dhi_sim"].iloc[0])
    ghi_sim = float(out["ghi_sim"].iloc[0])

    sza_rad = np.radians(sza)
    cos_sza = max(float(np.cos(sza_rad)), 0.01)
    bni_sim = edir_h / cos_sza

    return pd.Series({"ghi_sim": ghi_sim, "bni_sim": bni_sim, "dhi_sim": dhi_sim}, dtype=float)

def calculate_residuals_oe(
    x: np.ndarray, row: pd.Series, libradtran_dir: str,
    config: ClearskyConfig, alpha_m: float, o3_du_m: float,
) -> np.ndarray:
    """
    Calculates residuals between simulated and measured GHI, DNI, and DHI.

    Parameters:
    ------------
    x : np.ndarray
        Sample state vector [beta, h2o_mm].
    row : pd.Series
        Data row with measured 'ghi', 'bni', and 'dhi'.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig
        Reference configuration.
    alpha_m : float
        Fixed Angstrom alpha for this row.
    o3_du_m : float
        Fixed total column ozone for this row.

    Returns:
    ------------
    residuals : np.ndarray
        Array of (sim - meas) for GHI, DNI, DHI.
    """
    beta, h2o_mm = float(x[0]), float(x[1])
    sim = run_clearsky(
        row, libradtran_dir, config, angstrom_alpha=alpha_m, o3_du=o3_du_m,
        angstrom_beta=beta, h2o_mm=h2o_mm, quiet=True,
    )
    if pd.isna(sim["ghi_sim"]) or pd.isna(sim["bni_sim"]) or pd.isna(sim["dhi_sim"]):
        return np.full(3, FAILURE_PENALTY, dtype=float)

    y_meas = np.array([float(row["ghi"]), float(row["bni"]), float(row["dhi"])], dtype=float)
    y_sim = np.array([float(sim["ghi_sim"]), float(sim["bni_sim"]), float(sim["dhi_sim"])], dtype=float)
    
    if not np.isfinite(y_meas).all():
        return np.full(3, FAILURE_PENALTY, dtype=float)
    return y_sim - y_meas

def retrieve_beta_h2o_one_row(
    row: pd.Series, libradtran_dir: str, config: ClearskyConfig = DEFAULT_CLEARSKY_CONFIG,
) -> tuple[float, float, bool, float | None]:
    """
    Retrieves Angstrom beta and water vapor for one row using least_squares.

    Parameters:
    ------------
    row : pd.Series
        Data row containing measurements and MERRA priors.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig, optional
        Reference configuration.

    Returns:
    ------------
    (beta, h2o, success, cost) : tuple
        The retrieved values and optimization status.
    """
    if row_skip_oe(row):
        return np.nan, np.nan, False, None

    alpha_m, o3_du_m, beta0, h2o0 = merra_explicit_physics(row, config)
    x0 = np.array([beta0, h2o0], dtype=float)
    bounds = (np.array([BETA_MIN, H2O_MM_MIN]), np.array([BETA_MAX, H2O_MM_MAX]))
    diff_step = np.array([DIFF_STEP_BETA, DIFF_STEP_H2O_MM])

    try:
        result = least_squares(
            calculate_residuals_oe, x0, args=(row, libradtran_dir, config, alpha_m, o3_du_m),
            bounds=bounds, jac=JAC_MODE, diff_step=diff_step, max_nfev=MAX_NFEV,
        )
    except Exception as e:
        print(f"OE failed at {row.name}: {e}")
        return np.nan, np.nan, False, None

    cost = float(result.cost) if hasattr(result, "cost") else None
    if result.success:
        return float(result.x[0]), float(result.x[1]), True, cost
    return np.nan, np.nan, False, cost

def forward_merra_explicit(row: pd.Series, libradtran_dir: str, config: ClearskyConfig, quiet: bool = True) -> pd.Series:
    """
    Computes forward clear-sky fluxes using specific MERRA-2 optical priors.

    Parameters:
    ------------
    row : pd.Series
        Data row with MERRA-2 columns.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig
        Reference configuration.
    quiet : bool, optional
        Whether to suppress stderr on failure.

    Returns:
    ------------
    fluxes : pd.Series
        Simulated clear-sky fluxes.
    """
    alpha_m, o3_du_m, beta_m, h2o_mm_m = merra_explicit_physics(row, config)
    return run_clearsky(
        row, libradtran_dir, config, angstrom_alpha=alpha_m, o3_du=o3_du_m,
        angstrom_beta=beta_m, h2o_mm=h2o_mm_m, quiet=quiet,
    )

def process_row_ls(row: pd.Series, libradtran_dir: str, config: ClearskyConfig) -> pd.Series:
    """
    High-level handler: runs MERRA forward, performs LS retrieval, and runs LS forward.

    Parameters:
    ------------
    row : pd.Series
        Standardized BSRN+MERRA data row.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig
        Reference configuration.

    Returns:
    ------------
    result : pd.Series
        Series combining original data with simulated fluxes and retrieved params.
    """
    alpha_m, o3_du_m, beta_m, h2o_mm_m = merra_explicit_physics(row, config)

    merra_sim = forward_merra_explicit(row, libradtran_dir, config, quiet=True)
    ghi_m = float(merra_sim["ghi_sim"]) if pd.notna(merra_sim["ghi_sim"]) else np.nan
    bni_m = float(merra_sim["bni_sim"]) if pd.notna(merra_sim["bni_sim"]) else np.nan
    dhi_m = float(merra_sim["dhi_sim"]) if pd.notna(merra_sim["dhi_sim"]) else np.nan

    beta_hat, h2o_hat, ok, _cost = retrieve_beta_h2o_one_row(row, libradtran_dir, config)

    if ok and np.isfinite(beta_hat) and np.isfinite(h2o_hat):
        oe_sim = run_clearsky(
            row, libradtran_dir, config, angstrom_alpha=alpha_m, o3_du=o3_du_m,
            angstrom_beta=beta_hat, h2o_mm=h2o_hat, quiet=True,
        )
        ghi_o = float(oe_sim["ghi_sim"]) if pd.notna(oe_sim["ghi_sim"]) else np.nan
        bni_o = float(oe_sim["bni_sim"]) if pd.notna(oe_sim["bni_sim"]) else np.nan
        dhi_o = float(oe_sim["dhi_sim"]) if pd.notna(oe_sim["dhi_sim"]) else np.nan
    else:
        ghi_o = dni_o = dhi_o = np.nan

    return pd.Series({
        "ghi_merra": ghi_m, "bni_merra": bni_m, "dhi_merra": dhi_m,
        "ghi_ls": ghi_o, "bni_ls": bni_o, "dhi_ls": dhi_o,
        "beta_retrieved": beta_hat, "h2o_mm_retrieved": h2o_hat,
        "merra_ALPHA": alpha_m, "merra_BETA": beta_m,
        "merra_TO3": float(row["merra_TO3"]), "merra_TQV": float(row["merra_TQV"]),
    })


# =============================================================================
# End of 0.libRadtran.py (Library)
# =============================================================================
