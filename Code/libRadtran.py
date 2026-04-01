"""
libRadtran + Least Squares / Optimal Estimation: Forward modeling and parameter retrieval.
"""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass, replace
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
    """Defaults for ``build_uvspec_input`` / ``run_clearsky``.

    **Atmosphere profile:** override with env ``LRT_ATMOSPHERE`` (see ``_atmosphere_file_from_env``).
    Default is mid-latitude **winter** ``atmmod/afglmw.dat``; ``afglms`` selects mid-latitude **summer**.
    """
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
    # If True, ``build_uvspec_input`` picks ``atmmod/afglms.dat`` vs ``afglmw.dat`` from calendar month
    # (see ``_midlatitude_atmosphere_for_month``), using ``time_column`` or the row index name.
    # Ignored when env ``LRT_ATMOSPHERE`` is set (fixed profile for all rows).
    seasonal_atmosphere: bool = False
    time_column: str = "time_utc"
    # "north" — summer May–Sep → afglms; "south" — summer Nov–Mar → afglms
    seasonal_hemisphere: str = "north"


def _parse_atmosphere_token(v: str) -> str:
    """Path fragment under libRadtran ``data/`` (``atmosphere_file`` keyword)."""
    v = v.strip().lower()
    if not v or v in ("afglmw", "mw"):
        return "atmmod/afglmw.dat"
    if v in ("afglms", "ms"):
        return "atmmod/afglms.dat"
    if "/" in v or v.endswith(".dat"):
        return v
    return f"atmmod/{v}.dat"


def _atmosphere_file_from_env() -> str:
    """Path fragment from env ``LRT_ATMOSPHERE`` (see ``_parse_atmosphere_token``)."""
    return _parse_atmosphere_token(os.environ.get("LRT_ATMOSPHERE", ""))


def _midlatitude_atmosphere_for_month(month: int, hemisphere: str) -> str:
    """AFGL mid-latitude summer vs winter file from calendar month.

    Northern hemisphere: **May–September** → ``afglms.dat``; else ``afglmw.dat``.
    Southern hemisphere: **November–March** → ``afglms.dat``; else ``afglmw.dat``.
    """
    m = int(month)
    h = hemisphere.strip().lower()
    if h == "south":
        return "atmmod/afglms.dat" if m in (11, 12, 1, 2, 3) else "atmmod/afglmw.dat"
    return "atmmod/afglms.dat" if m in (5, 6, 7, 8, 9) else "atmmod/afglmw.dat"


def _timestamp_from_row(row: pd.Series, time_column: str) -> pd.Timestamp:
    """Timestamp for seasonal atmosphere: ``time_column`` or ``row.name`` (DatetimeIndex row)."""
    if time_column in row.index:
        return pd.Timestamp(row[time_column])
    if row.name is not None:
        return pd.Timestamp(row.name)
    raise KeyError(
        f"Cannot resolve time for seasonal atmosphere: no {time_column!r} in row and row has no .name"
    )


def _resolve_atmosphere_file_for_row(row: pd.Series, config: ClearskyConfig) -> str:
    """Fixed profile from ``LRT_ATMOSPHERE``, else optional month-based AFGL file, else ``atmosphere_file``."""
    env = os.environ.get("LRT_ATMOSPHERE", "").strip()
    if env:
        return _parse_atmosphere_token(env)
    if config.seasonal_atmosphere:
        ts = _timestamp_from_row(row, config.time_column)
        return _midlatitude_atmosphere_for_month(ts.month, config.seasonal_hemisphere)
    return config.atmosphere_file


def _seasonal_atmosphere_from_env() -> bool:
    return os.environ.get("LRT_SEASONAL_ATMOSPHERE", "").strip().lower() in {"1", "true", "yes"}


def _seasonal_hemisphere_from_env() -> str:
    v = os.environ.get("LRT_SEASON_HEMISPHERE", "").strip().lower()
    if v in ("south", "s", "sh"):
        return "south"
    return "north"


DEFAULT_CLEARSKY_CONFIG = ClearskyConfig()
CLEARSKY_CONFIG = replace(
    DEFAULT_CLEARSKY_CONFIG,
    atmosphere_file=_atmosphere_file_from_env(),
    seasonal_atmosphere=_seasonal_atmosphere_from_env(),
    seasonal_hemisphere=_seasonal_hemisphere_from_env(),
)

# LS Parameters
BETA_MIN = 1e-6
BETA_MAX = 1.1 # Chris' REST2 paper
ALPHA_MIN = 0.3
ALPHA_MAX = 2.0
W_MIN = 0.05
W_MAX = 100.0 # Chris' REST2 paper
DIFF_STEP_BETA = 0.002
DIFF_STEP_ALPHA = 0.01
DIFF_STEP_W = 0.05
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
    w: float | None = None,
    albedo: float | None = None,
    pressure_hpa: float | None = None,
    angstrom_alpha: float | None = None,
    angstrom_beta: float | None = None,
    sza_deg: float | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Acts as a parameter arbitrator that prepares the physical inputs required for a libRadtran simulation. 
    It resolves geometry, albedo, pressure, and gas/aerosol optics, prioritizing explicit overrides 
    (typically from Least Squares) over the background MERRA-2 or BSRN data columns.

    Parameters:
    ------------
    row : pd.Series
        Data row containing base measurements (merra_* and zenith).
    config : ClearskyConfig
        Reference for default headers and unit scaling factors.
    o3_du : float, optional
        Override for total column ozone (in DU).
    w : float, optional
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
    Return (sza, albedo, p_hpa, o3_du, w, alpha, beta). Overrides beat ``row``.
    """
    sza = float(row[config.sza_column]) if sza_deg is None else float(sza_deg)
    alb = float(row["merra_ALBEDO"]) if albedo is None else float(albedo)
    p = float(row["merra_PS"]) if pressure_hpa is None else float(pressure_hpa)
    o3 = float(row["merra_TO3"]) * config.to3_bsrn_to_du if o3_du is None else float(o3_du)
    w_v = float(row["merra_TQV"]) * config.tqv_bsrn_to_mm_pw if w is None else float(w)
    a = float(row["merra_ALPHA"]) if angstrom_alpha is None else float(angstrom_alpha)
    b = float(row["merra_BETA"]) if angstrom_beta is None else float(angstrom_beta)
    return sza, alb, p, o3, w_v, a, b

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
    (alpha, o3_du, beta, w) : tuple of 4 floats.
    """
    alpha_m = float(row["merra_ALPHA"])
    o3_du_m = float(row["merra_TO3"]) * config.to3_bsrn_to_du
    beta_m = float(row["merra_BETA"])
    w_m = float(row["merra_TQV"]) * config.tqv_bsrn_to_mm_pw
    return alpha_m, o3_du_m, beta_m, w_m

def row_skip_ls(row: pd.Series) -> bool:
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
    config: ClearskyConfig = CLEARSKY_CONFIG,
    *,
    o3_du: float | None = None,
    w: float | None = None,
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
    w : float, optional
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
    atm_rel = _resolve_atmosphere_file_for_row(row, config)
    atmo_path = f"{data_dir}/{atm_rel}"
    solar_path = f"{data_dir}/{config.source_solar}"

    sza, albedo_v, pressure_hpa_v, o3_v, w_v, a, b = _resolve_physics(
        row, config, o3_du=o3_du, w=w, albedo=albedo,
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
    if np.isfinite(w_v) and w_v > 0.0:
        lines.append(f"mol_modify h2o {w_v:.6f} MM")

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
    config: ClearskyConfig = CLEARSKY_CONFIG,
    *,
    o3_du: float | None = None,
    w: float | None = None,
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
    w : float, optional
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
        row, config, o3_du=o3_du, w=w, albedo=albedo,
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
        row, libradtran_dir, config, o3_du=o3_du, w=w, albedo=albedo,
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

def calculate_residuals_ls(
    x: np.ndarray, row: pd.Series, libradtran_dir: str,
    config: ClearskyConfig, o3_du_m: float, w_m: float,
) -> np.ndarray:
    """
    Calculates residuals between simulated and measured GHI, BNI, and DHI.

    Parameters:
    ------------
    x : np.ndarray
        State vector [beta, alpha].
    row : pd.Series
        Data row with measured 'ghi', 'bni', and 'dhi'.
    libradtran_dir : str
        Path to the libRadtran installation root.
    config : ClearskyConfig
        Reference configuration.
    o3_du_m : float
        Fixed total column ozone for this row.
    w_m : float
        Fixed precipitable water from MERRA for this row.

    Returns:
    ------------
    residuals : np.ndarray
        Array of (sim - meas) for GHI, BNI, DHI.
    """
    beta, alpha = float(x[0]), float(x[1])
    sim = run_clearsky(
        row, libradtran_dir, config, angstrom_alpha=alpha, o3_du=o3_du_m,
        angstrom_beta=beta, w=w_m, quiet=True,
    )
    if pd.isna(sim["ghi_sim"]) or pd.isna(sim["bni_sim"]) or pd.isna(sim["dhi_sim"]):
        return np.full(3, FAILURE_PENALTY, dtype=float)

    y_meas = np.array([float(row["ghi"]), float(row["bni"]), float(row["dhi"])], dtype=float)
    y_sim = np.array([float(sim["ghi_sim"]), float(sim["bni_sim"]), float(sim["dhi_sim"])], dtype=float)

    if not np.isfinite(y_meas).all():
        return np.full(3, FAILURE_PENALTY, dtype=float)
    return y_sim - y_meas

def retrieve_one_row_ls(
    row: pd.Series, libradtran_dir: str, config: ClearskyConfig = CLEARSKY_CONFIG,
) -> tuple[float, float, bool, float | None]:
    """
    Retrieves Angstrom beta and alpha for one row using least_squares.

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
    (beta, alpha, success, cost) : tuple
        The retrieved values and optimization status.
    """
    if row_skip_ls(row):
        return np.nan, np.nan, False, None

    alpha_m, o3_du_m, beta_m, w_m = merra_explicit_physics(row, config)
    x0 = np.array([beta_m, alpha_m], dtype=float)
    bounds = (np.array([BETA_MIN, ALPHA_MIN]), np.array([BETA_MAX, ALPHA_MAX]))
    diff_step = np.array([DIFF_STEP_BETA, DIFF_STEP_ALPHA])

    try:
        result = least_squares(
            calculate_residuals_ls,
            x0,
            args=(row, libradtran_dir, config, o3_du_m, w_m),
            bounds=bounds,
            jac=JAC_MODE,
            diff_step=diff_step,
            max_nfev=MAX_NFEV,
        )
    except Exception as e:
        print(f"LS failed at {row.name}: {e}")
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
    alpha_m, o3_du_m, beta_m, w_m = merra_explicit_physics(row, config)
    return run_clearsky(
        row, libradtran_dir, config, angstrom_alpha=alpha_m, o3_du=o3_du_m,
        angstrom_beta=beta_m, w=w_m, quiet=quiet,
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
    alpha_m, o3_du_m, beta_m, w_m = merra_explicit_physics(row, config)

    merra_sim = forward_merra_explicit(row, libradtran_dir, config, quiet=True)
    ghi_m = float(merra_sim["ghi_sim"]) if pd.notna(merra_sim["ghi_sim"]) else np.nan
    bni_m = float(merra_sim["bni_sim"]) if pd.notna(merra_sim["bni_sim"]) else np.nan
    dhi_m = float(merra_sim["dhi_sim"]) if pd.notna(merra_sim["dhi_sim"]) else np.nan

    beta_ls, alpha_ls, ok, _cost = retrieve_one_row_ls(row, libradtran_dir, config)

    if ok and np.isfinite(beta_ls) and np.isfinite(alpha_ls):
        ls_sim = run_clearsky(
            row, libradtran_dir, config, angstrom_alpha=alpha_ls, o3_du=o3_du_m,
            angstrom_beta=beta_ls, w=w_m, quiet=True,
        )
        ghi_o = float(ls_sim["ghi_sim"]) if pd.notna(ls_sim["ghi_sim"]) else np.nan
        bni_o = float(ls_sim["bni_sim"]) if pd.notna(ls_sim["bni_sim"]) else np.nan
        dhi_o = float(ls_sim["dhi_sim"]) if pd.notna(ls_sim["dhi_sim"]) else np.nan
    else:
        ghi_o = bni_o = dhi_o = np.nan

    return pd.Series({
        "ghi_merra": ghi_m, "bni_merra": bni_m, "dhi_merra": dhi_m,
        "ghi_ls": ghi_o, "bni_ls": bni_o, "dhi_ls": dhi_o,
        "beta_ls": beta_ls, "alpha_ls": alpha_ls,
        "merra_ALPHA": alpha_m, "merra_BETA": beta_m,
        "merra_TO3": float(row["merra_TO3"]), "merra_TQV": float(row["merra_TQV"]),
    })

def calculate_residuals_oe(
    x: np.ndarray, row: pd.Series, libradtran_dir: str,
    config: ClearskyConfig, o3_du_m: float, w_m: float,
    x_prior: np.ndarray, y_err: np.ndarray, x_err: np.ndarray
) -> np.ndarray:
    """
    Calculates the OE weighted residuals for both measurements and priors.

    State vector is [beta, alpha]; water vapour w is fixed from MERRA.

    Uses only BNI and DHI as the measurement vector (not GHI) because
    GHI ≈ BNI·cos(SZA) + DHI — including all three with a diagonal S_ε
    would over-count measurement information and implicitly down-weight
    the prior constraint.
    """
    beta, alpha = float(x[0]), float(x[1])

    sim = run_clearsky(
        row, libradtran_dir, config, angstrom_alpha=alpha, o3_du=o3_du_m,
        angstrom_beta=beta, w=w_m, quiet=True,
    )

    y_meas = np.array([float(row["bni"]), float(row["dhi"])], dtype=float)
    y_sim = np.array([float(sim["bni_sim"]), float(sim["dhi_sim"])], dtype=float)

    if not np.isfinite(y_sim).all() or not np.isfinite(y_meas).all():
        return np.full(4, FAILURE_PENALTY, dtype=float)

    res_y = (y_sim - y_meas) / y_err
    res_x = (x - x_prior) / x_err

    return np.concatenate((res_y, res_x))

def retrieve_one_row_oe(
    row: pd.Series, libradtran_dir: str, config: ClearskyConfig = CLEARSKY_CONFIG,
) -> tuple[float, float, bool, float | None]:
    """
    Retrieves Angstrom beta and alpha for one row using OE (least_squares with priors).

    Returns:
    ------------
    (beta, alpha, success, cost) : tuple
    """
    if row_skip_ls(row):
        return np.nan, np.nan, False, None

    alpha_m, o3_du_m, beta_m, w_m = merra_explicit_physics(row, config)

    x_prior = np.array([beta_m, alpha_m], dtype=float)

    # Dynamic measurement uncertainties (BNI + DHI only)
    bni_meas = float(row["bni"])
    dhi_meas = float(row["dhi"])
    bni_err = max(0.02 * abs(bni_meas), 2.0)
    dhi_err = max(0.05 * abs(dhi_meas), 5.0)
    y_err = np.array([bni_err, dhi_err], dtype=float)

    # Dynamic prior uncertainties
    # Aerosol turbidity: ~20% relative (MERRA-2 vs AERONET), floor 0.01
    beta_err = max(0.20 * beta_m, 0.01)
    # Ångström alpha: ~15% relative (MERRA-2 vs AERONET), floor 0.05
    alpha_err = max(0.15 * alpha_m, 0.05)
    x_err = np.array([beta_err, alpha_err], dtype=float)

    x0 = np.array([beta_m, alpha_m], dtype=float)
    bounds = (np.array([BETA_MIN, ALPHA_MIN]), np.array([BETA_MAX, ALPHA_MAX]))
    diff_step = np.array([DIFF_STEP_BETA, DIFF_STEP_ALPHA])

    try:
        result = least_squares(
            calculate_residuals_oe,
            x0,
            args=(row, libradtran_dir, config, o3_du_m, w_m, x_prior, y_err, x_err),
            bounds=bounds,
            jac=JAC_MODE,
            diff_step=diff_step,
            max_nfev=MAX_NFEV,
        )
    except Exception as e:
        print(f"OE failed at {row.name}: {e}")
        return np.nan, np.nan, False, None

    cost = float(result.cost) if hasattr(result, "cost") else None
    if result.success:
        return float(result.x[0]), float(result.x[1]), True, cost
    return np.nan, np.nan, False, cost

def process_row_oe(row: pd.Series, libradtran_dir: str, config: ClearskyConfig) -> pd.Series:
    """
    High-level handler: runs MERRA forward, performs OE retrieval, and runs OE forward.
    """
    alpha_m, o3_du_m, beta_m, w_m = merra_explicit_physics(row, config)

    merra_sim = forward_merra_explicit(row, libradtran_dir, config, quiet=True)
    ghi_m = float(merra_sim["ghi_sim"]) if pd.notna(merra_sim["ghi_sim"]) else np.nan
    bni_m = float(merra_sim["bni_sim"]) if pd.notna(merra_sim["bni_sim"]) else np.nan
    dhi_m = float(merra_sim["dhi_sim"]) if pd.notna(merra_sim["dhi_sim"]) else np.nan

    beta_oe, alpha_oe, ok, _cost = retrieve_one_row_oe(row, libradtran_dir, config)

    if ok and np.isfinite(beta_oe) and np.isfinite(alpha_oe):
        oe_sim = run_clearsky(
            row, libradtran_dir, config, angstrom_alpha=alpha_oe, o3_du=o3_du_m,
            angstrom_beta=beta_oe, w=w_m, quiet=True,
        )
        ghi_o = float(oe_sim["ghi_sim"]) if pd.notna(oe_sim["ghi_sim"]) else np.nan
        bni_o = float(oe_sim["bni_sim"]) if pd.notna(oe_sim["bni_sim"]) else np.nan
        dhi_o = float(oe_sim["dhi_sim"]) if pd.notna(oe_sim["dhi_sim"]) else np.nan
    else:
        ghi_o = bni_o = dhi_o = np.nan

    return pd.Series({
        "ghi_merra": ghi_m, "bni_merra": bni_m, "dhi_merra": dhi_m,
        "ghi_oe": ghi_o, "bni_oe": bni_o, "dhi_oe": dhi_o,
        "beta_oe": beta_oe, "alpha_oe": alpha_oe,
        "merra_ALPHA": alpha_m, "merra_BETA": beta_m,
        "merra_TO3": float(row["merra_TO3"]), "merra_TQV": float(row["merra_TQV"]),
    })


# =============================================================================
# End of libRadtran.py (Library)
# =============================================================================
