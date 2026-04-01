r"""
8.testing_analysis: error heatmap vs solar zenith on the **test** combined table.

**Motivation:** broadband retrieval sensitivity often varies with solar path length. A 2-D histogram
(heatmap) of how often each (zenith, error) bin occurs highlights blind spots (e.g. low sun vs noon)
and shows whether TabPFN stays stable across BSRN daytime geometry.

**Input:** ``TEST_COMBINED`` — default ``Data/<STATION>_<YEAR>_test_combined<suffix>.txt`` from step 6
(same ``LHS_N`` / ``_0.5k`` convention as other scripts). Requires ``zenith``, ``aeronet_aod550``,
``beta_pred_ls``, ``alpha_pred_ls``, ``beta_pred_oe``, ``alpha_pred_oe``.

**Fields:** X-axis = solar zenith angle :math:`\theta_z` (degrees, from the table). Y-axis =
retrieval error :math:`\Delta\tau_{550} = \tau_{550}^{\mathrm{pred}} - \tau_{550}^{\mathrm{AERONET}}`
with :math:`\tau_{550}` from Ångström :math:`\beta,\alpha` at reference wavelength (``ANGSTROM_BETA_REF_UM``).

**Output:** single figure with two panels (TabPFN LS and TabPFN OE, or retrieval labels if
``USE_TABPFN=0``), Viridis count density, vector PDF default.

Example:
    /opt/anaconda3/bin/python Code/8.testing_analysis.py
    TEST_COMBINED=Data/PAL_2024_test_combined_0.5k.txt /opt/anaconda3/bin/python Code/8.testing_analysis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

USE_TABPFN = os.environ.get("USE_TABPFN", "1").strip().lower() in {"1", "true", "yes"}
STATION = os.environ.get("STATION", "PAL")
YEAR = int(os.environ.get("YEAR", "2024"))
LHS_N = int(os.environ.get("LHS_N", "500"))
_n = LHS_N
_k_suffix = "_0.5k" if _n == 500 else f"_{_n / 1000:g}k" if _n >= 1000 else f"_{_n}"

LAM550_UM = 0.55
LAMBDA_REF_UM = float(os.environ.get("ANGSTROM_BETA_REF_UM", "1.0"))

_DEFAULT_COMBINED = PROJECT / "Data" / f"{STATION}_{YEAR}_test_combined{_k_suffix}.txt"
TEST_COMBINED = Path(os.environ.get("TEST_COMBINED", str(_DEFAULT_COMBINED)))

FIG_W_MM = float(os.environ.get("FIG_W_MM", "160"))
FIG_H_MM = float(os.environ.get("FIG_H_MM", "72"))
_PT = float(os.environ.get("FIG_PT", "8"))
_LW = 0.3

# Binning (override with env if needed)
ZENITH_BINS = int(os.environ.get("ZENITH_BINS", "28"))
ERROR_BINS = int(os.environ.get("ERROR_BINS", "40"))
# Symmetric |Δτ| limit at this percentile of pooled |errors| (after dropna)
ERROR_ABS_PERCENTILE = float(os.environ.get("ERROR_ABS_PERCENTILE", "99"))

_DEFAULT_FIG = PROJECT / "tex" / "figures" / f"{STATION}_{YEAR}_test_error_heatmap{_k_suffix}.pdf"
OUTPUT_FIG = Path(os.environ.get("OUTPUT_FIG", str(_DEFAULT_FIG)))

LABEL_LS = "TabPFN LS" if USE_TABPFN else "LS retrieval"
LABEL_OE = "TabPFN OE" if USE_TABPFN else "OE retrieval"


def aod550_angstrom(beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """τ₅₅₀ from Ångström β at ``LAMBDA_REF_UM`` and α (same as ``7.*`` / ``old/7``)."""
    return beta * (LAM550_UM / LAMBDA_REF_UM) ** (-alpha)


def _load_df(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"Missing input: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, sep="\t", parse_dates=["time_utc"])
    need = (
        "zenith",
        "aeronet_aod550",
        "beta_pred_ls",
        "alpha_pred_ls",
        "beta_pred_oe",
        "alpha_pred_oe",
    )
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"Columns missing in {path}: {miss}", file=sys.stderr)
        sys.exit(1)
    return df


def main() -> None:
    df = _load_df(TEST_COMBINED)
    z = df["zenith"].to_numpy(dtype=float)
    tau_ae = df["aeronet_aod550"].to_numpy(dtype=float)
    tau_ls = aod550_angstrom(
        df["beta_pred_ls"].to_numpy(dtype=float),
        df["alpha_pred_ls"].to_numpy(dtype=float),
    )
    tau_oe = aod550_angstrom(
        df["beta_pred_oe"].to_numpy(dtype=float),
        df["alpha_pred_oe"].to_numpy(dtype=float),
    )
    d_ls = tau_ls - tau_ae
    d_oe = tau_oe - tau_ae

    ok = np.isfinite(z) & np.isfinite(tau_ae) & np.isfinite(tau_ls) & np.isfinite(tau_oe)
    z, d_ls, d_oe = z[ok], d_ls[ok], d_oe[ok]
    if z.size == 0:
        print("No valid rows after filtering.", file=sys.stderr)
        sys.exit(1)

    lim = float(np.nanpercentile(np.abs(np.concatenate([d_ls, d_oe])), ERROR_ABS_PERCENTILE))
    lim = max(lim, 1e-4)
    y_edges = np.linspace(-lim, lim, ERROR_BINS + 1)
    z_lo = float(np.nanmin(z))
    z_hi = float(np.nanmax(z))
    x_edges = np.linspace(z_lo, z_hi, ZENITH_BINS + 1)

    H_ls, _, _ = np.histogram2d(z, d_ls, bins=[x_edges, y_edges])
    H_oe, _, _ = np.histogram2d(z, d_oe, bins=[x_edges, y_edges])
    vmax = max(float(H_ls.max()), float(H_oe.max()), 1.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    plt.rcParams.update(
        {
            "font.size": _PT,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": _LW,
            "figure.dpi": 150,
        }
    )

    w_in = FIG_W_MM / 25.4
    h_in = FIG_H_MM / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(w_in, h_in), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.88, bottom=0.18, wspace=0.12)

    cmap = plt.cm.viridis
    panels = [(H_ls, LABEL_LS), (H_oe, LABEL_OE)]
    ims = []
    for ax, (H, title) in zip(axes, panels):
        # histogram2d: first dim x (zenith), second dim y (error); pcolormesh wants H.T
        pc = ax.pcolormesh(
            x_edges,
            y_edges,
            H.T,
            cmap=cmap,
            norm=norm,
            shading="flat",
            linewidth=0,
            edgecolors="none",
        )
        ims.append(pc)
        ax.set_xlabel(r"Solar zenith angle $\theta_z$ (°)")
        ax.set_title(title)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.axhline(0.0, color="white", linewidth=0.6, linestyle="--", alpha=0.85)
        for spine in ax.spines.values():
            spine.set_linewidth(_LW)

    axes[0].set_ylabel(r"Retrieval error $\Delta\tau_{550}$")

    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Count")
    cbar.ax.tick_params(labelsize=_PT - 1)

    fig.suptitle(
        f"Test set — error vs solar geometry ({STATION} {YEAR}, n={len(z):d})",
        fontsize=_PT + 1,
        y=0.98,
    )

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_FIG,
        format=OUTPUT_FIG.suffix.lower().lstrip(".") or "pdf",
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"Wrote: {OUTPUT_FIG}  (n={len(z)}, |Δτ| limit ≈ ±{lim:.4f} at p{ERROR_ABS_PERCENTILE:g})")


if __name__ == "__main__":
    main()
