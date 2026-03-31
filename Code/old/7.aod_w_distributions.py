r"""
Distributions of **AOD at 550 nm** and **column water** :math:`w` (mm): MERRA-2 prior, AERONET
:math:`\tau_{550}`, and **TabPFN predictions** (default) or **physics retrievals** (optional).

**Default (``DATA_SOURCE=pred``)** — outputs of ``5.tabpfn.py``:

- ``PRED_LS`` — ``Data/<STATION>_<YEAR>_pred_ls_<suffix>.txt`` (``beta_pred_ls``, ``w_pred_ls``, …).
- ``PRED_OE`` — ``Data/<STATION>_<YEAR>_pred_oe_<suffix>.txt`` (``beta_pred_oe``, ``w_pred_oe``).

Merged on ``time_utc`` (test-pool predictions vs AERONET at those instants).

**Retrieval mode** (``DATA_SOURCE=retrieval``) — same layout using LHS + 4a/4b:

- ``LHS_TABLE``, ``RET_LS``, ``RET_OE`` (``beta_ls``/``w_ls``, ``beta_oe``/``w_oe``).

:math:`\tau_{550}` uses the Ångström law with ``merra_ALPHA`` and :math:`\beta` (retrieved or TabPFN):

    \tau(550\,\mathrm{nm}) = \beta\,(0.55/\lambda_{\mathrm{ref}})^{-\alpha}

MERRA prior uses ``merra_BETA`` / ``merra_ALPHA``; column water from ``merra_TQV`` (scaled to mm PW).

For the AOD panel, samples with :math:`\tau_{550}` **>** ``AOD_X_MAX`` are **dropped** for every
series (including AERONET). The **w** panel uses MERRA :math:`w` and retrieved ``w_ls``, ``w_oe``
(no AERONET :math:`w`).

**Second row:** ``facet_wrap`` of 1:1 scatters — AERONET :math:`\tau_{550}` (:math:`x`) vs MERRA prior,
LS, and OE :math:`\tau_{550}` (:math:`y`). Metrics (finite pairs, :math:`O` = AERONET, :math:`P` =
model): **MBE** = :math:`\overline{P-O}`; **RMSE** = :math:`\sqrt{\overline{(P-O)^2}}`; **FB**;
**FGE** (fractional gross error).

Env:
    DATA_SOURCE — ``pred`` (default, TabPFN ``pred_*.txt``) or ``retrieval`` (LHS + train_ls/oe).
    STATION, YEAR, LHS_N — default paths (same suffix rules as ``5.tabpfn.py``).
    PRED_LS, PRED_OE — override TabPFN outputs.
    LHS_TABLE, RET_LS, RET_OE — override retrieval inputs (retrieval mode only).
    OUTPUT_PNG — default ``tex/figures/aod_w_distributions.png``
    ANGSTROM_BETA_REF_UM — :math:`\beta` reference wavelength in µm (default **1.0**)
    DENSITY_N, RIBBON_ALPHA, AOD_X_MAX, DENSITY_LINE_WIDTH, PLOT_WIDTH_MM, PLOT_HEIGHT_MM
    SCATTER_PLOT_HEIGHT_MM — height of the scatter row (default **85**, three facets)
    PLOT_SHOW, PLOT_OPEN

Styling: ``.agents/SKILL.md`` (Times New Roman, 8 pt, 0.3 pt grid, Wong palette, 160 mm width).

Requires: pandas, numpy, plotnine, matplotlib.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from plotnine import (
    aes,
    after_stat,
    as_labeller,
    coord_fixed,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_abline,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    stat_density,
    theme,
)
from plotnine.themes.elements.margin import margin

from libRadtran import DEFAULT_CLEARSKY_CONFIG

PROJECT = Path(__file__).resolve().parent.parent

_PT = 8
_LW = 0.3
_DENSITY_LW = float(os.environ.get("DENSITY_LINE_WIDTH", "0.12"))

# Wong: MERRA, AERONET, LS, OE (AOD has four; W uses first, third, fourth)
COLOR_MERRA = "#E69F00"
COLOR_AERONET = "#56B4E9"
COLOR_LS = "#009E73"
COLOR_OE = "#CC79A7"

SRC_MERRA = "MERRA-2 prior"
SRC_AERONET = r"AERONET $\tau_{550}$"
LABEL_TABPFN_LS = "TabPFN (LS)"
LABEL_TABPFN_OE = "TabPFN (OE)"
LABEL_RETR_LS = "Retrieved (LS)"
LABEL_RETR_OE = "Retrieved (OE)"

_COLORS_AOD = [COLOR_MERRA, COLOR_AERONET, COLOR_LS, COLOR_OE]

_COLORS_SCATTER = [COLOR_MERRA, COLOR_LS, COLOR_OE]

PANEL_AOD = "aod"
PANEL_W = "w"
_PANEL_ORDER = [PANEL_AOD, PANEL_W]

LAM550_UM = 0.55
_PW_SCALE = DEFAULT_CLEARSKY_CONFIG.tqv_bsrn_to_mm_pw

LAMBDA_REF_UM = float(os.environ.get("ANGSTROM_BETA_REF_UM", "1.0"))
DENSITY_N = int(os.environ.get("DENSITY_N", "512"))
RIBBON_ALPHA = float(os.environ.get("RIBBON_ALPHA", "0.18"))
AOD_X_MAX = float(os.environ.get("AOD_X_MAX", "0.6"))
FIG_W_MM = float(os.environ.get("PLOT_WIDTH_MM", "160"))
FIG_H_MM = float(os.environ.get("PLOT_HEIGHT_MM", "80"))
FIG_H_SCATTER_MM = float(os.environ.get("SCATTER_PLOT_HEIGHT_MM", "85"))

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

_DEFAULT_LHS = PROJECT / "Data" / f"{STATION}_{YEAR}_train{_k_suffix}.txt"
_DEFAULT_LS = PROJECT / "Data" / f"{STATION}_{YEAR}_train_ls{_k_suffix}.txt"
_DEFAULT_OE = PROJECT / "Data" / f"{STATION}_{YEAR}_train_oe{_k_suffix}.txt"
_DEFAULT_PRED_LS = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_ls{_k_suffix}.txt"
_DEFAULT_PRED_OE = PROJECT / "Data" / f"{STATION}_{YEAR}_pred_oe{_k_suffix}.txt"

LHS_TABLE = Path(os.environ.get("LHS_TABLE", str(_DEFAULT_LHS)))
RET_LS = Path(os.environ.get("RET_LS", str(_DEFAULT_LS)))
RET_OE = Path(os.environ.get("RET_OE", str(_DEFAULT_OE)))
PRED_LS = Path(os.environ.get("PRED_LS", str(_DEFAULT_PRED_LS)))
PRED_OE = Path(os.environ.get("PRED_OE", str(_DEFAULT_PRED_OE)))
OUTPUT_PNG = Path(
    os.environ.get(
        "OUTPUT_PNG",
        str(PROJECT / "tex" / "figures" / "aod_w_distributions.png"),
    )
)


def aod550_angstrom(beta: np.ndarray, alpha: np.ndarray, lambda_ref_um: float) -> np.ndarray:
    """τ(550 nm) from Ångström law; λ_ref in µm."""
    return beta * (LAM550_UM / lambda_ref_um) ** (-alpha)


def aod_metrics_vs_ref(o: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Compare model :math:`P` to reference :math:`O` (AERONET). All finite pairs."""
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if len(o) == 0:
        return {"n": 0.0, "mbe": np.nan, "rmse": np.nan, "fb": np.nan, "fge": np.nan}
    mo, mp = float(np.mean(o)), float(np.mean(p))
    denom = mo + mp
    mbe = float(np.mean(p - o))
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    fb = float(2.0 * (mo - mp) / denom) if denom > 0 else float("nan")
    fge = float(2.0 * np.mean(np.abs(o - p)) / denom) if denom > 0 else float("nan")
    return {"n": float(len(o)), "mbe": mbe, "rmse": rmse, "fb": fb, "fge": fge}


def _metrics_text_block(m: dict[str, float]) -> str:
    """Multiline annotation for ``geom_text``."""
    return (
        f"$n$ = {int(m['n'])}\n"
        f"MBE = {m['mbe']:.4f}\n"
        f"RMSE = {m['rmse']:.4f}\n"
        f"FB = {m['fb']:.4f}\n"
        f"FGE = {m['fge']:.4f}"
    )


def main() -> None:
    _ds = os.environ.get("DATA_SOURCE", "pred").strip().lower()
    use_retrieval = _ds in ("retrieval", "ret", "train", "4a", "4b", "ls_oe")

    if use_retrieval:
        for pth, label in ((LHS_TABLE, "LHS_TABLE"), (RET_LS, "RET_LS"), (RET_OE, "RET_OE")):
            if not pth.is_file():
                print(f"ERROR: Missing {label}: {pth}", file=sys.stderr)
                sys.exit(1)
        src_ls = LABEL_RETR_LS
        src_oe = LABEL_RETR_OE
        need_lhs = [
            "time_utc",
            "merra_ALPHA",
            "merra_BETA",
            "merra_TQV",
            "aeronet_aod550",
        ]
        need_ls = ["time_utc", "beta_ls", "w_ls"]
        need_oe = ["time_utc", "beta_oe", "w_oe"]
        lhs = pd.read_csv(LHS_TABLE, sep="\t", comment="#", parse_dates=["time_utc"])
        ls = pd.read_csv(RET_LS, sep="\t", comment="#", parse_dates=["time_utc"])
        oe = pd.read_csv(RET_OE, sep="\t", comment="#", parse_dates=["time_utc"])
        for c in need_lhs:
            if c not in lhs.columns:
                print(f"ERROR: Missing {c!r} in {LHS_TABLE}", file=sys.stderr)
                sys.exit(1)
        for c in need_ls:
            if c not in ls.columns:
                print(f"ERROR: Missing {c!r} in {RET_LS}", file=sys.stderr)
                sys.exit(1)
        for c in need_oe:
            if c not in oe.columns:
                print(f"ERROR: Missing {c!r} in {RET_OE}", file=sys.stderr)
                sys.exit(1)
        df = lhs[need_lhs].merge(ls[need_ls], on="time_utc", how="inner").merge(
            oe[need_oe], on="time_utc", how="inner"
        )
        df = df.dropna(
            subset=[
                "merra_ALPHA",
                "merra_BETA",
                "merra_TQV",
                "aeronet_aod550",
                "beta_ls",
                "w_ls",
                "beta_oe",
                "w_oe",
            ]
        )
        title_rows = "LHS rows (4a/4b retrievals)"
    else:
        for pth, label in ((PRED_LS, "PRED_LS"), (PRED_OE, "PRED_OE")):
            if not pth.is_file():
                print(f"ERROR: Missing {label}: {pth}", file=sys.stderr)
                sys.exit(1)
        src_ls = LABEL_TABPFN_LS
        src_oe = LABEL_TABPFN_OE
        need_pls = [
            "time_utc",
            "merra_ALPHA",
            "merra_BETA",
            "merra_TQV",
            "aeronet_aod550",
            "beta_pred_ls",
            "w_pred_ls",
        ]
        need_poe = ["time_utc", "beta_pred_oe", "w_pred_oe"]
        pls = pd.read_csv(PRED_LS, sep="\t", comment="#", parse_dates=["time_utc"])
        poe = pd.read_csv(PRED_OE, sep="\t", comment="#", parse_dates=["time_utc"])
        for c in need_pls:
            if c not in pls.columns:
                print(f"ERROR: Missing {c!r} in {PRED_LS}", file=sys.stderr)
                sys.exit(1)
        for c in need_poe:
            if c not in poe.columns:
                print(f"ERROR: Missing {c!r} in {PRED_OE}", file=sys.stderr)
                sys.exit(1)
        df = pls[need_pls].merge(poe[need_poe], on="time_utc", how="inner")
        df = df.rename(
            columns={
                "beta_pred_ls": "beta_ls",
                "w_pred_ls": "w_ls",
                "beta_pred_oe": "beta_oe",
                "w_pred_oe": "w_oe",
            }
        )
        df = df.dropna(
            subset=[
                "merra_ALPHA",
                "merra_BETA",
                "merra_TQV",
                "aeronet_aod550",
                "beta_ls",
                "w_ls",
                "beta_oe",
                "w_oe",
            ]
        )
        title_rows = "TabPFN pred rows (5.tabpfn)"

    if len(df) == 0:
        print("ERROR: No rows after merge and dropna.", file=sys.stderr)
        sys.exit(1)

    order_aod = [SRC_MERRA, SRC_AERONET, src_ls, src_oe]
    order_w = [SRC_MERRA, src_ls, src_oe]
    order_scatter = [SRC_MERRA, src_ls, src_oe]

    alpha = df["merra_ALPHA"].to_numpy(dtype=float)
    beta_m = df["merra_BETA"].to_numpy(dtype=float)
    beta_ls = df["beta_ls"].to_numpy(dtype=float)
    beta_oe = df["beta_oe"].to_numpy(dtype=float)

    aod_m = aod550_angstrom(beta_m, alpha, LAMBDA_REF_UM)
    aod_ae = df["aeronet_aod550"].to_numpy(dtype=float)
    aod_ls = aod550_angstrom(beta_ls, alpha, LAMBDA_REF_UM)
    aod_oe = aod550_angstrom(beta_oe, alpha, LAMBDA_REF_UM)

    w_m = df["merra_TQV"].to_numpy(dtype=float) * _PW_SCALE
    w_ls = df["w_ls"].to_numpy(dtype=float)
    w_oe = df["w_oe"].to_numpy(dtype=float)

    def _aod_keep(a: np.ndarray) -> np.ndarray:
        m = np.isfinite(a) & (a <= AOD_X_MAX)
        return a[m]

    parts: list[pd.DataFrame] = []
    for src, aod_v in (
        (SRC_MERRA, aod_m),
        (SRC_AERONET, aod_ae),
        (src_ls, aod_ls),
        (src_oe, aod_oe),
    ):
        parts.append(
            pd.DataFrame(
                {"panel": PANEL_AOD, "source": src, "value": _aod_keep(aod_v)}
            )
        )
    for src, w_v in (
        (SRC_MERRA, w_m),
        (src_ls, w_ls),
        (src_oe, w_oe),
    ):
        parts.append(
            pd.DataFrame({"panel": PANEL_W, "source": src, "value": w_v})
        )

    long_df = pd.concat(parts, ignore_index=True)
    long_df["panel"] = pd.Categorical(long_df["panel"], categories=_PANEL_ORDER, ordered=True)

    # Per-panel category order (avoid empty factor levels breaking scales)
    aod_df = long_df.loc[long_df["panel"] == PANEL_AOD].copy()
    w_df = long_df.loc[long_df["panel"] == PANEL_W].copy()
    aod_df["source"] = pd.Categorical(aod_df["source"], categories=order_aod, ordered=True)
    w_df["source"] = pd.Categorical(w_df["source"], categories=order_w, ordered=True)

    if len(aod_df) == 0:
        print("ERROR: No AOD samples after τ ≤ AOD_X_MAX filter.", file=sys.stderr)
        sys.exit(1)

    _facet_lbl = as_labeller(
        {
            PANEL_AOD: rf"AOD$_{{550}}$ ($\lambda_{{\mathrm{{ref}}}}$ = {LAMBDA_REF_UM:g} $\mu$m)",
            PANEL_W: r"Column water $w$ (mm)",
        }
    )

    _theme = theme(
        text=element_text(family="Times New Roman", size=_PT),
        axis_text=element_text(size=_PT),
        axis_title=element_text(size=_PT),
        plot_title=element_text(size=_PT),
        plot_subtitle=element_text(size=_PT),
        legend_text=element_text(size=_PT, margin=margin(t=0, r=0, b=0, l=0, unit="pt")),
        legend_title=element_blank(),
        strip_text=element_text(
            size=_PT,
            margin=margin(t=0.3, r=1, b=0.3, l=1, unit="pt"),
        ),
        panel_spacing_x=0.02,
        panel_grid_major=element_line(color="black", size=_LW, alpha=0.35),
        panel_grid_minor=element_blank(),
        plot_background=element_rect(fill="white", color="none"),
        panel_background=element_rect(fill="white", color="none"),
        panel_border=element_rect(fill="none", color="black", size=_LW),
        axis_line=element_blank(),
        axis_ticks=element_line(color="black", size=_LW),
        legend_position="bottom",
        legend_direction="horizontal",
        legend_margin=0,
        legend_box_margin=0,
        legend_box_spacing=0,
        legend_spacing=4,
        legend_key_spacing_x=4,
        legend_key_spacing_y=0,
        legend_key_size=8 * 1.2,
        legend_background=element_blank(),
        legend_key=element_rect(fill="none", color="none", size=0),
        legend_frame=element_blank(),
        strip_background=element_rect(fill="white", color="black", size=_LW),
    )

    _dens = {"position": "identity", "n": DENSITY_N}
    # Two manual scales so AOD and W legends show only relevant sources.
    p = (
        ggplot(long_df, aes(x="value", fill="source", color="source"))
        + stat_density(
            aes(ymin=0, ymax=after_stat("density")),
            data=aod_df,
            geom="ribbon",
            bounds=(0.0, AOD_X_MAX),
            alpha=RIBBON_ALPHA,
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(y=after_stat("density")),
            data=aod_df,
            geom="line",
            bounds=(0.0, AOD_X_MAX),
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(ymin=0, ymax=after_stat("density")),
            data=w_df,
            geom="ribbon",
            alpha=RIBBON_ALPHA,
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            aes(y=after_stat("density")),
            data=w_df,
            geom="line",
            size=_DENSITY_LW,
            **_dens,
        )
        + facet_wrap("~ panel", nrow=1, scales="free", labeller=_facet_lbl)
        + scale_fill_manual(
            values=dict(zip(order_aod, _COLORS_AOD, strict=True)),
            breaks=order_aod,
            name="",
        )
        + scale_color_manual(
            values=dict(zip(order_aod, _COLORS_AOD, strict=True)),
            breaks=order_aod,
            name="",
        )
        + labs(
            x="",
            y="Density",
            title=(
                f"$n = {len(df)}$ {title_rows}; "
                r"$\tau_{550}=\beta\,(0.55/\lambda_{\mathrm{ref}})^{-\alpha}$"
            ),
        )
        + _theme
    )

    met_m = aod_metrics_vs_ref(aod_ae, aod_m)
    met_ls = aod_metrics_vs_ref(aod_ae, aod_ls)
    met_oe = aod_metrics_vs_ref(aod_ae, aod_oe)
    for tag, met in (("MERRA", met_m), ("LS", met_ls), ("OE", met_oe)):
        print(
            f"AOD550 vs AERONET ({tag}): n={int(met['n'])}, MBE={met['mbe']:.6f}, "
            f"RMSE={met['rmse']:.6f}, FB={met['fb']:.6f}, FGE={met['fge']:.6f}"
        )

    hi = float(np.nanmax(np.concatenate([aod_ae, aod_m, aod_ls, aod_oe])) * 1.05)
    hi = max(hi, 0.05)
    n_row = len(aod_ae)
    scatter_long = pd.DataFrame(
        {
            "aeronet": np.tile(aod_ae, 3),
            "tau_model": np.concatenate([aod_m, aod_ls, aod_oe]),
            "model": np.repeat(order_scatter, n_row),
        }
    )
    scatter_long["model"] = pd.Categorical(
        scatter_long["model"], categories=order_scatter, ordered=True
    )
    label_df = pd.DataFrame(
        {
            "model": order_scatter,
            "tx": [hi * 0.04] * 3,
            "ty": [hi * 0.95] * 3,
            "txt": [_metrics_text_block(m) for m in (met_m, met_ls, met_oe)],
        }
    )
    label_df["model"] = pd.Categorical(label_df["model"], categories=order_scatter, ordered=True)

    _scatter_lab = as_labeller(
        {
            SRC_MERRA: "MERRA-2 prior",
            src_ls: src_ls,
            src_oe: src_oe,
        }
    )
    p_scatter = (
        ggplot(scatter_long, aes(x="aeronet", y="tau_model", color="model"))
        + geom_point(alpha=0.45, size=1.5)
        + geom_abline(intercept=0, slope=1, color="#737373", linetype="dashed", size=_LW)
        + geom_text(
            aes(x="tx", y="ty", label="txt"),
            data=label_df,
            ha="left",
            va="top",
            size=8,
            color="black",
        )
        + facet_wrap("~ model", nrow=1, ncol=3, labeller=_scatter_lab, scales="fixed")
        + scale_color_manual(values=dict(zip(order_scatter, _COLORS_SCATTER, strict=True)))
        + scale_x_continuous(limits=(0.0, hi), expand=(0, 0))
        + scale_y_continuous(limits=(0.0, hi), expand=(0, 0))
        + coord_fixed(ratio=1)
        + labs(
            x=r"AERONET $\tau_{550}$",
            y=r"Model $\tau_{550}$",
            title="",
        )
        + _theme
        + theme(legend_position="none", panel_spacing_x=0.02)
    )

    buf = io.BytesIO()
    p.save(
        buf,
        format="png",
        dpi=300,
        width=FIG_W_MM / 25.4,
        height=FIG_H_MM / 25.4,
        units="in",
        verbose=False,
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    img = plt.imread(buf)

    buf_s = io.BytesIO()
    p_scatter.save(
        buf_s,
        format="png",
        dpi=300,
        width=FIG_W_MM / 25.4,
        height=FIG_H_SCATTER_MM / 25.4,
        units="in",
        verbose=False,
        facecolor="white",
        edgecolor="none",
    )
    buf_s.seek(0)
    img_s = plt.imread(buf_s)

    w_in = FIG_W_MM / 25.4
    h_top = FIG_H_MM / 25.4
    h_bot = FIG_H_SCATTER_MM / 25.4
    fig = plt.figure(figsize=(w_in, h_top + h_bot + 0.25), dpi=300, facecolor="white")
    gs_outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[FIG_H_MM, FIG_H_SCATTER_MM],
        hspace=0.08,
    )
    ax_img = fig.add_subplot(gs_outer[0])
    ax_img.imshow(img, aspect="auto")
    ax_img.axis("off")
    ax_s = fig.add_subplot(gs_outer[1])
    ax_s.imshow(img_s, aspect="auto")
    ax_s.axis("off")

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(OUTPUT_PNG),
        dpi=300,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(
        f"Wrote: {OUTPUT_PNG} (DATA_SOURCE={'retrieval' if use_retrieval else 'pred'}, "
        f"n={len(df)}, density + AOD scatter; τ≤{AOD_X_MAX} on density panel, "
        f"ANGSTROM_BETA_REF_UM={LAMBDA_REF_UM}, RIBBON_ALPHA={RIBBON_ALPHA})"
    )

    _env_truthy = {"1", "true", "yes"}
    if os.environ.get("PLOT_SHOW", "").strip().lower() in _env_truthy:
        print(p)
        print(p_scatter)

    if os.environ.get("PLOT_OPEN", "").strip().lower() in _env_truthy and sys.platform == "darwin":
        subprocess.run(["open", str(OUTPUT_PNG)], check=False)


if __name__ == "__main__":
    main()
