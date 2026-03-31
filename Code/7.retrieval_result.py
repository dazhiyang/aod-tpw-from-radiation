r"""7.retrieval_result: densities of AOD₅₅₀ and α, then scatter vs AERONET.

**Default (retrieval):** merges LHS sample + ``train_ls`` + ``train_oe`` from steps 3–4.

**TabPFN (``USE_TABPFN=1``):** merges ``pred_ls`` + ``pred_oe`` from step 5 (same layout; TabPFN LS/OE
labels). Defaults: ``Data/<STATION>_<YEAR>_pred_ls<suffix>.txt`` and ``_pred_oe<suffix>.txt``.

Four sources: AERONET, MERRA-2, LS (retrieval or TabPFN), OE (retrieval or TabPFN).

Row 1: density of AOD₅₅₀ and α (two panels).
Row 2: AOD₅₅₀ scatter (AERONET x vs model y) for MERRA, LS, OE (three panels).
Row 3: Ångström α scatter for MERRA, LS, OE (three panels).

Usage:
    /opt/anaconda3/bin/python Code/7.retrieval_result.py
    USE_TABPFN=1 /opt/anaconda3/bin/python Code/7.retrieval_result.py

**Figure output:** defaults to **PDF** (three **vector** pages: densities, AOD scatters, α scatters)
via ``plotnine`` ``draw()`` + ``PdfPages`` — zoom stays sharp. ``.png`` uses a single **raster**
composite (not lossless zoom). Override path with ``OUTPUT_FIG`` / ``OUTPUT_PNG``.
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
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_abline,
    guides,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    stat_density,
    theme,
)
from plotnine.themes.elements.margin import margin

PROJECT = Path(__file__).resolve().parent.parent
USE_TABPFN = os.environ.get("USE_TABPFN", "").strip().lower() in {"1", "true", "yes"}

_PT = 8
_LW = 0.3
_DENSITY_LW = 0.12
# AERONET-only density curve (line, no ribbon): thicker stroke than other density lines.
DENSITY_AERONET_LW = 0.55

# Wong palette (.agents/SKILL.md order 1–3 for MERRA / LS / OE; black for AERONET only).
WONG_BLACK = "#000000"
WONG_1 = "#E69F00"  # orange — MERRA-2
WONG_2 = "#56B4E9"  # sky blue — LS
WONG_3 = "#009E73"  # bluish green — OE

COLOR_AERONET = WONG_BLACK
COLOR_MERRA = WONG_1
COLOR_LS = WONG_2
COLOR_OE = WONG_3

SRC_MERRA = "MERRA-2"
SRC_AERONET = "AERONET"
SRC_LS = "TabPFN LS" if USE_TABPFN else "LS retrieval"
SRC_OE = "TabPFN OE" if USE_TABPFN else "OE retrieval"
_ORDER = [SRC_MERRA, SRC_AERONET, SRC_LS, SRC_OE]
_COLORS = [COLOR_MERRA, COLOR_AERONET, COLOR_LS, COLOR_OE]
_DENSITY_COLOR_MAP = {
    SRC_MERRA: WONG_1,
    SRC_AERONET: WONG_BLACK,
    SRC_LS: WONG_2,
    SRC_OE: WONG_3,
}
_RES_ORDER = [SRC_MERRA, SRC_LS, SRC_OE]
_RES_COLORS = [WONG_1, WONG_2, WONG_3]

LAM550_UM = 0.55
LAMBDA_REF_UM = 1.0
DENSITY_N = 512
RIBBON_ALPHA = 0.18
AOD_X_MAX = 0.6

STATION = os.environ.get("STATION", "PAL")
YEAR = int(os.environ.get("YEAR", "2024"))
LHS_N = int(os.environ.get("LHS_N", "500"))
_n = LHS_N
_k_suffix = "_0.5k" if _n == 500 else f"_{_n / 1000:g}k" if _n >= 1000 else f"_{_n}"

LHS_TABLE = Path(
    os.environ.get("LHS_TABLE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train{_k_suffix}.txt"))
)
RET_LS = Path(
    os.environ.get("RET_LS", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train_ls{_k_suffix}.txt"))
)
RET_OE = Path(
    os.environ.get("RET_OE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_train_oe{_k_suffix}.txt"))
)
PRED_LS = Path(
    os.environ.get("PRED_LS", str(PROJECT / "Data" / f"{STATION}_{YEAR}_pred_ls{_k_suffix}.txt"))
)
PRED_OE = Path(
    os.environ.get("PRED_OE", str(PROJECT / "Data" / f"{STATION}_{YEAR}_pred_oe{_k_suffix}.txt"))
)
_DEFAULT_FIG = (
    "retrieval_result_distributions_tabpfn.pdf"
    if USE_TABPFN
    else "retrieval_result_distributions.pdf"
)
_default_fig_path = PROJECT / "tex" / "figures" / _DEFAULT_FIG
OUTPUT_FIG = Path(
    os.environ.get("OUTPUT_FIG", os.environ.get("OUTPUT_PNG", str(_default_fig_path)))
)


# Figure heights (mm); composite gap = GS_HSPACE (matplotlib fraction of subplot height).
FIG_H_MM = 62
FIG_H_SCATTER_MM = 64
GS_HSPACE = 0.012
FIG_COMPOSITE_PAD_IN = 0.04


def aod550_angstrom(beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """τ(550 nm) from Ångström law; λ_ref in µm."""
    return beta * (LAM550_UM / LAMBDA_REF_UM) ** (-alpha)


def metrics_vs_ref(o: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Compare model P to reference O (AERONET). All finite pairs."""
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


def _metrics_text(m: dict[str, float]) -> str:
    return (
        f"$n$ = {int(m['n'])}\n"
        f"MBE = {m['mbe']:.4f}\n"
        f"RMSE = {m['rmse']:.4f}\n"
        f"FB = {m['fb']:.4f}\n"
        f"FGE = {m['fge']:.4f}"
    )


def main() -> None:
    if USE_TABPFN:
        for pth, label in ((PRED_LS, "PRED_LS"), (PRED_OE, "PRED_OE")):
            if not pth.is_file():
                print(f"ERROR: Missing {label}: {pth}", file=sys.stderr)
                sys.exit(1)

        pls = pd.read_csv(PRED_LS, sep="\t", comment="#", parse_dates=["time_utc"])
        poe = pd.read_csv(PRED_OE, sep="\t", comment="#", parse_dates=["time_utc"])

        need_base = ["time_utc", "merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha"]
        need_pls = ["time_utc", "beta_pred_ls", "alpha_pred_ls"]
        need_poe = ["time_utc", "beta_pred_oe", "alpha_pred_oe"]
        for c in need_base:
            if c not in pls.columns:
                print(f"ERROR: Missing {c!r} in {PRED_LS}", file=sys.stderr)
                sys.exit(1)
        for c in need_pls:
            if c not in pls.columns:
                print(f"ERROR: Missing {c!r} in {PRED_LS}", file=sys.stderr)
                sys.exit(1)
        for c in need_poe:
            if c not in poe.columns:
                print(f"ERROR: Missing {c!r} in {PRED_OE}", file=sys.stderr)
                sys.exit(1)

        df = pls[need_base + ["beta_pred_ls", "alpha_pred_ls"]].merge(
            poe[need_poe], on="time_utc", how="inner"
        )
        df = df.dropna(
            subset=[
                "merra_ALPHA",
                "merra_BETA",
                "aeronet_aod550",
                "aeronet_alpha",
                "beta_pred_ls",
                "alpha_pred_ls",
                "beta_pred_oe",
                "alpha_pred_oe",
            ]
        )
        mode_tag = "TabPFN pred_ls + pred_oe"
        b_ls, a_ls, b_oe, a_oe = "beta_pred_ls", "alpha_pred_ls", "beta_pred_oe", "alpha_pred_oe"
    else:
        for pth, label in ((LHS_TABLE, "LHS_TABLE"), (RET_LS, "RET_LS"), (RET_OE, "RET_OE")):
            if not pth.is_file():
                print(f"ERROR: Missing {label}: {pth}", file=sys.stderr)
                sys.exit(1)

        lhs = pd.read_csv(LHS_TABLE, sep="\t", comment="#", parse_dates=["time_utc"])
        ls = pd.read_csv(RET_LS, sep="\t", comment="#", parse_dates=["time_utc"])
        oe = pd.read_csv(RET_OE, sep="\t", comment="#", parse_dates=["time_utc"])

        need_lhs = ["time_utc", "merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha"]
        need_ls = ["time_utc", "beta_ls", "alpha_ls"]
        need_oe = ["time_utc", "beta_oe", "alpha_oe"]
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

        df = (lhs[need_lhs]
              .merge(ls[need_ls], on="time_utc", how="inner")
              .merge(oe[need_oe], on="time_utc", how="inner"))
        df = df.dropna(
            subset=[
                "merra_ALPHA",
                "merra_BETA",
                "aeronet_aod550",
                "aeronet_alpha",
                "beta_ls",
                "alpha_ls",
                "beta_oe",
                "alpha_oe",
            ]
        )
        mode_tag = "LS + OE retrieval"
        b_ls, a_ls, b_oe, a_oe = "beta_ls", "alpha_ls", "beta_oe", "alpha_oe"

    if len(df) == 0:
        print("ERROR: No rows after merge and dropna.", file=sys.stderr)
        sys.exit(1)
    print(f"Merged: {len(df)} rows ({mode_tag}).")

    # --- Compute AOD550 ---
    aod_merra = aod550_angstrom(df["merra_BETA"].values, df["merra_ALPHA"].values)
    aod_aeronet = df["aeronet_aod550"].values.astype(float)
    aod_ls = aod550_angstrom(df[b_ls].values, df[a_ls].values)
    aod_oe = aod550_angstrom(df[b_oe].values, df[a_oe].values)

    # --- Alpha ---
    alpha_merra = df["merra_ALPHA"].values.astype(float)
    alpha_aeronet = df["aeronet_alpha"].values.astype(float)
    alpha_ls_vals = df[a_ls].values.astype(float)
    alpha_oe_vals = df[a_oe].values.astype(float)

    # --- Build long-form dataframe ---
    PANEL_AOD = "aod"
    PANEL_ALPHA = "alpha"
    _PANEL_ORDER = [PANEL_AOD, PANEL_ALPHA]

    def _aod_keep(a: np.ndarray) -> np.ndarray:
        m = np.isfinite(a) & (a <= AOD_X_MAX)
        return a[m]

    parts: list[pd.DataFrame] = []
    for src, vals in ((SRC_MERRA, aod_merra), (SRC_AERONET, aod_aeronet),
                      (SRC_LS, aod_ls), (SRC_OE, aod_oe)):
        parts.append(pd.DataFrame({"panel": PANEL_AOD, "source": src, "value": _aod_keep(vals)}))
    for src, vals in ((SRC_MERRA, alpha_merra), (SRC_AERONET, alpha_aeronet),
                      (SRC_LS, alpha_ls_vals), (SRC_OE, alpha_oe_vals)):
        finite = vals[np.isfinite(vals)]
        parts.append(pd.DataFrame({"panel": PANEL_ALPHA, "source": src, "value": finite}))

    long_df = pd.concat(parts, ignore_index=True)
    long_df["panel"] = pd.Categorical(long_df["panel"], categories=_PANEL_ORDER, ordered=True)
    long_df["source"] = pd.Categorical(long_df["source"], categories=_ORDER, ordered=True)

    _facet_lbl = as_labeller({
        PANEL_AOD: r"AOD$_{550}$",
        PANEL_ALPHA: r"Ångström $\alpha$",
    })

    _theme = theme(
        text=element_text(family="Times New Roman", size=_PT),
        axis_text=element_text(size=_PT),
        axis_title=element_text(size=_PT),
        plot_title=element_text(size=_PT),
        plot_subtitle=element_text(size=_PT),
        legend_text=element_text(
            size=_PT, margin=margin(t=0, r=0, b=0, l=0, unit="pt")
        ),
        legend_title=element_blank(),
        strip_text=element_text(
            size=_PT, margin=margin(t=0.3, r=1, b=0.3, l=1, unit="pt")
        ),
        panel_spacing_x=0.012,
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
    FIG_W_MM = 160

    # Ribbon + line for MERRA / LS / OE; line only (no ribbon) for AERONET.
    _ribbon_src = {SRC_MERRA, SRC_LS, SRC_OE}
    long_df_ribbon = long_df[long_df["source"].astype(str).isin(_ribbon_src)].copy()
    long_df_aeronet = long_df[long_df["source"] == SRC_AERONET].copy()

    p_density = (
        ggplot()
        + stat_density(
            long_df_ribbon,
            aes(
                x="value",
                ymin=0,
                ymax=after_stat("density"),
                fill="source",
                color="source",
            ),
            geom="ribbon",
            alpha=RIBBON_ALPHA,
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            long_df_ribbon,
            aes(x="value", y=after_stat("density"), color="source"),
            geom="line",
            size=_DENSITY_LW,
            **_dens,
        )
        + stat_density(
            long_df_aeronet,
            aes(x="value", y=after_stat("density"), color="source"),
            geom="line",
            size=DENSITY_AERONET_LW,
            **_dens,
        )
        + facet_wrap("~ panel", nrow=1, scales="free", labeller=_facet_lbl)
        + scale_fill_manual(values=_DENSITY_COLOR_MAP, breaks=_ORDER, name="")
        + scale_color_manual(values=_DENSITY_COLOR_MAP, breaks=_ORDER, name="")
        + guides(fill="none")
        + labs(x="", y="Density")
        + _theme
    )

    # --- Metrics ---
    met_aod_m = metrics_vs_ref(aod_aeronet, aod_merra)
    met_aod_ls = metrics_vs_ref(aod_aeronet, aod_ls)
    met_aod_oe = metrics_vs_ref(aod_aeronet, aod_oe)
    met_alpha_m = metrics_vs_ref(alpha_aeronet, alpha_merra)
    met_alpha_ls = metrics_vs_ref(alpha_aeronet, alpha_ls_vals)
    met_alpha_oe = metrics_vs_ref(alpha_aeronet, alpha_oe_vals)
    _ls_tag = "TabPFN LS" if USE_TABPFN else "LS"
    _oe_tag = "TabPFN OE" if USE_TABPFN else "OE"
    for tag, met in (
        ("AOD MERRA", met_aod_m),
        (f"AOD {_ls_tag}", met_aod_ls),
        (f"AOD {_oe_tag}", met_aod_oe),
        ("Alpha MERRA", met_alpha_m),
        (f"Alpha {_ls_tag}", met_alpha_ls),
        (f"Alpha {_oe_tag}", met_alpha_oe),
    ):
        print(
            f"{tag} vs AERONET: n={int(met['n'])}, MBE={met['mbe']:.6f}, "
            f"RMSE={met['rmse']:.6f}, FB={met['fb']:.6f}, FGE={met['fge']:.6f}"
        )

    # --- Scatter rows: AOD (middle), α (bottom); three facets each ---
    n_row = len(df)
    n_src = len(_RES_ORDER)
    hi_aod = max(float(np.nanmax(np.concatenate(
        [aod_aeronet, aod_merra, aod_ls, aod_oe])) * 1.05), 0.05)
    aod_scatter = pd.DataFrame({
        "aeronet": np.tile(aod_aeronet, n_src),
        "model": np.concatenate([aod_merra, aod_ls, aod_oe]),
        "source": np.repeat(_RES_ORDER, n_row),
    })
    aod_scatter["source"] = pd.Categorical(
        aod_scatter["source"], categories=_RES_ORDER, ordered=True
    )
    aod_label = pd.DataFrame({
        "source": _RES_ORDER,
        "tx": [hi_aod * 0.04] * n_src,
        "ty": [hi_aod * 0.95] * n_src,
        "txt": [_metrics_text(met_aod_m), _metrics_text(met_aod_ls), _metrics_text(met_aod_oe)],
    })
    aod_label["source"] = pd.Categorical(
        aod_label["source"], categories=_RES_ORDER, ordered=True
    )

    all_alpha = np.concatenate([alpha_aeronet, alpha_merra, alpha_ls_vals, alpha_oe_vals])
    hi_alpha = max(float(np.nanmax(all_alpha) * 1.05), 0.5)
    alpha_scatter = pd.DataFrame({
        "aeronet": np.tile(alpha_aeronet, n_src),
        "model": np.concatenate([alpha_merra, alpha_ls_vals, alpha_oe_vals]),
        "source": np.repeat(_RES_ORDER, n_row),
    })
    alpha_scatter["source"] = pd.Categorical(
        alpha_scatter["source"], categories=_RES_ORDER, ordered=True
    )
    alpha_label = pd.DataFrame({
        "source": _RES_ORDER,
        "tx": [hi_alpha * 0.04] * n_src,
        "ty": [hi_alpha * 0.95] * n_src,
        "txt": [_metrics_text(met_alpha_m), _metrics_text(met_alpha_ls),
                _metrics_text(met_alpha_oe)],
    })
    alpha_label["source"] = pd.Categorical(
        alpha_label["source"], categories=_RES_ORDER, ordered=True
    )

    _scatter_theme = _theme + theme(legend_position="none", panel_spacing_x=0.01)

    p_aod_scatter = (
        ggplot(aod_scatter, aes(x="aeronet", y="model", color="source"))
        + geom_point(alpha=0.45, size=1.5)
        + geom_abline(intercept=0, slope=1, color="#737373", linetype="dashed", size=_LW)
        + geom_text(
            aes(x="tx", y="ty", label="txt"),
            data=aod_label, ha="left", va="top", size=8, color="black",
        )
        + facet_wrap("~ source", nrow=1, scales="free")
        + scale_color_manual(values=dict(zip(_RES_ORDER, _RES_COLORS)))
        + labs(
            x="AERONET (reference)",
            y="Model",
            title=r"AOD$_{550}$",
        )
        + _scatter_theme
    )

    p_alpha_scatter = (
        ggplot(alpha_scatter, aes(x="aeronet", y="model", color="source"))
        + geom_point(alpha=0.45, size=1.5)
        + geom_abline(intercept=0, slope=1, color="#737373", linetype="dashed", size=_LW)
        + geom_text(
            aes(x="tx", y="ty", label="txt"),
            data=alpha_label, ha="left", va="top", size=8, color="black",
        )
        + facet_wrap("~ source", nrow=1, scales="free")
        + scale_color_manual(values=dict(zip(_RES_ORDER, _RES_COLORS)))
        + labs(
            x="AERONET (reference)",
            y="Model",
            title=r"Ångström $\alpha$",
        )
        + _scatter_theme
    )

    # --- Output: vector PDF (3 pages) or raster composite (PNG / other) ---
    _fmt = OUTPUT_FIG.suffix.lower().lstrip(".") or "pdf"
    if _fmt not in ("pdf", "png", "svg", "ps", "eps"):
        _fmt = "pdf"

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    w_in = FIG_W_MM / 25.4

    if _fmt == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(str(OUTPUT_FIG)) as pdf:
            for p, h_mm in (
                (p_density, FIG_H_MM),
                (p_aod_scatter, FIG_H_SCATTER_MM),
                (p_alpha_scatter, FIG_H_SCATTER_MM),
            ):
                h_in = h_mm / 25.4
                fig = (p + theme(figure_size=(w_in, h_in))).draw()
                fig.patch.set_facecolor("white")
                pdf.savefig(
                    fig,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    facecolor="white",
                    edgecolor="none",
                )
                plt.close(fig)
        print(f"Wrote: {OUTPUT_FIG}  (3-page vector PDF)")
    else:

        def _save_gg(p: ggplot, h_mm: float) -> np.ndarray:
            buf = io.BytesIO()
            p.save(
                buf,
                format="png",
                dpi=300,
                width=w_in,
                height=h_mm / 25.4,
                units="in",
                verbose=False,
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            return plt.imread(buf)

        img_d = _save_gg(p_density, FIG_H_MM)
        img_aod = _save_gg(p_aod_scatter, FIG_H_SCATTER_MM)
        img_alpha = _save_gg(p_alpha_scatter, FIG_H_SCATTER_MM)

        h_top = FIG_H_MM / 25.4
        h_mid = FIG_H_SCATTER_MM / 25.4
        h_bot = FIG_H_SCATTER_MM / 25.4
        fig = plt.figure(
            figsize=(w_in, h_top + h_mid + h_bot + FIG_COMPOSITE_PAD_IN),
            dpi=300,
            facecolor="white",
        )
        gs = GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=[FIG_H_MM, FIG_H_SCATTER_MM, FIG_H_SCATTER_MM],
            hspace=GS_HSPACE,
        )
        ax_top = fig.add_subplot(gs[0])
        ax_top.imshow(img_d, aspect="auto")
        ax_top.axis("off")
        ax_mid = fig.add_subplot(gs[1])
        ax_mid.imshow(img_aod, aspect="auto")
        ax_mid.axis("off")
        ax_bot = fig.add_subplot(gs[2])
        ax_bot.imshow(img_alpha, aspect="auto")
        ax_bot.axis("off")

        fig.savefig(
            str(OUTPUT_FIG),
            format=_fmt,
            dpi=300,
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
        print(f"Wrote: {OUTPUT_FIG}  (single-page raster composite)")

    if os.environ.get("PLOT_OPEN", "").strip().lower() in {"1", "true", "yes"}:
        if sys.platform == "darwin":
            subprocess.run(["open", str(OUTPUT_FIG)], check=False)


if __name__ == "__main__":
    main()
