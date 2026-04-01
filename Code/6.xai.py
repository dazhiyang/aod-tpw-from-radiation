"""
7.xai: SHAP-style feature attributions for **TabPFN** (same setup as ``5.tabpfn.py``).

Uses `tabpfn_extensions.interpretability.shap.get_shap_values` with the local
``tabpfn.TabPFNRegressor`` (one model per target: ``beta_{MODE}``, ``alpha_{MODE}``). SHAP is
**expensive** (many model evaluations); by default ``N_SHAP`` rows from the test pool are explained.

**Dependency (not in base TabPFN):**

    pip install 'tabpfn-extensions[interpretability]'

**GPU** is used when available (same logic as step 5). CPU-only runs are valid but slow.

**Outputs:** per target, PDF + CSV under ``OUTPUT_DIR`` (default ``tex/figures``). Long tidy tables for
R go to ``SHAP_TXT_DIR`` (default ``Data/``) with short names:
``<STATION>_<YEAR>_shap_<MODE>_<beta|alpha><k_suffix>.txt``.

**Overrides:** ``TRAIN_IN``, ``TEST_POOL``, ``STATION``, ``YEAR``, ``LHS_N``, ``MODE``, ``N_SHAP``,
``N_TEST``, ``OUTPUT_DIR``, ``SHAP_TXT_DIR``, ``SHAP_SEED``. Optional: ``TARGETS`` = comma list.

Example::

    /opt/anaconda3/bin/python Code/7.xai.py
    MODE=oe N_SHAP=50 /opt/anaconda3/bin/python Code/7.xai.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

PROJECT = Path(__file__).resolve().parent.parent

if importlib.util.find_spec("tabpfn_extensions") is None:
    print(
        "Missing package: install interpretability extras, e.g.\n"
        "  pip install 'tabpfn-extensions[interpretability]'",
        file=sys.stderr,
    )
    sys.exit(1)

from tabpfn_extensions.interpretability.shap import get_shap_values  # noqa: E402

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

MODE = os.environ.get("MODE", "oe").lower()
if MODE not in ("ls", "oe"):
    print(f"ERROR: MODE must be ls or oe, got {MODE!r}", file=sys.stderr)
    sys.exit(1)

N_TEST = int(os.environ.get("N_TEST", "5000"))
N_SHAP = int(os.environ.get("N_SHAP", "100"))
SHAP_SEED = int(os.environ.get("SHAP_SEED", "42"))

_DEFAULT_TRAIN = PROJECT / "Data" / f"{STATION}_{YEAR}_train_{MODE}{_k_suffix}.txt"
TRAIN_IN = Path(os.environ.get("TRAIN_IN", str(_DEFAULT_TRAIN)))
_DEFAULT_TEST = PROJECT / "Data" / f"{STATION}_{YEAR}_testpool.txt"
TEST_POOL = Path(os.environ.get("TEST_POOL", str(_DEFAULT_TEST)))

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(PROJECT / "tex" / "figures")))
SHAP_TXT_DIR = Path(os.environ.get("SHAP_TXT_DIR", str(PROJECT / "Data")))

_CLEAR = ("ghi_clear", "bni_clear", "dhi_clear")
_MEAS = ("ghi", "bni", "dhi")
FEATURES = [
    *_MEAS,
    "zenith",
    "merra_ALPHA",
    "merra_BETA",
    "merra_ALBEDO",
    "merra_TQV",
    "merra_TO3",
    "merra_PS",
]
ALL_TARGETS = [f"beta_{MODE}", f"alpha_{MODE}"]

_tenv = os.environ.get("TARGETS", "").strip()
if _tenv:
    TARGETS = [t.strip() for t in _tenv.split(",") if t.strip()]
    bad = [t for t in TARGETS if t not in ALL_TARGETS]
    if bad:
        print(f"ERROR: TARGETS must be subset of {ALL_TARGETS}, got {bad}", file=sys.stderr)
        sys.exit(1)
else:
    TARGETS = ALL_TARGETS

_FIG_PT = float(os.environ.get("FIG_PT", "8"))
_FIG_W_MM = float(os.environ.get("FIG_W_MM", "160"))
_FIG_H_MM = float(os.environ.get("FIG_H_MM", "100"))


def _add_transmittance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ghi_trans"] = out["ghi"] / out["ghi_clear"]
    out["bni_trans"] = out["bni"] / out["bni_clear"]
    out["dhi_trans"] = out["dhi"] / out["dhi_clear"]
    return out


def _load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_IN.is_file():
        print(f"ERROR: Missing training data: {TRAIN_IN}", file=sys.stderr)
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_IN, sep="\t")
    for c in _CLEAR:
        if c not in train_df.columns:
            print(f"ERROR: Missing column {c!r} in {TRAIN_IN}", file=sys.stderr)
            sys.exit(1)
    train_df = _add_transmittance(train_df)
    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    if not TEST_POOL.is_file():
        print(f"ERROR: Missing test pool: {TEST_POOL}", file=sys.stderr)
        sys.exit(1)
    test_df = pd.read_csv(TEST_POOL, sep="\t", comment="#")
    for c in _CLEAR:
        if c not in test_df.columns:
            print(f"ERROR: Missing column {c!r} in {TEST_POOL}", file=sys.stderr)
            sys.exit(1)
    test_df = _add_transmittance(test_df)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    if len(test_df) > N_TEST:
        test_df = test_df.sample(n=N_TEST, random_state=SHAP_SEED).copy()

    train_df = train_df.dropna(subset=FEATURES + ALL_TARGETS)
    test_df = test_df.dropna(subset=FEATURES)
    return train_df, test_df


def _shap_to_array(raw: object) -> np.ndarray:
    """``get_shap_values`` may return ``numpy.ndarray`` or ``shap.Explanation``."""
    import shap

    if isinstance(raw, shap.Explanation):
        return np.asarray(raw.values)
    return np.asarray(raw, dtype=float)


class _NamedFeaturePredictor:
    """Adapter that guarantees DataFrame input with training feature names/order."""

    def __init__(self, model: TabPFNRegressor, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = list(feature_names)

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            x_df = x.reindex(columns=self.feature_names)
        else:
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            x_df = pd.DataFrame(arr, columns=self.feature_names)
        return np.asarray(self.model.predict(x_df), dtype=float)


def _save_summary(shap_vals: np.ndarray, x_df: pd.DataFrame, title: str, out_path: Path) -> None:
    import shap

    w_in = _FIG_W_MM / 25.4
    h_in = _FIG_H_MM / 25.4
    plt.rcParams.update(
        {
            "font.size": _FIG_PT,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "figure.dpi": 150,
        }
    )
    exp = shap.Explanation(
        values=shap_vals,
        data=x_df.to_numpy(dtype=float),
        feature_names=list(FEATURES),
    )
    shap.plots.beeswarm(exp, max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    fig.set_size_inches(w_in, h_in)
    if fig.axes:
        fig.axes[0].set_title(title, fontsize=_FIG_PT + 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.05,
    )
    plt.close(fig)


def _write_shap_long_txt(
    target: str,
    arr: np.ndarray,
    x_shap: pd.DataFrame,
    test_indices: np.ndarray,
    meta_df: pd.DataFrame | None,
    out_path: Path,
) -> None:
    """Tidy table for R: one row per (sample, feature)."""
    n_s, n_f = arr.shape
    rows: list[dict] = []
    for i in range(n_s):
        tpi = int(test_indices[i])
        row_base: dict = {
            "target": target,
            "sample_index": i,
            "test_pool_index": tpi,
        }
        if meta_df is not None and len(meta_df) == n_s:
            for col in meta_df.columns:
                row_base[col] = meta_df.iloc[i][col]
        for j, feat in enumerate(FEATURES):
            r = dict(row_base)
            r["feature"] = feat
            r["shap_value"] = float(arr[i, j])
            r["feature_value"] = float(x_shap.iloc[i, j])
            rows.append(r)
    long_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, sep="\t", index=False, float_format="%.8g")
    print(f"Wrote: {out_path}")


def _target_short_label(full: str) -> str:
    """``beta_ls`` / ``beta_oe`` → ``beta``; ``alpha_*`` → ``alpha``."""
    if full.startswith("beta_"):
        return "beta"
    if full.startswith("alpha_"):
        return "alpha"
    return full.split("_", 1)[0]


def main() -> None:
    train_df, test_df = _load_train_test()
    if len(test_df) < N_SHAP:
        print(f"WARNING: test rows ({len(test_df)}) < N_SHAP ({N_SHAP}); using all.", file=sys.stderr)
        n_shap = len(test_df)
    else:
        n_shap = N_SHAP

    rng = np.random.default_rng(SHAP_SEED)
    idx = np.sort(rng.choice(len(test_df), size=n_shap, replace=False))
    x_shap = test_df.iloc[idx][FEATURES].copy()
    meta_cols = [c for c in ("time_utc",) if c in test_df.columns]
    meta_shap = test_df.iloc[idx][meta_cols].reset_index(drop=True) if meta_cols else None

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}  |  train n={len(train_df)}  |  SHAP rows={n_shap}  |  targets={TARGETS}")

    x_train = train_df[FEATURES]

    for target in TARGETS:
        print(f"--- SHAP for target: {target}")
        y_tr = train_df[target]
        model = TabPFNRegressor(device=device)
        model.fit(x_train, y_tr)
        model_for_shap = _NamedFeaturePredictor(model, FEATURES)

        shap_kw: dict = {}
        algo = os.environ.get("SHAP_ALGORITHM", "").strip()
        if algo:
            shap_kw["algorithm"] = algo

        raw = get_shap_values(
            model_for_shap,
            x_shap,
            attribute_names=FEATURES,
            **shap_kw,
        )
        arr = _shap_to_array(raw)
        if arr.ndim != 2 or arr.shape[0] != n_shap:
            print(f"WARNING: unexpected SHAP shape {arr.shape}, expected ({n_shap}, n_features)", file=sys.stderr)

        stem = f"xai_shap_{STATION}_{YEAR}_{MODE}_{target}{_k_suffix}"
        out_pdf = OUTPUT_DIR / f"{stem}.pdf"
        title = f"SHAP ({STATION} {YEAR}, {MODE}) — {target}"
        _save_summary(arr, x_shap, title, out_pdf)
        print(f"Wrote: {out_pdf}")

        csv_path = OUTPUT_DIR / f"{stem}_mean_abs_shap.csv"
        mean_abs = np.mean(np.abs(arr), axis=0)
        pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs}).to_csv(
            csv_path, index=False
        )
        print(f"Wrote: {csv_path}")

        txt_name = f"{STATION}_{YEAR}_shap_{MODE}_{_target_short_label(target)}{_k_suffix}.txt"
        txt_path = SHAP_TXT_DIR / txt_name
        _write_shap_long_txt(target, arr, x_shap, idx, meta_shap, txt_path)


if __name__ == "__main__":
    main()
