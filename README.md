# AOD and Ångström Aerosol Retrieval from Surface Solar Radiation

Hybrid physical–ML pipeline for retrieving **Ångström turbidity (β)** and **Ångström exponent (α)** from clear-sky surface solar radiation (GHI, BNI, DHI), with **AOD₅₅₀** derived from (β, α). **MERRA-2** supplies priors and radiative-transfer inputs (including column water **TQV** used inside libRadtran); the current numbered TabPFN path predicts **β and α**, not a separate TPW state vector.

The pipeline combines **physics-based retrievals** and a **tabular ML** shortcut:

1. **[libRadtran](http://www.libradtran.org) retrievals** — same clear-sky stack on each row (`uvspec`, DISORT, Kato2 bands, Ångström aerosol), state **(β, α)**. Two variants in `Code/libRadtran.py`: **least-squares (LS)** — `4a.retrieval_ls.py`, `retrieve_one_row_ls` — and **optimal estimation (OE)** with MERRA priors on (β, α) — `4b.retrieval_oe.py`, `retrieve_one_row_oe`.
2. **[TabPFN](https://github.com/PriorLabs/TabPFN)** — trained on **LS- or OE-labeled** tables from **4a** / **4b** (`MODE=ls` / `MODE=oe`); at inference maps radiation + MERRA features to **β and α** without a per-row RT solve.

MERRA-2 supplies priors and ancillaries (ozone, surface pressure, surface albedo, column water, Ångström α and β as in the merged BSRN table).

## Pipeline

### Optional data helpers (before step 1)

| Step | Script | What it does |
|:----:|--------|--------------|
| 0a | `0a.aeronet.py` | AERONET Level 2.0 → 1-minute processed columns (PAL/TAT layout under `Data/AERONET/`). |
| 0b | `0b.bsrn.py` | BSRN FTP inventory / monthly downloads under `Data/BSRN/`. |

### Core chain (train TabPFN on retrievals, predict on holdout)

| Step | Script | What it does |
|:----:|--------|--------------|
| 1 | `1.arrange.py` | **PAL/TAT/…:** edit **CONFIG** (`STATION`, `YEAR`, paths); BSRN + MERRA + AERONET merge → `Data/<STATION>_<year>_all.txt`. **QIQ:** `Code/old/1.arrange.py` → `qiq_1min_merra_qc.txt`. |
| 2 | `2.create_holdout.py` | **PAL/TAT/…:** reads `_all.txt`; finite AERONET AOD/α; writes `_trainpool.txt` + `_testpool.txt` (50:50). **QIQ:** `Code/old/2.create_holdout.py` (no AERONET). |
| 3 | `3.latin_hypercube.py` | LHS sample (default 500) from train pool → `<STATION>_<YEAR>_train_0.5k.txt`. **QIQ:** `Code/old/3.latin_hypercube.py`. |
| 4a | `4a.retrieval_ls.py` | **LS:** libRadtran clear-sky forward model + least-squares inversion for (β, α) → `train_ls<suffix>.txt`. |
| 4b | `4b.retrieval_oe.py` | **OE:** libRadtran clear-sky forward model + optimal-estimation inversion for (β, α) with MERRA priors → `train_oe<suffix>.txt`. |
| 5 | `5.tabpfn.py` | TabPFN trained on **4a** or **4b** outputs; run twice (`MODE=ls`, `MODE=oe`) → `pred_ls<suffix>.txt`, `pred_oe<suffix>.txt` (predicted β/α on **test pool**). |

### After step 5 (optional analysis / figures)

These scripts assume the same `STATION`, `YEAR`, `LHS_N` / `_0.5k` naming as above.

| Script | Purpose |
|--------|---------|
| `6.xai.py` | TabPFN **SHAP** (needs `tabpfn-extensions[interpretability]`); writes tidy SHAP tables under `Data/` (e.g. `*_shap_oe_beta_0.5k.txt`) and figures under `tex/figures/`. Run with `MODE=ls` or `MODE=oe`. |
| `7.irradiance.py` | Clear-sky forward validation (GHI/BNI/DHI): **MERRA explicit**, **TabPFN OE**, **AERONET** aerosol. **Input:** default `pred_oe<suffix>.txt`. **Output:** `test_irradiance<suffix>.txt`. |
| `10.plot_retrieval_scatter.py` | Measured vs libRadtran forward **GHI/BNI/DHI** (plotnine). Prefers legacy `test_combined<suffix>.txt` if present; else `test_ls` / `test_oe` or `pred_*` tables — see script docstring. **`SKIP_LS=1`** (default) omits the **TabPFN LS** panel. |
| `11.train_analysis.py` | Train-side densities and AOD₅₅₀ / Ångström α scatters: physical **LS/OE** retrievals or `USE_TABPFN=1` with `pred_ls` + `pred_oe`. |
| `11.train_analysis.R` | R counterpart for train figures (match styling with step 11 Python if desired). |
| `12.test_analysis.py` | Composite **test** PDF: FGE violins, SHAP summaries, irradiance scatter (matplotlib + plotnine). Defaults: `pred_oe`, `train_oe`, SHAP files, `test_irradiance`. |
| `12.test_analysis.R` | R **patchwork** composite for the same story (e.g. `scattermore` on irradiance panel). |

**Note:** Older workflows used a bundled `6.evaluation.py` that wrote `test_combined*.txt` and a separate `7.retrieval_result.py`. Those files are **not** in this repository anymore. Use **`pred_ls` / `pred_oe`** from step 5, **`7.irradiance.py`** for flux validation, and **`12.test_analysis`** for the combined test figure. If you still have a legacy `test_combined*.txt`, `10.plot_retrieval_scatter.py` can read it via `PLOT_INPUT_COMBINED`.

**Legacy / archive:** older one-off scripts live under `Code/old/`.

### Data flow

```text
Raw BSRN + MERRA-2 (+ AERONET for PAL/TAT)
  └─[1]─► <STATION>_<year>_all.txt   (or QIQ: qiq_1min_merra_qc.txt via old/1.arrange)
            └─[2]─► <STATION>_<year>_trainpool.txt + <STATION>_<year>_testpool.txt
                      └─[3]─► <STATION>_<year>_train_0.5k.txt
                                └─[4a/4b]─► train_ls_{N}k.txt (LS), train_oe_{N}k.txt (OE) — retrieved (β, α)
                                          └─[5]─► pred_ls_{N}k.txt, pred_oe_{N}k.txt — TabPFN (β, α) on test pool
                                                    ├─► (optional) 6.xai.py → SHAP tables + figures
                                                    ├─► (optional) 7.irradiance.py → test_irradiance_{N}k.txt
                                                    └─► (optional) 10 / 11 / 12 → paper figures
```

## Core library

**`Code/libRadtran.py`** — shared, import-pure library containing:

- `ClearskyConfig` dataclass and `build_uvspec_input` for assembling libRadtran input decks.
- `run_clearsky` for calling the `uvspec` binary and parsing broadband fluxes.
- `merra_explicit_physics` for consistent MERRA scaling (O₃ DU, column water mm, Ångström β and α).
- `retrieve_one_row_ls` / `process_row_ls` — **LS** loop; `retrieve_one_row_oe` / `process_row_oe` — **OE** loop (both for β, α).

## Prerequisites

- **libRadtran 2.0.6** — the `uvspec` binary must be installed locally.
- **Python ≥ 3.10**

### Python dependencies

```text
numpy
pandas
scipy
matplotlib
scikit-learn
tabpfn
torch
tqdm
bsrn
plotnine          # scripts 10–12 (figures)
```

Optional:

```text
tabpfn-extensions[interpretability]   # 6.xai.py (SHAP)
```

For **`12.test_analysis.R`**: R packages `dplyr`, `tidyr`, `ggplot2`, `patchwork`, **`scattermore`**.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/dazhiyang/aod-tpw-from-radiation.git
cd aod-tpw-from-radiation
```

2. Use **Anaconda** Python (this repo is run with ``/opt/anaconda3`` on macOS). Put it on your ``PATH`` or call it explicitly:

```bash
export PATH="/opt/anaconda3/bin:$PATH"
# or: PY=/opt/anaconda3/bin/python
```

3. Install Python dependencies:

```bash
/opt/anaconda3/bin/pip install numpy pandas scipy matplotlib scikit-learn tabpfn torch tqdm bsrn plotnine
```

4. Point to your libRadtran installation (default `~/libRadtran-2.0.6`):

```bash
export LIBRADTRANDIR=/path/to/libRadtran-2.0.6
```

## Usage

Run steps **1–5** in order (step 1 requires raw BSRN data and is typically run once). Examples use Anaconda’s interpreter:

```bash
PY=/opt/anaconda3/bin/python
$PY Code/1.arrange.py
$PY Code/2.create_holdout.py
# QIQ holdout: $PY Code/old/2.create_holdout.py
# PAL/TAT step 5: TEST_POOL=Data/PAL_2024_testpool.txt
$PY Code/3.latin_hypercube.py
# QIQ LHS: $PY Code/old/3.latin_hypercube.py
# Step 4a with PAL file: INPUT_DATA=Data/PAL_2024_train_0.5k.txt $PY Code/4a.retrieval_ls.py
$PY Code/4a.retrieval_ls.py
$PY Code/4b.retrieval_oe.py
MODE=ls $PY Code/5.tabpfn.py
MODE=oe $PY Code/5.tabpfn.py

# Optional after step 5
# MODE=oe $PY Code/6.xai.py
# $PY Code/7.irradiance.py
# $PY Code/10.plot_retrieval_scatter.py
# USE_TABPFN=1 $PY Code/11.train_analysis.py
# $PY Code/12.test_analysis.py
# Rscript Code/12.test_analysis.R
```

### Environment variables (selected)

| Variable | Script(s) | Default / notes |
|----------|-----------|-----------------|
| `ARRANGE_OUTPUT` | 1 | `Data/qiq_1min_merra_qc.txt` (QIQ legacy path). |
| `ARRANGE_FIRST_MONTH_ONLY` | 1 | *(off)* Test subset. |
| `ARRANGE_ONE_FILE` | 1 | *(off)* Single input file. |
| `LHS_INPUT` | old/3 | `Data/trainpool.txt` |
| `LHS_TRAIN` | old/3 | `Data/train_0.5k.txt` |
| `LHS_N` | 3–5, 6–7, 10–12 | `500` → `_0.5k` suffix pattern. |
| `LHS_SEED` | old/3 | `42` |
| `LHS_ZENITH_MAX` | old/3 | `87` |
| `TEST_POOL` | 5, 6 | e.g. `Data/PAL_2024_testpool.txt` |
| `TRAIN_IN` | 5 | Default `train_{MODE}<suffix>.txt` from 4a/4b. |
| `PRED_OUT` | 5 | Default `pred_{MODE}<suffix>.txt` |
| `INPUT_DATA` | 4a, 4b | e.g. `Data/PAL_2024_train_0.5k.txt` |
| `OUTPUT_DATA` | 4a, 4b | `train_ls*.txt` / `train_oe*.txt` |
| `MODE` | 5, 6 | `ls` or `oe` |
| `N_TEST` | 5 | `5000` TabPFN prediction rows |
| `LRT_SEASONAL_ATMOSPHERE` | 4a, 4b, 7 | `1` default (month-based AFGL); set `0` to override. |
| `PRED_OE` | 7 | Default `pred_oe<suffix>.txt` |
| `VAL_OUT` | 7 | Default `test_irradiance<suffix>.txt` |
| `PRED_IN` | 7 | Single-table override instead of `PRED_OE` |
| `PLOT_INPUT_COMBINED` | 10 | Legacy `test_combined<suffix>.txt` if available |
| `SKIP_LS` | 10 | `1` = omit LS panel (default) |
| `USE_TABPFN` | 11 | `1` = use `pred_ls` + `pred_oe` |
| `TEST_COMBINED`, `TRAIN_OE`, `SHAP_*`, `IRRADIANCE_IN`, `OUTPUT_FIG` | 12 | See `12.test_analysis.py` / `.R` headers |
| `LIBRADTRANDIR` | lib | `~/libRadtran-2.0.6` |

## License

[MIT](LICENSE) — Copyright (c) 2026 Dazhi Yang
