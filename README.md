# AOD and TPW Retrieval from Surface Solar Radiation

Hybrid physical–ML pipeline for retrieving **Angstrom turbidity coefficient (beta)** and **total precipitable water (TPW, mm)** from clear-sky surface solar radiation measurements (GHI, BNI, DHI).

The pipeline combines two complementary approaches:

1. **Least-squares (LS) retrieval** — physics-based inversion using the [libRadtran](http://www.libradtran.org) radiative transfer model (`uvspec`, DISORT solver, Kato2 band parameterisation, Angstrom aerosol).
2. **[TabPFN](https://github.com/PriorLabs/TabPFN)** — a tabular foundation model that learns the mapping from radiation and meteorological inputs to **Ångström β and α**, replacing the per-row RT inversion at inference time.

MERRA-2 reanalysis supplies prior/ancillary information (ozone, surface pressure, Angstrom alpha, surface albedo, total precipitable water, Angstrom beta).

## Pipeline

| Step | Script | What it does |
|:----:|--------|--------------|
| 1 | `1.arrange.py` | **PAL/TAT/…:** edit **CONFIG** block (`STATION`, `YEAR`, optional paths); `Data/BSRN/<STATION>/`, AERONET `processed_<STATION>.txt`, out `Data/<STATION>_<year>_all.txt`. **QIQ:** `Code/old/1.arrange.py` → `qiq_1min_merra_qc.txt`. |
| 2 | `2.create_holdout.py` | **PAL/TAT/…:** edit **CONFIG**; reads `<STATION>_<year>_all.txt` (must include AERONET); keeps rows with finite `aeronet_aod550` and `aeronet_alpha`; writes `<STATION>_<year>_trainpool.txt` + `_testpool.txt` (50:50). **QIQ:** `Code/old/2.create_holdout.py` (no AERONET). |
| 3 | `3.latin_hypercube.py` | **PAL/TAT/…:** edit **CONFIG**; LHS sample (default 500) from `<STATION>_<year>_trainpool.txt` → `<STATION>_<year>_train_0.5k.txt`. **QIQ:** `Code/old/3.latin_hypercube.py` → `train_0.5k.txt`. |
| 4a | `4a.retrieval_ls.py` | libRadtran forward + least-squares inversion; retrieves **(β, α)** per training row → `train_ls*.txt`. |
| 4b | `4b.retrieval_oe.py` | Optimal-estimation retrieval **(β, α)** → `train_oe*.txt`. |
| 5 | `5.tabpfn.py` | Trains TabPFN on LS or OE labels; run twice with `MODE=ls` and `MODE=oe` → `pred_ls*.txt`, `pred_oe*.txt`. |
| 6 | `6.evaluation.py` | One pass per row: **MERRA explicit**, **TabPFN LS**, **TabPFN OE**, **AERONET** forwards (shared `merra_explicit_physics`). Merges `pred_ls` + `pred_oe` on `time_utc`; writes `test_combined*.txt`. Sets `LRT_SEASONAL_ATMOSPHERE=1` by default (same idea as 4a/4b). |
| 7 | `7.retrieval_result.py` | AOD₅₅₀ / α densities and scatter vs AERONET (retrieval or `USE_TABPFN=1`); figures in `tex/figures/retrieval_result_distributions*.pdf`. |
| 8 | `8.test_analysis.py` | **Test** set: **TabPFN OE** **Δτ₅₅₀** vs **solar zenith** heatmap from `test_combined*.txt`; default `tex/figures/<STATION>_<YEAR>_test_error_heatmap_oe*.pdf`. |

### Supporting scripts

| Script | Purpose |
|--------|---------|
| `10.plot_retrieval_scatter.py` | Measured vs forward **GHI/BNI/DHI** scatters (MERRA, TabPFN **OE**, optional AERONET). Prefers `test_combined*.txt`; **LS panel off by default** — set `SKIP_LS=0` to include TabPFN LS. |

**Legacy / archive:** older one-off scripts live under `Code/old/` (including former `7`–`9` analysis scripts moved out of the main numbered path).

### Data flow

```text
Raw BSRN + MERRA-2 (+ AERONET for PAL/TAT)
  └─[1]─► <STATION>_<year>_all.txt   (or QIQ: qiq_1min_merra_qc.txt via old/1.arrange)
            └─[2]─► <STATION>_<year>_trainpool.txt + <STATION>_<year>_testpool.txt
                      └─[3]─► <STATION>_<year>_train_0.5k.txt
                                └─[4a/4b]─► train_ls_{N}k.txt, train_oe_{N}k.txt  (retrieved β, α)
                                          └─[5]─► pred_ls_{N}k.txt, pred_oe_{N}k.txt  (TabPFN β, α on test pool)
                                                    └─[6]─► test_combined_{N}k.txt  (forward fluxes: MERRA, LS, OE, AERONET)
                                                              └─[8]─► optional Δτ₅₅₀ vs zenith heatmap (test diagnostics)
```

## Core library

**`Code/libRadtran.py`** — shared, import-pure library containing:

- `ClearskyConfig` dataclass and `build_uvspec_input` for assembling libRadtran input decks.
- `run_clearsky` for calling the `uvspec` binary and parsing broadband fluxes.
- `retrieve_one_row_ls` / `process_row_ls` for the least-squares retrieval loop (β, α).

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
```

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
/opt/anaconda3/bin/pip install numpy pandas scipy matplotlib scikit-learn tabpfn torch tqdm bsrn
```

4. Point to your libRadtran installation (default `~/libRadtran-2.0.6`):

```bash
export LIBRADTRANDIR=/path/to/libRadtran-2.0.6
```

## Usage

Run the scripts in order (step 1 requires raw BSRN data and is typically run once). Examples use Anaconda’s interpreter:

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
$PY Code/6.evaluation.py
$PY Code/7.retrieval_result.py
# TabPFN figures: USE_TABPFN=1 $PY Code/7.retrieval_result.py
$PY Code/8.test_analysis.py
# Optional: TEST_COMBINED=Data/PAL_2024_test_combined_0.5k.txt $PY Code/8.test_analysis.py
# Scatter plot: $PY Code/10.plot_retrieval_scatter.py
```

### Environment variables

| Variable | Script | Default | Description |
|----------|--------|---------|-------------|
| `ARRANGE_OUTPUT` | 1 | `Data/qiq_1min_merra_qc.txt` | Output path for the arranged dataset. |
| `ARRANGE_FIRST_MONTH_ONLY` | 1 | *(off)* | Process only the first month (for testing). |
| `ARRANGE_ONE_FILE` | 1 | *(off)* | Process a single input file. |
| `LHS_INPUT` | old/3 | `Data/trainpool.txt` | Input pool for legacy step 3. |
| `LHS_TRAIN` | old/3 | `Data/train_0.5k.txt` | LHS output path (legacy). |
| `LHS_N` | old/3 | `500` | LHS sample count (legacy). |
| `LHS_SEED` | old/3 | `42` | Random seed (legacy). |
| `LHS_ZENITH_MAX` | old/3 | `87` | Max zenith (legacy). |
| `TEST_POOL` | 5 | `Data/testpool.txt` | Holdout test pool for TabPFN (e.g. `Data/PAL_2024_testpool.txt`). |
| `INPUT_DATA` | 4 | `Data/train_0.5k.txt` | LS retrieval input (e.g. `Data/PAL_2024_train_0.5k.txt` for PAL). |
| `OUTPUT_DATA` | 4 | `Data/train_ls_0.5k.txt` | Output with retrieved (beta, H₂O). |
| `LHS_N` | 3–7 | `500` | LHS / pred / eval suffix (e.g. `_0.5k`); same as step 3. |
| `MODE` | 5 | `ls` | `ls` or `oe`; run step 5 twice for both TabPFN outputs (`pred_ls`, `pred_oe`). |
| `LRT_SEASONAL_ATMOSPHERE` | 4a, 4b, 6 | `1` in 4a/4b/6 | Month-based AFGL `afglms`/`afglmw`; override with `0` if needed. |
| `PLOT_INPUT_COMBINED` | 10 | `Data/..._test_combined<suffix>.txt` | Optional; defaults to combined step-6 output for scatter plots. |
| `N_TEST` | 5 | `5000` | Number of test rows to predict. |
| `LIBRADTRANDIR` | lib | `~/libRadtran-2.0.6` | Path to libRadtran installation. |
| `SKIP_LS` | 10 | `1` (omit LS) | Set `0` to **show** TabPFN (LS) panel; default omits LS (OE-focused). |

## License

[MIT](LICENSE) — Copyright (c) 2026 Dazhi Yang
