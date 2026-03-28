# AOD and TPW Retrieval from Surface Solar Radiation

Hybrid physical–ML pipeline for retrieving **Angstrom turbidity coefficient (beta)** and **total precipitable water (TPW, mm)** from clear-sky surface solar radiation measurements (GHI, BNI, DHI).

The pipeline combines two complementary approaches:

1. **Least-squares (LS) retrieval** — physics-based inversion using the [libRadtran](http://www.libradtran.org) radiative transfer model (`uvspec`, DISORT solver, Kato2 band parameterisation, Angstrom aerosol).
2. **[TabPFN](https://github.com/PriorLabs/TabPFN)** — a tabular foundation model that learns the mapping from radiation and meteorological inputs to (beta, H₂O), replacing the expensive per-row RT inversion at inference time.

MERRA-2 reanalysis supplies prior/ancillary information (ozone, surface pressure, Angstrom alpha, surface albedo, total precipitable water, Angstrom beta).

## Pipeline

| Step | Script | What it does |
|:----:|--------|--------------|
| 1 | `1.arrange.py` | Reads raw BSRN LR0100 files, merges MERRA-2 priors, applies quality control and clear-sky flagging. |
| 2 | `2.create_holdout.py` | Filters to clear-sky daytime (zenith ≤ 87°) and splits into a 70 % training pool and 30 % holdout test pool. |
| 3 | `3.latin_hypercube.py` | Selects a stratified training subset via Latin Hypercube Sampling, mapped to nearest real observations. |
| 4a | `4a.retrieval_ls.py` | Runs the libRadtran forward model and nonlinear least-squares inversion to retrieve (beta, H₂O) for each training row. |
| 5 | `5.tabpfn.py` | Trains a TabPFN regressor on the LS-retrieved labels and predicts (beta, H₂O) on the test pool. |
| 6 | `6.evaluation.py` | Validates predictions by re-running the forward model with the predicted (beta, H₂O) and comparing fluxes. |

### Supporting scripts

| Script | Purpose |
|--------|---------|
| `plot_retrieval_scatter.py` | 1:1 scatter plots (MERRA-forward vs measured vs LS-forward) with MBE, RMSE %, and R². |
| `example_merra_qiq.py` | Minimal one-day example of BSRN + MERRA-2 data access using the `bsrn` package. |

### Data flow

```text
Raw QIQ + MERRA-2
  └─[1]─► qiq_1min_merra_qc.txt
            └─[2]─► trainpool.txt + testpool.txt
                      └─[3]─► train_{N}k.txt
                                └─[4]─► train_ls_{N}k.txt   (with retrieved β, H₂O)
                                          └─[5]─► test_{N}k.txt    (with predicted β, H₂O)
                                                    └─[6]─► test_ls_{N}k.txt   (with forward fluxes)
```

## Core library

**`Code/libRadtran.py`** — shared, import-pure library containing:

- `ClearskyConfig` dataclass and `build_uvspec_input` for assembling libRadtran input decks.
- `run_clearsky` for calling the `uvspec` binary and parsing broadband fluxes.
- `retrieve_beta_h2o_one_row` and `process_row_ls` for the least-squares retrieval loop.

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

2. Install Python dependencies:

```bash
pip install numpy pandas scipy matplotlib scikit-learn tabpfn torch tqdm bsrn
```

3. Point to your libRadtran installation (default `~/libRadtran-2.0.6`):

```bash
export LIBRADTRANDIR=/path/to/libRadtran-2.0.6
```

## Usage

Run the scripts in order (step 1 requires raw BSRN data and is typically run once):

```bash
python Code/2.create_holdout.py
LHS_N=500 python Code/3.latin_hypercube.py
python Code/4a.retrieval_ls.py
python Code/5.tabpfn.py
python Code/6.evaluation.py
```

### Environment variables

| Variable | Script | Default | Description |
|----------|--------|---------|-------------|
| `ARRANGE_OUTPUT` | 1 | `Data/qiq_1min_merra_qc.txt` | Output path for the arranged dataset. |
| `ARRANGE_FIRST_MONTH_ONLY` | 1 | *(off)* | Process only the first month (for testing). |
| `ARRANGE_ONE_FILE` | 1 | *(off)* | Process a single input file. |
| `LHS_INPUT` | 3 | `Data/trainpool.txt` | Input training pool. |
| `LHS_N` | 3 | `500` | Number of LHS samples. |
| `LHS_TRAIN` | 3 | `Data/train_{N}k.txt` | Output path for training subset. |
| `LHS_SEED` | 3 | `42` | Random seed for LHS. |
| `LHS_ZENITH_MAX` | 3 | `87` | Maximum solar zenith angle. |
| `INPUT_DATA` | 4 | `Data/train_0.5k.txt` | Input for the LS retrieval. |
| `OUTPUT_DATA` | 4 | `Data/train_ls_0.5k.txt` | Output with retrieved (beta, H₂O). |
| `K_SUFFIX` | 5, 6 | `_0.5k` | Training-size suffix for file naming. |
| `N_TEST` | 5 | `5000` | Number of test rows to predict. |
| `LIBRADTRANDIR` | lib | `~/libRadtran-2.0.6` | Path to libRadtran installation. |
| `PLOT_INPUT` | plot | `Data/train_ls_0.5k.txt` | Input for scatter plots. |

## License

[MIT](LICENSE) — Copyright (c) 2026 Dazhi Yang
