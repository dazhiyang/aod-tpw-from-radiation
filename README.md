# AOD and TPW Retrieval from Solar Radiation

This project implements a hybrid physical-ML pipeline for retrieving **Atmospheric Optical Depth (AOD/Angstrom Beta)** and **Total Precipitable Water (TPW/H2O)** from surface solar radiation measurements.

The system combines:
1. **Optimal Estimation (OE)**: A rigorous physics-based retrieval using the **libRadtran** radiative transfer model.
2. **TabPFN**: A state-of-the-art transformer-based Tabular Foundation Model used for fast, high-accuracy regression.

## Pipeline Structure

The project is organized as a numbered sequence of scripts:

1. **`1.arrange.py`**: Prepares the raw data and merges it with MERRA-2 meteorological priors.
2. **`2.create_holdout.py`**: Extracts a static 30% global holdout test set (`testpool.txt`) to ensure standardized evaluations and prevent data leakage.
3. **`3.latin_hypercube.py`**: Generates training samples using Latin Hypercube Sampling (LHS).
4. **`4.retrieval.py`**: Performs the CPU-intensive libRadtran physics retrievals on the training samples.
5. **`5.tabpfn.py`**: Trains and runs the TabPFN regression model on GPU/MPS.
6. **`6.evaluation.py`**: Performs final validation by comparing model predictions back to observations using forward radiative transfer.

## Core Library

- **`Code/libRadtran.py`**: The central project library. Contains all physics configurations, `uvspec` input/output handlers, and Optimal Estimation residuals math.

## Prerequisites

- **libRadtran 2.0.6**: The radiative transfer binary (`uvspec`) must be locally installed.
- **Python 3.10+**: Recommended environment.
- **Dependencies**: `numpy`, `pandas`, `scipy`, `matplotlib`, `tabpfn`, `tqdm`.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install numpy pandas scipy matplotlib tabpfn tqdm`.
3. Configure the path to libRadtran in your environment or directly in `Code/libRadtran.py`.

## Usage
Run the scripts in order:
```bash
python Code/2.create_holdout.py
LHS_N=500 python Code/3.latin_hypercube.py
python Code/4.retrieval.py
python Code/5.tabpfn.py
python Code/6.evaluation.py
```

---
*Developed for Atmospheric Radiation Research.*
