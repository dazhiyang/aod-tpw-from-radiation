"""
4.retrieval: Minimalist driver for libRadtran + Optimal Estimation retrieval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from libRadtran import (
    LIBRADTRANDIR, CLEARSKY_CONFIG, process_row_oe
)

PROJECT = Path(__file__).resolve().parent.parent
INPUT_TXT = Path(os.environ.get("OE_INPUT", str(PROJECT / "Data" / "train_0.5k.txt")))

_out_name = INPUT_TXT.name.replace("train", "train_ls") if "train" in INPUT_TXT.name else f"{INPUT_TXT.stem}_ls.txt"
OUTPUT_TXT = Path(os.environ.get("OE_OUTPUT", str(PROJECT / "Data" / _out_name)))

def main():
    if not INPUT_TXT.is_file():
        print(f"ERROR: Missing input file: {INPUT_TXT}")
        sys.exit(1)

    print(f"Loading dataset: {INPUT_TXT.name}")
    df = pd.read_csv(INPUT_TXT, sep="\t", comment="#", parse_dates=["time_utc"])
    df = df.set_index("time_utc").sort_index()

    print(f"Starting retrieval for {len(df)} rows using libRadtran...")
    
    _oe_fn = lambda r: process_row_oe(r, LIBRADTRANDIR, CLEARSKY_CONFIG)
    if _tqdm is not None:
        _tqdm.pandas(desc="OE Beta + H2O", leave=True)
        results = df.progress_apply(_oe_fn, axis=1)
    else:
        results = df.apply(_oe_fn, axis=1)

    out = pd.concat([df, results], axis=1)
    
    cols = [
        "ghi", "bni", "dhi", "ghi_merra", "dni_merra", "dhi_merra",
        "ghi_ls", "dni_ls", "dhi_ls", "beta_retrieved", "h2o_mm_retrieved",
        "merra_ALPHA", "merra_BETA", "merra_TO3", "merra_TQV", "merra_ALBEDO", "merra_PS", "zenith",
    ]
    final_cols = [c for c in cols if c in out.columns]
    out.reset_index()[["time_utc"] + final_cols].to_csv(
        OUTPUT_TXT, sep="\t", index=False, float_format="%.8f"
    )

    print(f"Successfully wrote: {OUTPUT_TXT.name}")

if __name__ == "__main__":
    main()
