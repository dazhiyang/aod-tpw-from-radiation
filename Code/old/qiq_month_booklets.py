#!/opt/anaconda3/bin/python
"""
Monthly PDF booklets for QIQ (Jan–Dec) using REST2 clear-sky model.
Uses the bsrn library workflow: QC -> REST2 Clear-sky -> Booklet Plotting.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

import bsrn
import bsrn.qc.wrapper
import bsrn.modeling.clear_sky
import bsrn.visualization.timeseries as ts

PROJECT = Path(__file__).resolve().parent.parent.parent # script moved to Code/old/
QIQ_DIR = PROJECT / "Data" / "QIQ"
OUTPUT_DIR = PROJECT / "tex"
STATION = "QIQ"

def main():
    paths = sorted(QIQ_DIR.glob("qiq*.dat.gz"))
    if not paths:
        print(f"No monthly files found in {QIQ_DIR}")
        return
        
    print(f"Found {len(paths)} monthly file(s) in {QIQ_DIR}.")
    
    # Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for path in paths:
        # e.g. path = 'qiq0124.dat.gz'
        month_label = path.name.replace("qiq", "").replace(".dat.gz", "")
        # convert '0124' to '2024-01'
        if len(month_label) == 4:
            month_str = f"20{month_label[2:]}-{month_label[:2]}"
        else:
            month_str = month_label
            
        pdf_path = OUTPUT_DIR / f"qiq_month_booklet_{month_str}.pdf"
        print(f"--- Processing {path.name} -> {pdf_path.name} ---")
        
        # 1. Read monthly LR0100 file
        m_df = bsrn.io.reader.read_lr0100(str(path))
        if m_df is None or m_df.empty:
            continue
            
        # 2. Add solpos / zenith (required for REST2 and Plotter)
        m_df = bsrn.physics.geometry.add_solpos_columns(m_df, station_code=STATION)
        
        # 3. Add derived columns required by the plotter (gh_diff is closure diff)
        mu0 = np.cos(np.radians(m_df["apparent_zenith"]))
        sum_sw = m_df["bni"] * mu0 + m_df["dhi"]
        m_df["gh_diff"] = m_df["ghi"] - sum_sw
        m_df["gh_ratio"] = m_df["ghi"] / sum_sw
        
        # Ensure meteorological placeholders exist for the plotter
        for col in ["temp", "rh", "pressure"]:
            if col not in m_df.columns:
                m_df[col] = np.nan

        # 4. Run QC (ensures flags exist)
        print("  Running QC...")
        m_df = bsrn.qc.wrapper.run_qc(m_df, station_code=STATION)
        
        # 5. Add REST2 Clear-sky columns
        print("  Adding REST2 clear-sky columns...")
        m_df = bsrn.modeling.clear_sky.add_clearsky_columns(m_df, station_code=STATION, model='rest2')
        
        # 6. Generate PDF Booklet
        print(f"  Generating PDF pages...")
        with PdfPages(pdf_path) as pdf:
            # Group by day
            days = m_df.groupby(m_df.index.date)
            for day, day_df in days:
                day_zenith = day_df["zenith"]
                title = f"{STATION} Timeseries — {day.strftime('%Y-%m-%d')}"
                
                try:
                    p = ts._ggplot_bsrn_timeseries_one_day(
                        day_df,
                        day_zenith,
                        title=title,
                        station_code=STATION,
                        show_qc_markers=True
                    )
                    fig = p.draw()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"    Error plotting {day}: {e}")
                    
        print(f"  Saved booklet to {pdf_path.name}")

if __name__ == "__main__":
    main()
