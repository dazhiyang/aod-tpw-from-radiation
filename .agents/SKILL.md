# Coding Standards for Atmospheric Radiation Research

## Documentation Standards
- **Function Docstrings**: Every core function must have a clear Summary, Parameters, and Returns section.
- **Line Length**: Maintain a rigorous **110-character limit** for readability in side-by-side diffs.
- **Commenting**: Use inline comments for complex radiative transfer physics or math transforms.

## Project Structure
- `Code/`: Contains numbered scripts (1–7, plus `10.*` plots, etc.) and the `libRadtran.py` library.
- `Data/`: Contains raw measurements and generated training/test pools.
- `tex/`: Contains LaTeX figures and booklets.

## Python Requirements
- Type hints are preferred for all function signatures.
- Prefer `pathlib` over `os.path` for robust cross-platform path handling.

## Visualization & Aesthetics

- **Color palettes**
  - **Discrete variables**: **MUST** use the **Wong colorblind-friendly palette**. Colors **MUST** be
    used in this order by category index (use only as many as needed, in sequence):
    1. `#E69F00` (orange)
    2. `#56B4E9` (sky blue)
    3. `#009E73` (bluish green)
    4. `#CC79A7` (reddish purple)
    5. `#D55E00` (vermillion)
    6. `#F0E442` (yellow)
    7. `#0072B2` (blue)
  - **Continuous variables**: **MUST** use the **Viridis** palette (or another perceptually uniform
    colormap of equivalent intent).

- **Line size**: Default line width **MUST** be **0.3** for all plots (unless a journal or venue
  explicitly overrides).

- **Fonts**: **MUST** use **Times New Roman** for axis labels, titles, and legend text.

- **Size**
  - Figure width **MUST** be **160 mm** when a fixed journal single-column width applies.
  - **8 pt** for all text (titles, axes, legend, cell labels) in **faceted / tabular** plotnine figures.
