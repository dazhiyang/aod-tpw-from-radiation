# Coding Standards for Atmospheric Radiation Research

## Documentation Standards
- **Function Docstrings**: Every core function must have a clear Summary, Parameters, and Returns section.
- **Line Length**: Maintain a rigorous **110-character limit** for readability in side-by-side diffs.
- **Commenting**: Use inline comments for complex radiative transfer physics or math transforms.

## Project Structure
- `Code/`: Contains numbered scripts (1-6) and the `libRadtran.py` library.
- `Data/`: Contains raw measurements and generated training/test pools.
- `tex/`: Contains LaTeX figures and booklets.

## Python Requirements
- Type hints are preferred for all function signatures.
- Prefer `pathlib` over `os.path` for robust cross-platform path handling.
