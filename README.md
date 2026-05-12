# Ferroelectrics Project

Simulation and post-processing utilities for ferroelectric domain and grain-boundary (GB) polarization behavior, including trajectory analysis and Bezier-based trajectory classification.

## Repository Structure

- `test1_modular/`: Core simulation and post-processing package.
- `run_free_energy.py`: Generates free-energy landscape plot.
- `run_polar_trajectory.py`: Runs tracker and plots GB polarization trajectory.
- `run_timeseries.py`: Runs tracker and generates time-series and inflection-count plots.
- `run_vector_field.py`: Generates polarization vector-field evolution video.
- `run_bezier_classify.py`: Runs tracker, classifies trajectories with Bezier fitting + clustering, and plots clusters.
- `bezier_classify.py`: Bezier fitting, feature extraction, and clustering utilities.
- `postprocess_bezier.py`: Plotting utilities for Bezier classification output.

## Requirements

- Python 3.10+ recommended
- Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `scikit-learn`

Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-learn
```

## Quick Start

From the project root:

```bash
python run_free_energy.py
python run_polar_trajectory.py
python run_timeseries.py
python run_vector_field.py
python run_bezier_classify.py
```

## Typical Outputs

Generated files are named using the current simulation-angle suffix (for example, `0_45_deg`):

- `free_energy*.png`
- `gb_polar_trajectory*.png`
- `gb_Pc_mag_vs_t*.png`
- `gb_theta_vs_t*.png`
- `gb_inflection_count_distribution*.png`
- `gb_inflection_pair_distribution*.png`
- `bezier_classification_*.png`
- `efield_domain_evolution*.mp4`
- `polarization_vector_field_evolution.mp4`
- `gb_history*.npz`

These generated outputs are excluded from version control by `.gitignore`.

## Configuration Notes

Default simulation/material settings are in:

- `test1_modular/constants.py` (`SimParams`, `MaterialParams`)

If you want different angles, time steps, mesh size, or field settings, update those dataclass defaults (or add CLI arguments later).

## Suggested Next GitHub Additions

- Add a `LICENSE` file (for example, MIT or BSD-3-Clause).
- Optionally add a `requirements.txt` for reproducible installs.
- Optionally add a small example dataset or figure set for quick verification.
