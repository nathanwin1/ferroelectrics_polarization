import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test1_modular.constants import MaterialParams, SimParams
from test1_modular.simulation import run_vector_field_evolution


def main():
    # Same SimParams as run_timeseries / GB tracker; tracked curve indices are
    # gb_interface_tracked_indices(mesh, sim) in test1_modular.simulation.
    sim = SimParams()
    mat = MaterialParams()
    run_vector_field_evolution(
        sim=sim,
        mat=mat,
        output_dir=".",
        video_path="polarization_vector_field_evolution.mp4",
        fps=20,
    )


if __name__ == "__main__":
    main()
