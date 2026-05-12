import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test1_modular.constants import MaterialParams, SimParams
from test1_modular.postprocess import build_output_names
from test1_modular.simulation import run_free_energy_scan


def main():
    sim = SimParams()
    mat = MaterialParams()
    names = build_output_names(".", sim.theta_left_deg, sim.theta_right_deg)
    run_free_energy_scan(
        mat=mat,
        e0=sim.e0,
        phi_e_deg=sim.phi_e_deg,
        show=True,
        save_path=names["free_energy"],
    )


if __name__ == "__main__":
    main()
