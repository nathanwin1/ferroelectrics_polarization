import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test1_modular.constants import MaterialParams, SimParams
from test1_modular.postprocess import build_output_names
from test1_modular.simulation import run_tracker_only

def main():
    sim = SimParams()
    mat = MaterialParams()
    tracker = run_tracker_only(sim=sim, mat=mat)
    names = build_output_names(".", sim.theta_left_deg, sim.theta_right_deg)
    tracker.plot_polar_trajectory(
        theta_left_deg=sim.theta_left_deg,
        theta_right_deg=sim.theta_right_deg,
        save_path=names["polar"],
        show=True,
    )
    print(f"Saved polar trajectory: {names['polar']}")


if __name__ == "__main__":
    main()
