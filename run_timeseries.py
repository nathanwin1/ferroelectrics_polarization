import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test1_modular.constants import MaterialParams, SimParams
from test1_modular.postprocess import (
    build_output_names,
    compute_inflection_counts,
    plot_history_timeseries,
    plot_inflection_count_distribution,
    plot_inflection_pair_distribution,
    print_inflection_counts_table,
)
from test1_modular.simulation import run_tracker_only


def main():
    sim = SimParams()
    mat = MaterialParams()
    smooth_window = 1
    tracker = run_tracker_only(sim=sim, mat=mat)
    history = tracker.export_history()
    names = build_output_names(".", sim.theta_left_deg, sim.theta_right_deg)
    plot_history_timeseries(
        history=history,
        e0=sim.e0,
        column_subsample=sim.column_subsample,
        save_mag_path=names["p_mag_t"],
        save_theta_path=names["theta_t"],
    )
    print(f"Saved time-series plots: {names['p_mag_t']}, {names['theta_t']}")
    inflection = compute_inflection_counts(history, smooth_window=smooth_window)
    plot_inflection_count_distribution(
        inflection,
        save_path=names["inflection_hist"],
        smooth_window=smooth_window,
        e0=sim.e0,
    )
    print(f"Saved inflection count distribution: {names['inflection_hist']}")
    plot_inflection_pair_distribution(
        inflection,
        save_path=names["inflection_pair_hist"],
        smooth_window=smooth_window,
        e0=sim.e0,
    )
    print(f"Saved paired inflection-count distribution: {names['inflection_pair_hist']}")
    print_inflection_counts_table(history, smooth_window=smooth_window, counts=inflection)


if __name__ == "__main__":
    main()
