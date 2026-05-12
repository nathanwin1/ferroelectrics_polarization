import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test1_modular.constants import MaterialParams, SimParams
from test1_modular.simulation import run_tracker_only

from bezier_classify import classify_trajectories
from postprocess_bezier import plot_bezier_classification


def main():
    sim = SimParams()
    mat = MaterialParams()

    print("Running simulation (tracker only)...")
    tracker = run_tracker_only(sim=sim, mat=mat)
    history = tracker.export_history()

    print("Fitting Bézier curves and clustering...")
    ctrl_pts_list, labels, features_list = classify_trajectories(
        history=history,
        degree=4,
        n_clusters=4,
    )

    # Print summary table
    n_cells = len(labels)
    print(
        f"\n{'Cell':>5}  {'Cluster':>7}  {'displacement':>14}  "
        f"{'poly_len':>10}  {'winding':>9}  {'dtheta°':>9}  "
        f"{'hull_area':>10}  {'infθ':>5}"
    )
    print("-" * 92)
    for k in range(n_cells):
        lbl = labels[k]
        feats = features_list[k]
        if feats is not None:
            dtheta_deg = np.degrees(feats["net_angle_change_rad"])
            print(
                f"{k:>5}  {lbl:>7}  {feats['displacement']:>14.4f}  "
                f"{feats['poly_len']:>10.4f}  {feats['winding']:>9.4f}  "
                f"{dtheta_deg:>9.1f}  {feats['convex_hull_area']:>10.4f}  "
                f"{feats['n_inflections_theta']:>5.0f}"
            )
        else:
            print(f"{k:>5}  {'FAILED':>7}")

    # Build output path matching existing naming convention
    suffix = f"{sim.theta_left_deg:.0f}_{sim.theta_right_deg:.0f}_deg"
    save_path = f"bezier_classification_{suffix}.png"

    plot_bezier_classification(
        history=history,
        ctrl_pts_list=ctrl_pts_list,
        labels=labels,
        theta_left_deg=sim.theta_left_deg,
        theta_right_deg=sim.theta_right_deg,
        n_clusters=4,
        degree=4,
        save_path=save_path,
        show=True,
    )
    print(f"\nDone. Saved: {save_path}")


if __name__ == "__main__":
    main()
