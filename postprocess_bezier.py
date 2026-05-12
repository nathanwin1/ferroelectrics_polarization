"""
Visualization for Bézier-classified polarization trajectories.
Designed to slot alongside existing postprocess module.
"""

import matplotlib.pyplot as plt
import numpy as np

from bezier_classify import bezier_curve, history_cell_time_arrays


def plot_bezier_classification(
    history: dict,
    ctrl_pts_list: list,
    labels: np.ndarray,
    theta_left_deg: float,
    theta_right_deg: float,
    n_clusters: int = 3,
    degree: int = 4,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot original Cartesian trajectories color-coded by cluster,
    with fitted Bézier overlaid as black dashed curves.

    Layout: one subplot per cluster, showing all member trajectories.

    Parameters
    ----------
    history : dict
        From GBTracker.export_history().
    ctrl_pts_list : list
        From classify_trajectories(). One (degree+1, 2) array per cell or None.
    labels : np.ndarray
        Cluster label per cell (-1 = failed).
    theta_left_deg, theta_right_deg : float
        For plot title.
    n_clusters : int
        Number of clusters (sets subplot layout).
    degree : int
        Bézier degree used for fitting.
    save_path : str or None
        If given, saves figure to this path.
    show : bool
        If True, calls plt.show().
    """
    mag, ang = history_cell_time_arrays(history)
    n_cells = mag.shape[0]

    cmap = plt.cm.tab10
    t_eval = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), squeeze=False)
    axes = axes[0]

    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        member_indices = [k for k in range(n_cells) if labels[k] == cluster_id]

        for i, k in enumerate(member_indices):
            px = mag[k] * np.cos(ang[k])
            py = mag[k] * np.sin(ang[k])
            color = cmap(i % 10)
            ax.plot(px, py, color=color, alpha=0.5, lw=1.2, label=f"cell {k}")
            ax.scatter(px[0], py[0], color=color, marker="o", s=40, zorder=5)
            ax.scatter(px[-1], py[-1], color=color, marker="X", s=60, zorder=5)

            # Bézier overlay
            if ctrl_pts_list[k] is not None:
                curve = bezier_curve(ctrl_pts_list[k], t_eval)
                ax.plot(curve[:, 0], curve[:, 1], "k--", lw=1.0, alpha=0.7)

        ax.set_title(f"Cluster {cluster_id}\n({len(member_indices)} cells)")
        ax.set_xlabel("$P_x^c$ (C/m²)")
        ax.set_ylabel("$P_y^c$ (C/m²)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if member_indices:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Bézier-classified trajectories  |  "
        f"Left: {theta_left_deg:.0f}°  Right: {theta_right_deg:.0f}°\n"
        f"Degree={degree}, K={n_clusters}  |  ○=start  ✕=end  --=Bézier fit",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved Bézier classification plot: {save_path}")
    if show:
        plt.show()
    plt.close()
