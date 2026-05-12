import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

from .tracker import count_inflections_1d, tracked_cell_colors


def save_history_npz(history: dict, path: str):
    np.savez(path, **history)
    print(f"Saved history: {path}")


def load_history_npz(path: str) -> dict:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def save_video(images, path, fps=20):
    imageio.mimsave(path, images, fps=fps)
    print(f"Video saved as {path}")


def plot_history_timeseries(history, e0, column_subsample, save_mag_path, save_theta_path):
    t = np.asarray(history["times"])
    mag = np.asarray(history["mag_crys"])
    ang = np.unwrap(np.asarray(history["angle_crys_rad"]), axis=0)
    if mag.ndim == 1:
        mag = mag.reshape(-1, 1)
    if ang.ndim == 1:
        ang = ang.reshape(-1, 1)
    n_tr = mag.shape[1]
    colors = tracked_cell_colors(n_tr)
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    for k in range(n_tr):
        ax1.plot(t, mag[:, k], color=colors[k % len(colors)], linewidth=1.4, alpha=0.85, label=f"cell {k}")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r"$|P^c|$ (C/m²)")
    ax1.set_title(f"Crystal-frame polarization magnitude vs time ({n_tr} cells, stride={column_subsample}, E={e0} V/m)")
    ax1.grid(True, alpha=0.35)
    if n_tr <= 12:
        ax1.legend(loc="best", fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(save_mag_path, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for k in range(n_tr):
        ax2.plot(t, np.degrees(ang[:, k]), color=colors[k % len(colors)], linewidth=1.4, alpha=0.85, label=f"cell {k}")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(r"$\theta$ (deg)")
    ax2.set_title(f"Crystal-frame polarization angle vs time ({n_tr} cells, stride={column_subsample}, E={e0} V/m)")
    ax2.grid(True, alpha=0.35)
    if n_tr <= 12:
        ax2.legend(loc="best", fontsize=8, frameon=True)
    fig2.tight_layout()
    fig2.savefig(save_theta_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved plots: {save_mag_path}, {save_theta_path}")


def compute_inflection_counts(history, smooth_window=1):
    """
    Per-curve inflection counts matching the |P^c|(t) and θ(t) time-series plots
    (d²y/dt² sign changes on optionally smoothed series).

    Returns a dict with:
      - cell_indices: mesh cell index per tracked column (same order as plot curves)
      - inflection_count_P_mag: counts for |P^c|(t) (same as mag_crys / plot |P^c|)
      - inflection_count_theta: counts for θ(t) (unwrapped angle, same as plot after unwrap)
    """
    t = np.asarray(history["times"])
    mag = np.asarray(history["mag_crys"])
    ang = np.unwrap(np.asarray(history["angle_crys_rad"]), axis=0)
    if mag.ndim == 1:
        mag = mag.reshape(-1, 1)
    if ang.ndim == 1:
        ang = ang.reshape(-1, 1)
    n_cells = mag.shape[1]
    inflection_count_P_mag = np.empty(n_cells, dtype=int)
    inflection_count_theta = np.empty(n_cells, dtype=int)
    for k in range(n_cells):
        inflection_count_P_mag[k] = count_inflections_1d(t, mag[:, k], smooth_window=smooth_window)
        inflection_count_theta[k] = count_inflections_1d(t, ang[:, k], smooth_window=smooth_window)
    return {
        "cell_indices": np.asarray(history["cell_indices"], dtype=int).copy(),
        "inflection_count_P_mag": inflection_count_P_mag,
        "inflection_count_theta": inflection_count_theta,
    }


def print_inflection_counts_table(history, smooth_window=1, counts=None):
    """
    Spreadsheet-style table: Cell #, inflection_count (|P|(t)), inflection_count (θ(t)).
    Counts use crystal-frame |P^c| and θ (same series as the time-series plots); one row
    per GB-tracked curve index (same ``SimParams`` / ``gb_interface_tracked_indices`` as
    ``run_vector_field_evolution``).
    """
    d = counts if counts is not None else compute_inflection_counts(history, smooth_window=smooth_window)
    npm = d["inflection_count_P_mag"]
    nth = d["inflection_count_theta"]
    n_cells = npm.size
    sep = "\t"
    print("")
    print(
        "Inflection counts (GB interface cells; curve index = Cell #; "
        f"smooth_window={smooth_window})"
    )
    print(sep.join(["Cell #", "inflection_count (|P|(t))", "inflection_count (θ(t))"]))
    for k in range(n_cells):
        print(sep.join([str(k), str(int(npm[k])), str(int(nth[k]))]))
    print("")


def _inflection_count_frequencies(values):
    """Return (bin_centers, frequencies) for integer inflection counts."""
    v = np.asarray(values, dtype=int).ravel()
    if v.size == 0:
        return np.array([]), np.array([])
    vmin, vmax = int(v.min()), int(v.max())
    bc = np.bincount(v, minlength=vmax + 1)
    x = np.arange(vmin, vmax + 1, dtype=int)
    y = bc[vmin : vmax + 1].astype(int)
    return x, y


def plot_inflection_count_distribution(counts, save_path, smooth_window=1, e0=None):
    """
    Column chart: x = inflection count value, y = number of curves with that count.
    Left: |P^c|(t); right: θ(t). ``counts`` is the dict from ``compute_inflection_counts``.
    """
    npm = counts["inflection_count_P_mag"]
    nth = counts["inflection_count_theta"]
    x_p, y_p = _inflection_count_frequencies(npm)
    x_t, y_t = _inflection_count_frequencies(nth)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    w = 0.72
    ax1.bar(x_p, y_p, width=w, color="steelblue", edgecolor="0.35", linewidth=0.6, alpha=0.9)
    ax1.set_xlabel(r"inflection count ($|P^c|(t)$)")
    ax1.set_ylabel("number of curves")
    ax1.set_xticks(x_p)
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.set_title(r"Distribution of $|P^c|(t)$ inflection counts")
    ax2.bar(x_t, y_t, width=w, color="darkseagreen", edgecolor="0.35", linewidth=0.6, alpha=0.9)
    ax2.set_xlabel(r"inflection count ($\theta(t)$)")
    ax2.set_ylabel("number of curves")
    ax2.set_xticks(x_t)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.set_title(r"Distribution of $\theta(t)$ inflection counts")
    n_curves = int(npm.size)
    sub = f"{n_curves} GB curves, smooth_window={smooth_window}"
    if e0 is not None:
        sub += f", E={e0} V/m"
    fig.suptitle("How often each inflection count appears", fontsize=11, y=1.02)
    fig.text(0.5, 0.02, sub, ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_inflection_pair_distribution(counts, save_path, smooth_window=1, e0=None):
    """
    Column chart of paired counts:
      pair = (inflection_count(|P^c|(t)), inflection_count(θ(t)))
      height = number of curves with that pair.
    """
    npm = np.asarray(counts["inflection_count_P_mag"], dtype=int).ravel()
    nth = np.asarray(counts["inflection_count_theta"], dtype=int).ravel()
    if npm.size != nth.size:
        raise ValueError("inflection_count_P_mag and inflection_count_theta must have same length")
    if npm.size == 0:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title("Paired inflection-count frequencies (no curves)")
        ax.set_xlabel(r"pair $(n_{|P^c|}, n_{\theta})$")
        ax.set_ylabel("number of curves")
        ax.grid(True, axis="y", alpha=0.35)
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    pairs = np.stack([npm, nth], axis=1)
    uniq, freq = np.unique(pairs, axis=0, return_counts=True)
    order = np.argsort(-freq, kind="stable")
    uniq = uniq[order]
    freq = freq[order]

    labels = [f"({int(a)},{int(b)})" for a, b in uniq]
    x = np.arange(len(labels), dtype=int)

    fig_w = max(9.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(x, freq, width=0.75, color="mediumpurple", edgecolor="0.35", linewidth=0.6, alpha=0.9)
    ax.set_xlabel(r"paired inflection counts $(n_{|P^c|}, n_{\theta})$")
    ax.set_ylabel("number of curves")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_title("Most common (|P|, θ) inflection-count pairs")

    n_curves = int(npm.size)
    sub = f"{n_curves} GB curves, smooth_window={smooth_window}"
    if e0 is not None:
        sub += f", E={e0} V/m"
    fig.text(0.5, 0.02, sub, ha="center", fontsize=9, color="0.35")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_inflection_table(history, smooth_window=1):
    t = np.asarray(history["times"])
    mag = np.asarray(history["mag_crys"])
    ang = np.unwrap(np.asarray(history["angle_crys_rad"]), axis=0)
    if mag.ndim == 1:
        mag = mag.reshape(-1, 1)
    if ang.ndim == 1:
        ang = ang.reshape(-1, 1)
    n_cells = mag.shape[1]
    headers = f"{'Cell':>6}  {'Mesh idx':>10}  {'n_inf |P^c|':>14}  {'n_inf θ':>10}"
    print(headers)
    print("-" * len(headers))
    for k in range(n_cells):
        nim = count_inflections_1d(t, mag[:, k], smooth_window=smooth_window)
        nit = count_inflections_1d(t, ang[:, k], smooth_window=smooth_window)
        print(f"{k:>6}  {int(history['cell_indices'][k]):>10}  {nim:>14}  {nit:>10}")


def build_output_names(output_dir, left_deg, right_deg):
    suffix = f"{left_deg:.0f}_{right_deg:.0f}_deg"
    return {
        "video": os.path.join(output_dir, f"efield_domain_evolution{suffix}.mp4"),
        "polar": os.path.join(output_dir, f"gb_polar_trajectory{suffix}.png"),
        "p_mag_t": os.path.join(output_dir, f"gb_Pc_mag_vs_t{suffix}.png"),
        "theta_t": os.path.join(output_dir, f"gb_theta_vs_t{suffix}.png"),
        "inflection_hist": os.path.join(output_dir, f"gb_inflection_count_distribution{suffix}.png"),
        "inflection_pair_hist": os.path.join(output_dir, f"gb_inflection_pair_distribution{suffix}.png"),
        "history": os.path.join(output_dir, f"gb_history{suffix}.npz"),
        "free_energy": os.path.join(output_dir, f"free_energy{suffix}.png"),
    }
