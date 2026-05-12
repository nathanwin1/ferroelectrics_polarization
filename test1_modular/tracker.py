import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def interface_column_indices(mesh, x_mid, side="left", column_offset=0, tol_frac=0.05):
    x = np.asarray(mesh.x.value).ravel()
    y = np.asarray(mesh.y.value).ravel()
    if side == "left":
        mask_side = x < x_mid
        if not np.any(mask_side):
            return np.array([], dtype=int)
        x_unique = np.unique(x[mask_side])
        if column_offset >= x_unique.size:
            return np.array([], dtype=int)
        x_col = float(x_unique[-1 - int(column_offset)])
    else:
        mask_side = x >= x_mid
        if not np.any(mask_side):
            return np.array([], dtype=int)
        x_unique = np.unique(x[mask_side])
        if column_offset >= x_unique.size:
            return np.array([], dtype=int)
        x_col = float(x_unique[int(column_offset)])

    dx_mesh = getattr(mesh, "dx", None)
    if dx_mesh is not None:
        dx = float(np.asarray(dx_mesh).ravel()[0])
    else:
        dx = float((x.max() - x.min()) / max(1, int(mesh.nx) - 1))
    tol = max(tol_frac * dx, 1e-30)
    idx = np.where(mask_side & (np.abs(x - x_col) < tol))[0]
    order = np.argsort(y[idx])
    return idx[order]


def tracked_cell_colors(n_tr):
    return plt.cm.tab10(np.linspace(0, 1, min(10, max(int(n_tr), 1))))


def _moving_average_1d(y, window):
    y = np.asarray(y, dtype=float).ravel()
    w = int(window)
    if w <= 1:
        return y
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(yp, kernel, mode="valid")


def _d2y_smoothed(t, y, smooth_window):
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError("t and y must have same shape")
    if y.size < 3:
        return t, None, None
    y_s = _moving_average_1d(y, smooth_window)
    d1y = np.gradient(y_s, t)
    d2y = np.gradient(d1y, t)
    return t, y_s, d2y


def count_inflections_1d(t, y, smooth_window=1):
    t, y_s, d2y = _d2y_smoothed(t, y, smooth_window)
    if d2y is None:
        return 0
    return int(np.sum(d2y[:-1] * d2y[1:] < 0))


def inflection_points_1d(t, y, smooth_window=1):
    """
    Inflection points where d²y/dt² (on optionally smoothed y) changes sign.
    Returns (t_inflection, y_at_inflection) as 1d arrays; linear interpolation in t.
    """
    t, y_s, d2y = _d2y_smoothed(t, y, smooth_window)
    if d2y is None:
        return np.array([]), np.array([])
    j = np.where(d2y[:-1] * d2y[1:] < 0)[0]
    if j.size == 0:
        return np.array([]), np.array([])
    t0 = t[j]
    t1 = t[j + 1]
    d0 = d2y[j]
    d1 = d2y[j + 1]
    denom = d1 - d0
    safe = np.abs(denom) > 1e-30
    tc = np.empty(j.size, dtype=float)
    tc[safe] = t0[safe] - d0[safe] * (t1[safe] - t0[safe]) / denom[safe]
    tc[~safe] = 0.5 * (t0[~safe] + t1[~safe])
    y0 = y_s[j]
    y1 = y_s[j + 1]
    dt_seg = t1 - t0
    dt_seg = np.where(np.abs(dt_seg) > 1e-30, dt_seg, np.copysign(1e-30, dt_seg + 1e-300))
    yc = y0 + (y1 - y0) * (tc - t0) / dt_seg
    return tc, yc


class GBTracker:
    def __init__(self, mesh, c_arr, s_arr, ps, e0, x_mid, y_mid, cell_indices):
        self.mesh = mesh
        self.c_arr = np.asarray(c_arr)
        self.s_arr = np.asarray(s_arr)
        self.ps = float(ps)
        self.e0 = float(e0)
        self.x_mid = float(x_mid)
        self.y_mid = float(y_mid)
        self.cell_indices = np.asarray(cell_indices, dtype=int).ravel()
        self.gb_idx = int(self.cell_indices[0])
        x = np.asarray(mesh.x.value)
        y = np.asarray(mesh.y.value)
        self.cell_x = np.asarray(x[self.cell_indices], dtype=float)
        self.cell_y = np.asarray(y[self.cell_indices], dtype=float)
        self.steps, self.times = [], []
        self.P1_hist, self.P2_hist = [], []
        self.P1c_hist, self.P2c_hist = [], []
        self.angle_c_hist, self.mag_c_hist = [], []

    def record(self, step, dt, p1_flat, p2_flat):
        idxs = self.cell_indices
        c = self.c_arr[idxs]
        s = self.s_arr[idxs]
        p1 = np.asarray(p1_flat).ravel()[idxs].astype(float)
        p2 = np.asarray(p2_flat).ravel()[idxs].astype(float)
        p1c = c * p1 + s * p2
        p2c = -s * p1 + c * p2
        self.steps.append(int(step))
        self.times.append(float(step) * dt)
        self.P1_hist.append(p1)
        self.P2_hist.append(p2)
        self.P1c_hist.append(p1c)
        self.P2c_hist.append(p2c)
        self.angle_c_hist.append(np.arctan2(p2c, p1c))
        self.mag_c_hist.append(np.sqrt(p1c**2 + p2c**2))

    def export_history(self):
        ang = np.vstack(self.angle_c_hist)
        mag = np.vstack(self.mag_c_hist)
        return {
            "steps": np.array(self.steps),
            "times": np.array(self.times),
            "angle_crys_rad": ang,
            "angle_crys_deg": np.degrees(ang),
            "mag_crys": mag,
            "mag_over_ps": mag / self.ps,
            "cell_indices": self.cell_indices.copy(),
            "cell_x": self.cell_x.copy(),
            "cell_y": self.cell_y.copy(),
        }

    def inflection_counts(self, smooth_window=1):
        t = np.asarray(self.times)
        mag = np.vstack(self.mag_c_hist)
        ang = np.unwrap(np.vstack(self.angle_c_hist), axis=0)
        n_cells = mag.shape[1]
        n_inf_mag = np.empty(n_cells, dtype=int)
        n_inf_theta = np.empty(n_cells, dtype=int)
        for k in range(n_cells):
            n_inf_mag[k] = count_inflections_1d(t, mag[:, k], smooth_window=smooth_window)
            n_inf_theta[k] = count_inflections_1d(t, ang[:, k], smooth_window=smooth_window)
        return n_inf_mag, n_inf_theta

    def plot_polar_trajectory(self, theta_left_deg, theta_right_deg, save_path, show=False):
        theta = np.vstack(self.angle_c_hist)
        r_norm = np.vstack(self.mag_c_hist) / self.ps
        n_tr = theta.shape[1]
        colors = tracked_cell_colors(n_tr)
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        def _p_direction_marker(th, rn, color, zorder):
            d_r = 0.08
            r0 = max(0.0, float(rn) - 0.5 * d_r)
            r1 = float(rn) + 0.5 * d_r
            ax.annotate(
                "",
                xy=(float(th), r1),
                xytext=(float(th), r0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.8,
                    mutation_scale=12,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=zorder,
            )

        for k in range(n_tr):
            th_k = theta[:, k]
            r_k = r_norm[:, k]
            c_line = colors[k % len(colors)]
            ax.plot(th_k, r_k, color=c_line, linewidth=1.8, alpha=0.75, label=f"cell {k}")
            ax.scatter(th_k, r_k, color=c_line, s=12, alpha=0.7, zorder=3)
            _p_direction_marker(th_k[0], r_k[0], "lime", 5)
            _p_direction_marker(th_k[-1], r_k[-1], "red", 6)

        ax.set_rmax(1.25)
        ax.set_rticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25|P|", "0.5|P|", "0.75|P|", "|P|"], zorder=1)

        start_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=12, lw=1.8, color="lime", label="Start")
        end_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=12, lw=1.8, color="red", label="End")
        ax.legend(handles=[start_handle, end_handle], loc="lower right", bbox_to_anchor=(1.2, -0.02), fontsize=9, frameon=True)
        ax.set_title(
            "Polar crystal-frame trajectory of tracked grain-boundary cell(s)\n"
            f"Left grain: {theta_left_deg:.0f}°  |  Right grain: {theta_right_deg:.0f}°\n"
            f"Angle = direction, Radius = |P|= {self.ps} C/m²\n"
            f"E = {self.e0} V/m",
            pad=22,
            fontsize=11,
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
