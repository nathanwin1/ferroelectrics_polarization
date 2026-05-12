import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fipy import Grid2D, CellVariable, TransientTerm, DiffusionTerm

from .constants import MaterialParams, SimParams
from .tracker import GBTracker, interface_column_indices, tracked_cell_colors


def gb_interface_tracked_indices(mesh, sim: SimParams):
    """
    Mesh cell IDs recorded on the GB time-series / tracker curves — same ``SimParams``
    (column_subsample, offsets, nx/ny/dx) as ``run_vector_field_evolution`` and
    ``run_tracker_only``. Row ``Cell #`` k in the inflection table is curve index k
    (not necessarily the mesh's linear cell index).
    """
    x_mid = sim.nx * sim.dx / 2.0
    idx_left = interface_column_indices(mesh, x_mid, side="left", column_offset=sim.left_column_offset)[
        :: sim.column_subsample
    ]
    idx_right = interface_column_indices(mesh, x_mid, side="right", column_offset=sim.right_column_offset)[
        :: sim.column_subsample
    ]
    return np.unique(np.concatenate([idx_left, idx_right]))


def free_energy_crystal(P1c, P2c, E1c, E2c, mat: MaterialParams):
    return (
        mat.alpha_1 * (P1c**2 + P2c**2)
        + mat.alpha_11 * (P1c**4 + P2c**4)
        + mat.alpha_12 * (P1c**2 * P2c**2)
        + mat.alpha_111 * (P1c**6 + P2c**6)
        + mat.alpha_112 * (P1c**4 * P2c**2 + P2c**4 * P1c**2)
        - (P1c * E1c + P2c * E2c)
    )


def run_free_energy_scan(mat: MaterialParams, e0=0.0, phi_e_deg=90.0, show=True, save_path=None):
    phi_e = np.radians(phi_e_deg)
    ex_lab, ey_lab = e0 * np.cos(phi_e), e0 * np.sin(phi_e)
    px = np.linspace(-0.2, 0.2, 500)
    py = np.linspace(-0.2, 0.2, 500)
    X, Y = np.meshgrid(px, py)
    Z = free_energy_crystal(X, Y, ex_lab, ey_lab, mat)
    i_min = np.unravel_index(np.argmin(Z), Z.shape)
    p1_min, p2_min = X[i_min], Y[i_min]
    print(f"Lowest Energy (Z): {np.min(Z)}")
    print(f"Spontaneous Polarization (Ps): {p1_min}, {p2_min}")
    z_min_plot = -180000.0
    # Colormap and colorbar only span [0, z_cap]; Z > z_cap is masked (white / not drawn).
    z_cap = 180000.0
    levels = np.linspace(z_min_plot, z_cap, 21)
    Z_plot = np.clip(np.asarray(Z, dtype=float), z_min_plot, None)
    Z_plot = np.ma.masked_where(Z_plot > z_cap, Z_plot)
    cs = plt.contourf(
        X,
        Y,
        Z_plot,
        levels=levels,
        cmap="viridis",
        vmin=z_min_plot,
        vmax=z_cap,
    )
    cbar = plt.colorbar(cs, label="Volumetric Free Energy Density")
    cbar.set_ticks(np.linspace(z_min_plot, z_cap, 5))
    cbar.set_ticklabels(["-180000", "-90000", "0", "90000", "180000"])
    plt.xlabel("P1")
    plt.ylabel("P2")
    plt.title("Contour Plot Volumetric Free Energy Density")
    plt.gca().set_aspect("equal", adjustable="box")
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def _misalignment_score(P1_lab_flat, P2_lab_flat, c_arr, s_arr, nx, ny, ps):
    p1c = c_arr * P1_lab_flat + s_arr * P2_lab_flat
    p2c = -s_arr * P1_lab_flat + c_arr * P2_lab_flat
    mag = np.sqrt(p1c**2 + p2c**2)
    angle = np.degrees(np.arctan2(p2c, p1c)) % 360.0
    dist_to_nearest = np.abs(((angle % 90.0) + 45.0) % 90.0 - 45.0)
    score = dist_to_nearest / 45.0
    score = np.where(mag < 0.3 * ps, 0.25, 0.5 * score)
    return score.reshape(nx, ny)


def run_domain_simulation(sim: SimParams, mat: MaterialParams, output_dir="."):
    mesh = Grid2D(nx=sim.nx, ny=sim.ny, dx=sim.dx, dy=sim.dy)
    theta_left, theta_right = sim.theta_left_rad, sim.theta_right_rad
    cL, sL = np.cos(theta_left), np.sin(theta_left)
    cR, sR = np.cos(theta_right), np.sin(theta_right)
    x_coord, y_coord = mesh.x.value, mesh.y.value
    x_mid, y_mid = sim.nx * sim.dx / 2.0, sim.ny * sim.dy / 2.0
    c_arr = np.where(x_coord >= x_mid, cR, cL)
    s_arr = np.where(x_coord >= x_mid, sR, sL)
    np.random.seed(sim.random_seed)
    random_vector = np.random.randint(0, 4, mesh.numberOfCells)
    variants = np.array([[mat.ps, 0.0], [-mat.ps, 0.0], [0.0, mat.ps], [0.0, -mat.ps]])
    P1_init, P2_init = np.zeros(mesh.numberOfCells), np.zeros(mesh.numberOfCells)
    for i in range(mesh.numberOfCells):
        v = variants[random_vector[i]]
        ci, si = c_arr[i], s_arr[i]
        P1_init[i] = ci * v[0] - si * v[1]
        P2_init[i] = si * v[0] + ci * v[1]
    P1 = CellVariable(name="P1", mesh=mesh, value=P1_init, hasOld=True)
    P2 = CellVariable(name="P2", mesh=mesh, value=P2_init, hasOld=True)
    src1 = CellVariable(mesh=mesh, value=np.zeros(mesh.numberOfCells))
    src2 = CellVariable(mesh=mesh, value=np.zeros(mesh.numberOfCells))
    track_indices = gb_interface_tracked_indices(mesh, sim)
    tracker = GBTracker(mesh, c_arr, s_arr, mat.ps, sim.e0, x_mid, y_mid, track_indices)
    ex_lab = sim.e0 * np.cos(sim.phi_e_rad)
    ey_lab = sim.e0 * np.sin(sim.phi_e_rad)
    fig, ax = plt.subplots(figsize=(8, 8))
    X = mesh.x.value.reshape(sim.nx, sim.ny)
    Y = mesh.y.value.reshape(sim.nx, sim.ny)
    px_cmap = LinearSegmentedColormap.from_list("", [(0, (0, 0, 0, 0)), (1, (0, 0, 0, 0.25))])
    images = []
    for step in range(sim.steps):
        p1_lab, p2_lab = np.array(P1.value), np.array(P2.value)
        p1_crys = c_arr * p1_lab + s_arr * p2_lab
        p2_crys = -s_arr * p1_lab + c_arr * p2_lab
        ex_crys = c_arr * ex_lab + s_arr * ey_lab
        ey_crys = -s_arr * ex_lab + c_arr * ey_lab
        dF_dP1c = 2 * mat.alpha_1 * p1_crys + 4 * mat.alpha_11 * p1_crys**3 + 2 * mat.alpha_12 * p1_crys * p2_crys**2 + 6 * mat.alpha_111 * p1_crys**5 + mat.alpha_112 * (4 * p1_crys**3 * p2_crys**2 + 2 * p1_crys * p2_crys**4) - ex_crys
        dF_dP2c = 2 * mat.alpha_1 * p2_crys + 4 * mat.alpha_11 * p2_crys**3 + 2 * mat.alpha_12 * p2_crys * p1_crys**2 + 6 * mat.alpha_111 * p2_crys**5 + mat.alpha_112 * (2 * p1_crys**4 * p2_crys + 4 * p2_crys**3 * p1_crys**2) - ey_crys
        src1.value = c_arr * dF_dP1c - s_arr * dF_dP2c
        src2.value = s_arr * dF_dP1c + c_arr * dF_dP2c
        (TransientTerm(coeff=1 / mat.mobility, var=P1) == DiffusionTerm(coeff=mat.kp, var=P1) - src1).solve(var=P1, dt=sim.dt)
        (TransientTerm(coeff=1 / mat.mobility, var=P2) == DiffusionTerm(coeff=mat.kp, var=P2) - src2).solve(var=P2, dt=sim.dt)
        P1.updateOld()
        P2.updateOld()
        tracker.record(step, sim.dt, P1.value, P2.value)
        ax.clear()
        U = P1.value.reshape(sim.nx, sim.ny)
        V = P2.value.reshape(sim.nx, sim.ny)
        c_plot = c_arr.reshape(sim.nx, sim.ny)
        s_plot = s_arr.reshape(sim.nx, sim.ny)
        ftotal = free_energy_crystal(c_plot * U + s_plot * V, -s_plot * U + c_plot * V, c_plot * ex_lab + s_plot * ey_lab, -s_plot * ex_lab + c_plot * ey_lab, mat)
        q = sim.quiver_size
        ax.pcolormesh(X[::q, ::q], Y[::q, ::q], ftotal[::q, ::q], cmap=px_cmap, vmin=np.min(ftotal), vmax=np.max(ftotal))
        score_2d = _misalignment_score(P1.value, P2.value, c_arr, s_arr, sim.nx, sim.ny, mat.ps)
        ax.pcolormesh(X[::q, ::q], Y[::q, ::q], score_2d[::q, ::q], cmap="gray_r", vmin=0, vmax=1, shading="nearest", zorder=2)
        ax.quiver(X[::q, ::q], Y[::q, ::q], U[::q, ::q], V[::q, ::q], pivot="mid", color="blue", scale=5.0, zorder=3)
        for k in range(tracker.cell_indices.size):
            ax.plot(float(tracker.cell_x[k]), float(tracker.cell_y[k]), marker="o", markersize=7, color=tracked_cell_colors(tracker.cell_indices.size)[k % 10], markeredgecolor="black", markeredgewidth=0.6, zorder=5, linestyle="None")
        ax.axvline(x=x_mid, color="red", linewidth=1.5, linestyle="--")
        ax.set_title(f"time = {sim.dt * step:.4f} s | Left {sim.theta_left_deg:.0f}° | Right {sim.theta_right_deg:.0f}°", fontsize=10)
        ax.set_aspect("equal")
        frame_path = os.path.join(output_dir, f"frame_{step:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        images.append(imageio.imread(frame_path))
        os.remove(frame_path)
    plt.close(fig)
    return tracker, images


def run_vector_field_only(sim: SimParams, mat: MaterialParams, save_path="vector_field.png", show=True):
    """
    Build and plot only the polarization vector field from the initialized state.
    This is useful for quickly testing vector-field plotting without running
    the full time-stepping workflow and postprocessing outputs.
    """
    mesh = Grid2D(nx=sim.nx, ny=sim.ny, dx=sim.dx, dy=sim.dy)
    theta_left, theta_right = sim.theta_left_rad, sim.theta_right_rad
    cL, sL = np.cos(theta_left), np.sin(theta_left)
    cR, sR = np.cos(theta_right), np.sin(theta_right)
    x_coord = mesh.x.value
    x_mid = sim.nx * sim.dx / 2.0
    c_arr = np.where(x_coord >= x_mid, cR, cL)
    s_arr = np.where(x_coord >= x_mid, sR, sL)

    np.random.seed(sim.random_seed)
    random_vector = np.random.randint(0, 4, mesh.numberOfCells)
    variants = np.array([[mat.ps, 0.0], [-mat.ps, 0.0], [0.0, mat.ps], [0.0, -mat.ps]])
    p1_init = np.zeros(mesh.numberOfCells)
    p2_init = np.zeros(mesh.numberOfCells)
    for i in range(mesh.numberOfCells):
        v = variants[random_vector[i]]
        ci, si = c_arr[i], s_arr[i]
        p1_init[i] = ci * v[0] - si * v[1]
        p2_init[i] = si * v[0] + ci * v[1]

    X = mesh.x.value.reshape(sim.nx, sim.ny)
    Y = mesh.y.value.reshape(sim.nx, sim.ny)
    U = p1_init.reshape(sim.nx, sim.ny)
    V = p2_init.reshape(sim.nx, sim.ny)
    q = sim.quiver_size

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(
        X[::q, ::q],
        Y[::q, ::q],
        U[::q, ::q],
        V[::q, ::q],
        pivot="mid",
        color="blue",
        scale=5.0,
    )
    ax.axvline(x=x_mid, color="red", linewidth=1.5, linestyle="--", label="Grain boundary")
    ax.set_title(
        "Polarization vector field (initialized state)\n"
        f"Left {sim.theta_left_deg:.0f}°, Right {sim.theta_right_deg:.0f}°, E={sim.e0} V/m",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved vector-field plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_vector_field_evolution(sim: SimParams, mat: MaterialParams, output_dir=".", video_path=None, fps=20):
    """
    Run the full time evolution and render only the polarization vector field
    at each step into a video.
    """
    mesh = Grid2D(nx=sim.nx, ny=sim.ny, dx=sim.dx, dy=sim.dy)
    theta_left, theta_right = sim.theta_left_rad, sim.theta_right_rad
    cL, sL = np.cos(theta_left), np.sin(theta_left)
    cR, sR = np.cos(theta_right), np.sin(theta_right)
    x_coord = mesh.x.value
    x_mid = sim.nx * sim.dx / 2.0
    c_arr = np.where(x_coord >= x_mid, cR, cL)
    s_arr = np.where(x_coord >= x_mid, sR, sL)
    np.random.seed(sim.random_seed)
    random_vector = np.random.randint(0, 4, mesh.numberOfCells)
    variants = np.array([[mat.ps, 0.0], [-mat.ps, 0.0], [0.0, mat.ps], [0.0, -mat.ps]])
    p1_init = np.zeros(mesh.numberOfCells)
    p2_init = np.zeros(mesh.numberOfCells)
    for i in range(mesh.numberOfCells):
        v = variants[random_vector[i]]
        ci, si = c_arr[i], s_arr[i]
        p1_init[i] = ci * v[0] - si * v[1]
        p2_init[i] = si * v[0] + ci * v[1]

    P1 = CellVariable(name="P1", mesh=mesh, value=p1_init, hasOld=True)
    P2 = CellVariable(name="P2", mesh=mesh, value=p2_init, hasOld=True)
    src1 = CellVariable(mesh=mesh, value=np.zeros(mesh.numberOfCells))
    src2 = CellVariable(mesh=mesh, value=np.zeros(mesh.numberOfCells))
    ex_lab = sim.e0 * np.cos(sim.phi_e_rad)
    ey_lab = sim.e0 * np.sin(sim.phi_e_rad)
    X = mesh.x.value.reshape(sim.nx, sim.ny)
    Y = mesh.y.value.reshape(sim.nx, sim.ny)
    q = sim.quiver_size

    images = []
    fig, ax = plt.subplots(figsize=(8, 8))
    for step in range(sim.steps):
        p1_lab = np.array(P1.value)
        p2_lab = np.array(P2.value)
        p1_crys = c_arr * p1_lab + s_arr * p2_lab
        p2_crys = -s_arr * p1_lab + c_arr * p2_lab
        ex_crys = c_arr * ex_lab + s_arr * ey_lab
        ey_crys = -s_arr * ex_lab + c_arr * ey_lab
        dF_dP1c = 2 * mat.alpha_1 * p1_crys + 4 * mat.alpha_11 * p1_crys**3 + 2 * mat.alpha_12 * p1_crys * p2_crys**2 + 6 * mat.alpha_111 * p1_crys**5 + mat.alpha_112 * (4 * p1_crys**3 * p2_crys**2 + 2 * p1_crys * p2_crys**4) - ex_crys
        dF_dP2c = 2 * mat.alpha_1 * p2_crys + 4 * mat.alpha_11 * p2_crys**3 + 2 * mat.alpha_12 * p2_crys * p1_crys**2 + 6 * mat.alpha_111 * p2_crys**5 + mat.alpha_112 * (2 * p1_crys**4 * p2_crys + 4 * p2_crys**3 * p1_crys**2) - ey_crys
        src1.value = c_arr * dF_dP1c - s_arr * dF_dP2c
        src2.value = s_arr * dF_dP1c + c_arr * dF_dP2c
        (TransientTerm(coeff=1 / mat.mobility, var=P1) == DiffusionTerm(coeff=mat.kp, var=P1) - src1).solve(var=P1, dt=sim.dt)
        (TransientTerm(coeff=1 / mat.mobility, var=P2) == DiffusionTerm(coeff=mat.kp, var=P2) - src2).solve(var=P2, dt=sim.dt)
        P1.updateOld()
        P2.updateOld()

        U = P1.value.reshape(sim.nx, sim.ny)
        V = P2.value.reshape(sim.nx, sim.ny)
        ax.clear()
        score_2d = _misalignment_score(P1.value, P2.value, c_arr, s_arr, sim.nx, sim.ny, mat.ps)
        ax.pcolormesh(
            X[::q, ::q],
            Y[::q, ::q],
            score_2d[::q, ::q],
            cmap="gray_r",
            vmin=0,
            vmax=1,
            shading="nearest",
            zorder=2,
        )
        ax.quiver(
            X[::q, ::q],
            Y[::q, ::q],
            U[::q, ::q],
            V[::q, ::q],
            pivot="mid",
            color="blue",
            scale=5.0,
            zorder=3,
        )
        ax.axvline(x=x_mid, color="red", linewidth=1.5, linestyle="--", label="Grain boundary")
        ax.set_title(
            "Polarization vector field evolution\n"
            f"time = {sim.dt * step:.4f} s | Left {sim.theta_left_deg:.0f}° | Right {sim.theta_right_deg:.0f}° | E={sim.e0} V/m",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        frame_path = os.path.join(output_dir, f"vf_frame_{step:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        images.append(imageio.imread(frame_path))
        os.remove(frame_path)

    plt.close(fig)
    if video_path is None:
        video_path = os.path.join(output_dir, f"polarization_vector_field_evolution_{sim.theta_left_deg:.0f}_{sim.theta_right_deg:.0f}_deg.mp4")
    imageio.mimsave(video_path, images, fps=fps)
    print(f"Saved vector-field evolution video: {video_path}")
    return video_path


def run_tracker_only(sim: SimParams, mat: MaterialParams):
    """
    Run the evolution and record GB tracker history without rendering frames.
    Use this for standalone trajectory/time-series analyses.
    """
    mesh = Grid2D(nx=sim.nx, ny=sim.ny, dx=sim.dx, dy=sim.dy)
    theta_left, theta_right = sim.theta_left_rad, sim.theta_right_rad
    cL, sL = np.cos(theta_left), np.sin(theta_left)
    cR, sR = np.cos(theta_right), np.sin(theta_right)
    x_coord, y_coord = mesh.x.value, mesh.y.value
    x_mid, y_mid = sim.nx * sim.dx / 2.0, sim.ny * sim.dy / 2.0
    c_arr = np.where(x_coord >= x_mid, cR, cL)
    s_arr = np.where(x_coord >= x_mid, sR, sL)
    np.random.seed(sim.random_seed)
    random_vector = np.random.randint(0, 4, mesh.numberOfCells)
    variants = np.array([[mat.ps, 0.0], [-mat.ps, 0.0], [0.0, mat.ps], [0.0, -mat.ps]])
    p1_init, p2_init = np.zeros(mesh.numberOfCells), np.zeros(mesh.numberOfCells)
    for i in range(mesh.numberOfCells):
        v = variants[random_vector[i]]
        ci, si = c_arr[i], s_arr[i]
        p1_init[i] = ci * v[0] - si * v[1]
        p2_init[i] = si * v[0] + ci * v[1]

    P1 = CellVariable(name="P1", mesh=mesh, value=p1_init, hasOld=True)
    P2 = CellVariable(name="P2", mesh=mesh, value=p2_init, hasOld=True)
    src1 = CellVariable(mesh=mesh, value=np.zeros(mesh.numberOfCells))
    src2 = CellVariable(mesh=mesh, value=np.zeros(mesh.numberOfCells))
    track_indices = gb_interface_tracked_indices(mesh, sim)
    tracker = GBTracker(mesh, c_arr, s_arr, mat.ps, sim.e0, x_mid, y_mid, track_indices)

    ex_lab = sim.e0 * np.cos(sim.phi_e_rad)
    ey_lab = sim.e0 * np.sin(sim.phi_e_rad)
    for step in range(sim.steps):
        p1_lab = np.array(P1.value)
        p2_lab = np.array(P2.value)
        p1_crys = c_arr * p1_lab + s_arr * p2_lab
        p2_crys = -s_arr * p1_lab + c_arr * p2_lab
        ex_crys = c_arr * ex_lab + s_arr * ey_lab
        ey_crys = -s_arr * ex_lab + c_arr * ey_lab
        dF_dP1c = 2 * mat.alpha_1 * p1_crys + 4 * mat.alpha_11 * p1_crys**3 + 2 * mat.alpha_12 * p1_crys * p2_crys**2 + 6 * mat.alpha_111 * p1_crys**5 + mat.alpha_112 * (4 * p1_crys**3 * p2_crys**2 + 2 * p1_crys * p2_crys**4) - ex_crys
        dF_dP2c = 2 * mat.alpha_1 * p2_crys + 4 * mat.alpha_11 * p2_crys**3 + 2 * mat.alpha_12 * p2_crys * p1_crys**2 + 6 * mat.alpha_111 * p2_crys**5 + mat.alpha_112 * (2 * p1_crys**4 * p2_crys + 4 * p2_crys**3 * p1_crys**2) - ey_crys
        src1.value = c_arr * dF_dP1c - s_arr * dF_dP2c
        src2.value = s_arr * dF_dP1c + c_arr * dF_dP2c
        (TransientTerm(coeff=1 / mat.mobility, var=P1) == DiffusionTerm(coeff=mat.kp, var=P1) - src1).solve(var=P1, dt=sim.dt)
        (TransientTerm(coeff=1 / mat.mobility, var=P2) == DiffusionTerm(coeff=mat.kp, var=P2) - src2).solve(var=P2, dt=sim.dt)
        P1.updateOld()
        P2.updateOld()
        tracker.record(step, sim.dt, P1.value, P2.value)

    return tracker
