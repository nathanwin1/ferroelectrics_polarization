"""
bezier_classify.py
------------------
Bézier curve fitting and unsupervised classification of polarization
trajectories from GBTracker.export_history().

Dependencies: numpy, scipy, sklearn
"""

import numpy as np
from scipy.special import comb
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
    """Bernstein basis polynomial B_{k,n}(t). t must be in [0, 1]."""
    return comb(n, k, exact=False) * (t**k) * ((1 - t) ** (n - k))


def bezier_curve(control_pts: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """
    Evaluate a Bézier curve at t_vals.
    control_pts: shape (degree+1, 2)
    t_vals:      shape (M,)
    Returns:     shape (M, 2)
    """
    n = len(control_pts) - 1
    curve = np.zeros((len(t_vals), 2))
    for k, cp in enumerate(control_pts):
        b = bernstein(n, k, t_vals)[:, None]
        curve += b * cp
    return curve


def history_cell_time_arrays(history: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return mag_crys and angle_crys_rad as (n_cells, n_steps).

    The source tracker stores one row per recorded timestep in this project
    ((n_steps, n_cells)), while older notes may describe the transposed layout.
    Use cell_indices as the authoritative tracked-cell count.
    """
    mag = np.asarray(history["mag_crys"])
    ang = np.asarray(history["angle_crys_rad"])

    if mag.shape != ang.shape:
        raise ValueError(
            "history['mag_crys'] and history['angle_crys_rad'] must have matching shapes; "
            f"got {mag.shape} and {ang.shape}."
        )

    cell_indices = history.get("cell_indices")
    if cell_indices is None:
        return mag, ang

    n_cells = len(np.asarray(cell_indices).ravel())
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D trajectory arrays, got shape {mag.shape}.")

    if mag.shape[0] == n_cells:
        return mag, ang
    if mag.shape[1] == n_cells:
        return mag.T, ang.T

    raise ValueError(
        "Could not align history arrays with tracked cells: "
        f"mag_crys shape is {mag.shape}, but len(cell_indices) is {n_cells}."
    )


def fit_bezier(xy_points: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Fit a Bézier curve of given degree to xy_points (N, 2).

    Parameterization: chord-length (cumulative arc-length normalized to [0,1]).
    Endpoints are pinned: P0 = xy_points[0], P_n = xy_points[-1].
    Interior control points solved via least squares.

    Returns control_points of shape (degree+1, 2).
    Raises ValueError if N < degree + 2.
    """
    N = len(xy_points)
    n_ctrl = degree + 1
    if N < n_ctrl + 1:
        raise ValueError(
            f"Need at least {n_ctrl + 1} points to fit degree-{degree} Bézier, got {N}."
        )

    # Chord-length parameterization
    diffs = np.diff(xy_points, axis=0)
    chord_lens = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(chord_lens)])
    total = t[-1]
    if total < 1e-12:
        # Degenerate trajectory (no movement)
        ctrl = np.tile(xy_points[0], (n_ctrl, 1))
        return ctrl
    t /= total  # normalize to [0, 1]

    # Build Bernstein matrix B: shape (N, n_ctrl)
    B = np.column_stack([bernstein(degree, k, t) for k in range(n_ctrl)])

    # Pin endpoints: enforce P0 = xy_points[0], P_n = xy_points[-1]
    # Solve for interior control points (indices 1 to degree-1)
    if degree >= 2:
        B_inner = B[:, 1:-1]  # (N, degree-1)
        rhs = xy_points - B[:, [0]] * xy_points[0] - B[:, [-1]] * xy_points[-1]
        C_inner, _, _, _ = np.linalg.lstsq(B_inner, rhs, rcond=None)
        ctrl = np.vstack([xy_points[0], C_inner, xy_points[-1]])
    else:
        # Linear: only endpoints
        ctrl = np.vstack([xy_points[0], xy_points[-1]])

    return ctrl  # shape (degree+1, 2)


def extract_bezier_features(ctrl_pts: np.ndarray) -> dict:
    """
    Extract classification features from Bézier control points.

    Works for any degree >= 1, but designed for cubic (degree=3).

    Features:
      displacement   — ||P_n - P_0||, net start-to-end distance
      poly_len       — total length of control polygon (arc length proxy)
      ctrl_area      — |shoelace area| of control point polygon
      curvature_mid  — signed curvature at t=0.5 (cubic only, else 0)
      winding        — ctrl_area / poly_len^2  (loop detection ratio)
    """
    P = ctrl_pts  # (n_ctrl, 2)
    n = len(P) - 1  # degree

    displacement = float(np.linalg.norm(P[-1] - P[0]))

    segs = np.diff(P, axis=0)
    poly_len = float(np.sum(np.linalg.norm(segs, axis=1)))

    # Shoelace area of control polygon
    x, y = P[:, 0], P[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    # Signed curvature at t=0.5 for cubic Bézier
    if n == 3:
        P0, P1, P2, P3 = P
        t = 0.5
        # First derivative (tangent)
        tangent = (
            3 * (1 - t) ** 2 * (P1 - P0)
            + 6 * (1 - t) * t * (P2 - P1)
            + 3 * t**2 * (P3 - P2)
        )
        # Second derivative
        d2 = 6 * (1 - t) * (P2 - 2 * P1 + P0) + 6 * t * (P3 - 2 * P2 + P1)
        speed = np.linalg.norm(tangent)
        curvature = float(np.cross(tangent, d2)) / (speed**3 + 1e-12)
    else:
        curvature = 0.0

    winding = area / (poly_len**2 + 1e-12)

    return {
        "displacement": displacement,
        "poly_len": poly_len,
        "ctrl_area": area,
        "curvature_mid": curvature,
        "winding": winding,
    }


def count_inflections_1d(y: np.ndarray, smooth_window: int = 1) -> int:
    """
    Count sign changes in the second derivative of a 1D trajectory.

    This mirrors the tracker's discrete inflection proxy but keeps Bézier
    classification self-contained.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 3:
        return 0

    w = int(smooth_window)
    if w > 1:
        if w % 2 == 0:
            w += 1
        pad = w // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=float) / float(w)
        y = np.convolve(yp, kernel, mode="valid")

    d1 = np.gradient(y)
    d2 = np.gradient(d1)
    return int(np.sum(d2[:-1] * d2[1:] < 0))


def convex_hull_area(xy_points: np.ndarray) -> float:
    """Area enclosed by the trajectory's convex hull, or 0 for degenerate paths."""
    xy = np.asarray(xy_points, dtype=float)
    if xy.shape[0] < 3:
        return 0.0

    try:
        return float(ConvexHull(xy).volume)
    except QhullError:
        return 0.0


def extract_trajectory_features(
    mag_series: np.ndarray,
    ang_series: np.ndarray,
    xy_points: np.ndarray,
    ctrl_features: dict,
    smooth_window: int = 1,
) -> dict:
    """
    Add topology-aware trajectory features to the Bézier control-point features.

    These features capture switching angle, radial excursions, oscillations, and
    true phase-space area from the sampled trajectory rather than the control
    polygon alone.
    """
    mag_series = np.asarray(mag_series, dtype=float).ravel()
    ang_unwrapped = np.unwrap(np.asarray(ang_series, dtype=float).ravel())

    displacement = float(ctrl_features["displacement"])
    poly_len = float(ctrl_features["poly_len"])
    directness = displacement / (poly_len + 1e-12)

    feats = dict(ctrl_features)
    feats.update(
        {
            "net_angle_change_rad": float(ang_unwrapped[-1] - ang_unwrapped[0]),
            "abs_net_angle_change_rad": float(abs(ang_unwrapped[-1] - ang_unwrapped[0])),
            "max_radius": float(np.max(mag_series)),
            "radius_range": float(np.max(mag_series) - np.min(mag_series)),
            "n_inflections_mag": float(
                count_inflections_1d(mag_series, smooth_window=smooth_window)
            ),
            "n_inflections_theta": float(
                count_inflections_1d(ang_unwrapped, smooth_window=smooth_window)
            ),
            "convex_hull_area": convex_hull_area(xy_points),
            "path_directness": directness,
        }
    )
    return feats


def classify_trajectories(
    history: dict,
    degree: int = 3,
    n_clusters: int = 3,
    random_state: int = 0,
    smooth_window: int = 1,
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """
    Fit Bézier curves to all tracked cell trajectories and cluster them.

    Parameters
    ----------
    history : dict
        Output of GBTracker.export_history(). Expected keys:
          'mag_crys'       — shape (n_cells, n_steps)
          'angle_crys_rad' — shape (n_cells, n_steps)

    degree : int
        Bézier degree (default 3 = cubic).

    n_clusters : int
        Number of KMeans clusters.

    random_state : int
        KMeans random seed.

    smooth_window : int
        Optional moving-average window before inflection counting.

    Returns
    -------
    ctrl_pts_list : list of np.ndarray
        One (degree+1, 2) control point array per cell.
        Cells that failed fitting get None.

    labels : np.ndarray, shape (n_cells,)
        Cluster label per cell. Failed cells get label -1.

    features_list : list of dict
        Feature dicts per cell (None for failed cells).
    """
    mag, ang = history_cell_time_arrays(history)  # (n_cells, n_steps)

    # Convert polar (angle, mag) → Cartesian (Px_c, Py_c) in crystal frame
    n_cells = mag.shape[0]
    ctrl_pts_list = [None] * n_cells
    features_list = [None] * n_cells
    labels = np.full(n_cells, -1, dtype=int)

    valid_indices = []
    valid_features = []

    for k in range(n_cells):
        px = mag[k] * np.cos(ang[k])
        py = mag[k] * np.sin(ang[k])
        xy = np.column_stack([px, py])

        try:
            ctrl = fit_bezier(xy, degree=degree)
            ctrl_features = extract_bezier_features(ctrl)
            feats = extract_trajectory_features(
                mag_series=mag[k],
                ang_series=ang[k],
                xy_points=xy,
                ctrl_features=ctrl_features,
                smooth_window=smooth_window,
            )
            ctrl_pts_list[k] = ctrl
            features_list[k] = feats
            valid_indices.append(k)
            valid_features.append(list(feats.values()))
        except ValueError:
            # Too few points — leave as None / label -1
            continue

    if len(valid_indices) >= n_clusters:
        X = StandardScaler().fit_transform(valid_features)
        cluster_labels = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        ).fit_predict(X)
        for i, k in enumerate(valid_indices):
            labels[k] = cluster_labels[i]
    elif len(valid_indices) > 0:
        # Fewer valid trajectories than clusters — assign all to cluster 0
        for k in valid_indices:
            labels[k] = 0

    return ctrl_pts_list, labels, features_list
