"""Candelier cage-jump and chi4 helper functions."""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import minimum_image, wrap_points


def compute_p_function(traj):
    """Return Candelier/Zeng Appendix-B p(tc) for all interior cut times.

    The trajectory is split at frame ``tc`` into two successive subsets,
    ``S1 = traj[:tc+1]`` and ``S2 = traj[tc+1:]``.  For each candidate cut,
    this implements

        p(tc) = zeta(tc) * sqrt(<d1^2>_S2 * <d2^2>_S1)

    where ``d1`` is measured from the center of mass of ``S1`` and ``d2`` from
    the center of mass of ``S2``.  The returned ``tc`` values are integer frame
    indices inside the supplied sub-trajectory.
    """
    traj = np.asarray(traj, dtype=float)
    T = len(traj)
    if T < 4:
        return np.array([], dtype=int), np.array([], dtype=float)
    tc = np.arange(1, T - 1)
    n1 = tc + 1
    n2 = T - tc - 1
    prefix = np.zeros((T + 1, traj.shape[1]), dtype=float)
    prefix[1:] = np.cumsum(traj, axis=0)
    sq = np.einsum("ij,ij->i", traj, traj)
    sq_prefix = np.zeros(T + 1, dtype=float)
    sq_prefix[1:] = np.cumsum(sq)
    c1 = prefix[tc + 1] / n1[:, None]
    c2 = (prefix[T] - prefix[tc + 1]) / n2[:, None]
    mean_sq_s2 = (sq_prefix[T] - sq_prefix[tc + 1]) / n2
    mean_sq_s1 = sq_prefix[tc + 1] / n1
    d1 = mean_sq_s2 - 2.0 * np.einsum("ij,ij->i", c1, c2) + np.einsum("ij,ij->i", c1, c1)
    d2 = mean_sq_s1 - 2.0 * np.einsum("ij,ij->i", c2, c1) + np.einsum("ij,ij->i", c2, c2)
    d1 = np.maximum(d1, 0.0)
    d2 = np.maximum(d2, 0.0)
    zeta = np.sqrt((tc / float(T)) * (1.0 - tc / float(T)))
    return tc, zeta * np.sqrt(d1 * d2)


def find_cage_jumps_recursive(traj, lc2, offset=0, min_segment=4):
    traj = np.asarray(traj, dtype=float)
    if len(traj) < min_segment:
        return []
    tc, p = compute_p_function(traj)
    if len(p) == 0:
        return []
    imax = int(np.argmax(p))
    if float(p[imax]) < float(lc2):
        return []
    cut = int(tc[imax])
    left = find_cage_jumps_recursive(traj[: cut + 1], lc2, offset, min_segment)
    right = find_cage_jumps_recursive(traj[cut + 1 :], lc2, offset + cut + 1, min_segment)
    return [offset + cut] + left + right


def chi4_from_nonaffine(nonaffine, box, overlap_a, max_lag, origins=None, sample_n=0):
    """Compute chi4 lag curve using the shear-adjusted nonaffine positions."""
    arr = np.asarray(nonaffine, dtype=float)
    frames, particles = arr.shape[:2]
    max_lag = min(int(max_lag), frames - 2)
    if origins is None:
        origins = np.arange(0, frames - max_lag - 1)
    origins = np.asarray(origins, dtype=int)
    if sample_n and sample_n < particles:
        rng = np.random.default_rng(12345)
        sel = np.sort(rng.choice(particles, int(sample_n), replace=False))
    else:
        sel = slice(None)
    box = np.asarray(box, dtype=float)
    ref_trees = []
    for t0 in origins:
        ref = wrap_points(arr[t0, sel], box)
        ref_trees.append(cKDTree(ref, boxsize=box))
    chi4 = np.zeros(max_lag + 1, dtype=float)
    q_mean = np.zeros(max_lag + 1, dtype=float)
    for lag in range(1, max_lag + 1):
        qvals = np.zeros(len(origins), dtype=float)
        for oi, t0 in enumerate(origins):
            cur = wrap_points(arr[t0 + lag, sel], box)
            cur_tree = cKDTree(cur, boxsize=box)
            qvals[oi] = ref_trees[oi].count_neighbors(cur_tree, overlap_a)
        chi4[lag] = np.var(qvals)
        q_mean[lag] = np.mean(qvals)
    return chi4, q_mean


def cage_relative_displacement_2d(positions, neighbor_cutoff, box):
    """Appendix B Eq. (B2)-(B3) displacement correction for 2D samples."""
    pos = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    ref = pos[0]
    tree = cKDTree(wrap_points(ref, box), boxsize=box)
    neigh = tree.query_ball_point(wrap_points(ref, box), neighbor_cutoff)
    disp = minimum_image(pos - ref[None, :, :], box)
    out = np.zeros_like(disp)
    for i, ids in enumerate(neigh):
        ids = [j for j in ids if j != i]
        if ids:
            out[:, i, :] = disp[:, i, :] - np.mean(disp[:, ids, :], axis=1)
        else:
            out[:, i, :] = disp[:, i, :]
    return out
