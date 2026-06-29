"""Stress and local strain definitions from Eqs. (2)-(7)."""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import minimum_image, wrap_points
from .paper_constants import SYSTEM_BD, SYSTEM_MD2D, SYSTEM_MD3D


def particle_volumes(types, box, system, diameter_ratio_small_to_big=2.0 / 3.0):
    types = np.asarray(types, dtype=int)
    V = float(np.prod(box))
    nb = int(np.sum(types == 1))
    ns = int(np.sum(types == 2))
    dim = 2 if system == SYSTEM_MD2D else 3
    if system == SYSTEM_MD2D:
        # 2D MD has sigma_b/sigma_s=1.4 and units d0=sigma_s.
        vb_vs = 1.4 ** dim
    else:
        # BD: ds/db=2/3. 3D KA: use sigma_s/sigma_b=0.88 only for volume weights
        # if exact Voronoi volumes are absent.
        if system == SYSTEM_MD3D:
            small_to_big = 0.88
        else:
            small_to_big = diameter_ratio_small_to_big
        vb_vs = (1.0 / small_to_big) ** dim
    vs = V / (ns + nb * vb_vs)
    vb = vb_vs * vs
    return np.where(types == 1, vb, vs)


def dvdr(system, r, type_i, type_j):
    if system == SYSTEM_MD3D:
        eps, sig = lj_params_ka(type_i, type_j)
        sr = sig / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        return 4.0 * eps * (-12.0 * sr12 / r + 6.0 * sr6 / r)
    if system == SYSTEM_MD2D:
        sigma = soft2d_sigma(type_i, type_j)
        sr = sigma / r
        return -12.0 * (sr ** 12) / r
    if system == SYSTEM_BD:
        dij = bd_contact_distance(type_i, type_j)
        z = 4.86
        K = 9.69
        # V(r)=K*dij*exp[-z(r-dij)]/r
        expv = np.exp(-z * (r - dij))
        return K * dij * expv * (-z / r - 1.0 / (r * r))
    raise ValueError("unknown system {}".format(system))


def lj_params_ka(type_i, type_j):
    pair = tuple(sorted((int(type_i), int(type_j))))
    if pair == (1, 1):
        return 1.0, 1.0
    if pair == (1, 2):
        return 1.5, 0.8
    return 0.5, 0.88


def soft2d_sigma(type_i, type_j):
    sig = {1: 1.4, 2: 1.0}
    return 0.5 * (sig[int(type_i)] + sig[int(type_j)])


def bd_contact_distance(type_i, type_j):
    diam = {1: 1.0, 2: 2.0 / 3.0}
    return 0.5 * (diam[int(type_i)] + diam[int(type_j)])


def potential_cutoff(system):
    if system == SYSTEM_MD3D:
        return 2.5
    if system == SYSTEM_MD2D:
        return 4.5
    if system == SYSTEM_BD:
        return 5.0
    raise ValueError("unknown system {}".format(system))


def per_atom_stress_xy_from_positions(positions, types, box, system, particle_ids=None):
    """Compute Eq. (2) xy stress for selected particles from pair distances."""
    pos = np.asarray(positions, dtype=float)
    types = np.asarray(types, dtype=int)
    box = np.asarray(box, dtype=float)
    if particle_ids is None:
        particle_ids = np.arange(len(pos), dtype=int)
    else:
        particle_ids = np.asarray(particle_ids, dtype=int)
    vols = particle_volumes(types, box, system)
    tree = cKDTree(wrap_points(pos, box), boxsize=box)
    cutoff = potential_cutoff(system)
    out = np.zeros(len(particle_ids), dtype=float)
    for oi, i in enumerate(particle_ids):
        neigh = tree.query_ball_point(wrap_points(pos[i], box), cutoff)
        total = 0.0
        for j in neigh:
            if j == i:
                continue
            dr = minimum_image(pos[j] - pos[i], box)
            r = float(np.linalg.norm(dr))
            if r <= 1.0e-12 or r > cutoff:
                continue
            total += dvdr(system, r, types[i], types[j]) * dr[0] * dr[1] / r
        out[oi] = total / vols[i]
    return out


def cluster_stress_sigma_c(per_atom_stress_xy, volumes, cluster_ids):
    ids = np.asarray(cluster_ids, dtype=int)
    if len(ids) == 0:
        return np.nan
    vf = float(np.sum(volumes[ids]))
    if vf <= 0:
        return np.nan
    return float(np.sum(per_atom_stress_xy[ids] * volumes[ids]) / vf)


def local_shear_strain_gamma_c(pos_minus, pos_plus, cluster_ids):
    """Falk-Langer local strain xy component, Eqs. (5)-(7)."""
    ids = np.asarray(cluster_ids, dtype=int)
    if len(ids) < 3:
        return np.nan
    a = np.asarray(pos_plus[ids], dtype=float)
    b = np.asarray(pos_minus[ids], dtype=float)
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    X = a.T.dot(b)
    Y = b.T.dot(b)
    try:
        F = X.dot(np.linalg.pinv(Y))
    except np.linalg.LinAlgError:
        return np.nan
    if F.shape[0] < 2 or F.shape[1] < 2:
        return np.nan
    return float(F[0, 1])
