"""Geometry helpers for steady shear and cluster masks."""

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree

from .paper_constants import THRESHOLD_TOL


def affine_to_reference(points, point_times, shear_rate, reference_time):
    """Map lab-frame positions to the reference time by affine shear transport."""
    pts = np.asarray(points, dtype=float).copy()
    t = np.asarray(point_times, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    if t.ndim == 0:
        t = np.full(len(pts), float(t))
    pts[:, 0] += shear_rate * (reference_time - t) * pts[:, 1]
    return pts


def wrap_points(points, box):
    pts = np.asarray(points, dtype=float).copy()
    b = np.asarray(box, dtype=float)
    return np.mod(pts, b)


def minimum_image(displacements, box):
    disp = np.asarray(displacements, dtype=float).copy()
    b = np.asarray(box, dtype=float)
    return disp - b * np.round(disp / b)


def iterative_threshold(density, tol=THRESHOLD_TOL):
    rho = np.asarray(density, dtype=float).ravel()
    if len(rho) == 0:
        return 0.0
    rho_i = float(np.max(rho))
    if rho_i <= 0:
        return 0.0
    for _ in range(200):
        below = rho[rho < rho_i]
        if len(below) == 0:
            break
        rho_avg = float(np.mean(below))
        if rho_avg > 0 and abs(rho_avg - rho_i) / rho_avg < tol:
            return rho_avg
        if rho_i <= rho_avg:
            break
        rho_i = rho_avg
    return float(rho_i)


def histogram_density(points, box, grid_shape):
    pts = wrap_points(points, box)
    grid_shape = tuple(int(x) for x in grid_shape)
    idx = np.floor(pts / np.asarray(box) * np.asarray(grid_shape)).astype(int)
    for d in range(idx.shape[1]):
        idx[:, d] = np.clip(idx[:, d], 0, grid_shape[d] - 1)
    density = np.zeros(grid_shape, dtype=float)
    for row in idx:
        density[tuple(row)] += 1.0
    return density


def exponential_density(points, box, grid_shape, coarse_length):
    """MD coarse-graining with kernel proportional to exp(-r/d)."""
    grid_shape = tuple(int(x) for x in grid_shape)
    box = np.asarray(box, dtype=float)
    points = wrap_points(points, box)
    axes = []
    for L, n in zip(box, grid_shape):
        axes.append((np.arange(n, dtype=float) + 0.5) * L / n)
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_points = np.column_stack([m.ravel() for m in mesh])
    density = np.zeros(len(grid_points), dtype=float)
    if len(points) == 0:
        return density.reshape(grid_shape)
    tree = cKDTree(points, boxsize=box)
    cutoff = max(6.0 * coarse_length, coarse_length)
    for gi, gp in enumerate(grid_points):
        ids = tree.query_ball_point(gp, cutoff)
        if ids:
            dr = minimum_image(points[ids] - gp, box)
            rr = np.linalg.norm(dr, axis=1)
            density[gi] = np.sum(np.exp(-rr / coarse_length))
    return density.reshape(grid_shape)


def connected_components(density, threshold, min_voxels=1):
    binary = np.asarray(density) >= float(threshold)
    labeled, nlab = nd_label(binary)
    clusters = []
    for cid in range(1, nlab + 1):
        vox = np.argwhere(labeled == cid)
        if len(vox) < min_voxels:
            continue
        clusters.append({"id": cid, "voxels": vox, "n_voxels": len(vox)})
    clusters.sort(key=lambda c: c["n_voxels"], reverse=True)
    return labeled, clusters


class GridClusterMask(object):
    def __init__(self, box, grid_shape, labeled, cluster_id):
        self.box = np.asarray(box, dtype=float)
        self.grid_shape = np.asarray(grid_shape, dtype=int)
        self.labeled = np.asarray(labeled, dtype=int)
        self.cluster_id = int(cluster_id)

    def contains(self, points):
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        pts = wrap_points(points, self.box)
        idx = np.floor(pts / self.box * self.grid_shape).astype(int)
        for d in range(idx.shape[1]):
            idx[:, d] = np.clip(idx[:, d], 0, self.grid_shape[d] - 1)
        return self.labeled[tuple(idx.T)] == self.cluster_id

    def equivalent_radius(self):
        voxel_volume = float(np.prod(self.box / self.grid_shape))
        n_vox = int(np.sum(self.labeled == self.cluster_id))
        vol = n_vox * voxel_volume
        dim = len(self.box)
        if dim == 2:
            return float(np.sqrt(vol / np.pi))
        return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))

    def center(self):
        vox = np.argwhere(self.labeled == self.cluster_id)
        if len(vox) == 0:
            return np.zeros(len(self.box))
        return np.mean((vox + 0.5) * self.box / self.grid_shape, axis=0)
