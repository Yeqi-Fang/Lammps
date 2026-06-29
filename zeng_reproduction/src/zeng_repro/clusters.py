"""Dynamic-region demarcation following Appendix B."""

import numpy as np

from .geometry import (
    GridClusterMask,
    connected_components,
    exponential_density,
    histogram_density,
    iterative_threshold,
)
from .paper_constants import SYSTEM_BD, SYSTEM_MD2D, bd_coarse_length


def stz_min_voxels(system, box, grid_shape, c_prime=2.0, stz_diameter=3.0):
    dim = len(box)
    if dim == 2:
        vstz = np.pi * (0.5 * stz_diameter) ** 2
    else:
        vstz = 4.0 * np.pi * (0.5 * stz_diameter) ** 3 / 3.0
    voxel_volume = float(np.prod(np.asarray(box) / np.asarray(grid_shape)))
    return max(1, int(np.ceil(c_prime * vstz / voxel_volume)))


def demarcate_dynamic_regions(system, points_ref, box, shear_rate, gr_first_min=None,
                              grid_shape=None, cluster_id=None):
    points_ref = np.asarray(points_ref, dtype=float)
    box = np.asarray(box, dtype=float)
    dim = len(box)
    if grid_shape is None:
        if system == SYSTEM_MD2D:
            grid_shape = (80, 80)
        else:
            grid_shape = (30,) * dim

    if system == SYSTEM_BD:
        coarse = bd_coarse_length(float(np.max(box)), shear_rate)
        density = histogram_density(points_ref, box, grid_shape)
    else:
        if gr_first_min is None:
            raise ValueError("MD cluster demarcation requires gr_first_min as coarse length d")
        coarse = float(gr_first_min)
        density = exponential_density(points_ref, box, grid_shape, coarse)

    rho_th = iterative_threshold(density)
    min_vox = stz_min_voxels(system, box, grid_shape)
    labeled, clusters = connected_components(density, rho_th, min_voxels=min_vox)
    if not clusters:
        labeled, clusters = connected_components(density, rho_th * 0.5, min_voxels=1)
    if not clusters:
        raise RuntimeError("No cage-jump cluster found above threshold")

    if cluster_id is None:
        chosen = clusters[0]["id"]
        chosen_note = "largest cluster selected"
    else:
        chosen = int(cluster_id)
        chosen_note = "cluster_id selected from config"
        if chosen not in [c["id"] for c in clusters]:
            raise ValueError("cluster_id {} not present in connected components".format(chosen))

    mask = GridClusterMask(box, grid_shape, labeled, chosen)
    return {
        "mask": mask,
        "density": density,
        "rho_th": rho_th,
        "clusters": clusters,
        "coarse_length": coarse,
        "selected_cluster_id": chosen,
        "selection_note": chosen_note,
    }
