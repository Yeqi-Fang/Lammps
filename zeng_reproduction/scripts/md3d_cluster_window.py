#!/usr/bin/env python
"""Windowed cage-jump dynamic-region search for Zeng 3D MD.

This script is intentionally scoped to the first strict step after chi4:
select t_chi-long windows, coarse-grain cage-jump positions, and plot the
selected dynamic-region boundary in the shear plane.  It does not claim to
finish the full Fig. 6 stress/strain analysis; that requires tracing this
boundary and summing per-particle stresses in later steps.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generate_binary_structure, label, maximum_filter
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SRC))

from md3d_cage_jump_trial import (  # noqa: E402
    first_frame_selection,
    fold_positions_to_box,
    periodic_points,
    stream_window,
)


def load_first_frame_positions(dump_path):
    selection = first_frame_selection(dump_path, 0, 0, 42)
    window = stream_window(dump_path, selection["selected_ids"], 0, 2)
    box = window["boxes"][0]
    pos = fold_positions_to_box(window["wrapped"][0], box)
    lengths = np.asarray([box["Lx"], box["Ly"], box["Lz"]], dtype=np.float64)
    return periodic_points(pos, lengths), box


def load_particle_positions_at_frame(dump_path, particle_type, frame, seed=42):
    selection = first_frame_selection(dump_path, int(particle_type), 0, int(seed))
    window = stream_window(dump_path, selection["selected_ids"], int(frame), 2)
    box = window["boxes"][0]
    lengths = np.asarray([box["Lx"], box["Ly"], box["Lz"]], dtype=np.float64)
    pos = fold_positions_to_box(window["wrapped"][0], box)
    return periodic_points(pos, lengths), selection, box


def load_cage_source_summary(jumps_path):
    summary_path = Path(jumps_path).with_name("trial_summary.json")
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc), "path": str(summary_path)}


def count_particles_in_components(particle_positions, labels, components, box_lengths, grid_n, point_mode="centers"):
    if particle_positions is None or len(particle_positions) == 0:
        return 0
    selected = np.asarray([int(c) for c in components], dtype=np.int32)
    if selected.size == 0:
        return 0
    idx = assign_points_to_labeled_grid(particle_positions, box_lengths, int(grid_n), point_mode)
    particle_labels = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    return int(np.count_nonzero(np.isin(particle_labels, selected)))


def particle_count_frame(start_frame, window_frames, mode):
    mode = str(mode).lower()
    if mode == "start":
        return int(start_frame)
    if mode == "end":
        return int(start_frame + window_frames - 1)
    if mode == "center":
        return int(start_frame + window_frames // 2)
    raise ValueError("particle count frame must be start, center, or end")


def estimate_first_minimum_gr(pos, box_lengths, rmax=3.0, dr=0.01):
    tree = cKDTree(periodic_points(pos, box_lengths), boxsize=box_lengths)
    pairs = np.asarray(list(tree.query_pairs(rmax)), dtype=np.int32)
    if len(pairs) == 0:
        return 1.2, {"status": "fallback_no_pairs"}

    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    delta -= box_lengths * np.round(delta / box_lengths)
    dist = np.sqrt(np.sum(delta * delta, axis=1))
    bins = np.arange(0.0, rmax + dr, dr)
    counts, edges = np.histogram(dist, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])

    n = len(pos)
    volume = float(np.prod(box_lengths))
    rho = n / volume
    shell = 4.0 * math.pi * r * r * dr
    g = (2.0 * counts) / (n * rho * shell)

    kernel = np.ones(7, dtype=np.float64) / 7.0
    g_s = np.convolve(g, kernel, mode="same")
    valid_peak = np.where((r > 0.75) & (r < 1.35))[0]
    if len(valid_peak) == 0:
        return 1.2, {"status": "fallback_no_peak_window"}
    peak_i = int(valid_peak[np.argmax(g_s[valid_peak])])

    search = np.where((np.arange(len(r)) > peak_i) & (r < 2.2))[0]
    if len(search) < 3:
        return 1.2, {"status": "fallback_no_min_window", "peak_r": float(r[peak_i])}
    mins = []
    for i in search[1:-1]:
        if g_s[i] <= g_s[i - 1] and g_s[i] <= g_s[i + 1]:
            mins.append(int(i))
    if not mins:
        min_i = int(search[np.argmin(g_s[search])])
    else:
        min_i = mins[0]
    return float(r[min_i]), {
        "status": "estimated",
        "peak_r": float(r[peak_i]),
        "first_min_r": float(r[min_i]),
        "rmax": float(rmax),
        "dr": float(dr),
        "n_pairs": int(len(pairs)),
    }


def periodic_kernel(grid_n, box_lengths, coarse_d):
    spacing = box_lengths / float(grid_n)
    axes = []
    for n, dx in zip([grid_n] * 3, spacing):
        idx = np.arange(n, dtype=np.float64)
        axes.append(np.minimum(idx, n - idx) * dx)
    x, y, z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    ker = np.exp(-r / float(coarse_d))
    ker /= np.sum(ker)
    return ker


def coarse_density_kernel_metadata(grid_n, box_lengths, coarse_d):
    ker = periodic_kernel(int(grid_n), np.asarray(box_lengths, dtype=np.float64), float(coarse_d))
    return {
        "kernel": "periodic_fft_exp_minus_r_over_d",
        "normalization": "kernel_sum_one_then_divide_by_voxel_volume",
        "kernel_sum": float(np.sum(ker)),
        "grid_n": int(grid_n),
        "coarse_d": float(coarse_d),
        "voxel_volume": float(np.prod(np.asarray(box_lengths, dtype=np.float64) / float(grid_n))),
    }


def coarse_density(positions, box_lengths, grid_n, coarse_d):
    pts = periodic_points(positions, box_lengths)
    scaled = pts / box_lengths
    hist, _ = np.histogramdd(
        scaled,
        bins=(grid_n, grid_n, grid_n),
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    ker = periodic_kernel(grid_n, box_lengths, coarse_d)
    voxel_volume = float(np.prod(box_lengths / float(grid_n)))
    density = np.fft.ifftn(np.fft.fftn(hist) * np.fft.fftn(ker)).real / voxel_volume
    return density


def threshold_iterative(density, tol=0.10, max_iter=100):
    rho = np.asarray(density, dtype=np.float64).ravel()
    rho_i = float(np.max(rho))
    history = [rho_i]
    for _ in range(max_iter):
        below = rho[rho < rho_i]
        if below.size == 0:
            break
        rho_avg = float(np.mean(below))
        history.append(rho_avg)
        if rho_avg > 0 and abs(rho_avg - rho_i) / rho_avg < tol:
            return rho_avg, history
        rho_i = rho_avg
    return rho_i, history


def file_identity(path):
    path = Path(path)
    st = path.stat()
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def parse_auto_int(value, default_value):
    if value is None:
        return int(default_value)
    text = str(value).strip().lower()
    if text == "auto":
        return int(default_value)
    return int(value)


def build_rho_th_cache_key(
    args,
    jumps_path,
    source_frames,
    window_frames,
    box_lengths,
    coarse_d,
    kernel_info,
    start_min,
    start_max,
    start_stride,
):
    return {
        "version": 1,
        "jumps_file": file_identity(jumps_path),
        "particle_type": int(args.cluster_particle_type),
        "analysis_scope": "type1-only analysis, not all-N Appendix B definition",
        "t_chi": float(args.t_chi),
        "dt_frame": float(args.dt_frame),
        "window_frames": int(window_frames),
        "grid_n": int(args.grid_n),
        "box_lengths": [float(x) for x in np.asarray(box_lengths, dtype=np.float64)],
        "coarse_d": float(coarse_d),
        "threshold_tol": float(args.threshold_tol),
        "rho_th_start_min": int(start_min),
        "rho_th_start_max": int(start_max),
        "rho_th_start_stride": int(start_stride),
        "source_frames": int(source_frames),
        "coarse_kernel": kernel_info,
    }


def cache_key_matches(cached_key, expected_key):
    return cached_key == expected_key


def rho_reference_starts(start_min, start_max, start_stride):
    if int(start_max) < int(start_min):
        return np.zeros(0, dtype=np.int32)
    return np.arange(int(start_min), int(start_max) + 1, max(1, int(start_stride)), dtype=np.int32)


def compute_rho_th_reference(positions, frames, starts, window_frames, box_lengths, grid_n, coarse_d, tol):
    values = []
    starts_used = []
    n_events = []
    records = []
    n_skipped_zero = 0
    for start in starts:
        start = int(start)
        end = start + int(window_frames)
        event_mask = (frames >= start) & (frames < end)
        pos = positions[event_mask]
        if len(pos) == 0:
            n_skipped_zero += 1
            continue
        density = coarse_density(pos, box_lengths, int(grid_n), float(coarse_d))
        rho_final, history = threshold_iterative(density, float(tol))
        values.append(float(rho_final))
        starts_used.append(start)
        n_events.append(int(len(pos)))
        records.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "n_window_jumps": int(len(pos)),
                "rho_final": float(rho_final),
                "iterations": int(len(history) - 1),
            }
        )
    if not values:
        raise SystemExit("rho_th reference has no non-empty t_ini windows")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mode": "reference",
        "definition": "mean of final rho_avg values over different t_ini windows, following Zeng Appendix B",
        "rho_th": float(np.mean(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_tini_total": int(len(starts)),
        "n_tini_used": int(len(values)),
        "n_skipped_zero_jump": int(n_skipped_zero),
        "starts_used": [int(x) for x in starts_used],
        "n_window_jumps_used": [int(x) for x in n_events],
        "rho_values": [float(x) for x in values],
        "records": records,
    }


def load_or_compute_rho_th_reference(
    args,
    jumps_path,
    positions,
    frames,
    source_frames,
    window_frames,
    box_lengths,
    coarse_d,
    kernel_info,
):
    start_min = parse_auto_int(args.rho_th_start_min, 0)
    start_max_default = max(0, int(source_frames) - int(window_frames))
    start_max = parse_auto_int(args.rho_th_start_max, start_max_default)
    start_max = min(int(start_max), int(start_max_default))
    start_stride = max(1, int(args.rho_th_start_stride))
    starts = rho_reference_starts(start_min, start_max, start_stride)
    key = build_rho_th_cache_key(
        args,
        jumps_path,
        source_frames,
        window_frames,
        box_lengths,
        coarse_d,
        kernel_info,
        start_min,
        start_max,
        start_stride,
    )
    cache_path = Path(args.rho_th_cache) if str(args.rho_th_cache) else Path(args.output) / "rho_th_reference.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if isinstance(cached, dict) and cache_key_matches(cached.get("cache_key"), key):
            cached["cache_status"] = "hit"
            return cached
    result = compute_rho_th_reference(
        positions,
        frames,
        starts,
        int(window_frames),
        box_lengths,
        int(args.grid_n),
        float(coarse_d),
        float(args.threshold_tol),
    )
    result.update(
        {
            "cache_status": "computed",
            "cache_key": key,
            "cache_path": str(cache_path),
            "threshold_tol": float(args.threshold_tol),
            "window_frames": int(window_frames),
            "window_time": float(window_frames * args.dt_frame),
            "source_frames": int(source_frames),
            "particle_type": int(args.cluster_particle_type),
            "analysis_scope": "type1-only analysis, not all-N Appendix B definition",
            "coarse_kernel": kernel_info,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def positive_neighbor_offsets(ndim, connectivity=3):
    offsets = []
    for off in np.ndindex(*([3] * int(ndim))):
        vec = tuple(int(v) - 1 for v in off)
        if all(v == 0 for v in vec):
            continue
        if sum(abs(v) for v in vec) > int(connectivity):
            continue
        for v in vec:
            if v != 0:
                if v > 0:
                    offsets.append(vec)
                break
    return offsets


def neighbor_offsets(ndim, connectivity=3):
    offsets = []
    for off in np.ndindex(*([3] * int(ndim))):
        vec = tuple(int(v) - 1 for v in off)
        if all(v == 0 for v in vec):
            continue
        if sum(abs(v) for v in vec) > int(connectivity):
            continue
        offsets.append(vec)
    return offsets


def periodic_merge_labels(labels, n_lab, connectivity=3):
    """Merge component labels that touch through periodic boundaries."""
    n_lab = int(n_lab)
    if n_lab <= 1:
        return labels, n_lab, 0

    parent = np.arange(n_lab + 1, dtype=np.int32)

    def find(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    shape = np.asarray(labels.shape, dtype=np.int32)
    offsets = positive_neighbor_offsets(labels.ndim, connectivity)
    boundary = np.zeros(labels.shape, dtype=bool)
    for ax, n in enumerate(shape):
        lo = [slice(None)] * labels.ndim
        hi = [slice(None)] * labels.ndim
        lo[ax] = 0
        hi[ax] = int(n) - 1
        boundary[tuple(lo)] = True
        boundary[tuple(hi)] = True

    for idx_tuple in zip(*np.nonzero((labels > 0) & boundary)):
        lab = int(labels[idx_tuple])
        idx = np.asarray(idx_tuple, dtype=np.int32)
        for off in offsets:
            nb = idx.copy()
            wrapped = False
            for ax, delta in enumerate(off):
                if delta == 0:
                    continue
                nb[ax] += int(delta)
                if nb[ax] < 0:
                    nb[ax] = shape[ax] - 1
                    wrapped = True
                elif nb[ax] >= shape[ax]:
                    nb[ax] = 0
                    wrapped = True
            if not wrapped:
                continue
            other = int(labels[tuple(nb)])
            if other > 0 and other != lab:
                union(lab, other)

    root_to_new = {}
    mapping = np.zeros(n_lab + 1, dtype=np.int32)
    next_label = 1
    for cid in range(1, n_lab + 1):
        root = find(cid)
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        mapping[cid] = root_to_new[root]

    n_periodic = next_label - 1
    return mapping[labels], int(n_periodic), int(n_lab - n_periodic)


def component_stats(density, rho_th, box_lengths, grid_n, min_voxels, connectivity=3):
    mask = density >= float(rho_th)
    structure = generate_binary_structure(3, int(connectivity)).astype(np.int8)
    labels, n_lab_raw = label(mask, structure=structure)
    labels, n_lab, n_periodic_merges = periodic_merge_labels(labels, n_lab_raw, connectivity)
    label_info = {
        "periodic_components": True,
        "raw_components": int(n_lab_raw),
        "periodic_components_count": int(n_lab),
        "periodic_merges": int(n_periodic_merges),
    }
    if n_lab == 0:
        return labels, [], label_info

    voxel_volume = float(np.prod(box_lengths / float(grid_n)))
    stats = []
    for cid in range(1, n_lab + 1):
        vox = int(np.count_nonzero(labels == cid))
        if vox < int(min_voxels):
            continue
        volume = vox * voxel_volume
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        stats.append(
            {
                "component": int(cid),
                "voxels": vox,
                "volume": float(volume),
                "radius": float(radius),
                "density_sum": float(np.sum(density[labels == cid])),
            }
        )
    stats.sort(key=lambda x: (x["voxels"], x["density_sum"]), reverse=True)
    return labels, stats, label_info


def local_maximum_basin_stats(density, rho_th, box_lengths, grid_n, min_voxels, connectivity=3):
    """Segment rho>=rho_th into basins seeded by local density maxima.

    Zeng Appendix B describes finding local density maxima rmax and exploring
    outward from each rmax until rho drops below rho_th.  This seeded watershed-
    like implementation preserves separate maxima inside a broad above-threshold
    region, unlike a plain connected-component pass over rho>=rho_th.
    """
    density = np.asarray(density, dtype=np.float64)
    mask = density >= float(rho_th)
    if not np.any(mask):
        return np.zeros_like(density, dtype=np.int32), [], {
            "segmentation": "local_max",
            "local_maxima": 0,
            "periodic_components": True,
            "raw_components": 0,
            "periodic_components_count": 0,
            "periodic_merges": 0,
        }

    structure = generate_binary_structure(3, int(connectivity)).astype(bool)
    local_max = mask & (density >= maximum_filter(density, footprint=structure, mode="wrap"))
    offsets = neighbor_offsets(density.ndim, connectivity)
    shape = np.asarray(density.shape, dtype=np.int32)

    labels = np.zeros(density.shape, dtype=np.int32)
    flat_mask = np.flatnonzero(mask.ravel())
    order = flat_mask[np.argsort(density.ravel()[flat_mask])[::-1]]
    n_lab = 0

    for flat in order:
        idx = np.asarray(np.unravel_index(int(flat), density.shape), dtype=np.int32)
        idx_tuple = tuple(int(x) for x in idx)
        neighbor_labels = []
        neighbor_rhos = []
        for off in offsets:
            nb = (idx + np.asarray(off, dtype=np.int32)) % shape
            lab = int(labels[tuple(int(x) for x in nb)])
            if lab > 0:
                neighbor_labels.append(lab)
                neighbor_rhos.append(float(density[tuple(int(x) for x in nb)]))
        if bool(local_max[idx_tuple]) or not neighbor_labels:
            n_lab += 1
            labels[idx_tuple] = n_lab
        else:
            labels[idx_tuple] = int(neighbor_labels[int(np.argmax(neighbor_rhos))])

    voxel_volume = float(np.prod(box_lengths / float(grid_n)))
    stats = []
    for cid in range(1, n_lab + 1):
        vox = int(np.count_nonzero(labels == cid))
        if vox < int(min_voxels):
            continue
        volume = vox * voxel_volume
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        stats.append(
            {
                "component": int(cid),
                "voxels": vox,
                "volume": float(volume),
                "radius": float(radius),
                "density_sum": float(np.sum(density[labels == cid])),
            }
        )
    stats.sort(key=lambda x: (x["voxels"], x["density_sum"]), reverse=True)
    return labels, stats, {
        "segmentation": "local_max",
        "local_maxima": int(np.count_nonzero(local_max)),
        "periodic_components": True,
        "raw_components": int(n_lab),
        "periodic_components_count": int(n_lab),
        "periodic_merges": 0,
    }


def assign_points_to_grid(positions, box_lengths, grid_n):
    pts = periodic_points(positions, box_lengths)
    idx = np.floor(pts / box_lengths * float(grid_n)).astype(np.int32)
    return np.clip(idx, 0, grid_n - 1)


def assign_points_to_labeled_grid(positions, box_lengths, grid_n, point_mode):
    mode = str(point_mode).lower()
    if mode == "centers":
        return assign_points_to_grid(positions, box_lengths, int(grid_n))
    if mode == "vertices":
        pts = periodic_points(positions, box_lengths)
        idx = np.rint(pts / box_lengths * float(grid_n)).astype(np.int32) % int(grid_n)
        return idx
    raise ValueError("point_mode must be 'vertices' or 'centers'")


def undersize_grid_points(box_lengths, fine_grid_n, point_mode):
    mode = str(point_mode).lower()
    fine_grid_n = int(fine_grid_n)
    if mode == "centers":
        axes = [(np.arange(fine_grid_n, dtype=np.float64) + 0.5) * L / fine_grid_n for L in box_lengths]
    elif mode == "vertices":
        axes = [np.arange(fine_grid_n, dtype=np.float64) * L / fine_grid_n for L in box_lengths]
    else:
        raise ValueError("point_mode must be 'vertices' or 'centers'")
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def coarse_membership_for_undersize_points(
    coarse_labels,
    coarse_component,
    box_lengths,
    coarse_grid_n,
    fine_grid_n,
    point_mode,
):
    mode = str(point_mode).lower()
    if mode == "centers":
        grid_points = undersize_grid_points(box_lengths, fine_grid_n, mode)
        coarse_idx = assign_points_to_grid(grid_points, box_lengths, int(coarse_grid_n))
        return (
            coarse_labels[coarse_idx[:, 0], coarse_idx[:, 1], coarse_idx[:, 2]] == int(coarse_component)
        )
    if mode != "vertices":
        raise ValueError("point_mode must be 'vertices' or 'centers'")

    fine_grid_n = int(fine_grid_n)
    inside = np.zeros(fine_grid_n ** 3, dtype=bool)
    vertex_indices = np.column_stack(
        np.unravel_index(np.arange(fine_grid_n ** 3, dtype=np.int64), (fine_grid_n, fine_grid_n, fine_grid_n))
    ).astype(np.int32)
    for off in np.ndindex(2, 2, 2):
        # A periodic vertex touches eight small boxes.  Use the centers of
        # those boxes to decide whether the vertex belongs to the coarse region.
        fine_cell = (vertex_indices - np.asarray(off, dtype=np.int32)) % fine_grid_n
        centers = (fine_cell.astype(np.float64) + 0.5) * (box_lengths / float(fine_grid_n))
        coarse_idx = assign_points_to_grid(centers, box_lengths, int(coarse_grid_n))
        inside |= coarse_labels[coarse_idx[:, 0], coarse_idx[:, 1], coarse_idx[:, 2]] == int(coarse_component)
    return inside


def grid_plot_axes(box_lengths, grid_n, point_mode):
    return grid_axis(box_lengths[0], grid_n, point_mode), grid_axis(box_lengths[1], grid_n, point_mode)


def grid_axis(length, grid_n, point_mode):
    mode = str(point_mode).lower()
    if mode == "vertices":
        return np.arange(int(grid_n), dtype=np.float64) * float(length) / float(grid_n)
    return (np.arange(int(grid_n), dtype=np.float64) + 0.5) * float(length) / float(grid_n)


def projection_specs():
    return [
        ("xy", 0, 1, 2, "x", "y"),
        ("xz", 0, 2, 1, "x", "z"),
        ("yz", 1, 2, 0, "y", "z"),
    ]


def projected_mask(mask3, collapse_axis):
    return np.any(mask3, axis=int(collapse_axis)).astype(float)


def grid_point_coordinates_from_indices(indices, box_lengths, grid_n, point_mode):
    indices = np.asarray(indices, dtype=np.float64)
    if str(point_mode).lower() == "vertices":
        return indices * (box_lengths / float(grid_n))
    return (indices + 0.5) * (box_lengths / float(grid_n))


def undersize_refine(
    positions,
    coarse_labels,
    coarse_component,
    box_lengths,
    coarse_grid_n,
    rho_th,
    fine_grid_n,
    connectivity,
    point_mode,
):
    """Appendix B undersize sieve on a selected coarse component.

    Zeng states that the MD/BD cluster boundary is refined on a grid of
    small boxes with d' = L/30 for 3D.  At each small-grid point a sphere of
    radius sqrt(2) d' / 2 is used to compute the local cage-jump density, and
    the point is retained when that density is above rho_th.
    """
    fine_grid_n = int(fine_grid_n)
    point_mode = str(point_mode).lower()
    grid_points = undersize_grid_points(box_lengths, fine_grid_n, point_mode)
    inside_coarse = coarse_membership_for_undersize_points(
        coarse_labels,
        int(coarse_component),
        box_lengths,
        int(coarse_grid_n),
        fine_grid_n,
        point_mode,
    )

    d_prime = float(np.mean(box_lengths / float(fine_grid_n)))
    sphere_radius = math.sqrt(2.0) * d_prime / 2.0
    sphere_volume = (4.0 / 3.0) * math.pi * sphere_radius ** 3

    pts = periodic_points(positions, box_lengths)
    tree = cKDTree(pts, boxsize=box_lengths)
    counts = np.asarray([len(v) for v in tree.query_ball_point(grid_points, sphere_radius)], dtype=np.float64)
    local_density = counts / sphere_volume
    fine_mask = (inside_coarse & (local_density >= float(rho_th))).reshape((fine_grid_n, fine_grid_n, fine_grid_n))
    fine_density = local_density.reshape((fine_grid_n, fine_grid_n, fine_grid_n))

    labels, stats, fine_label_info = component_stats(
        fine_density * fine_mask,
        max(float(rho_th), np.nextafter(0.0, 1.0)),
        box_lengths,
        fine_grid_n,
        min_voxels=1,
        connectivity=connectivity,
    )
    if not stats:
        return None

    point_idx = assign_points_to_labeled_grid(positions, box_lengths, fine_grid_n, point_mode)
    point_labels = labels[point_idx[:, 0], point_idx[:, 1], point_idx[:, 2]]
    for comp in stats:
        cid = int(comp["component"])
        comp["n_cluster_jumps"] = int(np.count_nonzero(point_labels == cid))
        comp_points = positions[point_labels == cid]
        comp["center"] = [float(x) for x in circular_periodic_center(comp_points, box_lengths)] if len(comp_points) else []
        comp["density_sum"] = float(np.sum(fine_density[labels == cid]))
    stats.sort(key=lambda x: (x["n_cluster_jumps"], x["voxels"], x["density_sum"]), reverse=True)
    best = stats[0]
    in_best = point_labels == int(best["component"])
    return best, fine_density, labels, in_best, {
        "d_prime": float(d_prime),
        "sphere_radius": float(sphere_radius),
        "sphere_volume": float(sphere_volume),
        "point_mode": point_mode,
        "coarse_membership": "periodic adjacent coarse-cell membership" if point_mode == "vertices" else "center point coarse-cell membership",
        "fine_label_info": fine_label_info,
        "fine_component_stats": stats,
    }, point_labels


def aggregate_selected_components(stats, labels, density, point_labels, box_lengths, grid_n, selected_components):
    selected = [int(c) for c in selected_components]
    if not selected:
        raise ValueError("selected_components must not be empty")

    mask = np.isin(labels, selected)
    point_mask = np.isin(point_labels, selected)
    voxel_volume = float(np.prod(box_lengths / float(grid_n)))
    voxels = int(np.count_nonzero(mask))
    volume = float(voxels * voxel_volume)
    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0) if volume > 0.0 else 0.0
    selected_stats = [dict(comp) for comp in stats if int(comp["component"]) in selected]
    selected_stats.sort(key=lambda x: (x["n_cluster_jumps"], x["voxels"], x["density_sum"]), reverse=True)
    return {
        "component": int(selected_stats[0]["component"]),
        "selected_components": selected,
        "n_selected_components": int(len(selected)),
        "component_mode": "multi" if len(selected) > 1 else "single",
        "component_stats": selected_stats,
        "voxels": voxels,
        "volume": volume,
        "radius": float(radius),
        "density_sum": float(np.sum(density[mask])),
        "n_cluster_jumps": int(np.count_nonzero(point_mask)),
    }, point_mask


def choose_window_components(stats, args, box_lengths):
    stats = [dict(comp) for comp in stats if int(comp.get("n_cluster_jumps", 0)) > 0]
    stats.sort(key=lambda x: (x["n_cluster_jumps"], x["voxels"], x["density_sum"]), reverse=True)
    if not stats:
        return []
    if str(args.component_mode) == "single":
        return [int(stats[0]["component"])]

    selected = [
        comp
        for comp in stats
        if int(comp.get("n_cluster_jumps", 0)) >= int(args.multi_min_component_jumps)
    ]
    if float(args.multi_max_reference_distance) > 0.0 and selected:
        ref = np.asarray(selected[0].get("center", []), dtype=np.float64)
        if ref.size == 3:
            box_lengths = np.asarray(box_lengths, dtype=np.float64)
            cutoff = float(args.multi_max_reference_distance)
            kept = []
            for comp in selected:
                center = np.asarray(comp.get("center", []), dtype=np.float64)
                if center.size != 3:
                    continue
                delta = center - ref
                delta -= box_lengths * np.round(delta / box_lengths)
                if float(np.sqrt(np.sum(delta * delta))) <= cutoff:
                    kept.append(comp)
            selected = kept
    if int(args.multi_max_components) > 0:
        selected = selected[: int(args.multi_max_components)]
    if not selected:
        selected = stats[:1]
    return [int(comp["component"]) for comp in selected]


def analyze_window(
    positions,
    frames,
    start,
    window_frames,
    box_lengths,
    grid_n,
    coarse_d,
    tol,
    min_voxels,
    undersize_grid_n,
    args,
    rho_th_reference=None,
):
    end = int(start + window_frames)
    event_mask = (frames >= int(start)) & (frames < end)
    pos = positions[event_mask]
    if len(pos) == 0:
        return None
    density = coarse_density(pos, box_lengths, grid_n, coarse_d)
    window_rho_th, history = threshold_iterative(density, tol)
    if str(args.rho_th_mode) == "reference":
        if not isinstance(rho_th_reference, dict):
            raise ValueError("rho_th_mode=reference requires rho_th_reference")
        rho_th = float(rho_th_reference["rho_th"])
    else:
        rho_th = float(window_rho_th)
    if str(args.coarse_segmentation) == "local-max":
        labels, comps, coarse_label_info = local_maximum_basin_stats(
            density, rho_th, box_lengths, grid_n, min_voxels, connectivity=int(args.connectivity)
        )
    else:
        labels, comps, coarse_label_info = component_stats(
            density, rho_th, box_lengths, grid_n, min_voxels, connectivity=int(args.connectivity)
        )
    if not comps:
        if str(args.coarse_segmentation) == "local-max":
            labels, comps, coarse_label_info = local_maximum_basin_stats(
                density, rho_th, box_lengths, grid_n, 1, connectivity=int(args.connectivity)
            )
        else:
            labels, comps, coarse_label_info = component_stats(
                density, rho_th, box_lengths, grid_n, 1, connectivity=int(args.connectivity)
            )
    if not comps:
        return None
    coarse_best = comps[0]

    if int(undersize_grid_n) > 0:
        refined = undersize_refine(
            pos,
            labels,
            int(coarse_best["component"]),
            box_lengths,
            grid_n,
            rho_th,
            int(undersize_grid_n),
            int(args.connectivity),
            args.undersize_point_mode,
        )
    else:
        refined = None

    if refined is None:
        point_idx = assign_points_to_grid(pos, box_lengths, grid_n)
        point_labels = labels[point_idx[:, 0], point_idx[:, 1], point_idx[:, 2]]
        in_best = point_labels == coarse_best["component"]
        best = dict(coarse_best)
        best["coarse_label_info"] = coarse_label_info
        plot_density = density
        plot_labels = labels
        boundary_grid_n = int(grid_n)
        sieve_stage = "coarse_only"
        undersize_info = None
    else:
        refined_best, refined_density, refined_labels, in_best, undersize_info, point_labels = refined
        selected_components = choose_window_components(undersize_info["fine_component_stats"], args, box_lengths)
        best, in_best = aggregate_selected_components(
            undersize_info["fine_component_stats"],
            refined_labels,
            refined_density,
            point_labels,
            box_lengths,
            int(undersize_grid_n),
            selected_components,
        )
        best["coarse_label_info"] = coarse_label_info
        best["coarse_component"] = int(coarse_best["component"])
        best["coarse_voxels"] = int(coarse_best["voxels"])
        best["coarse_volume"] = float(coarse_best["volume"])
        best["coarse_radius"] = float(coarse_best["radius"])
        plot_density = refined_density
        plot_labels = refined_labels
        boundary_grid_n = int(undersize_grid_n)
        sieve_stage = "coarse_plus_undersize"

    best.update(
        {
            "start_frame": int(start),
            "end_frame": int(end),
            "center_frame": int(start + window_frames // 2),
            "n_window_jumps": int(len(pos)),
            "n_cluster_jumps": int(np.count_nonzero(in_best)),
            "rho_th": float(rho_th),
            "rho_th_mode": str(args.rho_th_mode),
            "rho_th_window_final": float(window_rho_th),
            "rho_th_reference_mean": (
                float(rho_th_reference["rho_th"]) if isinstance(rho_th_reference, dict) else None
            ),
            "threshold_history": [float(x) for x in history],
            "sieve_stage": sieve_stage,
            "boundary_grid_n": int(boundary_grid_n),
            "boundary_point_mode": str(args.undersize_point_mode if undersize_info is not None else "centers"),
            "volume_fraction": float(best["volume"] / np.prod(box_lengths)),
        }
    )
    if undersize_info is not None:
        fine_component_stats = undersize_info.pop("fine_component_stats", [])
        best["n_fine_components"] = int(len(fine_component_stats))
        best.update(undersize_info)
    return best, plot_density, plot_labels, event_mask, in_best


def add_selection_metrics(win, number_density):
    volume = max(float(win.get("volume", 0.0)), np.nextafter(0.0, 1.0))
    radius = max(float(win.get("radius", 0.0)), np.nextafter(0.0, 1.0))
    n_cluster = float(win.get("n_cluster_jumps", 0.0))
    win["event_density"] = float(n_cluster / volume)
    win["compact_count_score"] = float(n_cluster / (radius ** 3))
    win["estimated_particle_count"] = float(float(number_density) * volume)
    return win


def localized_candidate(win, args):
    n_cluster = int(win.get("n_cluster_jumps", 0))
    if n_cluster < int(args.localized_min_cluster_jumps):
        return False
    if int(args.localized_max_cluster_jumps) > 0 and n_cluster > int(args.localized_max_cluster_jumps):
        return False
    n_particles = float(win.get("estimated_particle_count", 0.0))
    if float(args.localized_min_particles) > 0.0 and n_particles < float(args.localized_min_particles):
        return False
    if float(args.localized_max_particles) > 0.0 and n_particles > float(args.localized_max_particles):
        return False
    if float(win.get("volume_fraction", 1.0)) > float(args.localized_max_volume_fraction):
        return False
    if float(win.get("radius", np.inf)) > float(args.localized_max_radius):
        return False
    return True


def localized_score(win, args):
    score_name = str(args.localized_score)
    if score_name == "count":
        return float(win.get("n_cluster_jumps", 0.0))
    if score_name == "density":
        return float(win.get("event_density", 0.0))
    if score_name == "compact-count":
        return float(win.get("compact_count_score", 0.0))
    raise ValueError("Unknown localized score: {}".format(score_name))


def choose_best_pack(packs, args):
    if not packs:
        return None, {"selection_mode": args.selection_mode, "status": "no_candidates"}

    mode = str(args.selection_mode)
    if mode == "manual":
        target = int(args.select_start_frame)
        if target < 0:
            raise SystemExit("--selection-mode manual requires --select-start-frame")
        matches = [pack for pack in packs if int(pack[0]["start_frame"]) == target]
        if not matches:
            raise SystemExit("No analyzed window starts at frame {}".format(target))
        pack = matches[0]
        info = {
            "selection_mode": mode,
            "status": "manual",
            "select_start_frame": target,
            "score": None,
        }
        return pack, info

    if mode == "largest":
        pack = max(
            packs,
            key=lambda p: (
                int(p[0].get("n_cluster_jumps", 0)),
                int(p[0].get("voxels", 0)),
                float(p[0].get("density_sum", 0.0)),
            ),
        )
        info = {
            "selection_mode": mode,
            "status": "largest_component_over_all_windows",
            "score": None,
        }
        return pack, info

    if mode == "localized":
        candidates = [pack for pack in packs if localized_candidate(pack[0], args)]
        if not candidates:
            pack = max(
                packs,
                key=lambda p: (
                    int(p[0].get("n_cluster_jumps", 0)),
                    int(p[0].get("voxels", 0)),
                    float(p[0].get("density_sum", 0.0)),
                ),
            )
            info = {
                "selection_mode": mode,
                "status": "fallback_largest_no_localized_candidate",
                "score": None,
                "filters": localized_filter_summary(args),
            }
            return pack, info
        pack = max(candidates, key=lambda p: localized_score(p[0], args))
        info = {
            "selection_mode": mode,
            "status": "localized_candidate",
            "score": localized_score(pack[0], args),
            "score_name": str(args.localized_score),
            "n_localized_candidates": int(len(candidates)),
            "filters": localized_filter_summary(args),
        }
        return pack, info

    raise ValueError("Unknown selection mode: {}".format(mode))


def localized_filter_summary(args):
    return {
        "min_cluster_jumps": int(args.localized_min_cluster_jumps),
        "max_cluster_jumps": int(args.localized_max_cluster_jumps),
        "min_estimated_particles": float(args.localized_min_particles),
        "max_estimated_particles": float(args.localized_max_particles),
        "max_volume_fraction": float(args.localized_max_volume_fraction),
        "max_radius": float(args.localized_max_radius),
    }


def circular_periodic_center(points, box_lengths):
    points = np.asarray(points, dtype=np.float64)
    box_lengths = np.asarray(box_lengths, dtype=np.float64)
    if len(points) == 0:
        return 0.5 * box_lengths
    center = np.empty(points.shape[1], dtype=np.float64)
    for ax, length in enumerate(box_lengths):
        theta = 2.0 * math.pi * (points[:, ax] % length) / length
        s = float(np.mean(np.sin(theta)))
        c = float(np.mean(np.cos(theta)))
        angle = math.atan2(s, c)
        if angle < 0.0:
            angle += 2.0 * math.pi
        center[ax] = angle * length / (2.0 * math.pi)
    return center


def recenter_periodic_points(points, center, box_lengths):
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    box_lengths = np.asarray(box_lengths, dtype=np.float64)
    return (points - center + 0.5 * box_lengths) % box_lengths - 0.5 * box_lengths


def plot_projection(
    out_dir,
    positions,
    frames,
    window_result,
    labels,
    event_mask,
    in_best,
    box_lengths,
    grid_n,
    gamma_dot,
    jump_point_size,
    cluster_point_size,
    grid_point_mode,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_w = positions[event_mask]
    cluster_pos = pos_w[in_best]
    selected_components = window_result.get("selected_components")
    if selected_components is None:
        selected_components = [int(window_result["component"])]
    selected_components = np.asarray(selected_components, dtype=np.int32)

    mask3 = np.isin(labels, selected_components)
    touch = {}
    for ax, name in enumerate(("x", "y", "z")):
        touch[name] = {
            "low": bool(np.take(mask3, 0, axis=ax).any()),
            "high": bool(np.take(mask3, mask3.shape[ax] - 1, axis=ax).any()),
        }
    y_mid = (pos_w[:, 1] > box_lengths[1] / 3.0) & (pos_w[:, 1] < 2.0 * box_lengths[1] / 3.0)
    y_mid_cluster = (
        (cluster_pos[:, 1] > box_lengths[1] / 3.0)
        & (cluster_pos[:, 1] < 2.0 * box_lengths[1] / 3.0)
        if len(cluster_pos)
        else np.zeros(0, dtype=bool)
    )
    for suffix, ax0, ax1, collapse_axis, lab0, lab1 in projection_specs():
        axis0 = grid_axis(box_lengths[ax0], grid_n, grid_point_mode)
        axis1 = grid_axis(box_lengths[ax1], grid_n, grid_point_mode)
        mask2 = projected_mask(mask3, collapse_axis)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(pos_w[:, ax0], pos_w[:, ax1], s=jump_point_size, c="black", alpha=0.50, linewidths=0)
        ax.contour(axis0, axis1, mask2.T, levels=[0.5], colors="crimson", linewidths=1.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, box_lengths[ax0])
        ax.set_ylim(0.0, box_lengths[ax1])
        ax.set_xlabel(r"${}/d_0$".format(lab0))
        ax.set_ylabel(r"${}/d_0$".format(lab1))
        ax.set_title(
            r"3D MD cage jumps, {} projection, $\dot\gamma={}$, $t_\chi={:.2f}$".format(
                suffix, gamma_dot, window_result["window_time"]
            )
        )
        ax.grid(True, ls=":", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "fig5b_like_cluster_projection_{}.png".format(suffix), dpi=300)
        fig.savefig(out_dir / "fig5b_like_cluster_projection_{}.pdf".format(suffix))
        if suffix == "xy":
            fig.savefig(out_dir / "fig5b_like_cluster_projection.png", dpi=300)
            fig.savefig(out_dir / "fig5b_like_cluster_projection.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.hist2d(
            pos_w[:, ax0],
            pos_w[:, ax1],
            bins=120,
            range=[[0, box_lengths[ax0]], [0, box_lengths[ax1]]],
            cmap="Greys",
        )
        ax.contour(axis0, axis1, mask2.T, levels=[0.5], colors="crimson", linewidths=1.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, box_lengths[ax0])
        ax.set_ylim(0.0, box_lengths[ax1])
        ax.set_xlabel(r"${}/d_0$".format(lab0))
        ax.set_ylabel(r"${}/d_0$".format(lab1))
        ax.set_title("Density projection with selected boundary ({})".format(suffix))
        fig.tight_layout()
        fig.savefig(out_dir / "cluster_density_projection_{}.png".format(suffix), dpi=300)
        fig.savefig(out_dir / "cluster_density_projection_{}.pdf".format(suffix))
        if suffix == "xy":
            fig.savefig(out_dir / "cluster_density_projection.png", dpi=300)
            fig.savefig(out_dir / "cluster_density_projection.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.hist2d(
            pos_w[:, ax0],
            pos_w[:, ax1],
            bins=120,
            range=[[0, box_lengths[ax0]], [0, box_lengths[ax1]]],
            cmap="Greys",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, box_lengths[ax0])
        ax.set_ylim(0.0, box_lengths[ax1])
        ax.set_xlabel(r"${}/d_0$".format(lab0))
        ax.set_ylabel(r"${}/d_0$".format(lab1))
        ax.set_title("Actual {} jump density".format(suffix))
        fig.tight_layout()
        fig.savefig(out_dir / "actual_{}_jump_density.png".format(suffix), dpi=300)
        fig.savefig(out_dir / "actual_{}_jump_density.pdf".format(suffix))
        if suffix == "xy":
            fig.savefig(out_dir / "actual_xy_jump_density.png", dpi=300)
            fig.savefig(out_dir / "actual_xy_jump_density.pdf")
        plt.close(fig)

    center = circular_periodic_center(cluster_pos if len(cluster_pos) else pos_w, box_lengths)
    pos_c = recenter_periodic_points(pos_w, center, box_lengths)
    cluster_c = recenter_periodic_points(cluster_pos, center, box_lengths) if len(cluster_pos) else cluster_pos
    for suffix, ax0, ax1, _collapse_axis, lab0, lab1 in projection_specs():
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(pos_c[:, ax0], pos_c[:, ax1], s=jump_point_size, c="0.65", alpha=0.38, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5 * box_lengths[ax0], 0.5 * box_lengths[ax0])
        ax.set_ylim(-0.5 * box_lengths[ax1], 0.5 * box_lengths[ax1])
        ax.set_xlabel(r"${}-{}_c$".format(lab0, lab0))
        ax.set_ylabel(r"${}-{}_c$".format(lab1, lab1))
        ax.set_title("PBC-recentered cluster {}".format(suffix))
        ax.grid(True, ls=":", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "pbc_recentered_cluster_{}.png".format(suffix), dpi=300)
        fig.savefig(out_dir / "pbc_recentered_cluster_{}.pdf".format(suffix))
        plt.close(fig)

    mask_idx = np.column_stack(np.nonzero(mask3))
    if len(mask_idx):
        mask_centers = grid_point_coordinates_from_indices(mask_idx, box_lengths, grid_n, grid_point_mode)
        mask_c = recenter_periodic_points(mask_centers, center, box_lengths)
        for suffix, ax0, ax1, _collapse_axis, lab0, lab1 in projection_specs():
            bins0 = np.linspace(-0.5 * box_lengths[ax0], 0.5 * box_lengths[ax0], grid_n + 1)
            bins1 = np.linspace(-0.5 * box_lengths[ax1], 0.5 * box_lengths[ax1], grid_n + 1)
            hist, e0, e1 = np.histogram2d(mask_c[:, ax0], mask_c[:, ax1], bins=[bins0, bins1])
            c0 = 0.5 * (e0[:-1] + e0[1:])
            c1 = 0.5 * (e1[:-1] + e1[1:])
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            ax.scatter(pos_c[:, ax0], pos_c[:, ax1], s=jump_point_size, c="0.72", alpha=0.38, linewidths=0)
            if np.max(hist) > 0:
                ax.contour(c0, c1, hist.T, levels=[0.5], colors="crimson", linewidths=1.6)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-0.5 * box_lengths[ax0], 0.5 * box_lengths[ax0])
            ax.set_ylim(-0.5 * box_lengths[ax1], 0.5 * box_lengths[ax1])
            ax.set_xlabel(r"${}-{}_c$".format(lab0, lab0))
            ax.set_ylabel(r"${}-{}_c$".format(lab1, lab1))
            ax.set_title("PBC-recentered cluster boundary {}".format(suffix))
            ax.grid(True, ls=":", alpha=0.22)
            fig.tight_layout()
            fig.savefig(out_dir / "pbc_recentered_cluster_boundary_{}.png".format(suffix), dpi=300)
            fig.savefig(out_dir / "pbc_recentered_cluster_boundary_{}.pdf".format(suffix))
            plt.close(fig)

    return {
        "mask_projection": "np.any(labels == selected_component, axis=2); this is a 3D-to-xy projection, not a 2D density contour",
        "grid_point_mode": str(grid_point_mode),
        "touches_periodic_boundaries": touch,
        "pbc_center": [float(x) for x in center],
        "y_middle_third_fraction_all_jumps": float(np.mean(y_mid)) if len(y_mid) else 0.0,
        "y_middle_third_fraction_cluster_jumps": float(np.mean(y_mid_cluster)) if len(y_mid_cluster) else 0.0,
        "diagnostic_plots": [
            "actual_xy_jump_density.png",
            "actual_xz_jump_density.png",
            "actual_yz_jump_density.png",
            "pbc_recentered_cluster_xy.png",
            "pbc_recentered_cluster_xz.png",
            "pbc_recentered_cluster_yz.png",
            "pbc_recentered_cluster_boundary_xy.png",
            "pbc_recentered_cluster_boundary_xz.png",
            "pbc_recentered_cluster_boundary_yz.png",
        ],
    }


def plot_all_candidate_clusters(
    out_dir,
    positions,
    frames,
    start,
    window_frames,
    box_lengths,
    grid_n,
    coarse_d,
    tol,
    min_voxels,
    undersize_grid_n,
    particle_positions,
    args,
    rho_th_reference=None,
):
    """Diagnostic plot of every candidate cluster in the selected t_chi window.

    The main analysis intentionally follows one selected dynamic region into
    Fig. 6.  This diagnostic keeps all candidate regions in the raw simulation
    box so we can check whether the cage-jump field contains multiple separated
    regions like Fig. 5.
    """
    out_dir = Path(out_dir)
    end = int(start + window_frames)
    event_mask = (frames >= int(start)) & (frames < end)
    pos = positions[event_mask]
    if len(pos) == 0:
        return {"status": "no_events"}

    density = coarse_density(pos, box_lengths, int(grid_n), coarse_d)
    window_rho_th, history = threshold_iterative(density, tol)
    if str(args.rho_th_mode) == "reference":
        if not isinstance(rho_th_reference, dict):
            raise ValueError("rho_th_mode=reference requires rho_th_reference")
        rho_th = float(rho_th_reference["rho_th"])
    else:
        rho_th = float(window_rho_th)
    if str(args.coarse_segmentation) == "local-max":
        labels, comps, coarse_label_info = local_maximum_basin_stats(
            density, rho_th, box_lengths, int(grid_n), min_voxels, connectivity=int(args.connectivity)
        )
    else:
        labels, comps, coarse_label_info = component_stats(
            density, rho_th, box_lengths, int(grid_n), min_voxels, connectivity=int(args.connectivity)
        )
    if not comps:
        return {"status": "no_components", "rho_th": float(rho_th)}

    point_idx = assign_points_to_grid(pos, box_lengths, int(grid_n))
    point_labels = labels[point_idx[:, 0], point_idx[:, 1], point_idx[:, 2]]
    particle_point_idx = None
    particle_labels = None
    if particle_positions is not None and len(particle_positions):
        particle_point_idx = assign_points_to_grid(particle_positions, box_lengths, int(grid_n))
        particle_labels = labels[particle_point_idx[:, 0], particle_point_idx[:, 1], particle_point_idx[:, 2]]
    coarse_rows = []
    for comp in comps:
        row = dict(comp)
        row["n_cluster_jumps"] = int(np.count_nonzero(point_labels == int(comp["component"])))
        if particle_labels is not None:
            row["n_particles"] = int(np.count_nonzero(particle_labels == int(comp["component"])))
        coarse_rows.append(row)
    coarse_rows.sort(key=lambda r: (r["n_cluster_jumps"], r["voxels"], r["density_sum"]), reverse=True)

    max_coarse = int(args.all_candidate_max_coarse)
    def passes_particle_filter(row):
        if "n_particles" not in row:
            return True
        n_particles = int(row["n_particles"])
        if int(args.all_candidate_min_particles) > 0 and n_particles < int(args.all_candidate_min_particles):
            return False
        if int(args.all_candidate_max_particles) > 0 and n_particles > int(args.all_candidate_max_particles):
            return False
        return True

    coarse_for_plot = [
        r
        for r in coarse_rows
        if int(r["n_cluster_jumps"]) >= int(args.all_candidate_min_jumps) and passes_particle_filter(r)
    ]
    if max_coarse > 0:
        coarse_for_plot = coarse_for_plot[:max_coarse]

    pos_plot = pos.copy()
    pos_plot -= 0.5 * box_lengths
    for suffix, ax0, ax1, collapse_axis, lab0, lab1 in projection_specs():
        axis0 = grid_axis(box_lengths[ax0], int(grid_n), "centers") - 0.5 * box_lengths[ax0]
        axis1 = grid_axis(box_lengths[ax1], int(grid_n), "centers") - 0.5 * box_lengths[ax1]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(pos_plot[:, ax0], pos_plot[:, ax1], s=args.jump_point_size, c="black", alpha=0.50, linewidths=0)
        for row in coarse_for_plot:
            mask2 = projected_mask(labels == int(row["component"]), collapse_axis)
            if np.count_nonzero(mask2) > 0:
                ax.contour(axis0, axis1, mask2.T, levels=[0.5], colors="crimson", linewidths=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5 * box_lengths[ax0], 0.5 * box_lengths[ax0])
        ax.set_ylim(-0.5 * box_lengths[ax1], 0.5 * box_lengths[ax1])
        ax.set_xlabel(r"${}/d_0$".format(lab0))
        ax.set_ylabel(r"${}/d_0$".format(lab1))
        ax.set_title("All coarse-sieve candidate basins ({})".format(suffix))
        ax.grid(True, ls=":", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "all_candidate_coarse_basins_{}.png".format(suffix), dpi=300)
        fig.savefig(out_dir / "all_candidate_coarse_basins_{}.pdf".format(suffix))
        plt.close(fig)

    refined_rows = []
    refined_masks = []
    if int(undersize_grid_n) > 0:
        coarse_inputs = coarse_rows
        if max_coarse > 0:
            coarse_inputs = coarse_inputs[:max_coarse]
        for coarse in coarse_inputs:
            refined = undersize_refine(
                pos,
                labels,
                int(coarse["component"]),
                box_lengths,
                int(grid_n),
                rho_th,
                int(undersize_grid_n),
                int(args.connectivity),
                args.undersize_point_mode,
            )
            if refined is None:
                continue
            _, refined_density, refined_labels, _, undersize_info, refined_point_labels = refined
            refined_particle_labels = None
            if particle_positions is not None and len(particle_positions):
                refined_particle_idx = assign_points_to_labeled_grid(
                    particle_positions,
                    box_lengths,
                    int(undersize_grid_n),
                    args.undersize_point_mode,
                )
                refined_particle_labels = refined_labels[
                    refined_particle_idx[:, 0],
                    refined_particle_idx[:, 1],
                    refined_particle_idx[:, 2],
                ]
            mask3 = refined_labels > 0
            if not np.any(mask3):
                continue
            voxel_volume = float(np.prod(box_lengths / float(undersize_grid_n)))
            voxels = int(np.count_nonzero(mask3))
            volume = float(voxels * voxel_volume)
            radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0) if volume > 0.0 else 0.0
            row = {
                "coarse_component": int(coarse["component"]),
                "n_cluster_jumps": int(np.count_nonzero(refined_point_labels > 0)),
                "n_particles": int(np.count_nonzero(refined_particle_labels > 0)) if refined_particle_labels is not None else None,
                "voxels": voxels,
                "volume": volume,
                "radius": float(radius),
                "density_sum": float(np.sum(refined_density[mask3])),
                "coarse_n_cluster_jumps": int(coarse["n_cluster_jumps"]),
                "n_fine_components": int(len(undersize_info["fine_component_stats"])),
                "undersize_point_mode": str(args.undersize_point_mode),
            }
            if int(row["n_cluster_jumps"]) < int(args.all_candidate_min_jumps):
                continue
            if not passes_particle_filter(row):
                continue
            refined_rows.append(row)
            refined_masks.append(mask3)
    refined_order = sorted(
        range(len(refined_rows)),
        key=lambda i: (
            int(refined_rows[i].get("n_cluster_jumps", 0)),
            int(refined_rows[i].get("voxels", 0)),
            float(refined_rows[i].get("density_sum", 0.0)),
        ),
        reverse=True,
    )
    if refined_order:
        refined_rows = [refined_rows[int(i)] for i in refined_order]
        refined_masks = [refined_masks[int(i)] for i in refined_order]
    max_refined = int(args.all_candidate_max_clusters)
    if max_refined > 0:
        refined_rows = refined_rows[:max_refined]
        refined_masks = refined_masks[:max_refined]

    if refined_rows:
        for suffix, ax0, ax1, collapse_axis, lab0, lab1 in projection_specs():
            axis0 = grid_axis(box_lengths[ax0], int(undersize_grid_n), args.undersize_point_mode) - 0.5 * box_lengths[ax0]
            axis1 = grid_axis(box_lengths[ax1], int(undersize_grid_n), args.undersize_point_mode) - 0.5 * box_lengths[ax1]
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            ax.scatter(pos_plot[:, ax0], pos_plot[:, ax1], s=args.jump_point_size, c="black", alpha=0.50, linewidths=0)
            for mask3 in refined_masks:
                mask2 = projected_mask(mask3, collapse_axis)
                if np.count_nonzero(mask2) > 0:
                    ax.contour(axis0, axis1, mask2.T, levels=[0.5], colors="crimson", linewidths=1.4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-0.5 * box_lengths[ax0], 0.5 * box_lengths[ax0])
            ax.set_ylim(-0.5 * box_lengths[ax1], 0.5 * box_lengths[ax1])
            ax.set_xlabel(r"${}/d_0$".format(lab0))
            ax.set_ylabel(r"${}/d_0$".format(lab1))
            ax.set_title("All undersize-refined candidate clusters ({})".format(suffix))
            ax.grid(True, ls=":", alpha=0.22)
            fig.tight_layout()
            fig.savefig(out_dir / "all_candidate_refined_clusters_{}.png".format(suffix), dpi=300)
            fig.savefig(out_dir / "all_candidate_refined_clusters_{}.pdf".format(suffix))
            plt.close(fig)

    with open(out_dir / "all_candidate_coarse_components.csv", "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["component", "n_cluster_jumps", "n_particles", "voxels", "volume", "radius", "density_sum"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in coarse_rows:
            writer.writerow(row)
    if refined_rows:
        with open(out_dir / "all_candidate_refined_clusters.csv", "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "coarse_component",
                "n_cluster_jumps",
                "n_particles",
                "voxels",
                "volume",
                "radius",
                "density_sum",
                "coarse_n_cluster_jumps",
                "n_fine_components",
                "undersize_point_mode",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in refined_rows:
                writer.writerow(row)

    return {
        "status": "ok",
        "start_frame": int(start),
        "end_frame": int(end),
        "n_window_jumps": int(len(pos)),
        "rho_th": float(rho_th),
        "rho_th_mode": str(args.rho_th_mode),
        "rho_th_window_final": float(window_rho_th),
        "rho_th_reference_mean": (
            float(rho_th_reference["rho_th"]) if isinstance(rho_th_reference, dict) else None
        ),
        "threshold_history": [float(x) for x in history],
        "undersize_point_mode": str(args.undersize_point_mode),
        "coarse_label_info": coarse_label_info,
        "n_coarse_components": int(len(coarse_rows)),
        "n_coarse_components_plotted": int(len(coarse_for_plot)),
        "n_refined_clusters_plotted": int(len(refined_rows)),
        "all_candidate_min_jumps": int(args.all_candidate_min_jumps),
        "all_candidate_min_particles": int(args.all_candidate_min_particles),
        "all_candidate_max_particles": int(args.all_candidate_max_particles),
        "outputs": [
            "all_candidate_coarse_basins_xy.png",
            "all_candidate_coarse_basins_xz.png",
            "all_candidate_coarse_basins_yz.png",
            "all_candidate_refined_clusters_xy.png" if refined_rows else None,
            "all_candidate_refined_clusters_xz.png" if refined_rows else None,
            "all_candidate_refined_clusters_yz.png" if refined_rows else None,
            "all_candidate_coarse_components.csv",
            "all_candidate_refined_clusters.csv" if refined_rows else None,
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Find a t_chi-window dynamic region for Zeng 3D MD")
    parser.add_argument("--jumps", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma-dot", type=float, default=0.015)
    parser.add_argument("--dt-frame", type=float, default=0.1)
    parser.add_argument("--t-chi", type=float, default=2.4)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--grid-n", type=int, default=64)
    parser.add_argument("--coarse-d", type=float, default=0.0, help="0 means estimate from g(r) first minimum")
    parser.add_argument("--threshold-tol", type=float, default=0.10)
    parser.add_argument("--rho-th-mode", choices=["reference", "window"], default="reference")
    parser.add_argument("--rho-th-cache", default="", help="Path to rho_th reference cache; default is output/rho_th_reference.json")
    parser.add_argument("--rho-th-start-min", default="0")
    parser.add_argument("--rho-th-start-max", default="auto")
    parser.add_argument("--rho-th-start-stride", type=int, default=1)
    parser.add_argument("--min-source-cage-frames", type=int, default=200)
    parser.add_argument(
        "--allow-short-cage-source",
        action="store_true",
        help="Diagnostic override only; formal clusters must use long-trajectory cage-jump sources",
    )
    parser.add_argument(
        "--coarse-segmentation",
        choices=["local-max", "connected"],
        default="local-max",
        help=(
            "local-max follows the Appendix B description of starting from each "
            "density maximum rmax; connected preserves the older diagnostic that "
            "labels connected rho>=rho_th regions"
        ),
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Voxel connectivity: 1=6-neighbor, 2=18-neighbor, 3=26-neighbor",
    )
    parser.add_argument("--stz-radius", type=float, default=3.0)
    parser.add_argument("--c-prime", type=float, default=2.0)
    parser.add_argument("--jump-point-size", type=float, default=24.0, help="Scatter marker area for all cage-jump points")
    parser.add_argument("--cluster-point-size", type=float, default=28.0, help="Scatter marker area for selected cluster points")
    parser.add_argument(
        "--plot-all-candidates",
        dest="plot_all_candidates",
        action="store_true",
        default=True,
        help="Plot every candidate cage-jump cluster in the selected t_chi window",
    )
    parser.add_argument(
        "--no-plot-all-candidates",
        dest="plot_all_candidates",
        action="store_false",
        help="Disable all-candidate diagnostic plots",
    )
    parser.add_argument("--all-candidate-min-jumps", type=int, default=10)
    parser.add_argument("--all-candidate-min-particles", type=int, default=0)
    parser.add_argument("--all-candidate-max-particles", type=int, default=0, help="0 disables the upper particle-count filter")
    parser.add_argument("--all-candidate-max-clusters", type=int, default=24, help="0 plots all refined candidates")
    parser.add_argument("--all-candidate-max-coarse", type=int, default=64, help="0 checks every coarse candidate")
    parser.add_argument("--undersize-grid-n", type=int, default=30, help="3D MD undersize sieve grid; 0 disables it")
    parser.add_argument("--undersize-point-mode", choices=["vertices", "centers"], default="vertices")
    parser.add_argument(
        "--component-mode",
        choices=["single", "multi"],
        default="single",
        help="single draws the strongest fine component; multi draws several fine components in the selected window",
    )
    parser.add_argument("--multi-min-component-jumps", type=int, default=10)
    parser.add_argument("--multi-max-components", type=int, default=12, help="0 keeps every component passing the jump cutoff")
    parser.add_argument(
        "--multi-max-reference-distance",
        type=float,
        default=0.0,
        help="0 disables this filter; otherwise keep only components within this PBC distance from the strongest component center",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["largest", "localized", "manual"],
        default="localized",
        help="localized applies explicit size/event-count filters; manual selects --select-start-frame; largest is legacy diagnostics only",
    )
    parser.add_argument("--select-start-frame", type=int, default=-1)
    parser.add_argument("--localized-min-cluster-jumps", type=int, default=20)
    parser.add_argument("--localized-max-cluster-jumps", type=int, default=400)
    parser.add_argument("--localized-min-particles", type=float, default=0.0)
    parser.add_argument("--localized-max-particles", type=float, default=0.0, help="0 disables the upper particle-count filter")
    parser.add_argument("--localized-max-volume-fraction", type=float, default=0.02)
    parser.add_argument("--localized-max-radius", type=float, default=4.0)
    parser.add_argument(
        "--localized-score",
        choices=["count", "density", "compact-count"],
        default="count",
        help="Score used after localized filters pass",
    )
    parser.add_argument("--cluster-particle-type", type=int, default=1, help="Particle type used for cluster particle-count filters")
    parser.add_argument(
        "--particle-count-frame",
        choices=["start", "center", "end"],
        default="center",
        help="Reference frame inside the t_chi window for exact cluster particle counts",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cage_source_summary = load_cage_source_summary(Path(args.jumps))
    source_frames = None
    if cage_source_summary is not None and "frames" in cage_source_summary:
        source_frames = int(cage_source_summary.get("frames", 0))
        if source_frames < int(args.min_source_cage_frames) and not bool(args.allow_short_cage_source):
            raise SystemExit(
                "Refusing short-window cage-jump source with frames={}. "
                "Use a long-trajectory jump file, then select t_chi windows. "
                "Use --allow-short-cage-source only for explicitly labeled diagnostics.".format(source_frames)
            )
        cage_algorithm = cage_source_summary.get("cage_jump_algorithm")
        if isinstance(cage_algorithm, dict):
            algorithm_name = str(cage_algorithm.get("name", ""))
            if algorithm_name and algorithm_name != "candelier_recursive_trajectory_segmentation":
                raise SystemExit(
                    "Refusing cage-jump source generated by unsupported algorithm '{}'. "
                    "Formal Zeng clusters require Candelier recursive trajectory segmentation.".format(
                        algorithm_name
                    )
                )
    jumps = np.load(args.jumps)
    frames = jumps["jump_frames"].astype(np.int32)
    positions = jumps["jump_positions"].astype(np.float64)
    if source_frames is None:
        source_frames = int(np.max(frames)) + 1 if len(frames) else 0

    first_pos, box = load_first_frame_positions(Path(args.dump))
    box_lengths = np.asarray([box["Lx"], box["Ly"], box["Lz"]], dtype=np.float64)
    particle_count_selection = first_frame_selection(Path(args.dump), int(args.cluster_particle_type), 0, 42)
    cluster_particle_density = float(len(particle_count_selection["selected_ids"]) / np.prod(box_lengths))

    if args.coarse_d > 0:
        coarse_d = float(args.coarse_d)
        gr_info = {"status": "user", "first_min_r": coarse_d}
    else:
        coarse_d, gr_info = estimate_first_minimum_gr(first_pos, box_lengths)

    window_frames = max(1, int(round(float(args.t_chi) / float(args.dt_frame))))
    kernel_info = coarse_density_kernel_metadata(int(args.grid_n), box_lengths, coarse_d)
    if str(args.rho_th_mode) == "reference":
        rho_th_reference = load_or_compute_rho_th_reference(
            args,
            Path(args.jumps),
            positions,
            frames,
            int(source_frames),
            int(window_frames),
            box_lengths,
            coarse_d,
            kernel_info,
        )
    else:
        rho_th_reference = {
            "mode": "window",
            "strict_zeng_appendix_b": False,
            "definition": "diagnostic only: rho_th is computed independently for each selected t_ini window",
            "n_skipped_zero_jump": None,
            "n_tini_used": 1,
        }
    voxel_volume = float(np.prod(box_lengths / float(args.grid_n)))
    vstz = (4.0 / 3.0) * math.pi * (0.5 * float(args.stz_radius)) ** 3
    min_voxels = max(1, int(math.ceil(float(args.c_prime) * vstz / voxel_volume)))

    max_start = max(0, int(source_frames) - int(window_frames))
    if str(args.selection_mode) == "manual" and int(args.select_start_frame) >= 0:
        manual_start = max(0, min(int(args.select_start_frame), int(max_start)))
        starts = np.asarray([manual_start], dtype=np.int32)
    else:
        starts = np.arange(0, max_start + 1, max(1, int(args.window_stride)), dtype=np.int32)
    rows = []
    packs = []
    for si, start in enumerate(starts):
        res = analyze_window(
            positions,
            frames,
            int(start),
            window_frames,
            box_lengths,
            int(args.grid_n),
            coarse_d,
            float(args.threshold_tol),
            min_voxels,
            int(args.undersize_grid_n),
            args,
            rho_th_reference,
        )
        if res is None:
            continue
        win, density, labels, event_mask, in_best = res
        win["start_time"] = float(win["start_frame"] * args.dt_frame)
        win["end_time"] = float(win["end_frame"] * args.dt_frame)
        win["center_time"] = float(win["center_frame"] * args.dt_frame)
        win["window_time"] = float(window_frames * args.dt_frame)
        add_selection_metrics(win, cluster_particle_density)
        rows.append(win)
        packs.append((win, density, labels, event_mask, in_best))
        if (si + 1) % max(1, len(starts) // 10) == 0:
            print("  scanned {}/{} windows".format(si + 1, len(starts)))

    best_pack, selection_info = choose_best_pack(packs, args)
    if best_pack is None:
        raise SystemExit("No cluster found")

    selected_start = int(best_pack[0]["start_frame"])
    selected_end = int(best_pack[0]["end_frame"])
    for row in rows:
        row["selected"] = int(int(row["start_frame"]) == selected_start and int(row["end_frame"]) == selected_end)

    with open(out_dir / "window_scan.csv", "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "start_frame",
            "end_frame",
            "center_frame",
            "start_time",
            "end_time",
            "center_time",
            "n_window_jumps",
            "n_cluster_jumps",
            "voxels",
            "volume",
            "volume_fraction",
            "radius",
            "estimated_particle_count",
            "rho_th",
            "rho_th_mode",
            "rho_th_window_final",
            "rho_th_reference_mean",
            "density_sum",
            "sieve_stage",
            "boundary_grid_n",
            "boundary_point_mode",
            "d_prime",
            "sphere_radius",
            "coarse_voxels",
            "coarse_volume",
            "coarse_radius",
            "event_density",
            "compact_count_score",
            "selected",
            "component_mode",
            "n_selected_components",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best, density, labels, event_mask, in_best = best_pack
    count_frame = particle_count_frame(int(best["start_frame"]), int(window_frames), args.particle_count_frame)
    count_positions, count_selection, count_box = load_particle_positions_at_frame(
        Path(args.dump),
        int(args.cluster_particle_type),
        int(count_frame),
        seed=42,
    )
    selected_components = best.get("selected_components")
    if selected_components is None:
        selected_components = [int(best["component"])]
    selected_exact_particles = count_particles_in_components(
        count_positions,
        labels,
        selected_components,
        box_lengths,
        int(best.get("boundary_grid_n", args.grid_n)),
        str(best.get("boundary_point_mode", "centers")),
    )
    best["particle_count"] = {
        "definition": "exact count of selected particle type inside the selected cluster mask",
        "particle_type": int(args.cluster_particle_type),
        "frame_policy": str(args.particle_count_frame),
        "frame": int(count_frame),
        "n_particles": int(selected_exact_particles),
        "type_count_total": int(len(count_selection["selected_ids"])),
        "number_density": float(cluster_particle_density),
        "estimated_from_volume": float(best.get("estimated_particle_count", 0.0)),
    }
    projection_diagnostics = plot_projection(
        out_dir,
        positions,
        frames,
        best,
        labels,
        event_mask,
        in_best,
        box_lengths,
        int(best.get("boundary_grid_n", args.grid_n)),
        args.gamma_dot,
        args.jump_point_size,
        args.cluster_point_size,
        str(best.get("boundary_point_mode", "centers")),
    )
    best["projection_diagnostics"] = projection_diagnostics
    if bool(args.plot_all_candidates):
        best["all_candidate_diagnostics"] = plot_all_candidate_clusters(
            out_dir,
            positions,
            frames,
            int(best["start_frame"]),
            int(window_frames),
            box_lengths,
            int(args.grid_n),
            coarse_d,
            float(args.threshold_tol),
            min_voxels,
            int(args.undersize_grid_n),
            count_positions,
            args,
            rho_th_reference,
        )

    np.savez_compressed(
        out_dir / "selected_cluster_window.npz",
        best_window=json.dumps(best, sort_keys=True),
        event_mask=event_mask,
        in_best_cluster=in_best,
        labels=labels.astype(np.int32),
        density=density.astype(np.float32),
        box_lengths=box_lengths,
        coarse_d=float(coarse_d),
        grid_n=int(args.grid_n),
        boundary_grid_n=int(best.get("boundary_grid_n", args.grid_n)),
        boundary_point_mode=str(best.get("boundary_point_mode", "centers")),
        window_frames=int(window_frames),
        rho_th_mode=str(args.rho_th_mode),
        rho_th_reference_json=json.dumps(rho_th_reference, sort_keys=True),
    )
    component_stats = best.get("component_stats", [])
    if component_stats:
        with open(out_dir / "selected_window_components.csv", "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "component",
                "n_cluster_jumps",
                "voxels",
                "volume",
                "radius",
                "density_sum",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in component_stats:
                writer.writerow(row)

    assumptions = [
        "MD coarse-graining uses periodic FFT convolution with phi(r) proportional to exp(-r/d).",
        "The coarse-graining kernel is normalized to sum to one before division by voxel volume.",
        "d is estimated from the total g(r) first minimum of the first production frame unless --coarse-d is provided.",
        "STZ volume for the undersize lower limit is not numerically specified by Zeng; stz_radius is an explicit tunable assumption.",
        "This script identifies the Fig.5(b)-style dynamic region and boundary; Fig.6 stress/strain curves require a later boundary-tracing step.",
        "Formal undersize sieve uses small-box vertices. Center points are diagnostic only when --undersize-point-mode=centers.",
        "For vertices, coarse membership uses periodic adjacent coarse-cell membership because Zeng does not specify a programmatic boundary rule.",
    ]
    summary = {
        "paper_checks": {
            "lc2_3d_md": 0.057,
            "chi4_source": "analysis/chi4_md3d_gdot0p015_type1_auto_cosheared_1500f_lag1",
            "t_chi": float(args.t_chi),
            "particle_scope": "type1-only analysis, not all-N Appendix B definition",
            "cage_jump_method": "Candelier recursive trajectory segmentation on a long trajectory; t_chi is applied only as an event-selection window",
            "threshold_tol": float(args.threshold_tol),
            "rho_th_definition": "mean of final rho_avg values over different t_ini windows" if str(args.rho_th_mode) == "reference" else "diagnostic per-window rho_th",
            "md_coarse_grain": "phi(r) proportional to exp(-r/d)",
            "coarse_kernel_normalization": kernel_info["normalization"],
            "undersize_d_prime": "L/30 for 3D systems",
            "undersize_radius": "sqrt(2) d_prime / 2",
            "undersize_point_mode": str(args.undersize_point_mode),
        },
        "assumptions": assumptions,
        "input": {
            "jumps": str(Path(args.jumps)),
            "dump": str(Path(args.dump)),
            "gamma_dot": float(args.gamma_dot),
            "dt_frame": float(args.dt_frame),
            "cage_source_frames": (
                int(cage_source_summary["frames"])
                if isinstance(cage_source_summary, dict) and "frames" in cage_source_summary
                else None
            ),
            "cage_source_start_frame": (
                int(cage_source_summary["start_frame"])
                if isinstance(cage_source_summary, dict) and "start_frame" in cage_source_summary
                else None
            ),
            "cage_jump_algorithm": (
                cage_source_summary.get("cage_jump_algorithm")
                if isinstance(cage_source_summary, dict)
                else None
            ),
        },
        "box_lengths": [float(x) for x in box_lengths],
        "coarse_d": float(coarse_d),
        "coarse_kernel": kernel_info,
        "gr_info": gr_info,
        "grid_n": int(args.grid_n),
        "connectivity": int(args.connectivity),
        "coarse_segmentation": str(args.coarse_segmentation),
        "rho_th_mode": str(args.rho_th_mode),
        "rho_th_reference": rho_th_reference,
        "undersize_grid_n": int(args.undersize_grid_n),
        "undersize_point_mode": str(args.undersize_point_mode),
        "component_mode": str(args.component_mode),
        "multi_component_filters": {
            "min_component_jumps": int(args.multi_min_component_jumps),
            "max_components": int(args.multi_max_components),
            "max_reference_distance": float(args.multi_max_reference_distance),
        },
        "all_candidate_plot": {
            "enabled": bool(args.plot_all_candidates),
            "min_jumps": int(args.all_candidate_min_jumps),
            "min_particles": int(args.all_candidate_min_particles),
            "max_particles": int(args.all_candidate_max_particles),
            "max_refined_clusters": int(args.all_candidate_max_clusters),
            "max_coarse_components_checked": int(args.all_candidate_max_coarse),
        },
        "window_frames": int(window_frames),
        "window_time": float(window_frames * args.dt_frame),
        "window_stride": int(args.window_stride),
        "selection": selection_info,
        "cluster_particle_count": {
            "particle_type": int(args.cluster_particle_type),
            "frame_policy": str(args.particle_count_frame),
            "type_count_total": int(len(particle_count_selection["selected_ids"])),
            "number_density": float(cluster_particle_density),
            "localized_filter_uses_estimated_count_during_window_scan": True,
            "selected_cluster_exact_count_is_in_best_window_particle_count": True,
        },
        "stz_radius_assumption": float(args.stz_radius),
        "c_prime": float(args.c_prime),
        "min_voxels": int(min_voxels),
        "n_windows_scanned": int(len(starts)),
        "n_windows_with_cluster": int(len(rows)),
        "best_window": best,
    }
    (out_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Done. Output: {}".format(out_dir))


if __name__ == "__main__":
    main()
