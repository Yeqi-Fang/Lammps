#!/usr/bin/env python3
"""Plot MD2D cluster candidates from completed cage-jump output."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from md2d_chi4_cage_cluster import (  # noqa: E402
    MD2DConfig,
    cluster_candidate_starts,
    component_from_density,
    exponential_density,
    threshold_iterative,
)


def output_dir(repo_root, run_group):
    return Path(repo_root) / "zeng_reproduction" / "data" / "MD" / "2D" / "processed" / "chi4_cage_cluster" / run_group / "gdot0p001"


def boundary_points(labels, component, box_lengths):
    from scipy.ndimage import binary_erosion

    if component <= 0:
        return np.empty((0, 2), dtype=np.float64)
    mask = labels == component
    boundary = mask & ~binary_erosion(mask)
    yy, xx = np.where(boundary.T)
    grid_n = labels.shape[0]
    pts = np.empty((len(xx), 2), dtype=np.float64)
    pts[:, 0] = (xx + 0.5) * box_lengths[0] / grid_n
    pts[:, 1] = (yy + 0.5) * box_lengths[1] / grid_n
    return pts


def plot_candidate(path, points, boundary, box_lengths, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=220)
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=3, c="black", alpha=0.72, linewidths=0)
    if len(boundary):
        ax.scatter(boundary[:, 0], boundary[:, 1], s=8, c="crimson", alpha=0.95, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(0, box_lengths[0])
    ax.set_ylim(0, box_lengths[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-group", default="md2d_prod_slim_20260629_1412")
    parser.add_argument("--cluster-n-starts", type=int, default=10)
    parser.add_argument("--cluster-start-times", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = output_dir(repo, args.run_group)
    figure_dir = out_dir / "cluster_candidates"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((out_dir / "summary_md2d_gdot0p001.json").read_text())
    t_chi = float(summary["t_chi"])
    times = np.load(out_dir / "times.npy")
    boxes = np.load(out_dir / "boxes.npy")
    box_lengths = boxes[0, 2:4].astype(np.float64)
    cfg = MD2DConfig(cluster_n_starts=int(args.cluster_n_starts))
    start_times = None
    if args.cluster_start_times:
        start_times = np.asarray([float(x) for x in args.cluster_start_times.split(",") if x.strip()], dtype=np.float64)
    starts = cluster_candidate_starts(times, t_chi, cfg, start_times=start_times)

    jumps = np.load(out_dir / "cage_jumps_md2d_gdot0p001.npz", allow_pickle=False)
    order = np.argsort(jumps["jump_time"])
    jump_time = jumps["jump_time"][order]
    jump_pos = np.mod(jumps["jump_position"][order].astype(np.float64), box_lengths)

    rows = []
    for idx, start in enumerate(starts):
        lo = np.searchsorted(jump_time, start, side="left")
        hi = np.searchsorted(jump_time, start + t_chi, side="left")
        points = jump_pos[lo:hi]
        if len(points) == 0:
            rows.append({"index": idx, "start": float(start), "end": float(start + t_chi), "n_jumps": 0, "component": 0})
            continue
        density = exponential_density(points, box_lengths, cfg.cluster_grid_n, cfg.cluster_d)
        rho_th = threshold_iterative(density, cfg.threshold_tol)
        labels, component, min_voxels = component_from_density(density, rho_th, cfg, box_lengths)
        point_idx = np.floor(points / box_lengths * cfg.cluster_grid_n).astype(int) % cfg.cluster_grid_n
        in_component = (labels[point_idx[:, 0], point_idx[:, 1]] == component) if component else np.zeros(len(points), dtype=bool)
        component_cells = int(np.count_nonzero(labels == component)) if component else 0
        area_fraction = component_cells / float(labels.size)
        boundary = boundary_points(labels, int(component), box_lengths)
        title = "candidate {:02d}, t=[{:.1f},{:.1f}], jumps={}, area={:.2f}".format(idx, start, start + t_chi, len(points), area_fraction)
        png = figure_dir / "cluster_candidate_{:02d}_t{:.1f}.png".format(idx, start)
        plot_candidate(png, points, boundary, box_lengths, title)
        rows.append(
            {
                "index": int(idx),
                "start": float(start),
                "end": float(start + t_chi),
                "n_jumps": int(len(points)),
                "rho_th": float(rho_th),
                "component": int(component),
                "component_cells": component_cells,
                "component_area_fraction": float(area_fraction),
                "n_component_jumps": int(np.count_nonzero(in_component)),
                "min_voxels": int(min_voxels),
                "figure": str(png),
            }
        )
        print("candidate {:02d}: jumps={} component={} area={:.3f} n_component={}".format(idx, len(points), component, area_fraction, int(np.count_nonzero(in_component))), flush=True)

    (figure_dir / "cluster_candidates.json").write_text(json.dumps({"t_chi": t_chi, "candidates": rows}, indent=2, sort_keys=True))
    print("wrote {}".format(figure_dir))


if __name__ == "__main__":
    main()
