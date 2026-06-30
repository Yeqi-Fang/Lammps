#!/usr/bin/env python3
"""MD2D chi4, Candelier cage jumps, and Fig.5(d)-style cluster workflow."""

import argparse
import csv
import gzip
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zeng_repro.cage_jump import find_cage_jumps_recursive  # noqa: E402


@dataclass
class MD2DConfig:
    gamma_dot: float = 0.001
    dt_lammps: float = 0.005
    dump_dt: float = 0.5
    shear_origin_y: float = 0.0
    temperature: float = 0.526
    overlap_a: float = 0.3
    lc2: float = 0.048
    n_particles: int = 20000
    expected_frames: int = 100001
    chi4_max_time: float = 1000.0
    chi4_lag_dt: float = 5.0
    chi4_origin_dt: float = 100.0
    cage_min_segment: int = 4
    jump_vector_window: int = 5
    cluster_d: float = 1.6
    cluster_grid_n: int = 80
    threshold_tol: float = 0.10
    stz_radius: float = 1.5
    c_prime: float = 2.0
    r_tilde_method_version: str = "wrapped_sawtooth_v2_shear_origin"


TRAJECTORY_CONFIG_KEYS = (
    "gamma_dot",
    "dt_lammps",
    "dump_dt",
    "shear_origin_y",
    "n_particles",
    "expected_frames",
    "r_tilde_method_version",
)


def log(message):
    print("[{}] {}".format(time.strftime("%F %T"), message), flush=True)


def trajectory_config(cfg):
    data = asdict(cfg)
    return {key: data[key] for key in TRAJECTORY_CONFIG_KEYS}


def assert_manifest_compatible(meta, cfg):
    old = meta.get("trajectory_config")
    new = trajectory_config(cfg)
    if old != new:
        raise RuntimeError("Stored r_tilde config differs from current config; rerun with --rebuild")


def open_dump_text(path):
    if shutil.which("gzip"):
        proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE)
        return proc, io.TextIOWrapper(proc.stdout, encoding="ascii", errors="replace", newline="")
    return None, gzip.open(str(path), "rt", encoding="ascii", errors="replace", newline="")


def close_dump_text(proc, stream):
    stream.close()
    if proc is None:
        return
    rc = proc.wait()
    if rc not in (0, -13, 141):
        raise RuntimeError("gzip exited with status {}".format(rc))


def parse_box(bounds):
    xlo_b, xhi_b, xy = bounds[0]
    ylo_b, yhi_b, _ = bounds[1]
    xlo = xlo_b - min(0.0, xy)
    xhi = xhi_b - max(0.0, xy)
    return {
        "xlo": float(xlo),
        "ylo": float(ylo_b),
        "lx": float(xhi - xlo),
        "ly": float(yhi_b - ylo_b),
        "xy": float(xy),
    }


def read_frame(stream):
    line = stream.readline()
    if not line:
        return None
    if not line.startswith("ITEM: TIMESTEP"):
        raise RuntimeError("Expected TIMESTEP, got {!r}".format(line[:80]))
    step = int(stream.readline())
    stream.readline()
    n_atoms = int(stream.readline())
    stream.readline()
    bounds = []
    for _ in range(3):
        vals = [float(x) for x in stream.readline().split()]
        while len(vals) < 3:
            vals.append(0.0)
        bounds.append(vals[:3])
    box = parse_box(bounds)
    columns = stream.readline().strip().split()[2:]
    col = {name: i for i, name in enumerate(columns)}
    text = "".join(stream.readline() for _ in range(n_atoms))
    data = np.fromstring(text, sep=" ", dtype=np.float64).reshape(n_atoms, len(columns))
    ids = data[:, col["id"]].astype(np.int64)
    if not (ids[0] == 1 and ids[-1] == n_atoms and np.all(np.diff(ids) == 1)):
        order = np.argsort(ids)
        data = data[order]
    types = data[:, col["type"]].astype(np.int16)
    xy = data[:, [col["x"], col["y"]]].astype(np.float64)
    return step, box, types, xy


def fold_to_cosheared_box(xy, box):
    sy = (xy[:, 1] - box["ylo"]) / box["ly"]
    sx = (xy[:, 0] - box["xlo"] - box["xy"] * sy) / box["lx"]
    return np.column_stack(((sx % 1.0) * box["lx"], (sy % 1.0) * box["ly"]))


def fold_to_triclinic_box(xy, box):
    sy = (xy[:, 1] - box["ylo"]) / box["ly"]
    sx = (xy[:, 0] - box["xlo"] - box["xy"] * sy) / box["lx"]
    sy = sy % 1.0
    sx = sx % 1.0
    return np.column_stack((box["xlo"] + sx * box["lx"] + box["xy"] * sy, box["ylo"] + sy * box["ly"]))


def wrapped_y_average(prev_y, curr_y, box):
    y0 = np.mod((prev_y - box["ylo"]) / box["ly"], 1.0)
    y1 = np.mod((curr_y - box["ylo"]) / box["ly"], 1.0)
    ds = y1 - y0
    ds -= np.rint(ds)
    y1c = y0 + ds
    avg = np.empty_like(y0)
    inside = (y1c >= 0.0) & (y1c < 1.0)
    avg[inside] = 0.5 * (y0[inside] + y1c[inside])
    up = y1c >= 1.0
    if np.any(up):
        alpha = (1.0 - y0[up]) / ds[up]
        y1w = y1c[up] - np.floor(y1c[up])
        avg[up] = 0.5 * (y0[up] + 1.0) * alpha + 0.5 * y1w * (1.0 - alpha)
    down = y1c < 0.0
    if np.any(down):
        alpha = -y0[down] / ds[down]
        y1w = y1c[down] - np.floor(y1c[down])
        avg[down] = 0.5 * y0[down] * alpha + 0.5 * (1.0 + y1w) * (1.0 - alpha)
    return box["ylo"] + box["ly"] * avg


def nonaffine_increment(prev_xy, curr_xy, box, gamma_dot, dt_frame, shear_origin_y=0.0):
    dx = curr_xy[:, 0] - prev_xy[:, 0]
    dy = curr_xy[:, 1] - prev_xy[:, 1]
    n_y = np.round(dy / box["ly"])
    dy -= n_y * box["ly"]
    dx -= n_y * box["xy"]
    dx -= np.round(dx / box["lx"]) * box["lx"]
    affine_y = wrapped_y_average(prev_xy[:, 1], curr_xy[:, 1], box) - shear_origin_y
    affine = gamma_dot * dt_frame * affine_y
    return np.column_stack((dx - affine, dy)), affine.astype(np.float32)


def find_dump(repo_root, run_group):
    run_dir = repo_root / "zeng_reproduction" / "data" / "MD" / "2D" / "raw" / run_group / "gdot0p001"
    files = sorted(run_dir.glob("*.lammpstrj.gz"))
    if len(files) != 1:
        raise RuntimeError("Expected one gdot0p001 dump under {}, found {}".format(run_dir, len(files)))
    return files[0]


def memmap_paths(out_dir):
    return {
        "positions": out_dir / "positions_wrapped.float32",
        "r_tilde": out_dir / "r_tilde.float32",
        "affine_x": out_dir / "affine_x.float32",
        "manifest": out_dir / "trajectory_manifest.json",
    }


def build_r_tilde(dump_path, out_dir, cfg, max_frames=0, rebuild=False):
    paths = memmap_paths(out_dir)
    if paths["manifest"].exists() and not rebuild:
        meta = json.loads(paths["manifest"].read_text())
        assert_manifest_compatible(meta, cfg)
        return open_trajectory(out_dir, meta)

    n_frames_alloc = int(max_frames) if max_frames else cfg.expected_frames
    pos = np.memmap(paths["positions"], dtype=np.float32, mode="w+", shape=(n_frames_alloc, cfg.n_particles, 2))
    rt = np.memmap(paths["r_tilde"], dtype=np.float32, mode="w+", shape=(n_frames_alloc, cfg.n_particles, 2))
    ax = np.memmap(paths["affine_x"], dtype=np.float32, mode="w+", shape=(n_frames_alloc, cfg.n_particles))
    steps = np.empty(n_frames_alloc, dtype=np.int64)
    times = np.empty(n_frames_alloc, dtype=np.float64)
    boxes = np.empty((n_frames_alloc, 5), dtype=np.float64)

    proc, stream = open_dump_text(dump_path)
    acc = np.zeros((cfg.n_particles, 2), dtype=np.float64)
    affine_acc = np.zeros(cfg.n_particles, dtype=np.float64)
    prev_xy = None
    types0 = None
    max_step = 0.0
    frame_i = 0
    try:
        while frame_i < n_frames_alloc:
            frame = read_frame(stream)
            if frame is None:
                break
            step, box, types, xy = frame
            if len(types) != cfg.n_particles:
                raise RuntimeError("Expected {} particles, got {}".format(cfg.n_particles, len(types)))
            if types0 is None:
                types0 = types.copy()
            if prev_xy is not None:
                inc, affine = nonaffine_increment(prev_xy, xy, box, cfg.gamma_dot, cfg.dump_dt, cfg.shear_origin_y)
                acc += inc
                affine_acc += affine
                max_step = max(max_step, float(np.max(np.linalg.norm(inc, axis=1))))
            pos[frame_i] = xy.astype(np.float32)
            rt[frame_i] = acc.astype(np.float32)
            ax[frame_i] = affine_acc.astype(np.float32)
            steps[frame_i] = step
            times[frame_i] = step * cfg.dt_lammps
            boxes[frame_i] = [box["xlo"], box["ylo"], box["lx"], box["ly"], box["xy"]]
            prev_xy = xy
            frame_i += 1
            if frame_i % 1000 == 0:
                log("stored trajectory frame {}/{}".format(frame_i, n_frames_alloc))
    finally:
        close_dump_text(proc, stream)

    meta = {
        "config": asdict(cfg),
        "trajectory_config": trajectory_config(cfg),
        "dump": str(dump_path),
        "n_frames": int(frame_i),
        "n_particles": int(cfg.n_particles),
        "dtype": "float32",
        "positions": str(paths["positions"]),
        "r_tilde": str(paths["r_tilde"]),
        "affine_x": str(paths["affine_x"]),
        "types": str(out_dir / "types.npy"),
        "steps": str(out_dir / "steps.npy"),
        "times": str(out_dir / "times.npy"),
        "boxes": str(out_dir / "boxes.npy"),
        "max_nonaffine_step": float(max_step),
        "affine_y": "wrapped_sawtooth_average",
        "shear_origin_y": float(cfg.shear_origin_y),
        "ylo_min": float(np.min(boxes[:frame_i, 1])) if frame_i else None,
        "ylo_max": float(np.max(boxes[:frame_i, 1])) if frame_i else None,
    }
    if frame_i and (abs(meta["ylo_min"] - cfg.shear_origin_y) > 1.0e-8 or abs(meta["ylo_max"] - cfg.shear_origin_y) > 1.0e-8):
        log("WARNING: ylo range [{:.6g}, {:.6g}] differs from shear_origin_y={:.6g}".format(meta["ylo_min"], meta["ylo_max"], cfg.shear_origin_y))
    np.save(out_dir / "types.npy", types0)
    np.save(out_dir / "steps.npy", steps[:frame_i])
    np.save(out_dir / "times.npy", times[:frame_i])
    np.save(out_dir / "boxes.npy", boxes[:frame_i])
    paths["manifest"].write_text(json.dumps(meta, indent=2, sort_keys=True))
    pos.flush()
    rt.flush()
    ax.flush()
    return open_trajectory(out_dir, meta)


def open_trajectory(out_dir, meta):
    n_frames = int(meta["n_frames"])
    n_particles = int(meta["n_particles"])
    return {
        "positions": np.memmap(out_dir / "positions_wrapped.float32", dtype=np.float32, mode="r", shape=(n_frames, n_particles, 2)),
        "r_tilde": np.memmap(out_dir / "r_tilde.float32", dtype=np.float32, mode="r", shape=(n_frames, n_particles, 2)),
        "affine_x": np.memmap(out_dir / "affine_x.float32", dtype=np.float32, mode="r", shape=(n_frames, n_particles)),
        "types": np.load(out_dir / "types.npy"),
        "steps": np.load(out_dir / "steps.npy"),
        "times": np.load(out_dir / "times.npy"),
        "boxes": np.load(out_dir / "boxes.npy"),
        "meta": meta,
    }


def box_dict(row):
    return {"xlo": row[0], "ylo": row[1], "lx": row[2], "ly": row[3], "xy": row[4]}


def periodic_images(points, box):
    shifts = np.asarray([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=float)
    lattice = np.asarray([[box["lx"], 0.0], [box["xy"], box["ly"]]], dtype=float)
    vectors = shifts @ lattice
    return (points[None, :, :] + vectors[:, None, :]).reshape(-1, 2)


def count_overlap_pairs(current, reference, box, cutoff):
    cur = fold_to_triclinic_box(current, box)
    ref = fold_to_triclinic_box(reference, box)
    tree = cKDTree(periodic_images(ref, box))
    return float(np.sum(tree.query_ball_point(cur, cutoff, return_length=True)))


def compute_chi4(traj, out_dir, cfg):
    n_frames, n_particles = traj["positions"].shape[:2]
    max_lag = min(n_frames - 1, int(round(cfg.chi4_max_time / cfg.dump_dt)))
    lag_stride = max(1, int(round(cfg.chi4_lag_dt / cfg.dump_dt)))
    origin_stride = max(1, int(round(cfg.chi4_origin_dt / cfg.dump_dt)))
    lags = np.arange(0, max_lag + 1, lag_stride, dtype=np.int32)
    if lags[-1] != max_lag:
        lags = np.append(lags, np.int32(max_lag))

    beta = 1.0 / cfg.temperature
    volume = float(traj["boxes"][0, 2] * traj["boxes"][0, 3])
    origins_all = np.arange(0, n_frames, origin_stride, dtype=np.int32)
    chi4 = np.empty(len(lags), dtype=np.float64)
    q_mean = np.empty(len(lags), dtype=np.float64)
    q_var = np.empty(len(lags), dtype=np.float64)
    n_origins = np.empty(len(lags), dtype=np.int32)
    q_values = np.full((len(lags), len(origins_all)), np.nan, dtype=np.float64)

    for li, lag in enumerate(lags):
        origins = origins_all[origins_all + lag < n_frames]
        q_this = np.empty(len(origins), dtype=np.float64)
        for oi, origin in enumerate(origins):
            end = int(origin + lag)
            box = box_dict(traj["boxes"][end])
            adv = traj["positions"][origin].astype(np.float64, copy=True)
            adv[:, 0] += traj["affine_x"][end].astype(np.float64) - traj["affine_x"][origin].astype(np.float64)
            q_this[oi] = count_overlap_pairs(traj["positions"][end], adv, box, cfg.overlap_a)
        n_origins[li] = len(origins)
        q_values[li, : len(q_this)] = q_this
        q_mean[li] = float(np.mean(q_this))
        q_var[li] = float(np.var(q_this))
        chi4[li] = beta * volume * q_var[li] / float(n_particles * n_particles)
        if li % max(1, len(lags) // 10) == 0:
            log("chi4 lag {}/{} t={:.3g} origins={} chi4={:.6g}".format(li + 1, len(lags), lag * cfg.dump_dt, len(origins), chi4[li]))

    peak = int(np.nanargmax(chi4))
    result = {
        "lags": lags,
        "lag_times": lags.astype(np.float64) * cfg.dump_dt,
        "chi4": chi4,
        "q_mean": q_mean,
        "q_var": q_var,
        "n_origins": n_origins,
        "q_values": q_values,
        "summary": {
            "method": "collective_advected_reference_triclinic_pbc",
            "t_chi": float(lags[peak] * cfg.dump_dt),
            "lag_frame_chi": int(lags[peak]),
            "chi4_peak": float(chi4[peak]),
            "q_mean_over_n_peak": float(q_mean[peak] / n_particles),
            "peak_at_lag_boundary": bool(peak == 0 or peak == len(lags) - 1),
            "beta": beta,
            "volume": volume,
            "overlap_a": cfg.overlap_a,
            "n_particles": n_particles,
        },
    }
    np.savez_compressed(out_dir / "chi4_md2d_gdot0p001.npz", **{k: v for k, v in result.items() if k != "summary"}, summary_json=np.array(json.dumps(result["summary"])))
    write_chi4_csv(out_dir / "chi4_md2d_gdot0p001.csv", result)
    (out_dir / "chi4_md2d_gdot0p001.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


def write_chi4_csv(path, data):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lag_frames", "lag_time", "chi4", "Q_mean", "Q_var", "n_origins"])
        for row in zip(data["lags"], data["lag_times"], data["chi4"], data["q_mean"], data["q_var"], data["n_origins"]):
            writer.writerow(row)


def jump_vector(r_tilde, particle, frame, window):
    i0 = max(0, frame - window)
    i1 = frame
    j0 = frame + 1
    j1 = min(r_tilde.shape[0], frame + 1 + window)
    if i1 <= i0 or j1 <= j0:
        k = min(max(1, frame), r_tilde.shape[0] - 2)
        return r_tilde[k + 1, particle] - r_tilde[k - 1, particle]
    return r_tilde[j0:j1, particle].mean(axis=0) - r_tilde[i0:i1, particle].mean(axis=0)


def detect_cage_jumps(traj, out_dir, cfg):
    r_tilde = traj["r_tilde"]
    positions = traj["positions"]
    boxes = traj["boxes"]
    particle_ids = []
    jump_frames = []
    jump_times = []
    jump_positions = []
    jump_vectors = []
    n_particles = r_tilde.shape[1]
    for pi in range(n_particles):
        frames = sorted(set(find_cage_jumps_recursive(r_tilde[:, pi, :].astype(np.float64), cfg.lc2, min_segment=cfg.cage_min_segment)))
        for frame in frames:
            box = box_dict(boxes[frame])
            particle_ids.append(pi + 1)
            jump_frames.append(frame)
            jump_times.append(float(traj["times"][frame]))
            jump_positions.append(fold_to_cosheared_box(positions[frame, pi][None, :], box)[0])
            jump_vectors.append(jump_vector(r_tilde, pi, frame, cfg.jump_vector_window))
        if (pi + 1) % max(1, n_particles // 20) == 0:
            log("cage detection {}/{} particles, events={}".format(pi + 1, n_particles, len(jump_frames)))

    jumps = {
        "particle_id": np.asarray(particle_ids, dtype=np.int32),
        "jump_frame": np.asarray(jump_frames, dtype=np.int32),
        "jump_time": np.asarray(jump_times, dtype=np.float64),
        "jump_position": np.asarray(jump_positions, dtype=np.float32).reshape((-1, 2)),
        "jump_vector": np.asarray(jump_vectors, dtype=np.float32).reshape((-1, 2)),
    }
    np.savez_compressed(out_dir / "cage_jumps_md2d_gdot0p001.npz", **jumps, lc2=np.array(cfg.lc2))
    return jumps


def empty_window(jumps, t_chi, times):
    start = float(times[0]) if len(times) else 0.0
    mask = np.zeros(len(jumps["jump_time"]), dtype=bool)
    return {"start": start, "end": start + max(0.0, float(t_chi)), "mask": mask, "n_jumps": 0}


def threshold_iterative(density, tol):
    rho_i = float(np.max(density))
    for _ in range(100):
        below = density[density < rho_i]
        if len(below) == 0:
            break
        rho_avg = float(np.mean(below))
        if rho_avg > 0.0 and abs(rho_avg - rho_i) / rho_avg < tol:
            return rho_avg
        rho_i = rho_avg
    return rho_i


def periodic_labels(mask):
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    parent = np.arange(nx * ny, dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    true = np.argwhere(mask)
    for i, j in true:
        a = i * ny + j
        for ni, nj in [((i + 1) % nx, j), (i, (j + 1) % ny)]:
            if mask[ni, nj]:
                union(a, ni * ny + nj)
    labels = np.zeros(nx * ny, dtype=np.int32)
    roots = {}
    next_label = 1
    for i, j in true:
        root = find(i * ny + j)
        if root not in roots:
            roots[root] = next_label
            next_label += 1
        labels[i * ny + j] = roots[root]
    return labels.reshape(mask.shape)


def exponential_density(points, box_lengths, grid_n, coarse_d, chunk=256):
    x_axis = (np.arange(grid_n, dtype=np.float64) + 0.5) * box_lengths[0] / grid_n
    y_axis = (np.arange(grid_n, dtype=np.float64) + 0.5) * box_lengths[1] / grid_n
    gx, gy = np.meshgrid(x_axis, y_axis, indexing="ij")
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    density = np.zeros(len(grid), dtype=np.float64)
    if len(points) == 0:
        return density.reshape((grid_n, grid_n))
    for start in range(0, len(grid), chunk):
        g = grid[start : start + chunk]
        dr = points[None, :, :] - g[:, None, :]
        dr -= box_lengths * np.round(dr / box_lengths)
        density[start : start + chunk] = np.exp(-np.linalg.norm(dr, axis=2) / coarse_d).sum(axis=1)
    return density.reshape((grid_n, grid_n))


def component_from_density(density, rho_th, cfg, box_lengths):
    labels = periodic_labels(density >= rho_th)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if len(ids) == 0:
        return labels, 0, 0
    min_area = cfg.c_prime * math.pi * cfg.stz_radius * cfg.stz_radius
    cell_area = float(np.prod(box_lengths / cfg.cluster_grid_n))
    min_voxels = max(1, int(math.ceil(min_area / cell_area)))
    valid = ids[counts >= min_voxels]
    if len(valid) == 0:
        return labels, 0, min_voxels
    valid_counts = np.asarray([counts[np.where(ids == v)[0][0]] for v in valid])
    return labels, int(valid[int(np.argmax(valid_counts))]), min_voxels


def window_points(jumps, start, t_chi, box_lengths):
    mask = (jumps["jump_time"] >= start) & (jumps["jump_time"] < start + t_chi)
    points = np.mod(jumps["jump_position"][mask].astype(np.float64), box_lengths)
    return mask, points


def density_selected_window(jumps, t_chi, times, cfg, box_lengths):
    if len(jumps["jump_time"]) == 0 or t_chi <= 0.0:
        return empty_window(jumps, t_chi, times), None, 0.0, "none"
    stride = max(cfg.dump_dt, t_chi / 10.0)
    starts = np.arange(float(times[0]), max(float(times[-1]) - t_chi, float(times[0])) + stride, stride)
    best_any = None
    best_valid = None
    for start in starts:
        mask, points = window_points(jumps, float(start), t_chi, box_lengths)
        if len(points) == 0:
            continue
        density = exponential_density(points, box_lengths, cfg.cluster_grid_n, cfg.cluster_d)
        rho_i = threshold_iterative(density, cfg.threshold_tol)
        _, component, _ = component_from_density(density, rho_i, cfg, box_lengths)
        score = float(np.max(density))
        candidate = {"start": float(start), "mask": mask, "points": points, "density": density, "score": score, "rho_th": rho_i}
        if best_any is None or score > best_any["score"]:
            best_any = candidate
        if component and (best_valid is None or score > best_valid["score"]):
            best_valid = candidate
    best = best_valid if best_valid is not None else best_any
    if best is None:
        return empty_window(jumps, t_chi, times), None, 0.0, "none"
    window = {"start": best["start"], "end": best["start"] + float(t_chi), "mask": best["mask"], "n_jumps": int(np.count_nonzero(best["mask"]))}
    selection = "max_density_window_valid_min_size" if best_valid is not None else "max_density_window_no_valid_component"
    return window, best, float(best["rho_th"]), selection


def build_cluster_density(jumps, t_chi, traj, out_dir, cfg):
    box_lengths = traj["boxes"][0, 2:4].astype(np.float64)
    window, best, rho_th, window_selection = density_selected_window(jumps, t_chi, traj["times"], cfg, box_lengths)
    reference_time = window["end"]
    points = best["points"] if best is not None else np.empty((0, 2), dtype=np.float64)
    if len(points) == 0:
        empty = {
            "points": points.astype(np.float32),
            "density": np.zeros((cfg.cluster_grid_n, cfg.cluster_grid_n), dtype=np.float32),
            "labels": np.zeros((cfg.cluster_grid_n, cfg.cluster_grid_n), dtype=np.int32),
            "rho_th": 0.0,
            "component": 0,
            "in_component": np.zeros(0, dtype=bool),
            "box_lengths": box_lengths,
            "reference_time": float(reference_time),
            "min_voxels": 0,
            "window_selection": window_selection,
            "coordinate_system": "co_sheared_orthogonal_box",
        }
        np.savez_compressed(out_dir / "cluster_md2d_gdot0p001.npz", **empty)
        return window, empty
    density = best["density"]
    labels, component, min_voxels = component_from_density(density, rho_th, cfg, box_lengths)
    point_idx = np.floor(points / box_lengths * cfg.cluster_grid_n).astype(int) % cfg.cluster_grid_n
    in_component = (labels[point_idx[:, 0], point_idx[:, 1]] == component) if component else np.zeros(len(points), dtype=bool)
    result = {
        "points": points.astype(np.float32),
        "density": density.astype(np.float32),
        "labels": labels.astype(np.int32),
        "rho_th": float(rho_th),
        "component": int(component),
        "in_component": in_component,
        "box_lengths": box_lengths,
        "reference_time": float(reference_time),
        "min_voxels": int(min_voxels),
        "window_selection": window_selection,
        "coordinate_system": "co_sheared_orthogonal_box",
    }
    np.savez_compressed(out_dir / "cluster_md2d_gdot0p001.npz", **result)
    return window, result


def plot_outputs(out_dir, cfg, chi4, jumps, window, cluster):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = chi4["lag_times"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=180)
    axes[0].plot(t, chi4["chi4"], "o-", ms=2.6, lw=1.0, color="black")
    axes[0].axvline(chi4["summary"]["t_chi"], color="crimson", ls="--", lw=1.0)
    axes[0].set_xlabel(r"lag time $t$")
    axes[0].set_ylabel(r"$\chi_4(t)$")
    axes[1].plot(t, chi4["q_mean"] / cfg.n_particles, "o-", ms=2.6, lw=1.0, color="steelblue")
    axes[1].set_xlabel(r"lag time $t$")
    axes[1].set_ylabel(r"$\langle Q(t)\rangle/N$")
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "chi4_md2d_gdot0p001.png")
    plt.close(fig)

    points = cluster["points"]
    labels = cluster["labels"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=220)
    has_component = int(cluster["component"]) > 0
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=3, c="0.55", alpha=0.45, linewidths=0)
        if has_component:
            ax.scatter(points[cluster["in_component"], 0], points[cluster["in_component"], 1], s=5, c="crimson", alpha=0.75, linewidths=0)
    if has_component:
        mask = labels == cluster["component"]
        boundary = mask & ~binary_erosion(mask)
        yy, xx = np.where(boundary.T)
        Lx, Ly = cluster["box_lengths"]
        ax.scatter((xx + 0.5) * Lx / cfg.cluster_grid_n, (yy + 0.5) * Ly / cfg.cluster_grid_n, s=2, c="black", alpha=0.75, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(0, cluster["box_lengths"][0])
    ax.set_ylim(0, cluster["box_lengths"][1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"MD2D cage jumps, $\dot\gamma=0.001$")
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_md2d_gdot0p001.png")
    plt.close(fig)

    summary = {
        "t_chi": chi4["summary"]["t_chi"],
        "window": {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in window.items() if k != "mask"},
        "window_selection": str(cluster["window_selection"]),
        "coordinate_system": str(cluster["coordinate_system"]),
        "n_total_jumps": int(len(jumps["jump_time"])),
        "n_window_jumps": int(window["n_jumps"]),
        "n_component_jumps": int(np.count_nonzero(cluster["in_component"])),
        "rho_th": float(cluster["rho_th"]),
        "component": int(cluster["component"]),
    }
    (out_dir / "summary_md2d_gdot0p001.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-group", default="md2d_prod_slim_20260629_1412")
    parser.add_argument("--max-frames", type=int, default=0, help="Diagnostic limit; 0 means full trajectory")
    parser.add_argument("--chi4-max-time", type=float, default=None)
    parser.add_argument("--chi4-lag-dt", type=float, default=None)
    parser.add_argument("--chi4-origin-dt", type=float, default=None)
    parser.add_argument("--shear-origin-y", type=float, default=None)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    cfg = MD2DConfig()
    if args.chi4_max_time is not None:
        cfg.chi4_max_time = args.chi4_max_time
    if args.chi4_lag_dt is not None:
        cfg.chi4_lag_dt = args.chi4_lag_dt
    if args.chi4_origin_dt is not None:
        cfg.chi4_origin_dt = args.chi4_origin_dt
    if args.shear_origin_y is not None:
        cfg.shear_origin_y = args.shear_origin_y
    suffix = "gdot0p001" if not args.max_frames else "gdot0p001_smoke_{}f".format(args.max_frames)
    out_dir = repo / "zeng_reproduction" / "data" / "MD" / "2D" / "processed" / "chi4_cage_cluster" / args.run_group / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = find_dump(repo, args.run_group)
    log("building/loading r_tilde from {}".format(dump))
    traj = build_r_tilde(dump, out_dir, cfg, max_frames=args.max_frames, rebuild=args.rebuild)
    log("computing chi4")
    chi4 = compute_chi4(traj, out_dir, cfg)
    if chi4["summary"]["peak_at_lag_boundary"]:
        log("WARNING: chi4 peak is on lag boundary; increase chi4_max_time before final use")
    log("detecting cage jumps")
    jumps = detect_cage_jumps(traj, out_dir, cfg)
    log("building cluster")
    window, cluster = build_cluster_density(jumps, chi4["summary"]["t_chi"], traj, out_dir, cfg)
    plot_outputs(out_dir, cfg, chi4, jumps, window, cluster)
    log("wrote outputs under {}".format(out_dir))


if __name__ == "__main__":
    main()
