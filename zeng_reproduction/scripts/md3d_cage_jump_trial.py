#!/usr/bin/env python
"""Streaming cage-jump trial for the Zeng 3D KA high-resolution trajectory.

This script is intentionally a trial tool, not the final Fig. 5/Fig. 6
pipeline. It reuses the Candelier recursive detector from zeng_repro and the
incremental Lees-Edwards nonaffine displacement logic used in the older MSD
scripts, but avoids loading the full 54 GB dump into memory.
"""

import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from zeng_repro.cage_jump import find_cage_jumps_recursive  # noqa: E402


REQUIRED_COLUMNS = ("id", "type", "x", "y", "z", "ix", "iy", "iz")


def open_dump_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="ascii", errors="replace")
    return open(path, "r", encoding="ascii", errors="replace")


def parse_box(header, lines):
    triclinic = "xy" in header
    b0 = lines[0].split()
    b1 = lines[1].split()
    b2 = lines[2].split()
    if triclinic:
        xlo_b, xhi_b, xy = float(b0[0]), float(b0[1]), float(b0[2])
        ylo_b, yhi_b, xz = float(b1[0]), float(b1[1]), float(b1[2])
        zlo_b, zhi_b, yz = float(b2[0]), float(b2[1]), float(b2[2])
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
        zlo, zhi = zlo_b, zhi_b
    else:
        xy = xz = yz = 0.0
        xlo, xhi = float(b0[0]), float(b0[1])
        ylo, yhi = float(b1[0]), float(b1[1])
        zlo, zhi = float(b2[0]), float(b2[1])
    return {
        "xlo": xlo,
        "xhi": xhi,
        "ylo": ylo,
        "yhi": yhi,
        "zlo": zlo,
        "zhi": zhi,
        "Lx": xhi - xlo,
        "Ly": yhi - ylo,
        "Lz": zhi - zlo,
        "xy": xy,
        "xz": xz,
        "yz": yz,
        "triclinic": triclinic,
    }


def read_frame_header(fh):
    line = fh.readline()
    while line and not line.startswith("ITEM: TIMESTEP"):
        line = fh.readline()
    if not line:
        return None
    timestep = int(fh.readline().strip())
    number_header = fh.readline()
    if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError("Unexpected dump format after timestep {}".format(timestep))
    natoms = int(fh.readline().strip())
    box_header = fh.readline().strip()
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError("Missing BOX BOUNDS at timestep {}".format(timestep))
    box = parse_box(box_header, [fh.readline(), fh.readline(), fh.readline()])
    atoms_header = fh.readline().strip()
    if not atoms_header.startswith("ITEM: ATOMS"):
        raise ValueError("Missing ATOMS header at timestep {}".format(timestep))
    columns = atoms_header.split()[2:]
    col = {name: i for i, name in enumerate(columns)}
    missing = [name for name in REQUIRED_COLUMNS if name not in col]
    if missing:
        raise ValueError("Dump is missing required columns: {}".format(", ".join(missing)))
    return timestep, natoms, box, columns, col


def skip_atom_lines(fh, natoms):
    for _ in range(natoms):
        fh.readline()


def first_frame_selection(dump_path, particle_type, max_particles, seed):
    with open_dump_text(dump_path) as fh:
        header = read_frame_header(fh)
        if header is None:
            raise ValueError("Empty dump: {}".format(dump_path))
        timestep, natoms, box, columns, col = header
        ids = np.empty(natoms, dtype=np.int32)
        types = np.empty(natoms, dtype=np.int16)
        for i in range(natoms):
            parts = fh.readline().split()
            ids[i] = int(parts[col["id"]])
            types[i] = int(parts[col["type"]])
    if int(particle_type) == 0:
        candidates = ids.copy()
        type_count = int(len(candidates))
        type_label = "all"
    else:
        candidates = ids[types == int(particle_type)]
        type_count = int(len(candidates))
        type_label = str(particle_type)
    candidates.sort()
    if max_particles and max_particles > 0 and len(candidates) > max_particles:
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(candidates, int(max_particles), replace=False))
        selection_note = "random subset with seed {}".format(seed)
    else:
        selected = candidates
        selection_note = "all type {} particles".format(type_label)
    return {
        "timestep": timestep,
        "natoms": int(natoms),
        "box": box,
        "columns": columns,
        "type_count": type_count,
        "selected_ids": selected.astype(np.int32),
        "selection_note": selection_note,
    }


def parse_selected_atoms(fh, natoms, col, selected_id_to_index, n_selected):
    pos = np.empty((n_selected, 3), dtype=np.float64)
    image = np.empty((n_selected, 3), dtype=np.int32)
    virial_xy = np.full(n_selected, np.nan, dtype=np.float64)
    found = 0
    stress_col = col.get("c_Svirial[4]")
    for _ in range(natoms):
        line = fh.readline()
        parts = line.split()
        atom_id = int(parts[col["id"]])
        out_i = selected_id_to_index.get(atom_id)
        if out_i is None:
            continue
        pos[out_i, 0] = float(parts[col["x"]])
        pos[out_i, 1] = float(parts[col["y"]])
        pos[out_i, 2] = float(parts[col["z"]])
        image[out_i, 0] = int(parts[col["ix"]])
        image[out_i, 1] = int(parts[col["iy"]])
        image[out_i, 2] = int(parts[col["iz"]])
        if stress_col is not None:
            virial_xy[out_i] = float(parts[stress_col])
        found += 1
    if found != n_selected:
        raise ValueError("Selected atom count mismatch: found {}, expected {}".format(found, n_selected))
    return pos, image, virial_xy


def stream_window(dump_path, selected_ids, start_frame, n_frames):
    selected_id_to_index = {int(atom_id): i for i, atom_id in enumerate(selected_ids)}
    n_selected = len(selected_ids)
    steps = np.empty(n_frames, dtype=np.int64)
    boxes = []
    wrapped = np.empty((n_frames, n_selected, 3), dtype=np.float32)
    images = np.empty((n_frames, n_selected, 3), dtype=np.int32)
    virial_xy = np.empty((n_frames, n_selected), dtype=np.float32)

    frame_i = 0
    stored = 0
    t0 = time.time()
    with open_dump_text(dump_path) as fh:
        while stored < n_frames:
            header = read_frame_header(fh)
            if header is None:
                break
            timestep, natoms, box, columns, col = header
            if frame_i < start_frame:
                skip_atom_lines(fh, natoms)
            else:
                pos, image, stress = parse_selected_atoms(
                    fh, natoms, col, selected_id_to_index, n_selected
                )
                steps[stored] = timestep
                boxes.append(box)
                wrapped[stored] = pos.astype(np.float32)
                images[stored] = image
                virial_xy[stored] = stress.astype(np.float32)
                stored += 1
                if stored % max(1, n_frames // 10) == 0:
                    elapsed = time.time() - t0
                    print("  stored {}/{} frames, current step {}, elapsed {:.1f}s".format(
                        stored, n_frames, timestep, elapsed
                    ))
            frame_i += 1

    if stored < n_frames:
        steps = steps[:stored]
        wrapped = wrapped[:stored]
        images = images[:stored]
        virial_xy = virial_xy[:stored]
    if stored < 2:
        raise ValueError("Need at least two frames in selected window, got {}".format(stored))
    return {
        "steps": steps,
        "boxes": boxes,
        "wrapped": wrapped,
        "images": images,
        "virial_xy": virial_xy,
        "frames_read_before_window": frame_i - stored,
    }


def affine_y_coordinate(pos, img, box, mode):
    mode = str(mode).lower()
    Ly = float(box["Ly"])
    if mode == "unwrapped":
        return pos[:, 1] + img[:, 1] * Ly
    if mode == "wrapped":
        return pos[:, 1]
    if mode == "centered":
        return pos[:, 1] - (float(box["ylo"]) + 0.5 * Ly)
    raise ValueError("affine_y must be 'centered', 'wrapped', or 'unwrapped'")


def nonaffine_incremental(wrapped, images, boxes, gamma_dot, dt_frame, affine_rule="trapezoid", affine_y="wrapped"):
    n_frames, n_particles, _ = wrapped.shape
    r_tilde = np.zeros((n_frames, n_particles, 3), dtype=np.float32)
    if n_frames < 2:
        return r_tilde
    affine_rule = str(affine_rule).lower()
    if affine_rule not in {"left", "trapezoid"}:
        raise ValueError("affine_rule must be 'left' or 'trapezoid'")

    box0 = boxes[0]
    Lx, Ly, Lz = float(box0["Lx"]), float(box0["Ly"]), float(box0["Lz"])
    acc = np.zeros((n_particles, 3), dtype=np.float64)
    prev = wrapped[0].astype(np.float64)
    prev_img = images[0].astype(np.float64)
    yu_prev = affine_y_coordinate(prev, prev_img, box0, affine_y)

    for fi in range(1, n_frames):
        cur = wrapped[fi].astype(np.float64)
        img = images[fi].astype(np.float64)
        xy = float(boxes[fi]["xy"])
        yu_curr = affine_y_coordinate(cur, img, boxes[fi], affine_y)

        dx = cur[:, 0] - prev[:, 0]
        dy = cur[:, 1] - prev[:, 1]
        dz = cur[:, 2] - prev[:, 2]

        n_y = np.round(dy / Ly)
        dy -= n_y * Ly

        dx -= n_y * xy
        n_x = np.round(dx / Lx)
        dx -= n_x * Lx

        n_z = np.round(dz / Lz)
        dz -= n_z * Lz

        if affine_rule == "left":
            affine_y_val = yu_prev
        else:
            affine_y_val = 0.5 * (yu_prev + yu_curr)
        dx -= float(gamma_dot) * float(dt_frame) * affine_y_val

        acc[:, 0] += dx
        acc[:, 1] += dy
        acc[:, 2] += dz
        r_tilde[fi] = acc.astype(np.float32)

        prev = cur
        prev_img = img
        yu_prev = yu_curr

    return r_tilde


def cumulative_affine_x(wrapped, images, boxes, gamma_dot, dt_frame, affine_rule="trapezoid", affine_y="wrapped"):
    """Cumulative x displacement from the imposed shear flow.

    Zeng Appendix B uses the shear-adjusted overlap distance with
    gamma_dot * integral y_j(t') dt' e_x.  This returns that integral
    multiplied by gamma_dot for every stored particle and frame.
    """
    n_frames, n_particles, _ = wrapped.shape
    affine_x = np.zeros((n_frames, n_particles), dtype=np.float32)
    if n_frames < 2:
        return affine_x
    affine_rule = str(affine_rule).lower()
    if affine_rule not in {"left", "trapezoid"}:
        raise ValueError("affine_rule must be 'left' or 'trapezoid'")

    Ly = float(boxes[0]["Ly"])
    prev = wrapped[0].astype(np.float64)
    prev_img = images[0].astype(np.float64)
    yu_prev = affine_y_coordinate(prev, prev_img, boxes[0], affine_y)
    acc = np.zeros(n_particles, dtype=np.float64)

    for fi in range(1, n_frames):
        cur = wrapped[fi].astype(np.float64)
        img = images[fi].astype(np.float64)
        yu_curr = affine_y_coordinate(cur, img, boxes[fi], affine_y)
        if affine_rule == "left":
            affine_y_val = yu_prev
        else:
            affine_y_val = 0.5 * (yu_prev + yu_curr)
        acc += float(gamma_dot) * float(dt_frame) * affine_y_val
        affine_x[fi] = acc.astype(np.float32)
        yu_prev = yu_curr

    return affine_x


def diagnose_nonaffine(r_tilde, wrapped, images, boxes):
    if r_tilde.shape[0] < 2:
        return {}

    dr = np.diff(r_tilde.astype(np.float64), axis=0)
    dr_len = np.sqrt(np.sum(dr * dr, axis=2))
    r2 = np.sum(r_tilde.astype(np.float64) ** 2, axis=2)

    box0 = boxes[0]
    Lx, Ly, Lz = float(box0["Lx"]), float(box0["Ly"]), float(box0["Lz"])
    raw_delta = np.diff(wrapped.astype(np.float64), axis=0)
    img_delta = np.diff(images.astype(np.int32), axis=0)

    n_y = np.round(raw_delta[:, :, 1] / Ly).astype(np.int32)
    y_mismatch = int(np.count_nonzero(img_delta[:, :, 1] + n_y))

    n_z = np.round(raw_delta[:, :, 2] / Lz).astype(np.int32)
    z_mismatch = int(np.count_nonzero(img_delta[:, :, 2] + n_z))

    n_x = np.round(raw_delta[:, :, 0] / Lx).astype(np.int32)
    x_mismatch_naive = int(np.count_nonzero(img_delta[:, :, 0] + n_x))

    denom = int(np.prod(n_y.shape))
    tilts = np.asarray([float(b["xy"]) for b in boxes], dtype=np.float64)
    return {
        "increment_rms": float(np.sqrt(np.mean(dr_len * dr_len))),
        "increment_p95": float(np.percentile(dr_len, 95.0)),
        "increment_max": float(np.max(dr_len)),
        "particle_r2_p95_last": float(np.percentile(r2[-1], 95.0)),
        "particle_r2_max_last": float(np.max(r2[-1])),
        "xy_tilt_min": float(np.min(tilts)),
        "xy_tilt_max": float(np.max(tilts)),
        "y_image_mismatch_fraction": float(y_mismatch / denom),
        "z_image_mismatch_fraction": float(z_mismatch / denom),
        "x_image_mismatch_naive_fraction": float(x_mismatch_naive / denom),
    }


def fold_cartesian_to_box(pos, box):
    x = float(pos[0])
    y = float(pos[1])
    z = float(pos[2])
    Lx, Ly, Lz = float(box["Lx"]), float(box["Ly"]), float(box["Lz"])
    xy, xz, yz = float(box["xy"]), float(box["xz"]), float(box["yz"])
    z_s = (z - float(box["zlo"])) / Lz
    y_s = (y - float(box["ylo"]) - yz * z_s) / Ly
    x_s = (x - float(box["xlo"]) - xy * y_s - xz * z_s) / Lx
    return np.asarray(
        [(x_s % 1.0) * Lx, (y_s % 1.0) * Ly, (z_s % 1.0) * Lz],
        dtype=np.float32,
    )


def fold_positions_to_box(pos, box):
    pos = np.asarray(pos, dtype=np.float64)
    Lx, Ly, Lz = float(box["Lx"]), float(box["Ly"]), float(box["Lz"])
    xy, xz, yz = float(box["xy"]), float(box["xz"]), float(box["yz"])
    z_s = (pos[:, 2] - float(box["zlo"])) / Lz
    y_s = (pos[:, 1] - float(box["ylo"]) - yz * z_s) / Ly
    x_s = (pos[:, 0] - float(box["xlo"]) - xy * y_s - xz * z_s) / Lx
    out = np.empty_like(pos, dtype=np.float32)
    out[:, 0] = (x_s % 1.0) * Lx
    out[:, 1] = (y_s % 1.0) * Ly
    out[:, 2] = (z_s % 1.0) * Lz
    return out


def periodic_points(pos, box_lengths):
    pts = np.asarray(pos, dtype=np.float64) % box_lengths
    # scipy cKDTree with boxsize requires coordinates strictly smaller than
    # the box length. Floating point folding can occasionally produce L.
    pts = np.where(pts >= box_lengths, 0.0, pts)
    pts = np.where(pts < 0.0, pts + box_lengths, pts)
    return pts


def jump_vector(r_tilde, particle_i, frame_i, window):
    n_frames = r_tilde.shape[0]
    i0 = max(0, frame_i - window)
    i1 = frame_i
    j0 = frame_i + 1
    j1 = min(n_frames, frame_i + 1 + window)
    if i1 <= i0 or j1 <= j0:
        safe = max(1, min(frame_i, n_frames - 2))
        return r_tilde[safe + 1, particle_i] - r_tilde[safe - 1, particle_i]
    before = r_tilde[i0:i1, particle_i].mean(axis=0)
    after = r_tilde[j0:j1, particle_i].mean(axis=0)
    return after - before


def detect_cage_jumps(r_tilde, selected_ids, steps, times, wrapped, boxes, lc2, min_segment, jvec_window):
    particle_ids = []
    local_indices = []
    jump_frames = []
    jump_steps = []
    jump_times = []
    jump_vectors = []
    jump_positions = []
    jump_positions_raw = []
    n_particles = r_tilde.shape[1]
    report_every = max(1, n_particles // 20)
    for pi in range(n_particles):
        if pi % report_every == 0:
            print("  cage detection {}/{} particles, events so far {}".format(
                pi, n_particles, len(jump_frames)
            ))
        frames = sorted(set(find_cage_jumps_recursive(
            r_tilde[:, pi, :].astype(np.float64),
            float(lc2),
            offset=0,
            min_segment=int(min_segment),
        )))
        for jf in frames:
            jf = max(1, min(int(jf), r_tilde.shape[0] - 2))
            vec = jump_vector(r_tilde, pi, jf, int(jvec_window))
            particle_ids.append(int(selected_ids[pi]))
            local_indices.append(int(pi))
            jump_frames.append(jf)
            jump_steps.append(int(steps[jf]))
            jump_times.append(float(times[jf]))
            jump_vectors.append(vec.astype(np.float32))
            raw_pos = wrapped[jf, pi].astype(np.float32)
            jump_positions_raw.append(raw_pos)
            jump_positions.append(fold_cartesian_to_box(raw_pos, boxes[jf]))
    if jump_vectors:
        vectors = np.vstack(jump_vectors).astype(np.float32)
        positions = np.vstack(jump_positions).astype(np.float32)
        positions_raw = np.vstack(jump_positions_raw).astype(np.float32)
    else:
        vectors = np.zeros((0, 3), dtype=np.float32)
        positions = np.zeros((0, 3), dtype=np.float32)
        positions_raw = np.zeros((0, 3), dtype=np.float32)
    return {
        "particle_id": np.asarray(particle_ids, dtype=np.int32),
        "local_index": np.asarray(local_indices, dtype=np.int32),
        "jump_frames": np.asarray(jump_frames, dtype=np.int32),
        "jump_steps": np.asarray(jump_steps, dtype=np.int64),
        "jump_times": np.asarray(jump_times, dtype=np.float64),
        "jump_vectors": vectors,
        "jump_positions": positions,
        "jump_positions_raw": positions_raw,
    }


def empty_jumps():
    return {
        "particle_id": np.zeros(0, dtype=np.int32),
        "local_index": np.zeros(0, dtype=np.int32),
        "jump_frames": np.zeros(0, dtype=np.int32),
        "jump_steps": np.zeros(0, dtype=np.int64),
        "jump_times": np.zeros(0, dtype=np.float64),
        "jump_vectors": np.zeros((0, 3), dtype=np.float32),
        "jump_positions": np.zeros((0, 3), dtype=np.float32),
        "jump_positions_raw": np.zeros((0, 3), dtype=np.float32),
    }


def estimate_times(steps, dt_frame):
    step_stride = int(steps[1] - steps[0])
    if step_stride <= 0:
        raise ValueError("Non-increasing timesteps in window")
    dt_lammps = float(dt_frame) / float(step_stride)
    return steps.astype(np.float64) * dt_lammps, dt_lammps, step_stride


def make_plots(out_dir, times_rel, r_tilde, jumps, lc2, gamma_dot):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r2 = np.sum(r_tilde.astype(np.float64) ** 2, axis=2)
    msd = np.mean(r2, axis=1)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(times_rel, msd, color="black", lw=1.3)
    ax.axhline(float(lc2), color="crimson", ls="--", lw=1.0, label=r"$l_c^2$")
    ax.set_xlabel(r"$t/\tau_0$ relative to trial window")
    ax.set_ylabel(r"$\langle \tilde r^2(t)\rangle$")
    ax.set_title(r"Trial nonaffine MSD, $\dot\gamma={}$".format(gamma_dot))
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trial_msd.png", dpi=220)
    plt.close(fig)

    vectors = jumps["jump_vectors"]
    positions = jumps["jump_positions"]
    if len(vectors):
        lengths = np.linalg.norm(vectors, axis=1)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.hist(lengths, bins=50, color="steelblue", edgecolor="white", alpha=0.9)
        ax.axvline(float(np.mean(lengths)), color="crimson", ls="--", lw=1.2)
        ax.set_xlabel(r"$l_{cj}$")
        ax.set_ylabel("count")
        ax.set_title("Trial cage-jump length distribution")
        ax.grid(True, ls=":", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "trial_jump_lengths.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.scatter(positions[:, 0], positions[:, 1], s=3, c="black", alpha=0.35, linewidths=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Trial cage-jump positions, xy")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "trial_jump_spatial_xy.png", dpi=220)
        plt.close(fig)

    return {
        "msd_first": float(msd[0]),
        "msd_last": float(msd[-1]),
        "msd_max": float(np.max(msd)),
    }


def extract_tau_alpha(times, fs, threshold=1.0 / math.e):
    times = np.asarray(times, dtype=np.float64)
    fs = np.asarray(fs, dtype=np.float64)
    valid = np.isfinite(times) & np.isfinite(fs)
    times = times[valid]
    fs = fs[valid]
    if len(times) < 2:
        return float("nan")
    below = np.where(fs <= threshold)[0]
    if len(below) == 0:
        return float("nan")
    i1 = int(below[0])
    if i1 == 0:
        return float(times[0])
    i0 = i1 - 1
    f0, f1 = float(fs[i0]), float(fs[i1])
    t0, t1 = float(times[i0]), float(times[i1])
    if abs(f1 - f0) < 1e-12:
        return t1
    return float(t0 + (threshold - f0) * (t1 - t0) / (f1 - f0))


def compute_msd_fsqt(r_tilde, times_rel, q_vals, max_lag, origin_stride, mode):
    arr = r_tilde.astype(np.float32, copy=False)
    n_frames = arr.shape[0]
    max_lag = min(int(max_lag), n_frames - 1)
    origin_stride = max(1, int(origin_stride))
    origins_all = np.arange(0, n_frames, origin_stride, dtype=np.int32)
    q_vals = np.asarray(q_vals, dtype=np.float64)

    lag_times = np.empty(max_lag + 1, dtype=np.float64)
    msd = np.empty(max_lag + 1, dtype=np.float64)
    fsqt = np.empty((len(q_vals), max_lag + 1), dtype=np.float64)
    n_origins = np.empty(max_lag + 1, dtype=np.int32)

    if mode == "xy":
        from scipy.special import j0
    component = {"x": 0, "y": 1, "z": 2}.get(mode)

    report_every = max(1, max_lag // 10)
    for lag in range(max_lag + 1):
        origins = origins_all[origins_all + lag < n_frames]
        n_origins[lag] = len(origins)
        if len(origins) == 0:
            lag_times[lag] = np.nan
            msd[lag] = np.nan
            fsqt[:, lag] = np.nan
            continue

        dt = times_rel[origins + lag] - times_rel[origins]
        lag_times[lag] = float(np.median(dt))
        dr = arr[origins + lag].astype(np.float64) - arr[origins].astype(np.float64)
        r2 = np.sum(dr * dr, axis=2)
        msd[lag] = float(np.mean(r2))

        if mode == "isotropic":
            metric = np.sqrt(r2)
            for qi, q in enumerate(q_vals):
                qr = q * metric
                fsqt[qi, lag] = float(np.sinc(qr / np.pi).mean())
        elif component is not None:
            disp = dr[:, :, component]
            for qi, q in enumerate(q_vals):
                fsqt[qi, lag] = float(np.cos(q * disp).mean())
        elif mode == "xy":
            metric = np.sqrt(dr[:, :, 0] ** 2 + dr[:, :, 1] ** 2)
            for qi, q in enumerate(q_vals):
                fsqt[qi, lag] = float(j0(q * metric).mean())
        else:
            raise ValueError("Unsupported fsqt mode: {}".format(mode))

        if lag % report_every == 0:
            print("  lag {}/{}: t={:.3f}, origins={}".format(
                lag, max_lag, lag_times[lag], len(origins)
            ))

    return lag_times, msd, fsqt, n_origins


def compute_chi4(
    r_tilde,
    wrapped,
    images,
    boxes,
    times_rel,
    overlap_a,
    temperature,
    max_lag,
    lag_stride,
    origin_stride,
    gamma_dot,
    dt_frame,
    affine_rule,
    affine_y,
    method,
):
    arr = r_tilde.astype(np.float32, copy=False)
    n_frames, n_particles = arr.shape[:2]
    max_lag = min(int(max_lag), n_frames - 1)
    lag_stride = max(1, int(lag_stride))
    origin_stride = max(1, int(origin_stride))
    lags = np.arange(0, max_lag + 1, lag_stride, dtype=np.int32)
    if lags[-1] != max_lag:
        lags = np.append(lags, np.int32(max_lag))

    box0 = boxes[0]
    box_lengths = np.asarray([box0["Lx"], box0["Ly"], box0["Lz"]], dtype=np.float64)
    volume = float(np.prod(box_lengths))
    beta = 1.0 / float(temperature)
    method = str(method).lower()
    valid_methods = {
        "auto",
        "co-sheared",
        "cartesian-advected-ref",
        "folded-advected-ref",
        "advected-ref",
        "self-rtilde",
    }
    if method not in valid_methods:
        raise ValueError("chi4 method must be one of {}".format(sorted(valid_methods)))
    has_triclinic_shear = any(
        bool(b.get("triclinic", False))
        or abs(float(b.get("xy", 0.0))) > 1e-12
        or abs(float(b.get("xz", 0.0))) > 1e-12
        or abs(float(b.get("yz", 0.0))) > 1e-12
        for b in boxes
    )
    if method == "auto":
        resolved_method = "co-sheared" if has_triclinic_shear else "cartesian-advected-ref"
    elif method == "advected-ref":
        print("  NOTE: chi4-method=advected-ref is deprecated; resolving with auto")
        resolved_method = "co-sheared" if has_triclinic_shear else "cartesian-advected-ref"
    else:
        resolved_method = method

    print("  precomputing folded positions for chi4")
    folded = np.empty_like(wrapped, dtype=np.float32)
    report_every_frames = max(1, n_frames // 10)
    for fi in range(n_frames):
        folded[fi] = fold_positions_to_box(wrapped[fi], boxes[fi])
        if fi and fi % report_every_frames == 0:
            print("    folded {}/{} frames".format(fi, n_frames))

    affine_x = None
    if resolved_method in {"cartesian-advected-ref", "folded-advected-ref"}:
        print("  precomputing affine x shifts for advected-reference chi4")
        affine_x = cumulative_affine_x(
            wrapped,
            images,
            boxes,
            gamma_dot,
            dt_frame,
            affine_rule,
            affine_y,
        )

    origins_all = np.arange(0, n_frames, origin_stride, dtype=np.int32)
    lag_times = np.empty(len(lags), dtype=np.float64)
    chi4 = np.empty(len(lags), dtype=np.float64)
    q_mean = np.empty(len(lags), dtype=np.float64)
    q_var = np.empty(len(lags), dtype=np.float64)
    n_origins = np.empty(len(lags), dtype=np.int32)
    q_values = np.full((len(lags), len(origins_all)), np.nan, dtype=np.float64)

    ref_tree_cache = {}
    report_every_lag = max(1, len(lags) // 10)
    for li, lag in enumerate(lags):
        origins = origins_all[origins_all + lag < n_frames]
        n_origins[li] = len(origins)
        if len(origins) == 0:
            lag_times[li] = np.nan
            chi4[li] = np.nan
            q_mean[li] = np.nan
            q_var[li] = np.nan
            continue

        lag_times[li] = float(np.median(times_rel[origins + lag] - times_rel[origins]))
        q_this = np.empty(len(origins), dtype=np.float64)
        for oi, origin in enumerate(origins):
            origin = int(origin)
            end = origin + int(lag)
            if resolved_method == "co-sheared":
                # LAMMPS triclinic shear dumps already encode the affine box
                # deformation in the cell tilt. fold_positions_to_box maps
                # both frames into the co-sheared orthogonal box; adding an
                # affine reference shift here would double count homogeneous
                # shear.
                ref_tree = ref_tree_cache.get(origin)
                if ref_tree is None:
                    ref = periodic_points(folded[origin], box_lengths)
                    ref_tree = cKDTree(ref, boxsize=box_lengths)
                    ref_tree_cache[origin] = ref_tree
                cur = periodic_points(folded[end], box_lengths)
                cur_tree = cKDTree(cur, boxsize=box_lengths)
                q_this[oi] = float(ref_tree.count_neighbors(cur_tree, float(overlap_a)))
            elif resolved_method == "cartesian-advected-ref":
                # Orthogonal-box representation with explicit affine particle
                # motion. This follows the advected-reference form directly.
                ref = wrapped[origin].astype(np.float64, copy=True)
                ref[:, 0] += (
                    affine_x[end].astype(np.float64)
                    - affine_x[origin].astype(np.float64)
                )
                ref = periodic_points(ref, box_lengths)
                ref_tree = cKDTree(ref, boxsize=box_lengths)
                cur = periodic_points(wrapped[end], box_lengths)
                cur_tree = cKDTree(cur, boxsize=box_lengths)
                q_this[oi] = float(ref_tree.count_neighbors(cur_tree, float(overlap_a)))
            elif resolved_method == "folded-advected-ref":
                # Historical diagnostic only: this double counts affine shear
                # for LAMMPS triclinic dumps and must not be used as the final
                # chi4 method for the high-resolution shear trajectory.
                ref = folded[origin].astype(np.float64, copy=True)
                ref[:, 0] += (
                    affine_x[end].astype(np.float64)
                    - affine_x[origin].astype(np.float64)
                )
                ref = periodic_points(ref, box_lengths)
                ref_tree = cKDTree(ref, boxsize=box_lengths)
                cur = periodic_points(folded[end], box_lengths)
                cur_tree = cKDTree(cur, boxsize=box_lengths)
                q_this[oi] = float(ref_tree.count_neighbors(cur_tree, float(overlap_a)))
            else:
                ref_tree = ref_tree_cache.get(origin)
                if ref_tree is None:
                    ref = periodic_points(folded[origin], box_lengths)
                    ref_tree = cKDTree(ref, boxsize=box_lengths)
                    ref_tree_cache[origin] = ref_tree

                dr = arr[end].astype(np.float64) - arr[origin].astype(np.float64)
                cur = folded[origin].astype(np.float64) + dr
                cur = periodic_points(cur, box_lengths)
                cur_tree = cKDTree(cur, boxsize=box_lengths)
                q_this[oi] = float(ref_tree.count_neighbors(cur_tree, float(overlap_a)))

        q_values[li, : len(q_this)] = q_this
        q_mean[li] = float(np.mean(q_this))
        q_var[li] = float(np.var(q_this))
        chi4[li] = float(beta * volume * q_var[li] / float(n_particles * n_particles))
        if li % report_every_lag == 0:
            print("  chi4 lag {}/{}: t={:.3f}, origins={}, chi4={:.6g}, Q/N={:.6g}".format(
                li + 1,
                len(lags),
                lag_times[li],
                len(origins),
                chi4[li],
                q_mean[li] / n_particles,
            ))

    peak_i = int(np.nanargmax(chi4)) if np.any(np.isfinite(chi4)) else 0
    summary = {
        "overlap_a": float(overlap_a),
        "temperature": float(temperature),
        "beta": float(beta),
        "volume": float(volume),
        "method": method,
        "resolved_method": resolved_method,
        "has_triclinic_shear": bool(has_triclinic_shear),
        "overlap_interpretation": (
            "co-sheared periodic overlap: LAMMPS triclinic positions are "
            "folded into the co-sheared orthogonal cell before counting "
            "overlap pairs; for Lees-Edwards triclinic dumps this is the "
            "wrapped-coordinate reference-advection interpretation"
            if resolved_method == "co-sheared"
            else "explicit advected-reference overlap"
            if resolved_method in {"cartesian-advected-ref", "folded-advected-ref"}
            else "diagnostic overlap based on nonaffine r_tilde"
        ),
        "affine_y_used_in_chi4": bool(resolved_method in {"cartesian-advected-ref", "folded-advected-ref"}),
        "affine_rule": str(affine_rule),
        "affine_y": str(affine_y),
        "n_particles_chi4": int(n_particles),
        "max_lag_frame": int(max_lag),
        "lag_stride": int(lag_stride),
        "origin_stride": int(origin_stride),
        "t_chi": float(lag_times[peak_i]),
        "lag_frame_chi": int(lags[peak_i]),
        "chi4_peak": float(chi4[peak_i]),
        "q_mean_peak": float(q_mean[peak_i]),
        "q_mean_over_n_peak": float(q_mean[peak_i] / n_particles),
        "n_origins_peak": int(n_origins[peak_i]),
        "peak_at_lag_boundary": bool(peak_i == 0 or peak_i == len(lags) - 1),
    }
    return {
        "lags": lags,
        "lag_times": lag_times,
        "chi4": chi4,
        "q_mean": q_mean,
        "q_var": q_var,
        "n_origins": n_origins,
        "q_values": q_values,
        "summary": summary,
    }


def plot_chi4(out_dir, chi4_data, gamma_dot):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = chi4_data["lag_times"]
    chi4 = chi4_data["chi4"]
    q_over_n = chi4_data["q_mean"] / chi4_data["summary"]["n_particles_chi4"]
    peak_t = chi4_data["summary"]["t_chi"]
    sr_str = str(gamma_dot).replace(".", "p")

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    mask = np.isfinite(t) & np.isfinite(chi4)
    ax.plot(t[mask], chi4[mask], "o-", ms=3.2, lw=1.2, color="black")
    ax.axvline(peak_t, color="crimson", ls="--", lw=1.1, label=r"$t_\chi={:.3g}$".format(peak_t))
    ax.set_xlabel(r"$t/\tau_0$")
    ax.set_ylabel(r"$\chi_4(t)$")
    ax.set_title(r"$\chi_4(t)$, $\dot\gamma={}$".format(gamma_dot))
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chi4_{}.png".format(sr_str), dpi=240)
    fig.savefig(out_dir / "chi4_{}.pdf".format(sr_str))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.plot(t[mask], q_over_n[mask], "o-", ms=3.2, lw=1.2, color="steelblue")
    ax.axvline(peak_t, color="crimson", ls="--", lw=1.1)
    ax.set_xlabel(r"$t/\tau_0$")
    ax.set_ylabel(r"$\langle Q(t)\rangle/N$")
    ax.set_title(r"Overlap decay, $\dot\gamma={}$".format(gamma_dot))
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_q_{}.png".format(sr_str), dpi=240)
    fig.savefig(out_dir / "overlap_q_{}.pdf".format(sr_str))
    plt.close(fig)


def plot_msd_fsqt(out_dir, lag_times, msd, fsqt, q_vals, gamma_dot, lc2, mode):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_vals = np.asarray(q_vals, dtype=np.float64)
    sr_str = str(gamma_dot).replace(".", "p")

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    mask = (lag_times > 0.0) & np.isfinite(msd) & (msd > 0.0)
    ax.loglog(lag_times[mask], msd[mask], color="black", lw=1.5)
    ax.axhline(float(lc2), color="crimson", ls="--", lw=1.1, label=r"$l_c^2$")
    ax.set_xlabel(r"$t/\tau_0$")
    ax.set_ylabel(r"$\langle \Delta \tilde r^2(t)\rangle$")
    ax.set_title(r"Time-origin averaged MSD, $\dot\gamma={}$".format(gamma_dot))
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "msd_time_averaged_{}.png".format(sr_str), dpi=240)
    fig.savefig(out_dir / "msd_time_averaged_{}.pdf".format(sr_str))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    tau = []
    for qi, q in enumerate(q_vals):
        fs = fsqt[qi]
        ax.semilogx(lag_times[1:], fs[1:], lw=1.7, label=r"$q={:.2f}$".format(q))
        tau_i = extract_tau_alpha(lag_times, fs)
        tau.append(tau_i)
        if np.isfinite(tau_i):
            ax.axvline(tau_i, color="0.45", ls=":", lw=0.8)
    ax.axhline(1.0 / math.e, color="crimson", ls="--", lw=1.1, label=r"$e^{-1}$")
    ax.set_xlabel(r"$t/\tau_0$")
    ax.set_ylabel(r"$F_s(q,t)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"$F_s(q,t)$, $\dot\gamma={}$, mode={}".format(gamma_dot, mode))
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fsqt_{}_{}.png".format(mode, sr_str), dpi=240)
    fig.savefig(out_dir / "fsqt_{}_{}.pdf".format(mode, sr_str))
    plt.close(fig)

    return [float(x) if np.isfinite(x) else float("nan") for x in tau]


def main():
    parser = argparse.ArgumentParser(description="Streaming MD3D cage-jump trial")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--gamma-dot", type=float, default=0.015)
    parser.add_argument("--dt-frame", type=float, default=0.1)
    parser.add_argument("--affine-rule", choices=["left", "trapezoid"], default="trapezoid")
    parser.add_argument(
        "--affine-y",
        choices=["centered", "wrapped", "unwrapped"],
        default="wrapped",
        help=(
            "y coordinate used in gamma_dot * integral y dt for r_tilde; "
            "wrapped is the Lees-Edwards/co-sheared periodic convention for "
            "this triclinic LAMMPS dump, unwrapped/centered are diagnostics"
        ),
    )
    parser.add_argument("--type", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=6667)
    parser.add_argument("--n-frames", type=int, default=5000)
    parser.add_argument("--max-particles", type=int, default=6000, help="0 means all selected type particles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lc2", type=float, default=0.057)
    parser.add_argument("--min-segment", type=int, default=4)
    parser.add_argument("--jvec-window", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-window", action="store_true", help="Save sampled r_tilde/wrapped arrays for debugging")
    parser.add_argument("--skip-cage", action="store_true", help="Only compute MSD/Fsqt diagnostics; do not run cage-jump detection")
    parser.add_argument(
        "--min-cage-frames",
        type=int,
        default=200,
        help="Minimum trajectory length for formal Candelier cage-jump detection",
    )
    parser.add_argument(
        "--allow-short-cage-window",
        action="store_true",
        help="Diagnostic override only; short-window Candelier detection is not valid for formal Zeng clusters",
    )
    parser.add_argument("--fsqt", action="store_true", help="Compute time-origin averaged MSD and F_s(q,t)")
    parser.add_argument("--q-vals", nargs="+", type=float, default=[7.25])
    parser.add_argument("--fsqt-mode", choices=["isotropic", "x", "y", "z", "xy"], default="isotropic")
    parser.add_argument("--fsqt-max-lag", type=int, default=800)
    parser.add_argument("--origin-stride", type=int, default=10)
    parser.add_argument("--chi4", action="store_true", help="Compute dynamic susceptibility chi4(t)")
    parser.add_argument("--overlap-a", type=float, default=0.25, help="Overlap cutoff; Zeng Appendix B uses 0.25 for 3D MD")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--chi4-max-lag", type=int, default=800)
    parser.add_argument("--chi4-lag-stride", type=int, default=5)
    parser.add_argument("--chi4-origin-stride", type=int, default=20)
    parser.add_argument(
        "--chi4-method",
        choices=[
            "auto",
            "co-sheared",
            "cartesian-advected-ref",
            "folded-advected-ref",
            "advected-ref",
            "self-rtilde",
        ],
        default="auto",
        help="auto uses co-sheared overlap for LAMMPS triclinic dumps; folded-advected-ref is a diagnostic only",
    )
    args = parser.parse_args()

    dump_path = Path(args.dump)
    if not dump_path.exists():
        raise SystemExit("Dump not found: {}".format(dump_path))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("[1] selecting particles from first frame")
    selection = first_frame_selection(dump_path, args.type, args.max_particles, args.seed)
    selected_ids = selection["selected_ids"]
    print("  natoms={} type{}={} selected={} ({})".format(
        selection["natoms"], args.type, selection["type_count"], len(selected_ids), selection["selection_note"]
    ))

    print("[2] streaming window: start_frame={} n_frames={}".format(args.start_frame, args.n_frames))
    window = stream_window(dump_path, selected_ids, args.start_frame, args.n_frames)
    steps = window["steps"]
    times_abs, dt_lammps, step_stride = estimate_times(steps, args.dt_frame)
    times_rel = times_abs - times_abs[0]
    print("  stored frames={} selected particles={} step_stride={} dt_lammps={:.6g}".format(
        len(steps), len(selected_ids), step_stride, dt_lammps
    ))

    print("[3] computing incremental nonaffine displacement")
    r_tilde = nonaffine_incremental(
        window["wrapped"],
        window["images"],
        window["boxes"],
        args.gamma_dot,
        args.dt_frame,
        args.affine_rule,
        args.affine_y,
    )
    diagnostics = diagnose_nonaffine(
        r_tilde,
        window["wrapped"],
        window["images"],
        window["boxes"],
    )
    print("  nonaffine diagnostics: {}".format(json.dumps(diagnostics, sort_keys=True)))

    fsqt_summary = None
    chi4_summary = None
    if args.fsqt:
        print("[4] computing MSD and F_s(q,t)")
        lag_times, msd_ta, fsqt, n_origins = compute_msd_fsqt(
            r_tilde,
            times_rel,
            args.q_vals,
            args.fsqt_max_lag,
            args.origin_stride,
            args.fsqt_mode,
        )
        tau_alpha = plot_msd_fsqt(
            out_dir,
            lag_times,
            msd_ta,
            fsqt,
            args.q_vals,
            args.gamma_dot,
            args.lc2,
            args.fsqt_mode,
        )
        np.savez_compressed(
            out_dir / "msd_fsqt_sampled.npz",
            lag_times=lag_times,
            msd=msd_ta,
            fsqt=fsqt,
            q_vals=np.asarray(args.q_vals, dtype=np.float64),
            n_origins=n_origins,
            gamma_dot=float(args.gamma_dot),
            mode=args.fsqt_mode,
            start_frame=int(args.start_frame),
            dt_frame=float(args.dt_frame),
            selected_ids=selected_ids,
        )
        fsqt_summary = {
            "q_vals": [float(q) for q in args.q_vals],
            "mode": args.fsqt_mode,
            "fsqt_max_lag": int(min(args.fsqt_max_lag, len(steps) - 1)),
            "origin_stride": int(args.origin_stride),
            "tau_alpha": tau_alpha,
            "msd_last": float(msd_ta[-1]),
            "lag_time_last": float(lag_times[-1]),
        }

    if args.chi4:
        print("[4b] computing chi4(t)")
        chi4_data = compute_chi4(
            r_tilde,
            window["wrapped"],
            window["images"],
            window["boxes"],
            times_rel,
            args.overlap_a,
            args.temperature,
            args.chi4_max_lag,
            args.chi4_lag_stride,
            args.chi4_origin_stride,
            args.gamma_dot,
            args.dt_frame,
            args.affine_rule,
            args.affine_y,
            args.chi4_method,
        )
        plot_chi4(out_dir, chi4_data, args.gamma_dot)
        np.savez_compressed(
            out_dir / "chi4_sampled.npz",
            lags=chi4_data["lags"],
            lag_times=chi4_data["lag_times"],
            chi4=chi4_data["chi4"],
            q_mean=chi4_data["q_mean"],
            q_var=chi4_data["q_var"],
            n_origins=chi4_data["n_origins"],
            q_values=chi4_data["q_values"],
            selected_ids=selected_ids,
            gamma_dot=float(args.gamma_dot),
            dt_frame=float(args.dt_frame),
            overlap_a=float(args.overlap_a),
            temperature=float(args.temperature),
            chi4_method=args.chi4_method,
            chi4_resolved_method=chi4_data["summary"].get("resolved_method"),
            affine_rule=args.affine_rule,
            affine_y=args.affine_y,
        )
        chi4_summary = chi4_data["summary"]

    if args.skip_cage:
        print("[5] skipping Candelier cage jumps")
        jumps = empty_jumps()
    else:
        if len(steps) < int(args.min_cage_frames) and not bool(args.allow_short_cage_window):
            raise SystemExit(
                "Refusing formal Candelier cage-jump detection on only {} frames. "
                "Detect jumps on a long trajectory first, then select t_chi windows. "
                "Use --allow-short-cage-window only for explicitly labeled diagnostics.".format(len(steps))
            )
        print("[5] detecting Candelier cage jumps")
        jumps = detect_cage_jumps(
            r_tilde,
            selected_ids,
            steps,
            times_abs,
            window["wrapped"],
            window["boxes"],
            args.lc2,
            args.min_segment,
            args.jvec_window,
        )

    cage_algorithm = {
        "name": "candelier_recursive_trajectory_segmentation",
        "paper_basis": "Zeng Appendix B Eq. (B1); Candelier et al. PRL 105 and EPAPS",
        "trajectory": "Yamamoto-Onuki nonaffine r_tilde(t)",
        "p_function": "p(tc)=zeta(tc)*sqrt(<d1^2>_S2*<d2^2>_S1)",
        "zeta": "sqrt((tc/T)*(1-tc/T))",
        "threshold_rule": "recursively split at argmax p(tc) until pmax < lc2",
        "lc2": float(args.lc2),
        "min_segment_frames": int(args.min_segment),
        "formal_short_window_detection_allowed": bool(args.allow_short_cage_window),
        "formal_source": bool((not args.skip_cage) and len(steps) >= int(args.min_cage_frames)),
    }

    print("[6] writing outputs")
    np.savez_compressed(
        out_dir / "trial_cage_jumps.npz",
        selected_ids=selected_ids,
        steps=steps,
        times_abs=times_abs,
        times_rel=times_rel,
        particle_id=jumps["particle_id"],
        local_index=jumps["local_index"],
        jump_frames=jumps["jump_frames"],
        jump_steps=jumps["jump_steps"],
        jump_times=jumps["jump_times"],
        jump_vectors=jumps["jump_vectors"],
        jump_positions=jumps["jump_positions"],
        jump_positions_raw=jumps["jump_positions_raw"],
        lc2=float(args.lc2),
        gamma_dot=float(args.gamma_dot),
        dt_frame=float(args.dt_frame),
        start_frame=int(args.start_frame),
        algorithm_json=json.dumps(cage_algorithm, sort_keys=True),
    )
    if args.save_window:
        np.savez_compressed(
            out_dir / "trial_window_sample.npz",
            selected_ids=selected_ids,
            steps=steps,
            times_abs=times_abs,
            times_rel=times_rel,
            wrapped=window["wrapped"],
            images=window["images"],
            virial_xy=window["virial_xy"],
            r_tilde=r_tilde,
        )

    plot_stats = make_plots(out_dir, times_rel, r_tilde, jumps, args.lc2, args.gamma_dot)
    vectors = jumps["jump_vectors"]
    if len(vectors):
        lengths = np.linalg.norm(vectors, axis=1)
        p_ext = float(np.mean(vectors[:, 0] * vectors[:, 1] > 0.0))
        mean_length = float(np.mean(lengths))
        max_length = float(np.max(lengths))
    else:
        p_ext = None
        mean_length = None
        max_length = None

    summary = {
        "dump": str(dump_path),
        "gamma_dot": float(args.gamma_dot),
        "dt_frame": float(args.dt_frame),
        "affine_rule": args.affine_rule,
        "affine_y": args.affine_y,
        "dt_lammps_inferred": float(dt_lammps),
        "step_stride": int(step_stride),
        "start_frame": int(args.start_frame),
        "frames": int(len(steps)),
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "start_time": float(times_abs[0]),
        "end_time": float(times_abs[-1]),
        "duration": float(times_rel[-1]),
        "natoms": int(selection["natoms"]),
        "particle_type": int(args.type),
        "type_count": int(selection["type_count"]),
        "selected_particles": int(len(selected_ids)),
        "selection_note": selection["selection_note"],
        "lc2": float(args.lc2),
        "min_segment": int(args.min_segment),
        "jvec_window": int(args.jvec_window),
        "cage_jump_algorithm": cage_algorithm,
        "n_jumps": int(len(jumps["jump_frames"])),
        "particles_with_jumps": int(len(np.unique(jumps["particle_id"])) if len(jumps["particle_id"]) else 0),
        "mean_jump_length": mean_length,
        "max_jump_length": max_length,
        "P_ext": p_ext,
        "diagnostics": diagnostics,
        "fsqt_summary": fsqt_summary,
        "chi4_summary": chi4_summary,
        "plot_stats": plot_stats,
        "elapsed_sec": float(time.time() - t_start),
    }
    (out_dir / "trial_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Done. Output: {}".format(out_dir))


if __name__ == "__main__":
    main()
