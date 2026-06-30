#!/usr/bin/env python3
"""Sample check for cage-relative Candelier detection in MD2D."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zeng_repro.cage_jump import find_cage_jumps_recursive  # noqa: E402


def load_paths(repo_root, run_group, suffix):
    out = Path(repo_root) / "zeng_reproduction" / "data" / "MD" / "2D" / "processed" / "chi4_cage_cluster" / run_group / suffix
    return out, json.loads((out / "trajectory_manifest.json").read_text()), json.loads((out / "summary_md2d_gdot0p001.json").read_text())


def open_arrays(out, meta):
    n_frames = int(meta["n_frames"])
    n_particles = int(meta["n_particles"])
    return {
        "positions": np.memmap(out / "positions_wrapped.float32", dtype=np.float32, mode="r", shape=(n_frames, n_particles, 2)),
        "r_tilde": np.memmap(out / "r_tilde.float32", dtype=np.float32, mode="r", shape=(n_frames, n_particles, 2)),
        "boxes": np.load(out / "boxes.npy"),
        "times": np.load(out / "times.npy"),
    }


def cosheared_positions(positions, box):
    sy = (positions[:, 1] - box[1]) / box[3]
    sx = (positions[:, 0] - box[0] - box[4] * sy) / box[2]
    return np.column_stack(((sx % 1.0) * box[2], (sy % 1.0) * box[3]))


def neighbor_lists(positions0, box_lengths, sample_particles, cutoff):
    tree = cKDTree(positions0, boxsize=box_lengths)
    neigh = tree.query_ball_point(positions0[sample_particles], cutoff)
    out = []
    for particle, ids in zip(sample_particles, neigh):
        ids = np.asarray([idx for idx in ids if idx != particle], dtype=np.int64)
        out.append(ids)
    return out


def cr_segment(r_tilde, frames, particle, neighbors):
    base_i = r_tilde[frames[0], particle].astype(np.float64)
    seg_i = r_tilde[frames, particle].astype(np.float64) - base_i
    if len(neighbors) == 0:
        return seg_i
    base_n = r_tilde[frames[0], neighbors].astype(np.float64)
    seg_n = r_tilde[frames[:, None], neighbors, :].astype(np.float64) - base_n[None, :, :]
    return seg_i - seg_n.mean(axis=1)


def plain_segment(r_tilde, frames, particle):
    return r_tilde[frames, particle].astype(np.float64) - r_tilde[frames[0], particle].astype(np.float64)


def count_window_jumps(traj, lc2, min_segment):
    frames = find_cage_jumps_recursive(traj, lc2, min_segment=min_segment)
    return len(set(frames))


def existing_counts(jump_path, sample_particles, starts, t_chi):
    jumps = np.load(jump_path)
    pids = jumps["particle_id"].astype(np.int64) - 1
    times = jumps["jump_time"].astype(np.float64)
    sample_mask = np.isin(pids, sample_particles)
    pids = pids[sample_mask]
    times = times[sample_mask]
    rows = []
    for start in starts:
        mask = (times >= start) & (times < start + t_chi)
        rows.append((int(np.count_nonzero(mask)), int(len(np.unique(pids[mask])))))
    return rows


def sample_starts(times, summary, t_chi, n_windows):
    starts = [0.0, float(summary["window"]["start"])]
    hi = float(times[-1] - t_chi)
    if n_windows > len(starts):
        starts.extend(np.linspace(0.0, hi, n_windows - len(starts)).tolist())
    starts = np.asarray(sorted(set(round(float(x), 6) for x in starts if 0.0 <= x <= hi)), dtype=np.float64)
    return starts[:n_windows]


def run_validation(out, meta, summary, sample_n, n_windows, cutoff, lc2, min_segment, seed):
    arrays = open_arrays(out, meta)
    rng = np.random.default_rng(seed)
    n_particles = int(meta["n_particles"])
    sample_particles = np.sort(rng.choice(n_particles, size=min(sample_n, n_particles), replace=False))
    t_chi = float(summary["t_chi"])
    dump_dt = float(meta["dump_stride_time"])
    lag_frames = int(round(t_chi / dump_dt))
    starts = sample_starts(arrays["times"], summary, t_chi, n_windows)
    existing = existing_counts(out / "cage_jumps_md2d_gdot0p001.npz", sample_particles, starts, t_chi)

    rows = []
    for wi, start in enumerate(starts):
        start_frame = int(np.searchsorted(arrays["times"], start, side="left"))
        frames = np.arange(start_frame, min(start_frame + lag_frames + 1, len(arrays["times"])), dtype=np.int64)
        box = arrays["boxes"][start_frame]
        box_lengths = box[2:4].astype(np.float64)
        pos0 = cosheared_positions(arrays["positions"][start_frame].astype(np.float64), box)
        neigh = neighbor_lists(pos0, box_lengths, sample_particles, cutoff)

        plain_counts = []
        cr_counts = []
        plain_particles = 0
        cr_particles = 0
        for particle, ids in zip(sample_particles, neigh):
            p_count = count_window_jumps(plain_segment(arrays["r_tilde"], frames, particle), lc2, min_segment)
            c_count = count_window_jumps(cr_segment(arrays["r_tilde"], frames, particle, ids), lc2, min_segment)
            plain_counts.append(p_count)
            cr_counts.append(c_count)
            plain_particles += int(p_count > 0)
            cr_particles += int(c_count > 0)

        plain_counts = np.asarray(plain_counts, dtype=np.int32)
        cr_counts = np.asarray(cr_counts, dtype=np.int32)
        rows.append(
            {
                "start": float(arrays["times"][start_frame]),
                "end": float(arrays["times"][frames[-1]]),
                "n_frames": int(len(frames)),
                "existing_full_events_sample": int(existing[wi][0]),
                "existing_full_particles_sample": int(existing[wi][1]),
                "plain_window_events_sample": int(plain_counts.sum()),
                "plain_window_particles_sample": int(plain_particles),
                "cr_window_events_sample": int(cr_counts.sum()),
                "cr_window_particles_sample": int(cr_particles),
                "plain_mean_per_particle": float(plain_counts.mean()),
                "cr_mean_per_particle": float(cr_counts.mean()),
                "mean_neighbors": float(np.mean([len(ids) for ids in neigh])),
            }
        )
    return {
        "method": "window_origin_cage_relative_r_tilde_sample",
        "sample_particles": sample_particles.tolist(),
        "sample_n": int(len(sample_particles)),
        "t_chi": t_chi,
        "dump_dt": dump_dt,
        "lc2": float(lc2),
        "neighbor_cutoff": float(cutoff),
        "min_segment": int(min_segment),
        "windows": rows,
    }


def write_text(path, report):
    lines = [
        "MD2D cage-relative Candelier sample validation",
        "",
        f"sample_n: {report['sample_n']}",
        f"t_chi: {report['t_chi']}",
        f"lc2: {report['lc2']}",
        f"neighbor_cutoff: {report['neighbor_cutoff']}",
        "",
        "Columns: existing_full = previous full-trajectory detector restricted to sampled particles;",
        "plain_window/cr_window = Candelier rerun only inside this t_chi window.",
        "",
    ]
    for row in report["windows"]:
        lines.append(
            "start={start:9.1f} existing_full={existing_full_events_sample:4d}/{existing_full_particles_sample:3d} "
            "plain={plain_window_events_sample:4d}/{plain_window_particles_sample:3d} "
            "CR={cr_window_events_sample:4d}/{cr_window_particles_sample:3d} "
            "mean_neighbors={mean_neighbors:.2f}".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["windows"]
    x = np.arange(len(rows))
    labels = [f"{r['start']:.0f}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=170)
    width = 0.26
    axes[0].bar(x - width, [r["existing_full_events_sample"] for r in rows], width, label="full existing", color="0.25")
    axes[0].bar(x, [r["plain_window_events_sample"] for r in rows], width, label="plain window", color="steelblue")
    axes[0].bar(x + width, [r["cr_window_events_sample"] for r in rows], width, label="CR window", color="crimson")
    axes[0].set_ylabel("events in sampled particles")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - width, [r["existing_full_particles_sample"] for r in rows], width, label="full existing", color="0.25")
    axes[1].bar(x, [r["plain_window_particles_sample"] for r in rows], width, label="plain window", color="steelblue")
    axes[1].bar(x + width, [r["cr_window_particles_sample"] for r in rows], width, label="CR window", color="crimson")
    axes[1].set_ylabel("jumping sampled particles")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    for ax in axes:
        ax.set_xlabel(r"window start $t$")
        ax.grid(True, ls=":", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-group", default="md2d_prod_slim_20260629_1412")
    parser.add_argument("--suffix", default="gdot0p001")
    parser.add_argument("--sample-n", type=int, default=300)
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--neighbor-cutoff", type=float, default=1.6)
    parser.add_argument("--lc2", type=float, default=0.048)
    parser.add_argument("--min-segment", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    out, meta, summary = load_paths(args.repo_root, args.run_group, args.suffix)
    report = run_validation(out, meta, summary, args.sample_n, args.n_windows, args.neighbor_cutoff, args.lc2, args.min_segment, args.seed)
    json_path = out / "cr_cage_sample_validation.json"
    txt_path = out / "cr_cage_sample_validation.txt"
    png_path = out / "cr_cage_sample_validation.png"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_text(txt_path, report)
    plot(png_path, report)
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
