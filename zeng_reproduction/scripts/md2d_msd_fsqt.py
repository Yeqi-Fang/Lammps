#!/usr/bin/env python3
"""Compute sampled 2D shear-corrected MSD and F_s(q,t) for Zeng MD runs.

The trajectory is converted to a co-sheared continuous coordinate by unwrapping
fractional coordinates in the instantaneous triclinic cell. This keeps boundary
crossings continuous without using ordinary unwrapped y as a shear-flow
coordinate.
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import j0


RUNS = [
    {"label": "gdot0p0005", "gdot": 0.0005, "dump_dt": 5.0},
    {"label": "gdot0p001", "gdot": 0.001, "dump_dt": 0.5},
    {"label": "gdot0p005", "gdot": 0.005, "dump_dt": 1.0},
    {"label": "gdot0p01", "gdot": 0.01, "dump_dt": 1.0},
]


def log(message):
    print("[{}] {}".format(time.strftime("%F %T"), message), flush=True)


def open_dump_text(path):
    """Open a gzip LAMMPS dump as text, using system gzip for speed."""
    proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE)
    stream = io.TextIOWrapper(proc.stdout, encoding="ascii", errors="replace", newline="")
    return proc, stream


def close_dump_text(proc, stream, allow_sigpipe=False):
    stream.close()
    rc = proc.wait()
    if allow_sigpipe and rc in (-13, 141):
        return
    if rc != 0:
        raise RuntimeError("gzip exited with status {} for dump stream".format(rc))


def parse_box(bounds):
    xlo_b, xhi_b, xy = bounds[0]
    ylo_b, yhi_b, _xz = bounds[1]
    xlo = xlo_b - min(0.0, xy)
    xhi = xhi_b - max(0.0, xy)
    ylo = ylo_b
    yhi = yhi_b
    return {
        "xlo": float(xlo),
        "xhi": float(xhi),
        "ylo": float(ylo),
        "yhi": float(yhi),
        "lx": float(xhi - xlo),
        "ly": float(yhi - ylo),
        "xy": float(xy),
    }


def read_frame(stream, parse_atoms):
    line = stream.readline()
    if not line:
        return None
    if not line.startswith("ITEM: TIMESTEP"):
        raise RuntimeError("Expected ITEM: TIMESTEP, got {!r}".format(line[:80]))
    step = int(stream.readline().strip())

    line = stream.readline()
    if not line.startswith("ITEM: NUMBER OF ATOMS"):
        raise RuntimeError("Expected ITEM: NUMBER OF ATOMS at step {}".format(step))
    n_atoms = int(stream.readline().strip())

    line = stream.readline()
    if not line.startswith("ITEM: BOX BOUNDS"):
        raise RuntimeError("Expected ITEM: BOX BOUNDS at step {}".format(step))
    bounds = []
    for _ in range(3):
        parts = [float(x) for x in stream.readline().split()]
        while len(parts) < 3:
            parts.append(0.0)
        bounds.append(parts[:3])
    box = parse_box(bounds)

    line = stream.readline()
    if not line.startswith("ITEM: ATOMS"):
        raise RuntimeError("Expected ITEM: ATOMS at step {}".format(step))
    columns = line.strip().split()[2:]

    if not parse_atoms:
        for _ in range(n_atoms):
            stream.readline()
        return step, n_atoms, box, columns, None

    text = "".join(stream.readline() for _ in range(n_atoms))
    data = np.fromstring(text, sep=" ", dtype=np.float64)
    ncols = len(columns)
    if data.size != n_atoms * ncols:
        raise RuntimeError(
            "Parsed {} values at step {}, expected {} values".format(
                data.size, step, n_atoms * ncols
            )
        )
    data = data.reshape((n_atoms, ncols))
    col_index = {name: i for i, name in enumerate(columns)}
    if "id" not in col_index or "x" not in col_index or "y" not in col_index:
        raise RuntimeError("Dump columns missing id/x/y: {}".format(columns))

    ids = data[:, col_index["id"]].astype(np.int64)
    if not (ids[0] == 1 and ids[-1] == n_atoms and np.all(np.diff(ids) == 1)):
        order = np.argsort(ids)
        data = data[order]
        ids = ids[order]
        if not (ids[0] == 1 and ids[-1] == n_atoms and np.all(np.diff(ids) == 1)):
            raise RuntimeError("Particle ids are not a complete sorted 1..N set at step {}".format(step))

    xy = data[:, [col_index["x"], col_index["y"]]]
    return step, n_atoms, box, columns, xy


def cosheared_fractional(xy, box):
    yrel = xy[:, 1] - box["ylo"]
    sy = yrel / box["ly"]
    sx = (xy[:, 0] - box["xlo"] - box["xy"] * sy) / box["lx"]
    sx = sx - np.floor(sx)
    sy = sy - np.floor(sy)
    return np.column_stack((sx, sy))


def expected_selected_frames(gdot, dump_dt, sample_dt, dt, prod_strain):
    n_prod = int(math.ceil(prod_strain / (gdot * dt)))
    dump_every = int(round(dump_dt / dt))
    raw_frames = n_prod // dump_every + 1
    stride = int(round(sample_dt / dump_dt))
    if stride < 1 or abs(stride * dump_dt - sample_dt) > 1.0e-7:
        raise ValueError(
            "sample_dt={} is not an integer multiple of dump_dt={}".format(sample_dt, dump_dt)
        )
    selected = (raw_frames - 1) // stride + 1
    return raw_frames, stride, selected


def find_dump(raw_base, run_group, label):
    run_dir = raw_base / run_group / label
    files = sorted(run_dir.glob("*.lammpstrj.gz"))
    if len(files) != 1:
        raise RuntimeError("Expected one dump under {}, found {}".format(run_dir, len(files)))
    return files[0]


def fft_sq_accumulate(hist, lx, ly, q_min, q_max, q_bin_width, accum):
    grid_x, grid_y = hist.shape
    rho = np.fft.fft2(hist)
    sq = (np.abs(rho) ** 2) / float(hist.sum())
    qx = 2.0 * np.pi * np.fft.fftfreq(grid_x, d=lx / grid_x)
    qy = 2.0 * np.pi * np.fft.fftfreq(grid_y, d=ly / grid_y)
    qxx, qyy = np.meshgrid(qx, qy, indexing="ij")
    q = np.sqrt(qxx * qxx + qyy * qyy)
    mask = (q >= q_min) & (q <= q_max)
    qvals = q[mask].ravel()
    svals = sq[mask].ravel()
    nbins = int(math.ceil((q_max - q_min) / q_bin_width))
    bins = np.floor((qvals - q_min) / q_bin_width).astype(np.int64)
    valid = (bins >= 0) & (bins < nbins)
    if accum["sum"] is None:
        accum["sum"] = np.zeros(nbins, dtype=np.float64)
        accum["count"] = np.zeros(nbins, dtype=np.float64)
    accum["sum"] += np.bincount(bins[valid], weights=svals[valid], minlength=nbins)
    accum["count"] += np.bincount(bins[valid], minlength=nbins)


def estimate_qstar(raw_base, run_group, out_dir, sq_frames, sq_grid, q_min, q_max, q_bin_width):
    log("estimating q* from first {} frames per run".format(sq_frames))
    accum = {"sum": None, "count": None}
    per_run = {}
    for run in RUNS:
        dump = find_dump(raw_base, run_group, run["label"])
        proc, stream = open_dump_text(dump)
        frames = 0
        local = {"sum": None, "count": None}
        try:
            while frames < sq_frames:
                frame = read_frame(stream, parse_atoms=True)
                if frame is None:
                    break
                _step, _n_atoms, box, _columns, xy = frame
                frac = cosheared_fractional(xy, box)
                x = frac[:, 0] * box["lx"]
                y = frac[:, 1] * box["ly"]
                hist, _xe, _ye = np.histogram2d(
                    x,
                    y,
                    bins=sq_grid,
                    range=[[0.0, box["lx"]], [0.0, box["ly"]]],
                )
                fft_sq_accumulate(hist, box["lx"], box["ly"], q_min, q_max, q_bin_width, accum)
                fft_sq_accumulate(hist, box["lx"], box["ly"], q_min, q_max, q_bin_width, local)
                frames += 1
        finally:
            close_dump_text(proc, stream, allow_sigpipe=True)
        per_run[run["label"]] = frames
        log("  q* source {}: {} frames".format(run["label"], frames))

    q_centers = q_min + (np.arange(len(accum["sum"])) + 0.5) * q_bin_width
    sq_avg = accum["sum"] / np.maximum(accum["count"], 1.0)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(sq_avg, kernel, mode="same")
    peak_mask = (q_centers >= 4.0) & (q_centers <= 8.5) & np.isfinite(smooth)
    if not np.any(peak_mask):
        raise RuntimeError("No valid q bins for q* search")
    peak_indices = np.where(peak_mask)[0]
    peak_i = peak_indices[int(np.argmax(smooth[peak_indices]))]
    qstar = float(q_centers[peak_i])

    sq_csv = out_dir / "structure_factor_qstar.csv"
    with sq_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["q", "S_q", "S_q_smooth"])
        for q, s, sm in zip(q_centers, sq_avg, smooth):
            writer.writerow([q, s, sm])

    return qstar, {
        "qstar": qstar,
        "source": "first production frames from all four shear runs; no separate equilibrium dump was available",
        "sq_frames_per_run_requested": int(sq_frames),
        "sq_frames_per_run_used": per_run,
        "sq_grid": int(sq_grid),
        "q_min": float(q_min),
        "q_max": float(q_max),
        "q_bin_width": float(q_bin_width),
        "qstar_search_window": [4.0, 8.5],
        "structure_factor_csv": str(sq_csv),
    }


def build_sampled_trajectory(dump, run, args, run_out):
    raw_frames, frame_stride, selected_expected = expected_selected_frames(
        run["gdot"], run["dump_dt"], args.sample_dt, args.dt, args.prod_strain
    )
    n_atoms = int(args.n_particles)
    mmap_path = run_out / "rtilda_sampled.float32"
    traj = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(selected_expected, n_atoms, 2))
    times = np.empty(selected_expected, dtype=np.float64)
    steps = np.empty(selected_expected, dtype=np.int64)
    max_abs_ds = 0.0
    first_columns = None
    first_box = None
    last_box = None

    proc, stream = open_dump_text(dump)
    raw_i = 0
    selected_i = 0
    prev_frac = None
    frac_unwrapped = None
    try:
        while True:
            parse = (raw_i % frame_stride == 0)
            frame = read_frame(stream, parse_atoms=parse)
            if frame is None:
                break
            step, n_frame_atoms, box, columns, xy = frame
            if n_frame_atoms != n_atoms:
                raise RuntimeError("Expected {} atoms, got {} at step {}".format(n_atoms, n_frame_atoms, step))
            if parse:
                if selected_i >= selected_expected:
                    raise RuntimeError("More selected frames than expected for {}".format(run["label"]))
                frac = cosheared_fractional(xy, box)
                if prev_frac is None:
                    frac_unwrapped = frac.copy()
                else:
                    ds = frac - prev_frac
                    ds -= np.rint(ds)
                    max_abs_ds = max(max_abs_ds, float(np.max(np.abs(ds))))
                    frac_unwrapped += ds
                rt = np.empty((n_atoms, 2), dtype=np.float32)
                rt[:, 0] = (frac_unwrapped[:, 0] * box["lx"]).astype(np.float32)
                rt[:, 1] = (frac_unwrapped[:, 1] * box["ly"]).astype(np.float32)
                traj[selected_i, :, :] = rt
                times[selected_i] = step * args.dt
                steps[selected_i] = step
                prev_frac = frac
                if first_columns is None:
                    first_columns = list(columns)
                    first_box = dict(box)
                last_box = dict(box)
                selected_i += 1
                if selected_i % 1000 == 0:
                    log("  {} sampled {}/{} frames".format(run["label"], selected_i, selected_expected))
            raw_i += 1
    finally:
        close_dump_text(proc, stream)

    if selected_i != selected_expected:
        raise RuntimeError(
            "{} selected {} frames, expected {}".format(run["label"], selected_i, selected_expected)
        )
    traj.flush()
    return traj, mmap_path, times, steps, {
        "raw_frames_expected": int(raw_frames),
        "frame_stride": int(frame_stride),
        "sampled_frames": int(selected_i),
        "sample_dt": float(args.sample_dt),
        "max_abs_fractional_step_between_sampled_frames": float(max_abs_ds),
        "dump_columns": first_columns,
        "first_box": first_box,
        "last_box": last_box,
    }


def make_lags(n_frames, max_log_lags):
    if n_frames < 2:
        return np.array([0], dtype=np.int64)
    end = n_frames - 1
    small = np.arange(0, min(50, end) + 1, dtype=np.int64)
    medium = np.arange(55, min(200, end) + 1, 5, dtype=np.int64)
    log_lags = np.unique(np.round(np.geomspace(1, end, max_log_lags)).astype(np.int64))
    lags = np.unique(np.concatenate((small, medium, log_lags, np.array([end], dtype=np.int64))))
    return lags[(lags >= 0) & (lags <= end)]


def choose_origins(n_available, max_origins):
    if n_available <= 0:
        return np.array([], dtype=np.int64)
    if n_available <= max_origins:
        return np.arange(n_available, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, n_available - 1, max_origins)).astype(np.int64))


def compute_msd_fsqt(traj, times, qstar, args, run_label):
    n_frames, n_atoms, _ndim = traj.shape
    lags = make_lags(n_frames, args.max_log_lags)
    lag_times = times[lags] - times[0]
    msd = np.empty(len(lags), dtype=np.float64)
    msd_x = np.empty(len(lags), dtype=np.float64)
    msd_y = np.empty(len(lags), dtype=np.float64)
    fsqt = np.empty(len(lags), dtype=np.float64)
    n_origins = np.empty(len(lags), dtype=np.int64)

    for li, lag in enumerate(lags):
        if lag == 0:
            msd[li] = 0.0
            msd_x[li] = 0.0
            msd_y[li] = 0.0
            fsqt[li] = 1.0
            n_origins[li] = n_frames
            continue
        origins = choose_origins(n_frames - int(lag), args.max_origins)
        sum_r2 = 0.0
        sum_dx2 = 0.0
        sum_dy2 = 0.0
        sum_fs = 0.0
        count = 0
        for start in range(0, len(origins), args.origin_chunk):
            chunk = origins[start : start + args.origin_chunk]
            disp = np.asarray(traj[chunk + lag], dtype=np.float64) - np.asarray(traj[chunk], dtype=np.float64)
            dx2 = disp[:, :, 0] * disp[:, :, 0]
            dy2 = disp[:, :, 1] * disp[:, :, 1]
            r2 = dx2 + dy2
            sum_dx2 += float(np.sum(dx2))
            sum_dy2 += float(np.sum(dy2))
            sum_r2 += float(np.sum(r2))
            sum_fs += float(np.sum(j0(qstar * np.sqrt(r2))))
            count += int(r2.size)
        msd[li] = sum_r2 / float(count)
        msd_x[li] = sum_dx2 / float(count)
        msd_y[li] = sum_dy2 / float(count)
        fsqt[li] = sum_fs / float(count)
        n_origins[li] = len(origins)
        if li % 25 == 0 or li == len(lags) - 1:
            log(
                "  {} lag {}/{}: t={:.3g}, origins={}, MSD={:.5g}, Fs={:.5g}".format(
                    run_label, li + 1, len(lags), lag_times[li], len(origins), msd[li], fsqt[li]
                )
            )

    return {
        "lags": lags,
        "lag_times": lag_times,
        "msd": msd,
        "msd_x": msd_x,
        "msd_y": msd_y,
        "fsqt": fsqt,
        "n_origins": n_origins,
    }


def save_run_outputs(run_out, run, qstar, times, steps, traj_meta, result):
    npz_path = run_out / "msd_fsqt_sampledt5.npz"
    np.savez_compressed(
        npz_path,
        gdot=np.array(run["gdot"], dtype=np.float64),
        qstar=np.array(qstar, dtype=np.float64),
        times_sampled=times,
        steps_sampled=steps,
        lags=result["lags"],
        lag_times=result["lag_times"],
        msd=result["msd"],
        msd_x=result["msd_x"],
        msd_y=result["msd_y"],
        fsqt=result["fsqt"],
        n_origins=result["n_origins"],
        sample_dt=np.array(traj_meta["sample_dt"], dtype=np.float64),
    )

    csv_path = run_out / "msd_fsqt_sampledt5.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lag_frames", "lag_time", "msd", "msd_x", "msd_y", "fsqt", "n_origins"])
        for row in zip(
            result["lags"],
            result["lag_times"],
            result["msd"],
            result["msd_x"],
            result["msd_y"],
            result["fsqt"],
            result["n_origins"],
        ):
            writer.writerow(row)

    summary = {
        "label": run["label"],
        "gdot": run["gdot"],
        "qstar": float(qstar),
        "npz": str(npz_path),
        "csv": str(csv_path),
        "tau_alpha_fs_eq_exp_minus_1": estimate_tau_alpha(result["lag_times"], result["fsqt"]),
        "msd_last": float(result["msd"][-1]),
        "fsqt_last": float(result["fsqt"][-1]),
        "trajectory": traj_meta,
    }
    (run_out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def estimate_tau_alpha(times, fsqt):
    target = math.exp(-1.0)
    finite = np.isfinite(times) & np.isfinite(fsqt)
    t = times[finite]
    f = fsqt[finite]
    if len(t) < 2:
        return None
    for i in range(1, len(t)):
        if f[i] <= target <= f[i - 1]:
            if f[i] == f[i - 1]:
                return float(t[i])
            frac = (target - f[i - 1]) / (f[i] - f[i - 1])
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def plot_outputs(out_dir, summaries):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log("plot skipped: {}".format(exc))
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=180)
    for summary in summaries:
        data = np.load(summary["npz"])
        t = data["lag_times"]
        msd = data["msd"]
        fs = data["fsqt"]
        label = r"$\dot\gamma={}$".format(summary["gdot"])
        mask_m = (t > 0) & np.isfinite(msd) & (msd > 0)
        axes[0].loglog(t[mask_m], msd[mask_m], label=label, lw=1.3)
        mask_f = (t > 0) & np.isfinite(fs)
        axes[1].semilogx(t[mask_f], fs[mask_f], label=label, lw=1.3)
    axes[0].set_xlabel(r"lag time $t$")
    axes[0].set_ylabel(r"MSD, co-sheared")
    axes[1].set_xlabel(r"lag time $t$")
    axes[1].set_ylabel(r"$F_s(q^*,t)$")
    axes[1].axhline(math.exp(-1.0), color="0.6", ls="--", lw=0.9)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = out_dir / "msd_fsqt_all_rates.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Server repository root")
    parser.add_argument("--run-group", default="md2d_prod_slim_20260629_1412")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--prod-strain", type=float, default=50.0)
    parser.add_argument("--sample-dt", type=float, default=5.0)
    parser.add_argument("--n-particles", type=int, default=20000)
    parser.add_argument("--max-origins", type=int, default=120)
    parser.add_argument("--origin-chunk", type=int, default=4)
    parser.add_argument("--max-log-lags", type=int, default=150)
    parser.add_argument("--sq-frames", type=int, default=64)
    parser.add_argument("--sq-grid", type=int, default=512)
    parser.add_argument("--q-min", type=float, default=2.5)
    parser.add_argument("--q-max", type=float, default=9.5)
    parser.add_argument("--q-bin-width", type=float, default=0.05)
    parser.add_argument("--keep-memmap", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    raw_base = repo / "zeng_reproduction" / "data" / "MD" / "2D" / "raw"
    out_dir = repo / "zeng_reproduction" / "data" / "MD" / "2D" / "processed" / "msd_fsqt" / args.run_group
    out_dir.mkdir(parents=True, exist_ok=True)

    qstar, qstar_meta = estimate_qstar(
        raw_base,
        args.run_group,
        out_dir,
        args.sq_frames,
        args.sq_grid,
        args.q_min,
        args.q_max,
        args.q_bin_width,
    )
    log("selected q* = {:.6f}".format(qstar))
    (out_dir / "qstar_metadata.json").write_text(json.dumps(qstar_meta, indent=2, sort_keys=True))

    summaries = []
    for run in RUNS:
        run_out = out_dir / run["label"]
        run_out.mkdir(parents=True, exist_ok=True)
        dump = find_dump(raw_base, args.run_group, run["label"])
        log("processing {} from {}".format(run["label"], dump))
        traj, mmap_path, times, steps, traj_meta = build_sampled_trajectory(dump, run, args, run_out)
        result = compute_msd_fsqt(traj, times, qstar, args, run["label"])
        summary = save_run_outputs(run_out, run, qstar, times, steps, traj_meta, result)
        summaries.append(summary)
        del traj
        if not args.keep_memmap:
            try:
                os.remove(str(mmap_path))
            except OSError:
                pass
        log("finished {}".format(run["label"]))

    plot_path = plot_outputs(out_dir, summaries)
    all_summary = {
        "run_group": args.run_group,
        "analysis": "sampled co-sheared MSD and isotropic 2D F_s(q,t)",
        "sample_dt": args.sample_dt,
        "qstar_metadata": qstar_meta,
        "max_origins_per_lag": args.max_origins,
        "max_log_lags": args.max_log_lags,
        "summaries": summaries,
        "plot": str(plot_path) if plot_path is not None else None,
        "notes": [
            "Coordinates are fractional-unwrapped in the instantaneous triclinic cell, then mapped to an orthogonal co-sheared frame.",
            "This avoids using ordinary unwrapped y as the shear-flow coordinate.",
            "q* is estimated from production frames because this run group did not save an equilibrium dump trajectory.",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, sort_keys=True))
    log("wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
