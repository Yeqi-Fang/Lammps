#!/usr/bin/env python3
"""Post-run diagnostics for MD2D cage-jump density."""

import argparse
import json
from pathlib import Path

import numpy as np


def threshold_iterative(density, tol=0.10):
    rho = float(np.max(density))
    for _ in range(100):
        below = density[density < rho]
        if len(below) == 0:
            break
        avg = float(np.mean(below))
        if avg > 0.0 and abs(avg - rho) / avg < tol:
            return avg
        rho = avg
    return rho


def periodic_exp_density(points, box_lengths, grid_n=80, coarse_d=1.6):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((grid_n, grid_n), dtype=np.float64)

    Lx, Ly = float(box_lengths[0]), float(box_lengths[1])
    ix = np.floor(points[:, 0] / Lx * grid_n).astype(np.int64) % grid_n
    iy = np.floor(points[:, 1] / Ly * grid_n).astype(np.int64) % grid_n
    counts = np.bincount(ix * grid_n + iy, minlength=grid_n * grid_n).reshape(grid_n, grid_n)

    gx = np.minimum(np.arange(grid_n), grid_n - np.arange(grid_n)) * Lx / grid_n
    gy = np.minimum(np.arange(grid_n), grid_n - np.arange(grid_n)) * Ly / grid_n
    kernel = np.exp(-np.sqrt(gx[:, None] ** 2 + gy[None, :] ** 2) / coarse_d)
    return np.fft.irfft2(np.fft.rfft2(counts) * np.fft.rfft2(kernel), s=(grid_n, grid_n)).real


def window_counts(times, t_chi, stride):
    starts = np.arange(float(times[0]), float(times[-1] - t_chi), float(stride))
    counts = np.searchsorted(times, starts + t_chi, side="left") - np.searchsorted(times, starts, side="left")
    return starts, counts


def sampled_msd(r_tilde_path, n_frames, n_particles, dump_dt, rng, n_particles_sample=800, n_origins=160):
    r_tilde = np.memmap(r_tilde_path, dtype=np.float32, mode="r", shape=(n_frames, n_particles, 2))
    particles = np.sort(rng.choice(n_particles, size=min(n_particles_sample, n_particles), replace=False))
    lag_frames = np.array([1, 2, 5, 10, 20, 40, 80, 160, 320], dtype=np.int64)
    lag_frames = lag_frames[lag_frames < n_frames]
    rows = []
    for lag in lag_frames:
        max_origin = n_frames - int(lag)
        step = max(1, max_origin // n_origins)
        origins = np.arange(0, max_origin, step, dtype=np.int64)[:n_origins]
        vals = []
        for origin in origins:
            dr = r_tilde[origin + lag, particles, :].astype(np.float64) - r_tilde[origin, particles, :].astype(np.float64)
            vals.append(np.einsum("ij,ij->i", dr, dr))
        vals = np.concatenate(vals)
        rows.append(
            {
                "lag_frames": int(lag),
                "lag_time": float(lag * dump_dt),
                "msd": float(np.mean(vals)),
                "r_p50": float(np.sqrt(np.percentile(vals, 50))),
                "r_p95": float(np.sqrt(np.percentile(vals, 95))),
            }
        )
    return rows


def density_window_report(jumps, starts, t_chi, box_lengths, grid_n, coarse_d):
    order = np.argsort(jumps["jump_time"])
    times = jumps["jump_time"][order]
    positions = np.mod(jumps["jump_position"][order].astype(np.float64), box_lengths)
    particles = jumps["particle_id"][order]
    rows = []
    for start in starts:
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, start + t_chi, side="left")
        pts = positions[lo:hi]
        density = periodic_exp_density(pts, box_lengths, grid_n=grid_n, coarse_d=coarse_d)
        rho_th = threshold_iterative(density)
        rows.append(
            {
                "start": float(start),
                "end": float(start + t_chi),
                "events": int(hi - lo),
                "unique_particles": int(len(np.unique(particles[lo:hi]))),
                "rho_th": float(rho_th),
                "density_min": float(np.min(density)),
                "density_median": float(np.median(density)),
                "density_max": float(np.max(density)),
                "grid_fraction_above_rho_th": float(np.mean(density >= rho_th)),
            }
        )
    return rows


def write_text_report(path, report):
    lines = []
    lines.append("MD2D cage-jump density diagnostic")
    lines.append("")
    lines.append(f"total_jumps: {report['total_jumps']}")
    lines.append(f"jumps_per_particle_mean: {report['per_particle']['mean']:.3f}")
    lines.append(f"jumps_per_particle_p50: {report['per_particle']['p50']:.3f}")
    lines.append(f"jumps_per_particle_p95: {report['per_particle']['p95']:.3f}")
    lines.append(f"t_chi: {report['t_chi']}")
    lines.append("")
    lines.append("window_count_percentiles:")
    for key, value in report["window_counts"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("sampled_nonaffine_msd:")
    for row in report["sampled_msd"]:
        lines.append(f"  t={row['lag_time']:8.3f}  msd={row['msd']:10.6g}  r50={row['r_p50']:8.4f}  r95={row['r_p95']:8.4f}")
    lines.append("")
    lines.append("density_windows:")
    for row in report["density_windows"]:
        lines.append(
            "  start={start:.3f} events={events} unique={unique_particles} "
            "rho_th={rho_th:.6g} above={grid_fraction_above_rho_th:.4f} "
            "dens[min/med/max]={density_min:.4g}/{density_median:.4g}/{density_max:.4g}".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_report(path, report, window_starts, window_event_counts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), dpi=170)
    axes[0, 0].plot(window_starts, window_event_counts, lw=0.8, color="black")
    axes[0, 0].set_xlabel(r"window start $t$")
    axes[0, 0].set_ylabel(r"events in $t_\chi$")

    pct = report["per_particle"]
    labels = ["p1", "p5", "p50", "p95", "p99"]
    axes[0, 1].bar(labels, [pct[k] for k in labels], color="0.35")
    axes[0, 1].set_ylabel("jumps per particle")

    msd = report["sampled_msd"]
    axes[1, 0].plot([r["lag_time"] for r in msd], [r["msd"] for r in msd], "o-", color="steelblue")
    axes[1, 0].axhline(report["lc2"], color="crimson", ls="--", lw=1.0)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"lag time $t$")
    axes[1, 0].set_ylabel(r"sampled $\langle \Delta \tilde r^2\rangle$")

    rows = report["density_windows"]
    axes[1, 1].bar([f"{r['start']:.0f}" for r in rows], [r["grid_fraction_above_rho_th"] for r in rows], color="0.35")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel(r"window start $t$")
    axes[1, 1].set_ylabel(r"fraction $\rho \geq \rho_{th}$")
    for ax in axes.ravel():
        ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-group", default="md2d_prod_slim_20260629_1412")
    parser.add_argument("--suffix", default="gdot0p001")
    parser.add_argument("--grid-n", type=int, default=80)
    parser.add_argument("--coarse-d", type=float, default=1.6)
    args = parser.parse_args()

    out = Path(args.repo_root) / "zeng_reproduction" / "data" / "MD" / "2D" / "processed" / "chi4_cage_cluster" / args.run_group / args.suffix
    summary = json.loads((out / "summary_md2d_gdot0p001.json").read_text())
    manifest = json.loads((out / "trajectory_manifest.json").read_text())
    jumps = np.load(out / "cage_jumps_md2d_gdot0p001.npz")
    boxes = np.load(out / "boxes.npy")

    t_chi = float(summary["t_chi"])
    jump_times = jumps["jump_time"].astype(np.float64)
    jump_particles = jumps["particle_id"].astype(np.int64)
    sorted_times = np.sort(jump_times)
    starts, counts = window_counts(sorted_times, t_chi, t_chi)
    per_particle = np.bincount(jump_particles, minlength=int(manifest["n_particles"]) + 1)[1:]

    special_starts = [0.0, float(summary["window"]["start"])]
    if len(starts) > 0:
        special_starts.extend(np.linspace(float(starts[0]), float(starts[-1]), 4).tolist())
    special_starts = np.unique(np.asarray(special_starts, dtype=np.float64))

    rng = np.random.default_rng(20260630)
    report = {
        "output_dir": str(out),
        "total_jumps": int(len(jump_times)),
        "t_chi": t_chi,
        "lc2": float(summary.get("lc2", 0.048)) if "lc2" in summary else 0.048,
        "per_particle": {
            "mean": float(np.mean(per_particle)),
            "min": float(np.min(per_particle)),
            "p1": float(np.percentile(per_particle, 1)),
            "p5": float(np.percentile(per_particle, 5)),
            "p50": float(np.percentile(per_particle, 50)),
            "p95": float(np.percentile(per_particle, 95)),
            "p99": float(np.percentile(per_particle, 99)),
            "max": float(np.max(per_particle)),
        },
        "window_counts": {
            "n_windows": int(len(counts)),
            "min": float(np.min(counts)),
            "p5": float(np.percentile(counts, 5)),
            "p50": float(np.percentile(counts, 50)),
            "p95": float(np.percentile(counts, 95)),
            "max": float(np.max(counts)),
            "mean": float(np.mean(counts)),
        },
        "sampled_msd": sampled_msd(out / "r_tilde.float32", int(manifest["n_frames"]), int(manifest["n_particles"]), float(manifest["dump_stride_time"]), rng),
        "density_windows": density_window_report(jumps, special_starts, t_chi, boxes[0, 2:4].astype(np.float64), args.grid_n, args.coarse_d),
    }

    report_path = out / "cage_jump_density_diagnostic.json"
    text_path = out / "cage_jump_density_diagnostic.txt"
    plot_path = out / "cage_jump_density_diagnostic.png"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_text_report(text_path, report)
    plot_report(plot_path, report, starts, counts)
    print(f"wrote {report_path}")
    print(f"wrote {text_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
