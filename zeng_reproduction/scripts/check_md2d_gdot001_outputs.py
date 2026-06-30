#!/usr/bin/env python3
"""Validate completed MD2D gdot=0.001 chi4/cage/cluster outputs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_FILES = {
    "manifest": "trajectory_manifest.json",
    "positions": "positions_wrapped.float32",
    "r_tilde": "r_tilde.float32",
    "affine_x": "affine_x.float32",
    "times": "times.npy",
    "boxes": "boxes.npy",
    "chi4_npz": "chi4_md2d_gdot0p001.npz",
    "chi4_json": "chi4_md2d_gdot0p001.json",
    "jumps": "cage_jumps_md2d_gdot0p001.npz",
    "cluster": "cluster_md2d_gdot0p001.npz",
    "summary": "summary_md2d_gdot0p001.json",
}


class ValidationReport:
    def __init__(self):
        self.items = []

    def add(self, status, code, message, **data):
        self.items.append({"status": status, "code": code, "message": message, "data": data})

    def pass_(self, code, message, **data):
        self.add("pass", code, message, **data)

    def warn(self, code, message, **data):
        self.add("warn", code, message, **data)

    def fail(self, code, message, **data):
        self.add("fail", code, message, **data)

    def counts(self):
        return {status: sum(1 for item in self.items if item["status"] == status) for status in ("pass", "warn", "fail")}

    def ok(self):
        return self.counts()["fail"] == 0

    def to_dict(self):
        return {"ok": self.ok(), "counts": self.counts(), "items": self.items}


def analysis_dir(repo_root, run_group):
    return Path(repo_root) / "zeng_reproduction" / "data" / "MD" / "2D" / "processed" / "chi4_cage_cluster" / run_group / "gdot0p001"


def read_json(path):
    return json.loads(Path(path).read_text())


def scalar_text(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def check_files(out_dir, report):
    paths = {key: out_dir / name for key, name in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        report.fail("missing_files", "Required output files are missing", missing=missing)
    else:
        report.pass_("required_files", "All required output files are present")
    return paths


def check_manifest(paths, report, expected_n, expected_gdot, expected_dump_dt):
    if not paths["manifest"].exists():
        return None
    meta = read_json(paths["manifest"])
    cfg = meta.get("config", {})
    n_frames = int(meta.get("n_frames", 0))
    n_particles = int(meta.get("n_particles", 0))
    if n_frames > 0 and n_particles == expected_n:
        report.pass_("manifest_shape", "Manifest frame and particle counts look valid", n_frames=n_frames, n_particles=n_particles)
    else:
        report.fail("manifest_shape", "Unexpected manifest frame or particle count", n_frames=n_frames, n_particles=n_particles, expected_n=expected_n)
    if math.isclose(float(cfg.get("gamma_dot", np.nan)), expected_gdot, rel_tol=0.0, abs_tol=1.0e-12):
        report.pass_("manifest_gamma_dot", "gamma_dot matches expected value", gamma_dot=cfg.get("gamma_dot"))
    else:
        report.fail("manifest_gamma_dot", "gamma_dot does not match expected value", gamma_dot=cfg.get("gamma_dot"), expected=expected_gdot)
    if math.isclose(float(cfg.get("dump_dt", np.nan)), expected_dump_dt, rel_tol=0.0, abs_tol=1.0e-12):
        report.pass_("manifest_dump_dt", "dump_dt matches expected value", dump_dt=cfg.get("dump_dt"))
    else:
        report.fail("manifest_dump_dt", "dump_dt does not match expected value", dump_dt=cfg.get("dump_dt"), expected=expected_dump_dt)
    if "trajectory_config" in meta:
        report.pass_("trajectory_config", "trajectory_config is recorded")
    else:
        report.fail("trajectory_config", "trajectory_config is missing")
    ylo_min = meta.get("ylo_min")
    ylo_max = meta.get("ylo_max")
    shear_origin_y = float(meta.get("shear_origin_y", cfg.get("shear_origin_y", 0.0)))
    if ylo_min is not None and ylo_max is not None and abs(float(ylo_min) - shear_origin_y) < 1.0e-8 and abs(float(ylo_max) - shear_origin_y) < 1.0e-8:
        report.pass_("shear_origin", "ylo range matches shear_origin_y", ylo_min=ylo_min, ylo_max=ylo_max, shear_origin_y=shear_origin_y)
    else:
        report.warn("shear_origin", "ylo range differs from shear_origin_y; confirm shear-flow origin", ylo_min=ylo_min, ylo_max=ylo_max, shear_origin_y=shear_origin_y)
    if "dump_stride_steps" in meta and "dump_stride_time" in meta:
        report.pass_("dump_stride_recorded", "dump stride metadata is recorded", steps=meta["dump_stride_steps"], time=meta["dump_stride_time"])
    else:
        report.warn("dump_stride_recorded", "dump stride metadata is missing; rerun with updated generator for strict validation")
    check_memmap_size(paths["positions"], n_frames * n_particles * 2, "positions_size", report)
    check_memmap_size(paths["r_tilde"], n_frames * n_particles * 2, "r_tilde_size", report)
    check_memmap_size(paths["affine_x"], n_frames * n_particles, "affine_x_size", report)
    return meta


def check_memmap_size(path, n_float32, code, report):
    if not path.exists():
        return
    expected = int(n_float32) * np.dtype(np.float32).itemsize
    actual = path.stat().st_size
    if actual == expected:
        report.pass_(code, "Binary array size matches manifest", bytes=actual)
    else:
        report.fail(code, "Binary array size does not match manifest", bytes=actual, expected_bytes=expected)


def check_chi4(paths, report, n_particles):
    if not paths["chi4_npz"].exists():
        return
    chi = np.load(paths["chi4_npz"], allow_pickle=False)
    chi4 = np.asarray(chi["chi4"], dtype=float)
    q_mean = np.asarray(chi["q_mean"], dtype=float)
    lag_times = np.asarray(chi["lag_times"], dtype=float)
    if len(q_mean) == 0 or len(chi4) == 0:
        report.fail("chi4_empty", "chi4 arrays are empty")
        return
    q0_over_n = float(q_mean[0] / n_particles)
    chi40 = float(chi4[0])
    if 0.9 <= q0_over_n <= 1.1:
        report.pass_("chi4_q0", "Q(0)/N is close to 1", q0_over_n=q0_over_n)
    else:
        report.fail("chi4_q0", "Q(0)/N is not close to 1", q0_over_n=q0_over_n)
    if abs(chi40) <= 1.0e-8:
        report.pass_("chi4_zero", "chi4(0) is close to 0", chi4_0=chi40)
    else:
        report.fail("chi4_zero", "chi4(0) is not close to 0", chi4_0=chi40)
    if len(q_mean) > 1 and float(q_mean[-1]) < float(q_mean[0]):
        report.pass_("chi4_decay", "Q_mean/N decays over sampled lags", q_start=float(q_mean[0] / n_particles), q_end=float(q_mean[-1] / n_particles))
    else:
        report.warn("chi4_decay", "Q_mean/N does not decay over sampled lags", q_start=float(q_mean[0] / n_particles), q_end=float(q_mean[-1] / n_particles))
    peak = int(np.nanargmax(chi4))
    if peak == 0 or peak == len(chi4) - 1:
        report.warn("chi4_peak_boundary", "chi4 peak lies on lag boundary", peak_index=peak, peak_time=float(lag_times[peak]))
    else:
        report.pass_("chi4_peak_internal", "chi4 peak is internal", peak_index=peak, peak_time=float(lag_times[peak]))


def check_jumps(paths, report, times, box_lengths, allow_empty):
    if not paths["jumps"].exists():
        return None
    jumps = np.load(paths["jumps"], allow_pickle=False)
    frames = np.asarray(jumps["jump_frame"], dtype=int)
    jump_times = np.asarray(jumps["jump_time"], dtype=float)
    positions = np.asarray(jumps["jump_position"], dtype=float)
    vectors = np.asarray(jumps["jump_vector"], dtype=float)
    mode = scalar_text(jumps["detection_mode"]) if "detection_mode" in jumps.files else "unknown"
    if len(frames) == 0:
        if allow_empty or mode == "skipped":
            report.warn("jumps_empty", "No cage jumps present; accepted for skipped/diagnostic output", detection_mode=mode)
        else:
            report.fail("jumps_empty", "No cage jumps present in formal output", detection_mode=mode)
        return jumps
    report.pass_("jumps_nonempty", "Cage jump table is nonempty", n_jumps=int(len(frames)), detection_mode=mode)
    in_range = np.all((frames >= 0) & (frames < len(times)))
    if in_range:
        report.pass_("jump_frames", "jump_frame values are within trajectory range")
    else:
        report.fail("jump_frames", "jump_frame values exceed trajectory range", min_frame=int(np.min(frames)), max_frame=int(np.max(frames)), n_frames=int(len(times)))
    if in_range and np.allclose(jump_times, times[frames], rtol=0.0, atol=1.0e-8):
        report.pass_("jump_times", "jump_time values match times[jump_frame]")
    else:
        report.fail("jump_times", "jump_time values do not match frame times")
    finite = np.all(np.isfinite(positions)) and np.all(np.isfinite(vectors))
    if finite:
        report.pass_("jump_finite", "jump positions and vectors are finite")
    else:
        report.fail("jump_finite", "jump positions or vectors contain NaN/Inf")
    if len(positions) and np.all((positions >= -1.0e-6) & (positions < box_lengths + 1.0e-6)):
        report.pass_("jump_positions_box", "jump positions lie inside co-sheared box")
    else:
        report.fail("jump_positions_box", "jump positions lie outside co-sheared box", box_lengths=box_lengths.tolist())
    return jumps


def check_cluster(paths, report, jumps):
    if not paths["summary"].exists() or not paths["cluster"].exists():
        return
    summary = read_json(paths["summary"])
    cluster = np.load(paths["cluster"], allow_pickle=False)
    coordinate_system = str(summary.get("coordinate_system", scalar_text(cluster["coordinate_system"]) if "coordinate_system" in cluster.files else ""))
    if coordinate_system == "co_sheared_orthogonal_box":
        report.pass_("cluster_coordinate_system", "cluster coordinate system is co-sheared orthogonal box")
    else:
        report.fail("cluster_coordinate_system", "Unexpected cluster coordinate system", coordinate_system=coordinate_system)
    if summary.get("window_selection"):
        report.pass_("cluster_window_selection", "window_selection is recorded", window_selection=summary.get("window_selection"))
    else:
        report.fail("cluster_window_selection", "window_selection is missing")
    if summary.get("cluster_analysis_scope") == "representative_visualization":
        report.pass_("cluster_scope", "cluster scope is recorded as representative visualization")
    else:
        report.warn("cluster_scope", "cluster scope is missing or not representative_visualization", value=summary.get("cluster_analysis_scope"))
    component = int(summary.get("component", int(cluster["component"]) if "component" in cluster.files else 0))
    n_component = int(summary.get("n_component_jumps", int(np.count_nonzero(cluster["in_component"])) if "in_component" in cluster.files else 0))
    rho_th = float(summary.get("rho_th", float(cluster["rho_th"]) if "rho_th" in cluster.files else 0.0))
    if np.isfinite(rho_th) and (rho_th >= 0.0):
        report.pass_("cluster_rho_th", "rho_th is finite and nonnegative", rho_th=rho_th)
    else:
        report.fail("cluster_rho_th", "rho_th is invalid", rho_th=rho_th)
    if component == 0 and n_component == 0:
        report.pass_("cluster_no_false_component", "component=0 has no component jumps")
    elif component == 0 and n_component > 0:
        report.fail("cluster_no_false_component", "component=0 but component jumps are marked", n_component_jumps=n_component)
    elif component > 0 and n_component > 0:
        report.pass_("cluster_component_nonempty", "selected cluster component has marked jumps", component=component, n_component_jumps=n_component)
    else:
        report.fail("cluster_component_nonempty", "component>0 but no jumps are marked", component=component, n_component_jumps=n_component)
    if jumps is not None and "in_component" in cluster.files and len(cluster["in_component"]) > len(jumps["jump_time"]):
        report.fail("cluster_in_component_length", "cluster in_component is longer than jump table", in_component_len=int(len(cluster["in_component"])), jumps=int(len(jumps["jump_time"])))


def validate_output(out_dir, expected_n=20000, expected_gdot=0.001, expected_dump_dt=0.5, allow_empty_jumps=False):
    out_dir = Path(out_dir)
    report = ValidationReport()
    paths = check_files(out_dir, report)
    meta = check_manifest(paths, report, expected_n, expected_gdot, expected_dump_dt)
    n_particles = int(meta.get("n_particles", expected_n)) if meta else expected_n
    check_chi4(paths, report, n_particles)
    times = np.load(paths["times"]) if paths["times"].exists() else np.empty(0, dtype=float)
    boxes = np.load(paths["boxes"]) if paths["boxes"].exists() else np.empty((0, 5), dtype=float)
    box_lengths = boxes[0, 2:4].astype(float) if len(boxes) else np.array([np.nan, np.nan])
    jumps = check_jumps(paths, report, times, box_lengths, allow_empty_jumps)
    check_cluster(paths, report, jumps)
    return report


def write_reports(out_dir, report):
    data = report.to_dict()
    (out_dir / "validation_report.json").write_text(json.dumps(data, indent=2, sort_keys=True))
    lines = ["MD2D gdot=0.001 validation", "ok: {}".format(data["ok"]), "counts: {}".format(data["counts"]), ""]
    for item in report.items:
        lines.append("[{status}] {code}: {message}".format(**item))
    (out_dir / "validation_report.txt").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-group", default="md2d_prod_slim_20260629_1412")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-n", type=int, default=20000)
    parser.add_argument("--expected-gdot", type=float, default=0.001)
    parser.add_argument("--expected-dump-dt", type=float, default=0.5)
    parser.add_argument("--allow-empty-jumps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else analysis_dir(Path(args.repo_root).resolve(), args.run_group)
    report = validate_output(out_dir, args.expected_n, args.expected_gdot, args.expected_dump_dt, args.allow_empty_jumps)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_reports(out_dir, report)
    counts = report.counts()
    print("validation ok={} pass={} warn={} fail={}".format(report.ok(), counts["pass"], counts["warn"], counts["fail"]))
    raise SystemExit(0 if report.ok() else 1)


if __name__ == "__main__":
    main()
