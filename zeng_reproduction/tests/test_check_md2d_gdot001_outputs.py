from pathlib import Path
import json
import tempfile
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_md2d_gdot001_outputs import validate_output  # noqa: E402


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def make_valid_output(out_dir, chi4=None, q_mean=None, jump_position=None, jump_frame=None, component=1, n_component_jumps=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 3
    n_particles = 4
    write_json(
        out_dir / "trajectory_manifest.json",
        {
            "config": {"gamma_dot": 0.001, "dump_dt": 0.5, "shear_origin_y": 0.0},
            "trajectory_config": {"gamma_dot": 0.001, "dump_dt": 0.5},
            "n_frames": n_frames,
            "n_particles": n_particles,
            "dump_stride_steps": 100,
            "dump_stride_time": 0.5,
            "shear_origin_y": 0.0,
            "ylo_min": 0.0,
            "ylo_max": 0.0,
        },
    )
    np.zeros((n_frames, n_particles, 2), dtype=np.float32).tofile(out_dir / "positions_wrapped.float32")
    np.zeros((n_frames, n_particles, 2), dtype=np.float32).tofile(out_dir / "r_tilde.float32")
    np.zeros((n_frames, n_particles), dtype=np.float32).tofile(out_dir / "affine_x.float32")
    np.save(out_dir / "times.npy", np.array([0.0, 0.5, 1.0], dtype=np.float64))
    np.save(out_dir / "boxes.npy", np.array([[0.0, 0.0, 10.0, 10.0, 0.0]] * n_frames, dtype=np.float64))

    q_mean = np.array([4.0, 2.0, 1.0], dtype=np.float64) if q_mean is None else np.asarray(q_mean, dtype=np.float64)
    chi4 = np.array([0.0, 1.0, 0.5], dtype=np.float64) if chi4 is None else np.asarray(chi4, dtype=np.float64)
    np.savez_compressed(
        out_dir / "chi4_md2d_gdot0p001.npz",
        chi4=chi4,
        q_mean=q_mean,
        q_var=np.zeros_like(chi4),
        lag_times=np.arange(len(chi4), dtype=np.float64) * 0.5,
        lags=np.arange(len(chi4), dtype=np.int32),
        n_origins=np.ones(len(chi4), dtype=np.int32),
    )
    write_json(out_dir / "chi4_md2d_gdot0p001.json", {"peak_at_lag_boundary": False, "t_chi": 0.5})

    jump_frame = np.array([1], dtype=np.int32) if jump_frame is None else np.asarray(jump_frame, dtype=np.int32)
    jump_time = np.array([0.5], dtype=np.float64) if np.all((jump_frame >= 0) & (jump_frame < 3)) else np.array([99.0], dtype=np.float64)
    jump_position = np.array([[1.0, 2.0]], dtype=np.float32) if jump_position is None else np.asarray(jump_position, dtype=np.float32)
    np.savez_compressed(
        out_dir / "cage_jumps_md2d_gdot0p001.npz",
        particle_id=np.array([1], dtype=np.int32),
        jump_frame=jump_frame,
        jump_time=jump_time,
        jump_position=jump_position,
        jump_vector=np.array([[0.1, 0.2]], dtype=np.float32),
        detection_mode=np.array("formal_full"),
    )

    in_component = np.array([n_component_jumps > 0], dtype=bool)
    np.savez_compressed(
        out_dir / "cluster_md2d_gdot0p001.npz",
        points=jump_position,
        density=np.ones((2, 2), dtype=np.float32),
        labels=np.ones((2, 2), dtype=np.int32) * component,
        rho_th=np.array(0.1, dtype=np.float64),
        component=np.array(component, dtype=np.int32),
        in_component=in_component,
        box_lengths=np.array([10.0, 10.0], dtype=np.float64),
        window_selection=np.array("max_density_window_valid_min_size"),
        coordinate_system=np.array("co_sheared_orthogonal_box"),
    )
    write_json(
        out_dir / "summary_md2d_gdot0p001.json",
        {
            "component": component,
            "coordinate_system": "co_sheared_orthogonal_box",
            "window_selection": "max_density_window_valid_min_size",
            "cluster_analysis_scope": "representative_visualization",
            "n_component_jumps": n_component_jumps,
            "n_total_jumps": 1,
            "n_window_jumps": 1,
            "rho_th": 0.1,
        },
    )


def statuses(report, code):
    return [item["status"] for item in report.items if item["code"] == code]


def test_validator_missing_files_fail():
    with tempfile.TemporaryDirectory() as tmp:
        report = validate_output(Path(tmp), expected_n=4)
        assert "fail" in statuses(report, "missing_files")


def test_validator_warns_when_chi4_peak_on_boundary():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        make_valid_output(out_dir, chi4=np.array([0.0, 1.0]), q_mean=np.array([4.0, 1.0]))
        report = validate_output(out_dir, expected_n=4)
        assert "warn" in statuses(report, "chi4_peak_boundary")


def test_validator_fails_bad_q0():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        make_valid_output(out_dir, q_mean=np.array([8.0, 2.0, 1.0]))
        report = validate_output(out_dir, expected_n=4)
        assert "fail" in statuses(report, "chi4_q0")


def test_validator_fails_bad_jump_position():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        make_valid_output(out_dir, jump_position=np.array([[12.0, 2.0]], dtype=np.float32))
        report = validate_output(out_dir, expected_n=4)
        assert "fail" in statuses(report, "jump_positions_box")


def test_validator_fails_bad_jump_frame():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        make_valid_output(out_dir, jump_frame=np.array([5], dtype=np.int32))
        report = validate_output(out_dir, expected_n=4)
        assert "fail" in statuses(report, "jump_frames")


def test_validator_fails_false_component_count():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        make_valid_output(out_dir, component=0, n_component_jumps=1)
        report = validate_output(out_dir, expected_n=4)
        assert "fail" in statuses(report, "cluster_no_false_component")


def test_validator_fails_empty_selected_component():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        make_valid_output(out_dir, component=1, n_component_jumps=0)
        report = validate_output(out_dir, expected_n=4)
        assert "fail" in statuses(report, "cluster_component_nonempty")
