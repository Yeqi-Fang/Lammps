from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from md3d_cluster_window import (  # noqa: E402
    analyze_window,
    build_rho_th_cache_key,
    cache_key_matches,
    coarse_density_kernel_metadata,
    coarse_membership_for_undersize_points,
    compute_rho_th_reference,
    threshold_iterative,
    undersize_refine,
    undersize_grid_points,
)


def test_threshold_iterative_matches_hand_calculation():
    density = np.asarray([1.0, 2.0, 3.0, 4.0])
    rho, history = threshold_iterative(density, tol=0.10)
    assert rho == 1.0
    assert history == [4.0, 2.0, 1.0]


def test_rho_th_reference_uses_multiple_tini_and_skips_zero_jump_windows():
    positions = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [1.1, 1.1, 1.1],
        ],
        dtype=float,
    )
    frames = np.asarray([0, 1, 2], dtype=np.int32)
    starts = np.asarray([0, 1, 3], dtype=np.int32)
    result = compute_rho_th_reference(
        positions,
        frames,
        starts,
        window_frames=1,
        box_lengths=np.asarray([4.0, 4.0, 4.0]),
        grid_n=4,
        coarse_d=1.0,
        tol=0.10,
    )
    assert result["n_tini_total"] == 3
    assert result["n_tini_used"] == 2
    assert result["n_skipped_zero_jump"] == 1
    assert result["rho_th"] > 0.0


def test_rho_th_cache_key_rejects_parameter_mismatch():
    with tempfile.NamedTemporaryFile(suffix=".npz") as fh:
        args = SimpleNamespace(
            cluster_particle_type=1,
            t_chi=2.4,
            dt_frame=0.1,
            grid_n=4,
            threshold_tol=0.10,
        )
        kernel = coarse_density_kernel_metadata(4, np.asarray([4.0, 4.0, 4.0]), 1.0)
        key_a = build_rho_th_cache_key(
            args,
            fh.name,
            source_frames=10,
            window_frames=2,
            box_lengths=np.asarray([4.0, 4.0, 4.0]),
            coarse_d=1.0,
            kernel_info=kernel,
            start_min=0,
            start_max=8,
            start_stride=1,
        )
        args.grid_n = 5
        key_b = build_rho_th_cache_key(
            args,
            fh.name,
            source_frames=10,
            window_frames=2,
            box_lengths=np.asarray([4.0, 4.0, 4.0]),
            coarse_d=1.0,
            kernel_info=kernel,
            start_min=0,
            start_max=8,
            start_stride=1,
        )
    assert cache_key_matches(key_a, key_a)
    assert not cache_key_matches(key_a, key_b)


def test_vertex_grid_has_periodic_unique_points():
    points = undersize_grid_points(np.asarray([2.0, 2.0, 2.0]), 2, "vertices")
    assert points.shape == (8, 3)
    assert len(np.unique(points, axis=0)) == 8
    assert np.all(points >= 0.0)
    assert np.all(points < 2.0)


def test_vertex_membership_uses_periodic_adjacent_coarse_cells():
    coarse_labels = np.zeros((2, 2, 2), dtype=np.int32)
    coarse_labels[1, 1, 1] = 7
    inside = coarse_membership_for_undersize_points(
        coarse_labels,
        coarse_component=7,
        box_lengths=np.asarray([2.0, 2.0, 2.0]),
        coarse_grid_n=2,
        fine_grid_n=2,
        point_mode="vertices",
    )
    assert inside.reshape((2, 2, 2))[0, 0, 0]


def test_undersize_refine_runs_with_vertex_points():
    positions = np.asarray(
        [
            [0.02, 0.02, 0.02],
            [0.12, 0.03, 0.03],
            [0.05, 0.14, 0.04],
        ],
        dtype=float,
    )
    coarse_labels = np.ones((2, 2, 2), dtype=np.int32)
    result = undersize_refine(
        positions,
        coarse_labels,
        coarse_component=1,
        box_lengths=np.asarray([2.0, 2.0, 2.0]),
        coarse_grid_n=2,
        rho_th=0.01,
        fine_grid_n=4,
        connectivity=3,
        point_mode="vertices",
    )
    assert result is not None
    best, fine_density, labels, in_best, info, point_labels = result
    assert info["point_mode"] == "vertices"
    assert labels.shape == (4, 4, 4)
    assert fine_density.shape == (4, 4, 4)
    assert point_labels.shape == (3,)
    assert int(best["n_cluster_jumps"]) >= 1


def test_analyze_window_uses_reference_rho_and_vertex_undersize_mode():
    positions = np.asarray(
        [
            [0.02, 0.02, 0.02],
            [0.12, 0.03, 0.03],
            [0.05, 0.14, 0.04],
            [1.2, 1.2, 1.2],
        ],
        dtype=float,
    )
    frames = np.asarray([0, 0, 1, 1], dtype=np.int32)
    args = SimpleNamespace(
        rho_th_mode="reference",
        coarse_segmentation="local-max",
        connectivity=3,
        undersize_point_mode="vertices",
        component_mode="single",
        multi_min_component_jumps=1,
        multi_max_components=0,
        multi_max_reference_distance=0.0,
    )
    result = analyze_window(
        positions,
        frames,
        start=0,
        window_frames=2,
        box_lengths=np.asarray([2.0, 2.0, 2.0]),
        grid_n=4,
        coarse_d=1.0,
        tol=0.10,
        min_voxels=1,
        undersize_grid_n=4,
        args=args,
        rho_th_reference={"rho_th": 0.01},
    )
    assert result is not None
    best = result[0]
    assert best["rho_th_mode"] == "reference"
    assert best["rho_th"] == 0.01
    assert best["boundary_point_mode"] == "vertices"


if __name__ == "__main__":
    test_threshold_iterative_matches_hand_calculation()
    test_rho_th_reference_uses_multiple_tini_and_skips_zero_jump_windows()
    test_rho_th_cache_key_rejects_parameter_mismatch()
    test_vertex_grid_has_periodic_unique_points()
    test_vertex_membership_uses_periodic_adjacent_coarse_cells()
    test_undersize_refine_runs_with_vertex_points()
    test_analyze_window_uses_reference_rho_and_vertex_undersize_mode()
