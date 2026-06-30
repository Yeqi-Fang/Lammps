from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from md2d_chi4_cage_cluster import (  # noqa: E402
    MD2DConfig,
    box_dict,
    component_from_density,
    count_overlap_pairs,
    fold_to_triclinic_box,
    nonaffine_increment,
    periodic_labels,
    window_points,
    wrapped_y_average,
)


def test_wrapped_y_average_handles_periodic_crossings():
    box = {"ylo": 0.0, "ly": 10.0}
    np.testing.assert_allclose(wrapped_y_average(np.array([2.0]), np.array([4.0]), box), [3.0])
    np.testing.assert_allclose(wrapped_y_average(np.array([9.0]), np.array([1.0]), box), [5.0])
    np.testing.assert_allclose(wrapped_y_average(np.array([1.0]), np.array([9.0]), box), [5.0])
    np.testing.assert_allclose(wrapped_y_average(np.array([-0.001]), np.array([9.999]), box), [9.999])


def test_nonaffine_increment_removes_pure_affine_motion():
    box = {"xlo": 0.0, "ylo": 0.0, "lx": 20.0, "ly": 20.0, "xy": 0.0}
    prev = np.array([[5.0, 4.0], [7.0, 9.0]])
    gamma_dot = 0.01
    dt = 0.5
    curr = prev.copy()
    curr[:, 0] += gamma_dot * dt * prev[:, 1]
    inc, affine = nonaffine_increment(prev, curr, box, gamma_dot, dt)
    np.testing.assert_allclose(inc, np.zeros_like(prev), atol=1e-12)
    np.testing.assert_allclose(affine, gamma_dot * dt * prev[:, 1], atol=1e-7)


def test_nonaffine_increment_respects_shear_origin():
    box = {"xlo": 0.0, "ylo": 0.0, "lx": 20.0, "ly": 20.0, "xy": 0.0}
    prev = np.array([[5.0, 7.0]])
    gamma_dot = 0.01
    dt = 0.5
    curr = prev.copy()
    curr[:, 0] += gamma_dot * dt * (prev[:, 1] - 5.0)
    inc, _ = nonaffine_increment(prev, curr, box, gamma_dot, dt, shear_origin_y=5.0)
    np.testing.assert_allclose(inc, np.zeros_like(prev), atol=1e-12)


def test_collective_overlap_lag_zero_counts_self_pairs():
    box = {"xlo": 0.0, "ylo": 0.0, "lx": 10.0, "ly": 10.0, "xy": 0.0}
    points = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [4.0, 4.0]])
    assert count_overlap_pairs(points, points, box, 0.3) == 4.0


def test_fold_to_triclinic_box_returns_primary_cell_points():
    box = {"xlo": 0.0, "ylo": 0.0, "lx": 10.0, "ly": 8.0, "xy": 2.0}
    points = np.array([[13.0, 9.0], [-2.0, -1.0]])
    folded = fold_to_triclinic_box(points, box)
    sy = folded[:, 1] / box["ly"]
    sx = (folded[:, 0] - box["xy"] * sy) / box["lx"]
    assert np.all((sx >= 0.0) & (sx < 1.0))
    assert np.all((sy >= 0.0) & (sy < 1.0))


def test_periodic_labels_connects_across_box_edges():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = True
    mask[3, 1] = True
    labels = periodic_labels(mask)
    assert labels[0, 1] == labels[3, 1] != 0


def test_component_from_density_rejects_undersize_cluster():
    cfg = MD2DConfig(cluster_grid_n=10, stz_radius=1.5, c_prime=2.0)
    density = np.zeros((10, 10), dtype=float)
    density[3, 3] = 2.0
    labels, component, min_voxels = component_from_density(density, 1.0, cfg, np.array([10.0, 10.0]))
    assert min_voxels > 1
    assert component == 0
    assert labels[3, 3] != 0


def test_window_points_keeps_cosheared_positions_without_transport():
    jumps = {
        "jump_time": np.array([1.0]),
        "jump_position": np.array([[9.0, 2.0]], dtype=np.float32),
    }
    mask, points = window_points(jumps, 0.0, 2.0, np.array([10.0, 10.0]))
    assert mask.tolist() == [True]
    np.testing.assert_allclose(points, [[9.0, 2.0]])


def test_box_dict_uses_expected_order():
    box = box_dict(np.array([1.0, 2.0, 3.0, 4.0, 0.5]))
    assert box == {"xlo": 1.0, "ylo": 2.0, "lx": 3.0, "ly": 4.0, "xy": 0.5}
