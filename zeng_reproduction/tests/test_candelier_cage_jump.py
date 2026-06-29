from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeng_repro.cage_jump import compute_p_function, find_cage_jumps_recursive  # noqa: E402


def brute_force_p_function(traj):
    traj = np.asarray(traj, dtype=float)
    T = len(traj)
    cuts = np.arange(1, T - 1)
    out = []
    for cut in cuts:
        s1 = traj[: cut + 1]
        s2 = traj[cut + 1 :]
        c1 = s1.mean(axis=0)
        c2 = s2.mean(axis=0)
        d1 = np.mean(np.sum((s2 - c1) ** 2, axis=1))
        d2 = np.mean(np.sum((s1 - c2) ** 2, axis=1))
        zeta = np.sqrt((cut / float(T)) * (1.0 - cut / float(T)))
        out.append(zeta * np.sqrt(d1 * d2))
    return cuts, np.asarray(out)


def test_p_function_matches_zeng_appendix_b_formula():
    traj = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 0.9, 0.0],
            [1.1, 1.0, 0.0],
            [1.0, 1.1, 0.0],
        ]
    )
    cuts, p = compute_p_function(traj)
    expected_cuts, expected_p = brute_force_p_function(traj)
    np.testing.assert_array_equal(cuts, expected_cuts)
    np.testing.assert_allclose(p, expected_p, rtol=1e-13, atol=1e-13)


def test_recursive_detector_finds_synthetic_plateau_jumps():
    traj = np.vstack(
        [
            np.zeros((8, 3)),
            np.ones((8, 3)) * np.array([2.0, 0.0, 0.0]),
            np.ones((8, 3)) * np.array([4.0, 0.0, 0.0]),
        ]
    )
    jumps = sorted(find_cage_jumps_recursive(traj, lc2=0.1, min_segment=4))
    assert any(abs(j - 7) <= 1 for j in jumps)
    assert any(abs(j - 15) <= 1 for j in jumps)
