#!/usr/bin/env python
"""Convert existing local MD analysis NPZ files to the zeng_reproduction schema.

This helper is for MD convenience only. It does not create exact Fig. 6 stress,
because the current MD dumps do not contain per-particle stress or ICE averages.
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, ROOT)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--msd-npz", required=True, help="Existing msd_data_*.npz")
    p.add_argument("--jumps-npz", required=True, help="Existing cage_jumps_*.npz")
    p.add_argument("--trajectory-out", required=True)
    p.add_argument("--jumps-out", required=True)
    p.add_argument("--dt-frame", type=float, required=True)
    args = p.parse_args()

    msd = np.load(args.msd_npz, allow_pickle=True)
    jumps = np.load(args.jumps_npz, allow_pickle=True)

    required_msd = ["r_tilde", "types", "box_Lx", "box_Ly"]
    missing = [k for k in required_msd if k not in msd]
    if missing:
        raise SystemExit("MSD NPZ missing keys: {}".format(", ".join(missing)))

    nonaffine = np.asarray(msd["r_tilde"], dtype=np.float32)
    frames, particles, dim = nonaffine.shape
    times = np.arange(frames, dtype=float) * float(args.dt_frame)
    box = [float(msd["box_Lx"]), float(msd["box_Ly"])]
    if "box_Lz" in msd:
        box.append(float(msd["box_Lz"]))
    elif dim == 3:
        box.append(float(msd["box_Lx"]))

    # The old MSD NPZ does not preserve lab-frame positions. Store nonaffine
    # as positions only for workflows that explicitly tolerate this conversion.
    positions = nonaffine.copy()
    os.makedirs(os.path.dirname(args.trajectory_out), exist_ok=True)
    np.savez_compressed(
        args.trajectory_out,
        positions=positions,
        nonaffine=nonaffine,
        times=times,
        types=np.asarray(msd["types"], dtype=np.int16),
        box=np.asarray(box, dtype=float),
        converted_note=(
            "positions are copied from r_tilde; exact convective-boundary and "
            "stress analysis require lab-frame positions and per-atom stress"
        ),
    )

    required_jumps = ["jump_frames", "particle_idx", "jump_vectors"]
    missing = [k for k in required_jumps if k not in jumps]
    if missing:
        raise SystemExit("Jumps NPZ missing keys: {}".format(", ".join(missing)))
    frames_j = np.asarray(jumps["jump_frames"], dtype=int)
    jump_times = times[np.clip(frames_j, 0, len(times) - 1)]
    save = {
        "jump_frames": frames_j,
        "particle_idx": np.asarray(jumps["particle_idx"], dtype=int),
        "jump_vectors": np.asarray(jumps["jump_vectors"], dtype=np.float32),
        "jump_times": jump_times,
    }
    if "jump_pos_xy" in jumps:
        xy = np.asarray(jumps["jump_pos_xy"], dtype=np.float32)
        pos = np.zeros((len(xy), dim), dtype=np.float32)
        pos[:, :2] = xy[:, :2]
        save["jump_positions"] = pos
    elif "positions" in jumps:
        save["jump_positions"] = np.asarray(jumps["positions"], dtype=np.float32)
    os.makedirs(os.path.dirname(args.jumps_out), exist_ok=True)
    np.savez_compressed(args.jumps_out, **save)
    print("Wrote {}".format(args.trajectory_out))
    print("Wrote {}".format(args.jumps_out))
    print("Note: exact Fig. 6 still needs lab-frame positions and per_atom_stress_xy/ICE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
