#!/usr/bin/env python3
"""Create a low-overlap 2D binary soft-disk LAMMPS data file.

This is an implementation choice for the Zeng 2D MD reproduction workflow.
The paper specifies the composition, density, masses, size ratio, and
interaction, but not the exact initial configuration generator.
"""

import argparse
import json
import math
import os
import random


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_positions(natoms, box_length, seed, jitter_fraction):
    rng = random.Random(seed)
    ncols = int(math.ceil(math.sqrt(float(natoms))))
    nrows = int(math.ceil(float(natoms) / float(ncols)))
    spacing_x = box_length / float(ncols)
    spacing_y = box_length / float(nrows)
    jitter_x = jitter_fraction * spacing_x
    jitter_y = jitter_fraction * spacing_y

    positions = []
    for row in range(nrows):
        for col in range(ncols):
            if len(positions) >= natoms:
                break
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            if jitter_fraction > 0.0:
                x += rng.uniform(-jitter_x, jitter_x)
                y += rng.uniform(-jitter_y, jitter_y)
            x %= box_length
            y %= box_length
            parity = (row + col) % 2
            positions.append((parity, x, y))
    return positions


def assign_types(positions, nb, ns, seed, layout):
    natoms = nb + ns
    if layout == "random":
        atom_types = [1] * nb + [2] * ns
        random.Random(seed + 7919).shuffle(atom_types)
        return atom_types

    even = [idx for idx, item in enumerate(positions) if item[0] == 0]
    odd = [idx for idx, item in enumerate(positions) if item[0] == 1]
    big_candidates = even + odd
    atom_types = [2] * natoms
    for idx in big_candidates[:nb]:
        atom_types[idx] = 1
    return atom_types


def write_lammps_data(path, positions, atom_types, box_length, mass_big, mass_small):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii") as handle:
        handle.write("LAMMPS data file for Zeng 2D binary soft disks\n\n")
        handle.write("{} atoms\n".format(len(atom_types)))
        handle.write("2 atom types\n\n")
        handle.write("{:.16g} {:.16g} xlo xhi\n".format(0.0, box_length))
        handle.write("{:.16g} {:.16g} ylo yhi\n".format(0.0, box_length))
        handle.write("{:.16g} {:.16g} zlo zhi\n\n".format(-0.5, 0.5))
        handle.write("Masses\n\n")
        handle.write("1 {:.16g}\n".format(mass_big))
        handle.write("2 {:.16g}\n\n".format(mass_small))
        handle.write("Atoms # atomic\n\n")
        for atom_id, (item, atom_type) in enumerate(zip(positions, atom_types), start=1):
            _, x, y = item
            handle.write(
                "{} {} {:.16g} {:.16g} {:.16g}\n".format(
                    atom_id, atom_type, x, y, 0.0
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--nb", type=positive_int, default=10000)
    parser.add_argument("--ns", type=positive_int, default=10000)
    parser.add_argument("--rho", type=positive_float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--mass-big", type=positive_float, default=2.0)
    parser.add_argument("--mass-small", type=positive_float, default=1.0)
    parser.add_argument("--jitter-fraction", type=float, default=0.03)
    parser.add_argument(
        "--layout",
        choices=("checkerboard", "random"),
        default="checkerboard",
        help="checkerboard reduces big-big nearest-neighbor contacts initially",
    )
    args = parser.parse_args()

    if args.jitter_fraction < 0.0 or args.jitter_fraction >= 0.45:
        raise SystemExit("--jitter-fraction must be in [0, 0.45)")

    natoms = args.nb + args.ns
    box_length = math.sqrt(float(natoms) / args.rho)
    positions = build_positions(natoms, box_length, args.seed, args.jitter_fraction)
    atom_types = assign_types(positions, args.nb, args.ns, args.seed, args.layout)
    write_lammps_data(
        args.output, positions, atom_types, box_length, args.mass_big, args.mass_small
    )

    manifest = {
        "generator": os.path.basename(__file__),
        "implementation_note": "Initial configuration generator is not specified by Zeng et al.",
        "nb": args.nb,
        "ns": args.ns,
        "natoms": natoms,
        "rho": args.rho,
        "box_length": box_length,
        "seed": args.seed,
        "mass_big": args.mass_big,
        "mass_small": args.mass_small,
        "jitter_fraction": args.jitter_fraction,
        "layout": args.layout,
        "output": os.path.abspath(args.output),
    }
    if args.manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
        with open(args.manifest, "w", encoding="ascii") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(
        "Wrote {} atoms at rho={} to {}".format(
            natoms, args.rho, os.path.abspath(args.output)
        )
    )


if __name__ == "__main__":
    main()
