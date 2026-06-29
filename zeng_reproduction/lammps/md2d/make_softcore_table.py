#!/usr/bin/env python3
"""Generate a LAMMPS pair_style table for V(r)=(sigma/r)^12."""

import argparse
import math
import os


PAIR_SECTIONS = (
    ("BB", 1.4),
    ("BS", 1.2),
    ("SS", 1.0),
)


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed <= 1:
        raise argparse.ArgumentTypeError("value must be greater than 1")
    return parsed


def energy_force(radius, sigma):
    energy = math.pow(sigma / radius, 12)
    force = 12.0 * energy / radius
    return energy, force


def write_table(path, npoints, rmin, rmax):
    if rmin >= rmax:
        raise ValueError("rmin must be less than rmax")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii") as handle:
        handle.write("# Zeng 2D soft-core potential table: V(r)=(sigma/r)^12\n")
        handle.write("# Columns: index r energy force, with force=-dE/dr\n\n")
        for name, sigma in PAIR_SECTIONS:
            handle.write("{}\n".format(name))
            handle.write("N {} R {:.16g} {:.16g}\n\n".format(npoints, rmin, rmax))
            for idx in range(1, npoints + 1):
                frac = float(idx - 1) / float(npoints - 1)
                radius = rmin + frac * (rmax - rmin)
                energy, force = energy_force(radius, sigma)
                handle.write(
                    "{} {:.16e} {:.16e} {:.16e}\n".format(
                        idx, radius, energy, force
                    )
                )
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--npoints", type=positive_int, default=5000)
    parser.add_argument("--rmin", type=positive_float, default=0.45)
    parser.add_argument("--rmax", type=positive_float, default=4.5)
    args = parser.parse_args()
    write_table(args.output, args.npoints, args.rmin, args.rmax)
    print("Wrote soft-core table to {}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
