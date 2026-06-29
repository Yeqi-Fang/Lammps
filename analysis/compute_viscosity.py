"""
Compute steady-state shear viscosity from LAMMPS fix ave/time thermo files.

For the Zeng KA shear workflow, thermo.shear_<rate>.dat is written as:

    # time gamma temp press pxy
    <fix_step> <time> <gamma> <temp> <press> <pxy>

LAMMPS fix ave/time prepends the output timestep, so the parser always uses
the last column as Pxy and the last three columns as temp, press, Pxy. Older
4-column files of the form step temp press pxy are also supported.

This script intentionally stays compatible with older Python 3 versions on
the cluster login nodes.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISCARD_FRACTION = 0.20
BLOCK_SIZE = 200


class ViscosityResult(object):
    def __init__(
        self,
        shear_rate,
        eta,
        eta_err,
        pxy_mean,
        temp_mean,
        n_samples,
        n_steady,
        gamma_end,
    ):
        self.shear_rate = shear_rate
        self.eta = eta
        self.eta_err = eta_err
        self.pxy_mean = pxy_mean
        self.temp_mean = temp_mean
        self.n_samples = n_samples
        self.n_steady = n_steady
        self.gamma_end = gamma_end


def _numeric_rows(filename):
    rows = []
    with open(filename, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in stripped.split()])
            except ValueError:
                continue

    if not rows:
        raise ValueError("No numeric data found in {}".format(filename))

    ncols = set(len(row) for row in rows)
    if len(ncols) != 1:
        raise ValueError(
            "Inconsistent column count in {}: {}".format(filename, sorted(ncols))
        )

    return np.asarray(rows, dtype=float)


def read_thermo(filename):
    """Read a LAMMPS thermo.shear file.

    Returns arrays keyed by step/time/gamma/temp/press/pxy where available.
    The Pxy column is always interpreted as the last numeric column.
    """

    arr = _numeric_rows(filename)
    ncols = arr.shape[1]
    out = {"ncols": ncols}

    if ncols >= 6:
        # fix ave/time timestep + values: time gamma temp press pxy
        out["step"] = arr[:, 0]
        out["time"] = arr[:, 1]
        out["gamma"] = arr[:, 2]
        out["temp"] = arr[:, -3]
        out["press"] = arr[:, -2]
        out["pxy"] = arr[:, -1]
    elif ncols == 5:
        # Values only: time gamma temp press pxy.
        out["step"] = np.arange(arr.shape[0], dtype=float)
        out["time"] = arr[:, 0]
        out["gamma"] = arr[:, 1]
        out["temp"] = arr[:, -3]
        out["press"] = arr[:, -2]
        out["pxy"] = arr[:, -1]
    elif ncols == 4:
        # Older helper output: step temp press pxy.
        out["step"] = arr[:, 0]
        out["temp"] = arr[:, 1]
        out["press"] = arr[:, 2]
        out["pxy"] = arr[:, 3]
    elif ncols == 2:
        # Minimal time/pxy or step/pxy data.
        out["step"] = arr[:, 0]
        out["pxy"] = arr[:, 1]
    else:
        raise ValueError("Need at least 2 numeric columns in {}, got {}".format(filename, ncols))

    return out


def compute_viscosity(
    thermo_file,
    shear_rate,
    discard_fraction=DISCARD_FRACTION,
    block_size=BLOCK_SIZE,
):
    d = read_thermo(thermo_file)
    pxy = np.asarray(d["pxy"], dtype=float)
    n = len(pxy)
    if n < 2:
        raise ValueError("Not enough samples in {}".format(thermo_file))

    start = min(max(int(n * discard_fraction), 0), n - 1)
    pxy_ss = pxy[start:]
    pxy_mean = float(np.mean(pxy_ss))
    eta = -pxy_mean / shear_rate

    n_blocks = len(pxy_ss) // block_size if block_size > 0 else 0
    if n_blocks >= 2:
        block_means = np.array(
            [
                np.mean(pxy_ss[i * block_size : (i + 1) * block_size])
                for i in range(n_blocks)
            ],
            dtype=float,
        )
        eta_err = float(np.std(-block_means / shear_rate, ddof=1) / np.sqrt(n_blocks))
    else:
        eta_err = float(np.std(pxy_ss, ddof=1) / np.sqrt(len(pxy_ss)) / shear_rate)

    temp_mean = None
    if "temp" in d:
        temp = np.asarray(d["temp"], dtype=float)
        temp_mean = float(np.mean(temp[start:]))

    gamma_end = None
    if "gamma" in d:
        gamma = np.asarray(d["gamma"], dtype=float)
        gamma_end = float(gamma[-1])

    print(
        "  gdot={:g}: eta={:.6g} +/- {:.3g}, <Pxy>={:.6g}, Nss={}".format(
            shear_rate, eta, eta_err, pxy_mean, len(pxy_ss)
        )
    )

    return ViscosityResult(
        shear_rate=shear_rate,
        eta=eta,
        eta_err=eta_err,
        pxy_mean=pxy_mean,
        temp_mean=temp_mean,
        n_samples=n,
        n_steady=len(pxy_ss),
        gamma_end=gamma_end,
    )


def parse_rate_from_filename(path):
    name = os.path.basename(path)
    match = re.search(r"shear_([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", name)
    if not match:
        raise ValueError("Cannot extract shear rate from {}; pass --rates".format(path))
    return float(match.group(1))


def fit_power_law(shear_rates, etas):
    if len(shear_rates) < 3:
        return None
    ok = (shear_rates > 0.0) & (etas > 0.0)
    if ok.sum() < 3:
        return None
    slope, intercept = np.polyfit(np.log(shear_rates[ok]), np.log(etas[ok]), deg=1)
    lam = -float(slope)
    eta0 = float(np.exp(intercept))
    return eta0, lam


def plot_pxy_timeseries(
    thermo_file,
    shear_rate,
    output_dir,
    discard_fraction=DISCARD_FRACTION,
):
    os.makedirs(output_dir, exist_ok=True)
    d = read_thermo(thermo_file)
    pxy = np.asarray(d["pxy"], dtype=float)
    n = len(pxy)
    start = min(max(int(n * discard_fraction), 0), n - 1)

    x_key = "time" if "time" in d else "step"
    x = np.asarray(d[x_key], dtype=float)
    xlabel = "Time" if x_key == "time" else "Step"
    pxy_mean = float(np.mean(pxy[start:]))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    ax = axes[0]
    ax.plot(x, pxy, lw=0.7, color="0.55", alpha=0.75)
    ax.plot(x[start:], pxy[start:], lw=0.8, color="steelblue", label="steady window")
    ax.axhline(pxy_mean, color="crimson", ls="--", label="<Pxy>={:.4f}".format(pxy_mean))
    ax.axvline(x[start], color="seagreen", ls=":", label="discard boundary")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pxy")
    ax.set_title("Pxy time series, gdot={:g}".format(shear_rate))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    pxy_ss = pxy[start:]
    n_blocks = len(pxy_ss) // BLOCK_SIZE
    if n_blocks:
        block_idx = np.arange(n_blocks) * BLOCK_SIZE
        block_pxy = np.array(
            [np.mean(pxy_ss[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]) for i in range(n_blocks)]
        )
        ax2.plot(block_idx, block_pxy, "o-", ms=4, color="darkorange")
        ax2.axhline(pxy_mean, color="crimson", ls="--")
    ax2.set_xlabel("Block start index in steady window")
    ax2.set_ylabel("Block <Pxy>")
    ax2.set_title("Block-averaged Pxy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    sr_str = "{:g}".format(shear_rate).replace(".", "p")
    out = os.path.join(output_dir, "pxy_timeseries_{}.png".format(sr_str))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("  saved {}".format(out))


def plot_viscosity(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = sorted(results, key=lambda r: r.shear_rate)
    sr = np.array([r.shear_rate for r in results], dtype=float)
    eta = np.array([r.eta for r in results], dtype=float)
    err = np.array([r.eta_err for r in results], dtype=float)

    fit = fit_power_law(sr, eta)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(sr, eta, yerr=err, fmt="o", ms=7, capsize=4, color="steelblue")
    if fit is not None:
        eta0, lam = fit
        sr_fit = np.logspace(np.log10(sr.min()), np.log10(sr.max()), 200)
        ax.plot(sr_fit, eta0 * sr_fit ** (-lam), "r--", lw=2)
        ax.text(
            0.05,
            0.12,
            r"$\lambda={:.3f}$, $\lambda/4={:.3f}$".format(lam, lam / 4.0),
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
        )
        print(
            "\nPower-law fit: eta = {:.6g} * gdot^(-{:.4f}); lambda/4={:.4f}".format(
                eta0, lam, lam / 4.0
            )
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\dot\gamma\ [\tau_0^{-1}]$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title("3D KA shear viscosity, T=0.45")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()

    png = os.path.join(output_dir, "viscosity_vs_shearrate.png")
    pdf = os.path.join(output_dir, "viscosity_vs_shearrate.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print("  saved {}".format(png))

    csv = os.path.join(output_dir, "viscosity_data.csv")
    table = np.column_stack([sr, eta, err, [r.pxy_mean for r in results]])
    np.savetxt(csv, table, header="shear_rate eta eta_err pxy_mean", comments="# ")
    print("  saved {}".format(csv))


def expand_files(patterns):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute NEMD shear viscosity")
    parser.add_argument("--thermo", nargs="+", default=["thermo.shear_0.015.dat"])
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--output", default="figures")
    parser.add_argument("--discard", type=float, default=DISCARD_FRACTION)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = parser.parse_args(argv)

    files = expand_files(args.thermo)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print("Missing thermo files:")
        for f in missing:
            print("  {}".format(f))
        return 1

    if args.rates is not None and len(args.rates) != len(files):
        print("--rates must have the same length as --thermo files")
        return 1

    print("\nAnalyzing {} thermo file(s)".format(len(files)))
    results = []
    for i, path in enumerate(files):
        sr = args.rates[i] if args.rates is not None else parse_rate_from_filename(path)
        print("\n[{}] gdot={:g}".format(path, sr))
        result = compute_viscosity(path, sr, args.discard, args.block_size)
        results.append(result)
        plot_pxy_timeseries(path, sr, args.output, args.discard)

    if results:
        plot_viscosity(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
