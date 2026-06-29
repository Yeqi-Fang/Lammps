#!/usr/bin/env python
"""Analyze completed 3D KA high-rate server runs.

Input directory should contain the synced server small files:

    raw/lammps/log.equilibrate
    raw/lammps/log.shear_<rate>
    raw/lammps/thermo.shear_<rate>.dat
    raw/lammps/visc_blockave.shear_<rate>.dat

The script writes summary CSV/JSON and viscosity plots without touching the
large trajectory dumps.
"""

import argparse
import csv
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RATES = [0.005, 0.015, 0.030, 0.060]
DT_BY_RATE = {0.005: 0.003, 0.015: 0.001, 0.030: 0.001, 0.060: 0.001}
DISCARD = 0.20
BLOCK_SIZE = 200


def read_numeric(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                pass
    if not rows:
        raise ValueError("no numeric rows in {}".format(path))
    return np.asarray(rows, dtype=float)


def parse_log(path):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    dangerous = [int(x) for x in re.findall(r"Dangerous builds\s*=\s*(\d+)", text)]
    loops = re.findall(r"Loop time of\s+([0-9.eE+-]+).*?for\s+(\d+)\s+steps with\s+(\d+)\s+atoms", text)
    errors = re.findall(r"ERROR[^\n]*", text)
    dump_every = None
    m = re.search(r"dump_every\s*=\s*([0-9]+)\s+steps", text)
    if m:
        dump_every = int(m.group(1))
    n_prod = None
    m = re.search(r"N_prod steps\s*=\s*([0-9]+)", text)
    if m:
        n_prod = int(m.group(1))
    return {
        "dangerous": dangerous,
        "dangerous_max": max(dangerous) if dangerous else None,
        "loops": loops,
        "errors": errors,
        "dump_every": dump_every,
        "n_prod": n_prod,
    }


def parse_equil(log_path):
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    created = [int(x) for x in re.findall(r"Created\s+(\d+)\s+atoms", text)]
    density = None
    m = re.search(r"Density:\s+rho\s+=\s+([0-9.eE+-]+)", text)
    if m:
        density = float(m.group(1))
    dangerous = [int(x) for x in re.findall(r"Dangerous builds\s*=\s*(\d+)", text)]
    box = None
    m = re.search(r"Created orthogonal box = .*? to \(([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\)", text)
    if m:
        box = [float(m.group(i)) for i in range(1, 4)]
    return {
        "created_atoms": created,
        "total_atoms": sum(created),
        "density": density,
        "box": box,
        "dangerous": dangerous,
        "dangerous_max": max(dangerous) if dangerous else None,
    }


def viscosity_from_thermo(path, rate):
    arr = read_numeric(path)
    # fix ave/time prepends timestep; last three columns are temp, press, pxy.
    gamma = arr[:, 2] if arr.shape[1] >= 6 else None
    temp = arr[:, -3] if arr.shape[1] >= 4 else None
    press = arr[:, -2] if arr.shape[1] >= 3 else None
    pxy = arr[:, -1]
    start = int(len(pxy) * DISCARD)
    pxy_ss = pxy[start:]
    temp_ss = temp[start:] if temp is not None else None
    eta = -float(np.mean(pxy_ss)) / rate
    n_blocks = len(pxy_ss) // BLOCK_SIZE
    if n_blocks >= 2:
        means = np.array([np.mean(pxy_ss[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]) for i in range(n_blocks)])
        eta_err = float(np.std(-means / rate, ddof=1) / math.sqrt(n_blocks))
    else:
        eta_err = float(np.std(pxy_ss, ddof=1) / math.sqrt(len(pxy_ss)) / rate)
    return {
        "rows": int(len(arr)),
        "gamma_end": float(gamma[-1]) if gamma is not None else None,
        "temp_mean": float(np.mean(temp_ss)) if temp_ss is not None else None,
        "temp_std": float(np.std(temp_ss, ddof=1)) if temp_ss is not None and len(temp_ss) > 1 else None,
        "press_mean": float(np.mean(press[start:])) if press is not None else None,
        "pxy_mean": float(np.mean(pxy_ss)),
        "pxy_std": float(np.std(pxy_ss, ddof=1)),
        "eta": eta,
        "eta_err": eta_err,
        "n_steady": int(len(pxy_ss)),
    }


def block_viscosity(path, rate):
    arr = read_numeric(path)
    minus_pxy = arr[:, -1]
    start = int(len(minus_pxy) * DISCARD)
    eta_blocks = minus_pxy[start:] / rate
    return {
        "eta_block_mean": float(np.mean(eta_blocks)),
        "eta_block_std": float(np.std(eta_blocks, ddof=1)),
        "rows": int(len(arr)),
    }


def fit_power_law(rates, etas, eta_errs=None):
    x = np.log(np.asarray(rates, dtype=float))
    y = np.log(np.asarray(etas, dtype=float))
    if eta_errs is not None:
        sigma_y = np.asarray(eta_errs, dtype=float) / np.asarray(etas, dtype=float)
        w = 1.0 / np.maximum(sigma_y, 1.0e-12) ** 2
        coeff, cov = np.polyfit(x, y, 1, w=np.sqrt(w), cov=True)
    else:
        coeff, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = coeff
    lambda_val = -float(slope)
    lambda_err = float(np.sqrt(cov[0, 0])) if cov.shape == (2, 2) else float("nan")
    eta0 = float(np.exp(intercept))
    pred = np.exp(intercept + slope * x)
    ss_res = float(np.sum((y - np.log(pred)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "eta0": eta0,
        "lambda": lambda_val,
        "lambda_err": lambda_err,
        "lambda_over_4": lambda_val / 4.0,
        "lambda_over_4_err": lambda_err / 4.0,
        "r2_log": r2,
    }


def plot_viscosity(rows, fit, out_png, out_pdf):
    rates = np.array([r["rate"] for r in rows])
    eta = np.array([r["eta"] for r in rows])
    err = np.array([r["eta_err"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.errorbar(rates, eta, yerr=err, fmt="o", capsize=4, color="black", label="server run")
    xs = np.logspace(np.log10(rates.min()), np.log10(rates.max()), 200)
    ys = fit["eta0"] * xs ** (-fit["lambda"])
    ax.plot(xs, ys, color="crimson", lw=1.8, label=r"$\eta\sim\dot\gamma^{-%.3f}$" % fit["lambda"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\dot\gamma$")
    ax.set_ylabel(r"$\eta=-\langle P_{xy}\rangle/\dot\gamma$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    ax.text(
        0.05,
        0.08,
        r"$\lambda/4=%.4f$" % fit["lambda_over_4"],
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_diagnostics(raw_lammps, rows, out_png, out_pdf):
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(rows)))
    for row, color in zip(rows, colors):
        rate = row["rate"]
        rs = "{:.3f}".format(rate)
        arr = read_numeric(os.path.join(raw_lammps, "thermo.shear_{}.dat".format(rs)))
        if arr.shape[1] < 6:
            gamma = np.arange(len(arr), dtype=float)
            temp = arr[:, -3]
            pxy = arr[:, -1]
        else:
            gamma = arr[:, 2]
            temp = arr[:, -3]
            pxy = arr[:, -1]
        start = int(len(arr) * DISCARD)
        label = r"$\dot\gamma={}$".format(rs)
        axes[0].plot(gamma, -pxy, color=color, lw=0.8, alpha=0.75, label=label)
        axes[0].hlines(-row["pxy_mean"], gamma[start], gamma[-1], color=color, lw=1.6)
        axes[1].plot(gamma, temp, color=color, lw=0.8, alpha=0.75)
        axes[1].hlines(row["temp_mean"], gamma[start], gamma[-1], color=color, lw=1.6)
        axes[2].plot(gamma, -pxy / rate, color=color, lw=0.8, alpha=0.75)
        axes[2].hlines(row["eta"], gamma[start], gamma[-1], color=color, lw=1.6)

    for ax in axes:
        ax.axvspan(0.0, 50.0 * DISCARD, color="0.9", alpha=0.7, lw=0)
        ax.grid(True, ls=":", alpha=0.35)
    axes[0].set_ylabel(r"$-P_{xy}$")
    axes[1].set_ylabel(r"$T$")
    axes[2].set_ylabel(r"$\eta_{inst}$")
    axes[2].set_xlabel(r"strain $\gamma$")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(0.45, color="black", lw=0.8, ls="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    raw_lammps = os.path.join(args.raw_dir, "lammps")
    os.makedirs(args.output_dir, exist_ok=True)

    equil = parse_equil(os.path.join(raw_lammps, "log.equilibrate"))
    rows = []
    for rate in RATES:
        rs = "{:.3f}".format(rate)
        thermo = os.path.join(raw_lammps, "thermo.shear_{}.dat".format(rs))
        visc = os.path.join(raw_lammps, "visc_blockave.shear_{}.dat".format(rs))
        log = os.path.join(raw_lammps, "log.shear_{}".format(rs))
        tv = viscosity_from_thermo(thermo, rate)
        bv = block_viscosity(visc, rate)
        lg = parse_log(log)
        row = {
            "rate": rate,
            "dt": DT_BY_RATE[rate],
            "expected_n_pre": int(round(20.0 / (rate * DT_BY_RATE[rate]))),
            "expected_n_prod": int(round(50.0 / (rate * DT_BY_RATE[rate]))),
        }
        row.update(tv)
        row.update(bv)
        row.update({
            "log_dangerous_max": lg["dangerous_max"],
            "log_errors": "; ".join(lg["errors"]),
            "log_dump_every": lg["dump_every"],
            "log_n_prod": lg["n_prod"],
        })
        rows.append(row)

    fit_unweighted = fit_power_law([r["rate"] for r in rows], [r["eta"] for r in rows])
    fit_weighted = fit_power_law([r["rate"] for r in rows], [r["eta"] for r in rows], [r["eta_err"] for r in rows])

    csv_path = os.path.join(args.output_dir, "md3d_highrate_summary.csv")
    fieldnames = [
        "rate", "dt", "rows", "n_steady", "gamma_end", "temp_mean", "temp_std",
        "pxy_mean", "pxy_std", "eta", "eta_err", "eta_block_mean", "eta_block_std",
        "expected_n_pre", "expected_n_prod", "log_n_prod", "log_dump_every",
        "log_dangerous_max", "log_errors",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    json_path = os.path.join(args.output_dir, "md3d_highrate_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "equilibration": equil,
            "rates": rows,
            "fit_unweighted": fit_unweighted,
            "fit_weighted": fit_weighted,
        }, fh, indent=2, sort_keys=True)

    plot_viscosity(
        rows,
        fit_unweighted,
        os.path.join(args.output_dir, "md3d_highrate_viscosity.png"),
        os.path.join(args.output_dir, "md3d_highrate_viscosity.pdf"),
    )
    plot_diagnostics(
        raw_lammps,
        rows,
        os.path.join(args.output_dir, "md3d_highrate_diagnostics.png"),
        os.path.join(args.output_dir, "md3d_highrate_diagnostics.pdf"),
    )

    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(json_path))
    print("unweighted lambda={:.6f}, lambda/4={:.6f}, R2={:.6f}".format(
        fit_unweighted["lambda"], fit_unweighted["lambda_over_4"], fit_unweighted["r2_log"]
    ))
    print("weighted lambda={:.6f}, lambda/4={:.6f}, R2={:.6f}".format(
        fit_weighted["lambda"], fit_weighted["lambda_over_4"], fit_weighted["r2_log"]
    ))


if __name__ == "__main__":
    main()
