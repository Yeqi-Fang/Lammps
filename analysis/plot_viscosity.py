"""
plot_viscosity.py  —  Shear thinning: η vs γ̇ for KA binary LJ (SLLOD)
Extracts <P_xy> from LAMMPS log files, computes η = -<P_xy> / γ_dot.

Usage:
    python plot_viscosity.py

Edit LOG_FILES below to match your actual log file names/paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
# Each entry: (shear_rate, log_file_path)
LOG_FILES = [
    (0.003, "../lammps/log.shear_0.003"),
    (0.005, "../lammps/log.shear_0.005"),
    (0.015, "../lammps/log.shear_0.015"),
    (0.030, "../lammps/log.shear_0.030"),
    (0.060, "../lammps/log.shear_0.060"),
]

# Column name for shear stress in thermo output
# If you use "pxy" in thermo_style, set PXY_COL = "pxy"
# If you use "press" and compute Pdeform, adjust accordingly
PXY_COL = "c_Pdeform[4]"

# Fraction of run to skip as equilibration (e.g. 0.3 = skip first 30%)
SKIP_FRAC = 0.3
# ──────────────────────────────────────────────────────────────────────────────


def parse_log(fname, col):
    """Extract a column from LAMMPS thermo output in a log file."""
    data = []
    header = None
    col_idx = None
    in_run = False

    with open(fname, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Detect thermo header line
            if line.startswith("Step") or line.startswith("   Step"):
                cols = line.split()
                if col in cols:
                    col_idx = cols.index(col)
                    header = cols
                    in_run = True
                    continue

            if in_run and col_idx is not None:
                # Stop at end of run block
                if line.startswith("Loop time") or line.startswith("ERROR"):
                    in_run = False
                    continue
                parts = line.split()
                if len(parts) == len(header):
                    try:
                        data.append(float(parts[col_idx]))
                    except ValueError:
                        pass

    return np.array(data)


# ─── Extract η for each shear rate ───────────────────────────────────────────
rates = []
eta   = []
eta_err = []

for (gdot, logfile) in LOG_FILES:
    try:
        pxy = parse_log(logfile, PXY_COL)
    except FileNotFoundError:
        print(f"  WARNING: {logfile} not found, skipping.")
        continue

    if len(pxy) == 0:
        print(f"  WARNING: no '{PXY_COL}' column found in {logfile}, skipping.")
        print(f"           Check thermo_style — try PXY_COL = 'v_pxy' or 'c_myStress[4]'")
        continue

    # Skip equilibration
    n_skip = int(len(pxy) * SKIP_FRAC)
    pxy_prod = pxy[n_skip:]

    eta_val = -np.mean(pxy_prod) / gdot
    eta_std = sem(pxy_prod) / gdot   # standard error of mean

    rates.append(gdot)
    eta.append(eta_val)
    eta_err.append(eta_std)
    print(f"  γ̇ = {gdot:.3f}  <Pxy> = {np.mean(pxy_prod):.4f}  η = {eta_val:.4f} ± {eta_std:.4f}")

rates   = np.array(rates)
eta     = np.array(eta)
eta_err = np.array(eta_err)

if len(rates) < 2:
    print("\nNeed at least 2 data points to plot. Check log file paths.")
    exit()

# ─── Power-law fit: η = A * γ̇^(-λ)  ←→  log η = log A - λ log γ̇ ───────────
log_g = np.log10(rates)
log_e = np.log10(eta)
coeffs = np.polyfit(log_g, log_e, 1)
lam   = -coeffs[0]        # shear-thinning exponent (positive)
log_A = coeffs[1]
A     = 10**log_A

g_fit = np.logspace(np.log10(rates.min()) - 0.2,
                    np.log10(rates.max()) + 0.2, 100)
eta_fit = A * g_fit**(-lam)

print(f"\nPower-law fit: η = {A:.3f} × γ̇^(-{lam:.3f})")
print(f"  (Paper Zeng et al. 2025 reports λ ≈ 0.182 for 3D KA at T=0.45)")

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.errorbar(rates, eta, yerr=eta_err,
            fmt='o', color='#2c7bb6', ms=7, capsize=4, lw=1.5,
            label='MD data')

ax.plot(g_fit, eta_fit, '--', color='#d7191c', lw=1.8,
        label=rf'fit: $\eta \propto \dot\gamma^{{-{lam:.3f}}}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\dot\gamma\ [\tau_0^{-1}]$', fontsize=13)
ax.set_ylabel(r'$\eta$',                        fontsize=13)
ax.set_title(r'Shear thinning — KA binary LJ, $T=0.45$', fontsize=12)
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('viscosity_vs_shearrate.png', dpi=150)
print("\nSaved → viscosity_vs_shearrate.png")
plt.show()
