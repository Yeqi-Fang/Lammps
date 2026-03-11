"""
compute_gr.py  —  Radial Distribution Function g(r) for KA binary LJ
Uses scipy.spatial for fast vectorized distance computation.

Usage:
    python compute_gr.py --dump dump.equil_coarse.lammpstrj --n_frames 50
    python compute_gr.py --dump dump.shear_0.015.lammpstrj  --n_frames 50
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, re
from scipy.spatial.distance import pdist, cdist

# ─── parser ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dump",     required=True)
parser.add_argument("--n_frames", type=int,   default=50)
parser.add_argument("--n_bins",   type=int,   default=300)
parser.add_argument("--rmax",     type=float, default=None)
parser.add_argument("--out",      default=None)
args = parser.parse_args()

# ─── read dump ────────────────────────────────────────────────────────────────
def read_dump(fname):
    frames = []
    with open(fname) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            ts = int(f.readline())
            f.readline()
            N  = int(f.readline())
            f.readline()
            bx = list(map(float, f.readline().split()))
            by = list(map(float, f.readline().split()))
            bz = list(map(float, f.readline().split()))
            L  = np.array([bx[1]-bx[0], by[1]-by[0], bz[1]-bz[0]])
            header = f.readline().split()[2:]
            data   = np.array([f.readline().split() for _ in range(N)], dtype=float)
            col    = {h: i for i, h in enumerate(header)}
            types  = data[:, col['type']].astype(int)
            xyz    = data[:, [col['x'], col['y'], col['z']]]
            frames.append({'ts': ts, 'L': L, 'types': types, 'xyz': xyz})
    return frames

print(f"Reading {args.dump} ...")
all_frames = read_dump(args.dump)
print(f"  Total frames: {len(all_frames)}")

idx    = np.linspace(0, len(all_frames)-1, min(args.n_frames, len(all_frames)), dtype=int)
frames = [all_frames[i] for i in idx]
print(f"  Using {len(frames)} frames")

# ─── MIC distance helpers ─────────────────────────────────────────────────────
def mic_pdist(pos, L):
    """Same-species all-pair distances with MIC, fully vectorized via scipy."""
    pos = pos % L
    dx = pdist(pos[:, 0:1]); dx -= L[0] * np.round(dx / L[0])
    dy = pdist(pos[:, 1:2]); dy -= L[1] * np.round(dy / L[1])
    dz = pdist(pos[:, 2:3]); dz -= L[2] * np.round(dz / L[2])
    return np.sqrt(dx**2 + dy**2 + dz**2)

def mic_cdist(pos_a, pos_b, L):
    """Cross-species all-pair distances with MIC."""
    pos_a = pos_a % L; pos_b = pos_b % L
    dx = cdist(pos_a[:, 0:1], pos_b[:, 0:1]); dx -= L[0] * np.round(dx / L[0])
    dy = cdist(pos_a[:, 1:2], pos_b[:, 1:2]); dy -= L[1] * np.round(dy / L[1])
    dz = cdist(pos_a[:, 2:3], pos_b[:, 2:3]); dz -= L[2] * np.round(dz / L[2])
    return np.sqrt(dx**2 + dy**2 + dz**2).ravel()

# ─── g(r) per pair ────────────────────────────────────────────────────────────
def compute_gr(frames, t1, t2, n_bins, rmax):
    all_hist = np.zeros(n_bins)
    Na_tot = Nb_tot = nf = 0

    for i, fr in enumerate(frames):
        L   = fr['L']
        _rmax = rmax or np.min(L) / 2.0
        pos_a = fr['xyz'][fr['types'] == t1]
        pos_b = fr['xyz'][fr['types'] == t2]
        if len(pos_a) == 0 or len(pos_b) == 0:
            continue

        r = mic_pdist(pos_a, L) if t1 == t2 else mic_cdist(pos_a, pos_b, L)

        hist, _ = np.histogram(r, bins=n_bins, range=(0, _rmax))
        all_hist += hist * (2 if t1 == t2 else 1)
        Na_tot += len(pos_a); Nb_tot += len(pos_b); nf += 1
        print(f"    frame {i+1}/{len(frames)}", end='\r')

    print()
    if nf == 0:
        return None, None

    L      = frames[0]['L']
    _rmax  = rmax or np.min(L) / 2.0
    r_edges= np.linspace(0, _rmax, n_bins + 1)
    r_mid  = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr_bin = r_edges[1] - r_edges[0]

    rho_b  = (Nb_tot / nf) / np.prod(L)
    ideal  = (Na_tot / nf) * rho_b * 4 * np.pi * r_mid**2 * dr_bin
    gr     = (all_hist / nf) / np.where(ideal > 0, ideal, 1)
    return r_mid, gr

# ─── compute all pairs ────────────────────────────────────────────────────────
rmax_val  = args.rmax or np.min(frames[0]['L']) / 2.0
type_list = np.unique(frames[0]['types']).astype(int)
labels    = {1: 'A', 2: 'B'}
results   = {}

for t1 in type_list:
    for t2 in type_list:
        if t2 < t1:
            continue
        key = f"{labels.get(t1,t1)}{labels.get(t2,t2)}"
        print(f"Computing g_{key}(r) ...")
        r, gr = compute_gr(frames, t1, t2, args.n_bins, rmax_val)
        if r is not None:
            r_gt = r[r > 0.5]
            gr_gt = gr[r > 0.5]
            if len(gr_gt) > 0:
                print(f"  first peak at r = {r_gt[np.argmax(gr_gt[:50])]:.3f}")
            results[key] = (r, gr)

# number-weighted total
N_tot = len(frames[0]['types'])
x_A = np.sum(frames[0]['types'] == 1) / N_tot
x_B = np.sum(frames[0]['types'] == 2) / N_tot
if all(k in results for k in ('AA', 'AB', 'BB')):
    r_ref  = results['AA'][0]
    gr_tot = (x_A**2       * results['AA'][1] +
              2*x_A*x_B    * results['AB'][1] +
              x_B**2       * results['BB'][1])
    results['total'] = (r_ref, gr_tot)

# ─── plot ─────────────────────────────────────────────────────────────────────
colors = {'AA': '#1f77b4', 'AB': '#ff7f0e', 'BB': '#2ca02c', 'total': 'k'}
ls_map = {'AA': '-',       'AB': '--',       'BB': '-.',      'total': '-'}
lw_map = {'AA': 1.5,       'AB': 1.5,        'BB': 1.5,       'total': 2.2}

XMAX = 5.0   # fixed x-axis upper limit

fig, ax = plt.subplots(figsize=(7, 4.5))
peak_annot_offset = {}   # track vertical offsets to avoid label overlap

for key in ['total', 'AA', 'AB', 'BB']:
    if key not in results:
        continue
    r, gr = results[key]
    label = r'$g(r)$' if key == 'total' else rf'$g_{{\rm {key}}}(r)$'
    ax.plot(r, gr, color=colors[key], ls=ls_map[key], lw=lw_map[key], label=label)

    if key == 'total':
        continue   # annotate only the partial g(r) curves

    # ── find all peaks within [0.5, XMAX] ───────────────────────────────────
    from scipy.signal import find_peaks
    mask = (r >= 0.5) & (r <= XMAX)
    r_m, gr_m = r[mask], gr[mask]
    # prominence > 0.1 avoids marking tiny wiggles
    peak_idx, props = find_peaks(gr_m, prominence=0.1, distance=5)

    for pi in peak_idx:
        r_pk  = r_m[pi]
        gr_pk = gr_m[pi]

        # vertical dashed line to peak
        ax.axvline(r_pk, color=colors[key], lw=0.7, ls=':', alpha=0.6)

        # label: slightly above the peak, alternate offset to avoid overlap
        y_offset = gr_pk + 0.08
        ax.annotate(
            f'{r_pk:.2f}',
            xy=(r_pk, gr_pk),
            xytext=(r_pk + 0.03, y_offset),
            fontsize=8,
            color=colors[key],
            ha='left',
            arrowprops=dict(arrowstyle='-', color=colors[key],
                            lw=0.8, alpha=0.5),
        )

ax.axhline(1.0, color='gray', lw=0.8, ls=':')
ax.set_xlabel(r'$r\ [\sigma]$', fontsize=13)
ax.set_ylabel(r'$g(r)$',        fontsize=13)
ax.set_xlim(0, XMAX)
ax.set_ylim(0)

fname = os.path.basename(args.dump)
m = re.search(r'(\d+\.\d+)', fname)
if 'equil' in fname:
    title = r'$g(r)$   KA binary LJ,  $\dot\gamma=0$,  $T=0.45$'
elif m:
    title = rf'$g(r)$   KA binary LJ,  $\dot\gamma={m.group(1)}$,  $T=0.45$'
else:
    title = r'$g(r)$   KA binary LJ'
ax.set_title(title, fontsize=12)
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
plt.tight_layout()

outname = args.out or fname.replace('.lammpstrj', '_gr.png')
plt.savefig(outname, dpi=150)
print(f"Saved -> {outname}")
plt.show()