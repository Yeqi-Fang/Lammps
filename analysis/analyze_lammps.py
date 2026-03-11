#!/usr/bin/env python3
import argparse
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


# ----------------------------
# 1) 读取 LAMMPS lammpstrj
# ----------------------------
def iter_lammpstrj(path, need_cols=("id", "x", "y"), verbose=False):
    """
    Yields: (timestep:int, h: (dim,dim) box matrix, bounds: (dim,2) bounds, data: dict of np arrays)
    Supports orthorhombic and triclinic (BOX BOUNDS ... xy xz yz).
    """
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(f.readline().strip())

            # NUMBER OF ATOMS
            _ = f.readline()
            n = int(f.readline().strip())

            # BOX BOUNDS
            box_line = f.readline().strip()
            if not box_line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: missing BOX BOUNDS.")

            triclinic = ("xy" in box_line) or ("xz" in box_line) or ("yz" in box_line)

            # read 3 lines always in dump; for 2D you still often get 3 with thin z
            bounds_raw = []
            tilts = [0.0, 0.0, 0.0]  # xy, xz, yz
            for i in range(3):
                parts = f.readline().split()
                if triclinic:
                    lo, hi, tilt = map(float, parts[:3])
                    bounds_raw.append((lo, hi))
                    tilts[i] = tilt  # xline->xy, yline->xz, zline->yz in LAMMPS dump convention
                else:
                    lo, hi = map(float, parts[:2])
                    bounds_raw.append((lo, hi))

            xlo, xhi = bounds_raw[0]
            ylo, yhi = bounds_raw[1]
            zlo, zhi = bounds_raw[2]
            lx = xhi - xlo
            ly = yhi - ylo
            lz = zhi - zlo

            if triclinic:
                xy = tilts[0]
                xz = tilts[1]
                yz = tilts[2]
                h = np.array([[lx, xy, xz],
                              [0.0, ly, yz],
                              [0.0, 0.0, lz]], dtype=float)
            else:
                h = np.array([[lx, 0.0, 0.0],
                              [0.0, ly, 0.0],
                              [0.0, 0.0, lz]], dtype=float)

            bounds = np.array([[xlo, xhi],
                               [ylo, yhi],
                               [zlo, zhi]], dtype=float)

            # ATOMS header
            atoms_line = f.readline().strip()
            if not atoms_line.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: missing ATOMS.")

            cols = atoms_line.split()[2:]
            col_index = {c: i for i, c in enumerate(cols)}

            missing = [c for c in need_cols if c not in col_index]
            if missing:
                raise ValueError(f"Dump missing required columns: {missing}. "
                                 f"Available: {cols}")

            # read atom lines
            arr = np.empty((n, len(cols)), dtype=float)
            for i in range(n):
                arr[i] = np.fromstring(f.readline(), sep=" ")

            # sort by id to ensure consistent particle ordering
            if "id" in col_index:
                ids = arr[:, col_index["id"]].astype(np.int64)
                order = np.argsort(ids)
                arr = arr[order]
            else:
                ids = None

            data = {c: arr[:, col_index[c]] for c in cols}

            if verbose:
                print(f"Read step={step} N={n} triclinic={triclinic}")

            yield step, h, bounds, data


def get_dim(dim_arg, h):
    if dim_arg is not None:
        return dim_arg
    # auto: if lz is tiny, treat as 2D
    lz = h[2, 2]
    return 2 if lz < 1e-8 else 3


# ----------------------------
# 2) msqt: transverse MSD
# ----------------------------
def compute_msqt(dump_path, direction="y", max_lag=500, dt_frame=1.0, dim=None,
                 com_remove=False, use_unwrapped=True):
    """
    msqt(lag) = < (r_dir(t+lag)-r_dir(t))^2 > averaged over particles and time origins.
    direction: 'y' or 'z' (transverse directions are safest under shear).
    """
    dir_map = {"x": 0, "y": 1, "z": 2}
    if direction not in dir_map:
        raise ValueError("direction must be one of 'x','y','z'.")

    # prefer unwrapped
    coord_name = f"{direction}u" if use_unwrapped else direction
    need_cols = ("id", coord_name)

    sums = np.zeros(max_lag + 1, dtype=float)
    counts = np.zeros(max_lag + 1, dtype=np.int64)

    ring = [None] * (max_lag + 1)
    frame_idx = -1
    steps = []

    for step, h, bounds, data in iter_lammpstrj(dump_path, need_cols=need_cols):
        frame_idx += 1
        steps.append(step)

        r = data[coord_name].copy()

        if com_remove:
            r -= r.mean()

        ring[frame_idx % (max_lag + 1)] = r

        # accumulate
        max_l = min(max_lag, frame_idx)
        for lag in range(1, max_l + 1):
            r_prev = ring[(frame_idx - lag) % (max_lag + 1)]
            dr = r - r_prev
            sums[lag] += np.mean(dr * dr)
            counts[lag] += 1

    lags = np.arange(max_lag + 1)
    msqt = np.full_like(lags, np.nan, dtype=float)
    msqt[0] = 0.0
    valid = counts > 0
    msqt[valid] = sums[valid] / counts[valid]

    t = lags * dt_frame
    return t, msqt


# ----------------------------
# 3) g(r): RDF  (supports triclinic via fractional KDTree)
# ----------------------------
def frac_coords_from_cart(pos, h, bounds):
    """
    Convert Cartesian pos (N,dim) to fractional coords s in [0,1)^dim using h and bounds.
    Use bounds lo as origin (origin shift doesn't affect ds, but helps keep s reasonable).
    """
    dim = pos.shape[1]
    origin = bounds[:dim, 0]
    hinv = np.linalg.inv(h[:dim, :dim])
    s = (pos - origin) @ hinv.T
    s = s - np.floor(s)  # wrap to [0,1)
    return s


def compute_gr(dump_path, rmax, bins=200, dim=None, average_frames=None):
    """
    Compute g(r) by accumulating pair-distance histogram across frames.
    Uses periodic KDTree in fractional space (boxsize=1) then exact real distance check with h.
    """
    if cKDTree is None:
        raise ImportError("scipy is required for fast g(r). Install scipy or compute RDF in LAMMPS.")

    need_cols = ("id", "x", "y", "z")
    hist = np.zeros(bins, dtype=np.float64)
    nframes = 0
    last_N = None
    last_vol = None

    edges = np.linspace(0.0, rmax, bins + 1)
    dr = edges[1] - edges[0]
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    for step, h, bounds, data in iter_lammpstrj(dump_path, need_cols=need_cols):
        this_dim = get_dim(dim, h)
        pos = np.stack([data["x"], data["y"], data["z"]], axis=1)[:, :this_dim]

        # fractional coordinates
        s = frac_coords_from_cart(pos, h, bounds)

        # safe fractional search radius
        # If real dist <= rmax => ||ds|| <= rmax / sigma_min(h)
        singvals = np.linalg.svd(h[:this_dim, :this_dim], compute_uv=False)
        sigma_min = np.min(singvals)
        r_frac = rmax / sigma_min

        tree = cKDTree(s, boxsize=1.0)
        coo = tree.sparse_distance_matrix(tree, r_frac, output_type="coo_matrix")

        i = coo.row
        j = coo.col
        mask = (i < j)  # keep unique pairs
        i = i[mask]
        j = j[mask]

        ds = s[j] - s[i]
        ds -= np.rint(ds)  # minimum image in fractional
        # real displacement
        dr_vec = ds @ h[:this_dim, :this_dim]
        dist = np.linalg.norm(dr_vec, axis=1)

        dist = dist[dist < rmax]
        htmp, _ = np.histogram(dist, bins=edges)
        hist += htmp
        nframes += 1

        N = pos.shape[0]
        vol = abs(np.linalg.det(h[:this_dim, :this_dim]))
        last_N = N
        last_vol = vol

        if average_frames is not None and nframes >= average_frames:
            break

    if nframes == 0:
        raise RuntimeError("No frames read from dump.")

    rho = last_N / last_vol

    if get_dim(dim, h) == 2:
        shell = 2.0 * np.pi * r_centers * dr
    else:
        shell = 4.0 * np.pi * (r_centers ** 2) * dr

    # hist counts unique pairs (i<j)
    g = (2.0 * hist) / (nframes * last_N * rho * shell)
    return r_centers, g


# ----------------------------
# 4) viscosity from shear stress time series
# ----------------------------
def load_two_col(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError("Need at least 2 columns: time/step and value.")
    return arr[:, 0], arr[:, 1]


def viscosity_nemd(pxy_path, gamma_dot, skip=0.3):
    """
    eta = - <Pxy> / gamma_dot using steady-state average.
    skip: fraction of initial samples to discard as transient.
    """
    t, pxy = load_two_col(pxy_path)
    n = len(pxy)
    i0 = int(np.floor(skip * n))
    pxy_ss = pxy[i0:]
    avg = np.mean(pxy_ss)
    eta = -avg / gamma_dot
    return eta, avg, (t[i0], t[-1])


def autocorr_fft(x):
    """
    Unbiased autocorrelation using FFT.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    denom = np.arange(n, 0, -1, dtype=float)  # unbiased
    return acf / denom


def viscosity_gk(pxy_path, dt, volume, temp, kB=1.0, tmax=None):
    """
    Green-Kubo: eta = V/(kB*T) * integral_0^{tmax} <Pxy(0)Pxy(t)> dt
    """
    t, pxy = load_two_col(pxy_path)
    acf = autocorr_fft(pxy)
    if tmax is not None:
        m = int(np.floor(tmax / dt))
        acf = acf[:m]
    eta = volume / (kB * temp) * np.trapz(acf, dx=dt)
    return eta


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, required=True, help="LAMMPS dump (lammpstrj)")
    ap.add_argument("--dim", type=int, default=None, choices=[2, 3], help="2 or 3. default:auto")
    ap.add_argument("--dt_frame", type=float, default=1.0, help="time between saved frames (same units as LAMMPS time)")
    ap.add_argument("--max_lag", type=int, default=500)
    ap.add_argument("--msqt_dir", type=str, default="y", choices=["x", "y", "z"])
    ap.add_argument("--rmax", type=float, required=True)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--out", type=str, default="analysis_out.npz")

    # viscosity (optional)
    ap.add_argument("--pxy", type=str, default=None, help="2-col file: time/step  Pxy")
    ap.add_argument("--gamma_dot", type=float, default=None, help="shear rate")
    ap.add_argument("--skip", type=float, default=0.3, help="fraction to skip for steady-state average")

    args = ap.parse_args()

    # msqt
    t_msqt, msqt = compute_msqt(
        args.dump, direction=args.msqt_dir, max_lag=args.max_lag,
        dt_frame=args.dt_frame, dim=args.dim, com_remove=False, use_unwrapped=True
    )

    # g(r)
    r, g = compute_gr(args.dump, rmax=args.rmax, bins=args.bins, dim=args.dim)

    results = {
        "t_msqt": t_msqt,
        "msqt": msqt,
        "r": r,
        "g_r": g,
    }

    # viscosity
    if args.pxy is not None and args.gamma_dot is not None:
        eta, avg_pxy, (t0, t1) = viscosity_nemd(args.pxy, args.gamma_dot, skip=args.skip)
        results["eta_nemd"] = np.array([eta])
        results["avg_pxy_ss"] = np.array([avg_pxy])
        results["steady_window"] = np.array([t0, t1])

        print(f"[viscosity NEMD] eta = {eta:.6g}  (avg Pxy={avg_pxy:.6g}, window {t0} -> {t1})")

    np.savez(args.out, **results)
    print(f"Saved: {args.out}")
    print(f"msqt: {len(msqt)} points, g(r): {len(g)} bins")


if __name__ == "__main__":
    main()