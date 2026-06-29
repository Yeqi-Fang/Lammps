"""Fig. 4 PHM and cage-jump pair-correlation workflow."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .convective import prepare_convective_data, characteristic_times
from .geometry import minimum_image
from .io import MissingInputError, ensure_dir, load_config, output_paths, save_json


def load_phm_modes(path):
    if not path:
        raise MissingInputError("Fig. 4(a) requires phm_npz with PHM modes")
    data = np.load(path, allow_pickle=True)
    for key in ["phm_eigenvectors", "phm_eigenvalues", "phm_particle_ids"]:
        if key not in data:
            raise MissingInputError("PHM file missing {}".format(key))
    return data


def pair_correlation_cage_jumps(data, chars, cfg):
    dim = int(data["dimension"])
    box = np.asarray(cfg.get("box_for_correlation", [np.inf] * dim), dtype=float)
    event_times = data["event_times"]
    pos = data["event_positions_ref"][:, :dim]
    in_cluster = data["event_in_cluster"]
    vectors = data["event_vectors"][:, :dim]
    t1 = float(chars["t1"])
    t2 = float(chars["t2"])
    dt = float(cfg.get("analysis_delta_t", (t2 - t1) / 10.0))
    ref_sel = in_cluster & (event_times >= t1 - 0.5 * dt) & (event_times <= t1 + 0.5 * dt)
    tar_sel = (event_times >= t1) & (event_times <= t2)
    ref = pos[ref_sel]
    tar = pos[tar_sel]
    if len(ref) == 0 or len(tar) == 0:
        raise RuntimeError("Not enough cage jumps for Fig. 4 pair correlation")

    dr_list = []
    for r0 in ref:
        dr = tar - r0
        if np.all(np.isfinite(box)):
            dr = minimum_image(dr, box)
        dr_list.append(dr)
    dr = np.vstack(dr_list)
    rr = np.linalg.norm(dr, axis=1)
    rmax = float(cfg.get("r_max", np.nanpercentile(rr, 99.0)))
    nbins = int(cfg.get("r_bins", 80))
    bins = np.linspace(0.0, rmax, nbins + 1)
    counts, _ = np.histogram(rr, bins=bins)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    if dim == 2:
        shell = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    else:
        shell = 4.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3) / 3.0
    gcj = counts / np.maximum(shell * len(ref), 1.0)

    ntheta = int(cfg.get("angular_bins", 72))
    alpha = np.zeros_like(rmid)
    if dim >= 2:
        theta = np.arctan2(dr[:, 1], dr[:, 0])
        for i in range(nbins):
            m = (rr >= bins[i]) & (rr < bins[i + 1])
            if np.sum(m) < 4 or gcj[i] <= 0:
                alpha[i] = np.nan
                continue
            hist, _ = np.histogram(theta[m], bins=ntheta, range=(-np.pi, np.pi))
            local = hist.astype(float) / max(np.mean(hist), 1.0e-12)
            alpha[i] = float(np.mean((local - 1.0) ** 2))
    xi_alpha = cfg.get("xi_alpha")
    xi_note = "config"
    if xi_alpha is None:
        xi_alpha = estimate_xi_alpha(rmid, alpha)
        xi_note = "automatic_log_slope_change"
    return {
        "r": rmid,
        "gcj": gcj,
        "alpha": alpha,
        "xi_alpha": float(xi_alpha) if xi_alpha is not None else np.nan,
        "xi_alpha_note": xi_note,
        "n_ref": int(len(ref)),
        "n_target": int(len(tar)),
    }


def estimate_xi_alpha(r, alpha):
    r = np.asarray(r, dtype=float)
    a = np.asarray(alpha, dtype=float)
    ok = np.isfinite(a) & (a > 0) & (r > 0)
    if np.sum(ok) < 8:
        return np.nan
    lr = np.log(r[ok])
    la = np.log(a[ok])
    slope = np.gradient(la, lr)
    ds = np.gradient(slope, lr)
    idx = int(np.nanargmin(ds))
    return float(r[ok][idx])


def plot_fig4(cfg):
    data = prepare_convective_data(cfg)
    chars, char_note = characteristic_times(cfg, data)
    corr = pair_correlation_cage_jumps(data, chars, cfg)
    paths = output_paths(cfg, "fig4_{}".format(cfg.get("name", data["system"]).lower()))
    ensure_dir(paths["root"])

    phm = None
    phm_note = "missing"
    if cfg.get("phm_npz"):
        phm = load_phm_modes(cfg["phm_npz"])
        phm_note = "loaded"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.set_title("(a) PHM projection")
    if phm is None:
        ax.text(0.5, 0.5, "PHM input required", ha="center", va="center", transform=ax.transAxes)
    else:
        ids = np.asarray(phm["phm_particle_ids"], dtype=int)
        eigvec = np.asarray(phm["phm_eigenvectors"], dtype=float)
        mode_index = int(cfg.get("phm_mode_index", 0))
        vec = eigvec[mode_index].reshape((-1, data["dimension"]))
        pos0 = data["event_positions_ref"][: len(ids), :2]
        mag = np.linalg.norm(vec[:, :2], axis=1)
        keep = mag >= np.percentile(mag, float(cfg.get("phm_percentile", 80)))
        ax.quiver(pos0[keep, 0], pos0[keep, 1], vec[keep, 0], vec[keep, 1], color="black")
    early = (data["event_times"] >= chars["t0"]) & (data["event_times"] <= chars["t1"]) & data["event_in_cluster"]
    pts = data["event_positions_ref"][early]
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c="red", lw=0, alpha=0.75)
    ax.set_aspect("equal")

    axes[0, 1].plot(corr["r"], corr["gcj"], color="black")
    axes[0, 1].set_title(r"(b) orientation-averaged $g_{cj}(r)$")
    axes[0, 1].set_xlabel(r"$r/d_0$")
    axes[0, 1].set_ylabel(r"$g_{cj}$")
    axes[1, 0].plot(corr["r"], corr["alpha"], color="black")
    if np.isfinite(corr["xi_alpha"]):
        axes[1, 0].axvline(corr["xi_alpha"], color="crimson", ls="--")
    axes[1, 0].set_title(r"(c) anisotropic factor $\alpha(r)$")
    axes[1, 0].set_xlabel(r"$r/d_0$")
    axes[1, 0].set_ylabel(r"$\alpha$")
    axes[1, 1].scatter([data["shear_rate"]], [corr["xi_alpha"]], label=r"$\xi_\alpha$")
    axes[1, 1].scatter([data["shear_rate"]], [data["cluster_radius"]], label=r"$\xi_c$")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(r"(d) $\xi_\alpha$ and $\xi_c$")
    axes[1, 1].set_xlabel(r"$\dot\gamma\tau_0$")
    axes[1, 1].legend()
    for a in axes.ravel():
        a.grid(True, alpha=0.25)
    fig.suptitle("Fig. 4 preparation: {}".format(data["system"]))
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    np.savez(paths["npz"], **corr)
    save_json(paths["metadata"], {
        "characteristic_times": chars,
        "characteristic_time_source": char_note,
        "phm_note": phm_note,
        "xi_alpha": corr["xi_alpha"],
        "xi_alpha_note": corr["xi_alpha_note"],
        "n_ref": corr["n_ref"],
        "n_target": corr["n_target"],
    })
    return paths
