"""Fig. 2 and Fig. 6 convective-cluster analysis."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .clusters import demarcate_dynamic_regions
from .geometry import affine_to_reference
from .io import MissingInputError, ensure_dir, load_jumps, load_trajectory, output_paths, save_json
from .paper_constants import BIG_TYPE, analysis_dt, DIMENSION, LC2
from .stress_strain import cluster_stress_sigma_c, local_shear_strain_gamma_c, particle_volumes


def _frame_at_time(times, t):
    return int(np.argmin(np.abs(times - float(t))))


def _window_from_config(cfg, times):
    if cfg.get("window_start_time") is not None and cfg.get("window_end_time") is not None:
        return float(cfg["window_start_time"]), float(cfg["window_end_time"]), "config"
    if cfg.get("t_chi") is not None:
        center = float(cfg.get("t_chi_center", times[len(times) // 2]))
        half = 0.5 * float(cfg["t_chi"])
        return center - half, center + half, "t_chi_center_or_mid_trajectory"
    raise MissingInputError(
        "Need window_start_time/window_end_time or t_chi for the paper t_chi accumulation window"
    )


def _event_times(jumps, times):
    if jumps["jump_times"] is not None:
        return jumps["jump_times"]
    return times[np.clip(jumps["jump_frames"], 0, len(times) - 1)]


def _event_positions(jumps, traj_positions):
    if jumps["jump_positions"] is not None:
        return jumps["jump_positions"][:, : traj_positions.shape[2]]
    frames = np.clip(jumps["jump_frames"], 0, traj_positions.shape[0] - 1)
    return traj_positions[frames, jumps["particle_idx"]]


def prepare_convective_data(cfg):
    system = cfg["system"]
    shear_rate = float(cfg["shear_rate"])
    traj = load_trajectory(cfg["trajectory_npz"])
    jumps = load_jumps(cfg["jumps_npz"])

    positions = np.asarray(traj["positions"], dtype=float)
    times = np.asarray(traj["times"], dtype=float)
    types = np.asarray(traj["types"], dtype=int)
    box = np.asarray(traj["box"], dtype=float)
    dim = positions.shape[2]

    start, end, window_note = _window_from_config(cfg, times)
    t_ref = float(cfg.get("reference_time", 0.5 * (start + end)))
    event_times = _event_times(jumps, times)
    event_pos = _event_positions(jumps, positions)
    in_time = (event_times >= start) & (event_times <= end)
    if cfg.get("type_filter", BIG_TYPE) is not None:
        in_time &= types[jumps["particle_idx"]] == int(cfg.get("type_filter", BIG_TYPE))
    ref_points = affine_to_reference(event_pos[in_time], event_times[in_time], shear_rate, t_ref)
    if len(ref_points) == 0:
        raise RuntimeError("No cage jumps in the selected t_chi window")

    regions = demarcate_dynamic_regions(
        system=system,
        points_ref=ref_points,
        box=box,
        shear_rate=shear_rate,
        gr_first_min=cfg.get("gr_first_min"),
        grid_shape=cfg.get("grid_shape"),
        cluster_id=cfg.get("cluster_id"),
    )
    mask = regions["mask"]

    dt = float(cfg.get("analysis_delta_t", analysis_dt(system, shear_rate)))
    sample_times = np.arange(start, end + 0.5 * dt, dt)
    sample_frames = np.array([_frame_at_time(times, t) for t in sample_times], dtype=int)

    volumes = particle_volumes(types, box, system)
    per_atom_stress = traj["per_atom_stress_xy"] if "per_atom_stress_xy" in traj else None
    if per_atom_stress is not None:
        per_atom_stress = np.asarray(per_atom_stress, dtype=float)
    elif cfg.get("require_exact_stress", True):
        raise MissingInputError(
            "Exact sigma_c requires per_atom_stress_xy or ICE-averaged per_atom_stress_xy"
        )

    sigma_c = []
    gamma_c = []
    ncj = []
    pext = []
    R = []
    member_sets = []

    initial_members = None
    for t, fr in zip(sample_times, sample_frames):
        pos_ref = affine_to_reference(positions[fr], times[fr], shear_rate, t_ref)
        inside = mask.contains(pos_ref)
        members = np.where(inside)[0]
        member_sets.append(members)
        if initial_members is None:
            initial_members = set(members.tolist())
        if per_atom_stress is None:
            sigma_c.append(np.nan)
        else:
            sigma_c.append(cluster_stress_sigma_c(per_atom_stress[fr], volumes, members))

        half = 0.5 * dt
        event_mask = (event_times >= t - half) & (event_times <= t + half)
        evt_ref = affine_to_reference(event_pos[event_mask], event_times[event_mask], shear_rate, t_ref)
        event_inside = mask.contains(evt_ref)
        selected = np.where(event_mask)[0][event_inside]
        ncj.append(float(len(selected)))
        if len(selected):
            vec = jumps["jump_vectors"][selected]
            pext.append(float(np.mean(vec[:, 0] * vec[:, 1] > 0.0)))
        else:
            pext.append(np.nan)

        if initial_members:
            still = len(initial_members.intersection(set(members.tolist())))
            R.append(1.0 - still / float(len(initial_members)))
        else:
            R.append(np.nan)

        df = max(1, int(round(0.5 * dt / np.median(np.diff(times)))))
        fm = max(0, fr - df)
        fp = min(len(times) - 1, fr + df)
        gamma_c.append(local_shear_strain_gamma_c(positions[fm], positions[fp], members))

    event_ref_all = affine_to_reference(event_pos, event_times, shear_rate, t_ref)
    event_in_cluster = mask.contains(event_ref_all)

    out = {
        "system": system,
        "dimension": DIMENSION[system],
        "shear_rate": shear_rate,
        "times": sample_times,
        "sigma_c": np.asarray(sigma_c, dtype=float),
        "gamma_c": np.asarray(gamma_c, dtype=float),
        "ncj": np.asarray(ncj, dtype=float),
        "pext": np.asarray(pext, dtype=float),
        "R": np.asarray(R, dtype=float),
        "event_times": event_times,
        "event_positions_ref": event_ref_all,
        "event_in_cluster": event_in_cluster,
        "event_frames": jumps["jump_frames"],
        "event_particle_idx": jumps["particle_idx"],
        "event_vectors": jumps["jump_vectors"],
        "t_ref": t_ref,
        "window_start": start,
        "window_end": end,
        "window_note": window_note,
        "cluster_radius": mask.equivalent_radius(),
        "cluster_center": mask.center(),
        "rho_th": regions["rho_th"],
        "coarse_length": regions["coarse_length"],
        "selected_cluster_id": regions["selected_cluster_id"],
        "selection_note": regions["selection_note"],
    }
    return out


def characteristic_times(cfg, data):
    char = cfg.get("characteristic_times") or {}
    names = ["t_prime", "t0", "t1", "t2", "t3"]
    if all(name in char and char[name] is not None for name in names):
        return {name: float(char[name]) for name in names}, "config"
    t = data["times"]
    ncj = np.nan_to_num(data["ncj"], nan=0.0)
    sig = data["sigma_c"]
    if np.any(np.isfinite(sig)):
        finite = np.nan_to_num(sig, nan=np.nanmedian(sig[np.isfinite(sig)]))
        peak = int(np.nanargmax(finite))
    else:
        peak = int(np.argmax(ncj))
    t2 = t[peak]
    t1 = t[max(0, peak - max(1, len(t) // 5))]
    t3 = t[min(len(t) - 1, peak + max(1, len(t) // 6))]
    t0 = t[max(0, peak - max(2, len(t) // 3))]
    tp = t[max(0, np.searchsorted(t, t1) - max(1, len(t) // 8))]
    return {"t_prime": float(tp), "t0": float(t0), "t1": float(t1), "t2": float(t2), "t3": float(t3)}, "heuristic"


def plot_convective(cfg, data, figure_name):
    paths = output_paths(cfg, figure_name)
    ensure_dir(paths["root"])
    chars, char_note = characteristic_times(cfg, data)
    t = data["times"]
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.45, 1.0], hspace=0.25, wspace=0.24)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    panels = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(t, data["sigma_c"], color="black", lw=1.4, marker="o", ms=2.5, label=r"$\sigma_c$")
    ax2.plot(t, data["gamma_c"], color="crimson", lw=1.2, marker="^", ms=2.5, label=r"$\gamma_c$")
    ax.set_ylabel(r"$\sigma_c/\sigma_0$")
    ax2.set_ylabel(r"$\gamma_c$", color="crimson")
    ax2.tick_params(colors="crimson")

    axes[1].plot(t, data["ncj"], color="black", lw=1.4, marker="o", ms=2.5)
    axes[1].set_ylabel(r"$n_{cj}$")
    axes[2].plot(t, data["pext"], color="black", lw=1.4, marker="s", ms=2.5)
    axes[2].axhline(0.5, color="0.5", ls="--", lw=1.0)
    axes[2].set_ylabel(r"$P_{ext}$")
    axes[2].set_ylim(0.0, 1.05)
    axes[3].plot(t, data["R"], color="black", lw=1.4, marker="^", ms=2.5)
    axes[3].set_ylabel(r"$R$")
    axes[3].set_xlabel(r"$t/\tau_0$")
    axes[3].set_ylim(-0.02, 1.05)
    for i, a in enumerate(axes):
        a.text(-0.13, 0.92, "({})".format(chr(ord("a") + i)), transform=a.transAxes, fontweight="bold")
        for val in [chars["t0"], chars["t1"], chars["t2"], chars["t3"]]:
            a.axvline(val, color="0.35", ls="--", lw=0.8)
        a.grid(True, alpha=0.25)

    labels = [("e", "t_prime"), ("f", "t1"), ("g", "t2"), ("h", "t3")]
    pts = data["event_positions_ref"]
    in_cluster = data["event_in_cluster"]
    event_times = data["event_times"]
    center = data["cluster_center"]
    radius = data["cluster_radius"]
    dim = min(2, pts.shape[1])
    pad = max(1.5 * radius, 5.0)
    for axp, (letter, key) in zip(panels, labels):
        cutoff = chars[key]
        sel = in_cluster & (event_times <= cutoff)
        axp.set_facecolor("black")
        if np.any(sel):
            axp.scatter(pts[sel, 0], pts[sel, 1], s=5, c="red", lw=0, alpha=0.75)
        axp.set_xlim(center[0] - pad, center[0] + pad)
        axp.set_ylim(center[1] - pad, center[1] + pad)
        axp.set_aspect("equal")
        axp.text(0.04, 0.92, "({}) {}".format(letter, key), transform=axp.transAxes,
                 color="white", fontweight="bold")
        axp.set_xlabel(r"$x/d_0$")
        axp.set_ylabel(r"$y/d_0$")

    title = "{} convective cluster, gdot={}".format(data["system"], data["shear_rate"])
    fig.suptitle(title, fontsize=13)
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    np.savez(paths["npz"], **data)
    metadata = {
        "figure": figure_name,
        "system": data["system"],
        "shear_rate": data["shear_rate"],
        "characteristic_times": chars,
        "characteristic_time_source": char_note,
        "window_note": data["window_note"],
        "selected_cluster_id": int(data["selected_cluster_id"]),
        "selection_note": data["selection_note"],
        "cluster_radius": float(data["cluster_radius"]),
        "rho_th": float(data["rho_th"]),
        "coarse_length": float(data["coarse_length"]),
        "lc2": LC2[data["system"]],
    }
    save_json(paths["metadata"], metadata)
    return paths


def run_convective_figure(cfg, figure_name):
    data = prepare_convective_data(cfg)
    return plot_convective(cfg, data, figure_name)
