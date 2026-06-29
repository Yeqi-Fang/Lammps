"""I/O and schema validation."""

import json
import os

import numpy as np


class MissingInputError(RuntimeError):
    pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_npz_required(path, required_keys, label):
    if not path:
        raise MissingInputError("{} path is empty".format(label))
    if not os.path.exists(path):
        raise MissingInputError("{} not found: {}".format(label, path))
    data = np.load(path, allow_pickle=True)
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise MissingInputError("{} is missing keys: {}".format(path, ", ".join(missing)))
    return data


def get_array(data, primary, fallbacks=None, default=None):
    names = [primary] + list(fallbacks or [])
    for name in names:
        if name in data:
            return data[name]
    if default is not None:
        return default
    raise MissingInputError("Missing array; tried {}".format(names))


def load_trajectory(path):
    required = ["positions", "times", "types", "box"]
    data = load_npz_required(path, required, "trajectory NPZ")
    positions = np.asarray(data["positions"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    types = np.asarray(data["types"], dtype=int)
    box = np.asarray(data["box"], dtype=float)
    if positions.ndim != 3:
        raise ValueError("positions must have shape (frames, particles, dim)")
    if len(times) != positions.shape[0]:
        raise ValueError("times length does not match positions frames")
    if len(types) != positions.shape[1]:
        raise ValueError("types length does not match positions particles")
    if len(box) != positions.shape[2]:
        raise ValueError("box dimension does not match positions dim")
    return data


def load_jumps(path):
    data = load_npz_required(path, ["jump_frames", "particle_idx", "jump_vectors"], "cage-jump NPZ")
    jump_frames = np.asarray(data["jump_frames"], dtype=int)
    particle_idx = np.asarray(data["particle_idx"], dtype=int)
    jump_vectors = np.asarray(data["jump_vectors"], dtype=float)
    if "jump_times" in data:
        jump_times = np.asarray(data["jump_times"], dtype=float)
    else:
        jump_times = None
    if "jump_positions" in data:
        jump_positions = np.asarray(data["jump_positions"], dtype=float)
    elif "jump_pos_xy" in data:
        xy = np.asarray(data["jump_pos_xy"], dtype=float)
        jump_positions = np.zeros((len(xy), 3), dtype=float)
        jump_positions[:, :2] = xy[:, :2]
    elif "positions" in data and np.asarray(data["positions"]).ndim == 2:
        jump_positions = np.asarray(data["positions"], dtype=float)
    else:
        jump_positions = None
    if not (len(jump_frames) == len(particle_idx) == len(jump_vectors)):
        raise ValueError("jump arrays have inconsistent lengths")
    return {
        "raw": data,
        "jump_frames": jump_frames,
        "particle_idx": particle_idx,
        "jump_vectors": jump_vectors,
        "jump_times": jump_times,
        "jump_positions": jump_positions,
    }


def output_paths(cfg, figure_name):
    root = cfg.get("output_dir")
    if not root:
        root = os.path.join(os.path.dirname(cfg["_config_path"]), "..", "..", "data", "figures")
    root = os.path.abspath(root)
    ensure_dir(root)
    return {
        "root": root,
        "metadata": os.path.join(root, "{}_metadata.json".format(figure_name)),
        "npz": os.path.join(root, "{}_data.npz".format(figure_name)),
        "png": os.path.join(root, "{}.png".format(figure_name)),
        "pdf": os.path.join(root, "{}.pdf".format(figure_name)),
    }
