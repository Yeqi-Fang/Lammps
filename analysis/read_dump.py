"""
read_dump.py
============
LAMMPS dump 文件解析器（支持正交/斜交盒子，Lees-Edwards BC）

Bug 修复记录（compute_nonaffine_displacement）：
─────────────────────────────────────────────
旧版错误：
    xu = x + ix*Lx + iy*xy
    r_tilde_x = xu(t) - xu(0) - γ̇*t*y0

  当粒子穿越 y 周期边界时，LAMMPS 把粒子的 x 向左/右偏移了 xy（LE 边界条件）。
  这导致 xu 每次穿越 y 边界时都跳变 +2*xy。
  对于 γ̇=0.005，xy ≈ Lx/2 约在 t~100τ₀ 时触发一次 LE remap，产生图中的阶梯。

修复方案：逐帧累积（incremental）+ LE 正确最近像惯例
─────────────────────────────────────────────────────
  对连续两帧的 wrapped 坐标差 Δr = r(t+dt) - r(t)：

  1. y 方向：n_y = round(Δy/Ly)；Δy -= n_y*Ly
  2. x 方向（LE 耦合）：
       Δx -= n_y * xy_current   # 穿越 y 边界时 x 有 ±xy 的 LE 偏移，要扣掉
       n_x = round(Δx/Lx)；Δx -= n_x*Lx
  3. z 方向：普通最近像
  4. 仿射减除：Δx_na = Δx - γ̇ * dt * yu_prev
       yu_prev = y_prev + iy_prev*Ly  （y 方向 unwrap，对 iy 变化连续）

  累积到 r_tilde。γ̇=0 时仿射项为零，退化为标准增量 MSD。
"""

import numpy as np
from typing import List, Dict, Optional


def read_lammps_dump(filename: str,
                     max_frames: Optional[int] = None,
                     stride: int = 1) -> List[Dict]:
    """
    读取 LAMMPS custom dump 文件。

    支持格式：id type x y z ix iy iz [vx vy vz]
              id type xu yu zu [...]
    """
    frames = []
    frame_count = 0
    read_count  = 0

    with open(filename, 'r') as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if 'ITEM: TIMESTEP' not in line:
                continue
            timestep = int(fh.readline().strip())

            fh.readline()
            natoms = int(fh.readline().strip())

            box_header = fh.readline()
            triclinic  = ('xy xz yz' in box_header)

            if triclinic:
                v = fh.readline().split()
                xlo_b, xhi_b, xy = float(v[0]), float(v[1]), float(v[2])
                v = fh.readline().split()
                ylo_b, yhi_b, xz = float(v[0]), float(v[1]), float(v[2])
                v = fh.readline().split()
                zlo_b, zhi_b, yz = float(v[0]), float(v[1]), float(v[2])
                xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                xhi = xhi_b - max(0.0, xy, xz, xy + xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                zlo, zhi = zlo_b, zhi_b
            else:
                xy = xz = yz = 0.0
                v = fh.readline().split(); xlo, xhi = float(v[0]), float(v[1])
                v = fh.readline().split(); ylo, yhi = float(v[0]), float(v[1])
                v = fh.readline().split(); zlo, zhi = float(v[0]), float(v[1])

            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo
            box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi,
                       Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)

            atom_header = fh.readline()
            col_names   = atom_header.strip().split()[2:]
            col_idx     = {name: i for i, name in enumerate(col_names)}

            raw = np.zeros((natoms, len(col_names)), dtype=float)
            for i in range(natoms):
                raw[i] = fh.readline().split()

            frame_count += 1
            if (frame_count - 1) % stride != 0:
                continue

            def get(name, dtype=float):
                return raw[:, col_idx[name]].astype(dtype) if name in col_idx else None

            ids   = get('id',   dtype=int)
            types = get('type', dtype=int)

            has_wrapped = ('x' in col_idx and 'y' in col_idx and 'z' in col_idx)
            if has_wrapped:
                x  = get('x');  y  = get('y');  z  = get('z')
                ix = get('ix', dtype=int) if 'ix' in col_idx else np.zeros(natoms, int)
                iy = get('iy', dtype=int) if 'iy' in col_idx else np.zeros(natoms, int)
                iz = get('iz', dtype=int) if 'iz' in col_idx else np.zeros(natoms, int)
            else:
                x  = get('xu'); y  = get('yu'); z  = get('zu')
                ix = iy = iz = np.zeros(natoms, int)

            vx = get('vx') if 'vx' in col_idx else np.zeros(natoms)
            vy = get('vy') if 'vy' in col_idx else np.zeros(natoms)
            vz = get('vz') if 'vz' in col_idx else np.zeros(natoms)

            order = np.argsort(ids)
            def srt(a): return a[order] if a is not None else a
            ids   = srt(ids);  types = srt(types)
            x = srt(x); y = srt(y); z = srt(z)
            ix = srt(ix); iy = srt(iy); iz = srt(iz)
            vx = srt(vx); vy = srt(vy); vz = srt(vz)

            # xu/yu/zu 供 F_s(q,t) 使用（LE 下穿越 y 边界时不连续，不用于 MSD）
            xu = x + ix * Lx + iy * xy + iz * xz
            yu = y + iy * Ly + iz * yz
            zu = z + iz * Lz

            frames.append(dict(
                timestep=timestep, natoms=natoms, box=box,
                id=ids, type=types,
                x=x, y=y, z=z,
                ix=ix, iy=iy, iz=iz,
                vx=vx, vy=vy, vz=vz,
                xu=xu, yu=yu, zu=zu,
                _has_image_flags=has_wrapped,
            ))
            read_count += 1
            if max_frames is not None and read_count >= max_frames:
                break

    print(f"[read_dump] 读取 {len(frames)} 帧，文件: {filename}")
    return frames


def compute_nonaffine_displacement(frames: List[Dict],
                                   shear_rate: float,
                                   dt_frame: float) -> np.ndarray:
    """
    计算 Yamamoto-Onuki 非仿射位移 r̃(t)。

    使用逐帧累积 + LE 正确最近像，避免 xu 重建在 LE 下的不连续问题。

    Parameters
    ----------
    frames     : list of frame dicts
    shear_rate : float，γ̇（γ̇=0 时退化为标准增量 MSD）
    dt_frame   : float，帧间物理时间（LJ 单位）

    Returns
    -------
    r_tilde : (N_frames, N_atoms, 3) float64
    """
    N_frames = len(frames)
    N_atoms  = frames[0]['natoms']
    r_tilde  = np.zeros((N_frames, N_atoms, 3), dtype=np.float64)

    if not frames[0].get('_has_image_flags', True):
        print("  ⚠ 无 image flags，回退到 xu 差分法")
        return _nonaffine_from_xu(frames, shear_rate, dt_frame)

    box0 = frames[0]['box']
    Lx, Ly, Lz = box0['Lx'], box0['Ly'], box0['Lz']

    acc     = np.zeros((N_atoms, 3), dtype=np.float64)
    x_prev  = frames[0]['x'].copy().astype(np.float64)
    y_prev  = frames[0]['y'].copy().astype(np.float64)
    z_prev  = frames[0]['z'].copy().astype(np.float64)
    # unwrapped y：y + iy*Ly，对 iy 跳变连续，用于仿射减除
    yu_prev = (frames[0]['y'] + frames[0]['iy'] * Ly).astype(np.float64)

    for fi in range(1, N_frames):
        f  = frames[fi]
        xy = f['box']['xy']   # 当前帧 LE tilt

        x_curr  = f['x'].astype(np.float64)
        y_curr  = f['y'].astype(np.float64)
        z_curr  = f['z'].astype(np.float64)
        yu_curr = y_curr + f['iy'].astype(np.float64) * Ly

        dx = x_curr - x_prev
        dy = y_curr - y_prev
        dz = z_curr - z_prev

        # ── LE 正确最近像 ──────────────────────────────────────────────
        # 1. y 方向（普通周期）
        n_y = np.round(dy / Ly)
        dy -= n_y * Ly

        # 2. x 方向：先扣掉 LE 的 y 穿越偏移，再取 x 最近像
        #    粒子穿越 y+ 边界时，LAMMPS 把 x 向右偏移了 +xy；
        #    n_y 次穿越共偏移 n_y * xy，需先扣掉才能正确取 x 模
        dx -= n_y * xy
        n_x = np.round(dx / Lx)
        dx -= n_x * Lx

        # 3. z 方向（普通周期）
        n_z = np.round(dz / Lz)
        dz -= n_z * Lz

        # ── 减去仿射贡献（增量形式）────────────────────────────────────
        #   使用上一帧的 unwrapped y（y + iy*Ly），对边界穿越连续
        dx -= shear_rate * dt_frame * yu_prev

        acc[:, 0] += dx
        acc[:, 1] += dy
        acc[:, 2] += dz
        r_tilde[fi] = acc.copy()

        x_prev  = x_curr
        y_prev  = y_curr
        z_prev  = z_curr
        yu_prev = yu_curr

    return r_tilde


def _nonaffine_from_xu(frames, shear_rate, dt_frame):
    """旧方法（无 image flags 时 fallback）。"""
    N = len(frames)
    M = frames[0]['natoms']
    rt = np.zeros((N, M, 3), dtype=np.float64)
    x0 = frames[0]['xu'].astype(np.float64)
    y0 = frames[0]['yu'].astype(np.float64)
    z0 = frames[0]['zu'].astype(np.float64)
    for fi, f in enumerate(frames):
        t = fi * dt_frame
        rt[fi, :, 0] = f['xu'] - x0 - shear_rate * t * y0
        rt[fi, :, 1] = f['yu'] - y0
        rt[fi, :, 2] = f['zu'] - z0
    return rt


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        frames = read_lammps_dump(sys.argv[1], max_frames=3)
        print(f"Frames: {len(frames)}, Atoms: {frames[0]['natoms']}")
        b = frames[0]['box']
        print(f"Box: Lx={b['Lx']:.4f}, Ly={b['Ly']:.4f}, xy={b['xy']:.6f}")
        print(f"Image flags: {frames[0].get('_has_image_flags', False)}")