"""
read_dump.py
============
LAMMPS dump 文件解析器（支持正交/斜交盒子，Lees-Edwards BC）

输出格式：
    frames : list of dict, 每帧包含
        - 'timestep'  : int
        - 'natoms'    : int
        - 'box'       : dict  {xlo,xhi,ylo,yhi,zlo,zhi, xy,xz,yz}  (triclinic)
        - 'id'        : (N,) int array
        - 'type'      : (N,) int array
        - 'x','y','z' : (N,) float  (wrapped coordinates)
        - 'ix','iy','iz': (N,) int  (image flags)
        - 'vx','vy','vz': (N,) float
        - 'xu','yu','zu': (N,) float  (unwrapped coordinates, computed here)
"""

import numpy as np
import re
from typing import List, Dict, Optional


def _parse_box_bounds(lines):
    """解析盒子边界，支持正交和斜交(tilt)格式。"""
    box = {}
    tilt = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
    for line in lines:
        m = re.match(r'ITEM: BOX BOUNDS', line)
        if m:
            # 检查是否有 tilt 关键字
            if 'xy xz yz' in line:
                return 'triclinic'
    return 'orthogonal'


def read_lammps_dump(filename: str,
                     max_frames: Optional[int] = None,
                     stride: int = 1) -> List[Dict]:
    """
    读取 LAMMPS custom dump 文件。

    Parameters
    ----------
    filename : str
        dump 文件路径
    max_frames : int, optional
        最多读取帧数（None = 全部）
    stride : int
        每隔 stride 帧读一帧

    Returns
    -------
    frames : list of dict
    """
    frames = []
    frame_count = 0
    read_count = 0

    with open(filename, 'r') as f:
        while True:
            # --- ITEM: TIMESTEP ---
            line = f.readline()
            if not line:
                break
            if 'ITEM: TIMESTEP' not in line:
                continue
            timestep = int(f.readline().strip())

            # --- ITEM: NUMBER OF ATOMS ---
            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())

            # --- ITEM: BOX BOUNDS ---
            box_header = f.readline()  # ITEM: BOX BOUNDS ...
            triclinic = ('xy xz yz' in box_header)

            xlo, xhi = 0.0, 0.0
            ylo, yhi = 0.0, 0.0
            zlo, zhi = 0.0, 0.0
            xy, xz, yz = 0.0, 0.0, 0.0

            if triclinic:
                vals = f.readline().split()
                xlo_b, xhi_b, xy = float(vals[0]), float(vals[1]), float(vals[2])
                vals = f.readline().split()
                ylo_b, yhi_b, xz = float(vals[0]), float(vals[1]), float(vals[2])
                vals = f.readline().split()
                zlo_b, zhi_b, yz = float(vals[0]), float(vals[1]), float(vals[2])
                # 转换为真实盒子边长（LAMMPS convention）
                xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                xhi = xhi_b - max(0.0, xy, xz, xy + xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                zlo = zlo_b
                zhi = zhi_b
            else:
                vals = f.readline().split(); xlo, xhi = float(vals[0]), float(vals[1])
                vals = f.readline().split(); ylo, yhi = float(vals[0]), float(vals[1])
                vals = f.readline().split(); zlo, zhi = float(vals[0]), float(vals[1])

            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo

            box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi,
                       Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)

            # --- ITEM: ATOMS ---
            atom_header = f.readline()  # ITEM: ATOMS id type x y z ...
            col_names = atom_header.strip().split()[2:]  # 去掉 "ITEM:" "ATOMS"
            col_idx = {name: i for i, name in enumerate(col_names)}

            # 读取原子数据
            data = np.zeros((natoms, len(col_names)), dtype=float)
            for i in range(natoms):
                data[i] = f.readline().split()

            # 检查是否使用此帧（stride）
            frame_count += 1
            if (frame_count - 1) % stride != 0:
                continue

            # 提取各列
            def get_col(name):
                if name in col_idx:
                    return data[:, col_idx[name]]
                return None

            ids    = get_col('id').astype(int)
            types  = get_col('type').astype(int)
            x = get_col('x')
            y = get_col('y')
            z = get_col('z')
            ix = get_col('ix')
            iy = get_col('iy')
            iz = get_col('iz')
            vx = get_col('vx')
            vy = get_col('vy')
            vz = get_col('vz')

            # 按 id 排序（LAMMPS 输出顺序可能不一致）
            sort_idx = np.argsort(ids)
            ids   = ids[sort_idx]
            types = types[sort_idx]
            x = x[sort_idx]; y = y[sort_idx]; z = z[sort_idx]
            if ix is not None:
                ix = ix[sort_idx].astype(int)
                iy = iy[sort_idx].astype(int)
                iz = iz[sort_idx].astype(int)
            else:
                ix = np.zeros(natoms, int)
                iy = np.zeros(natoms, int)
                iz = np.zeros(natoms, int)
            if vx is not None:
                vx = vx[sort_idx]; vy = vy[sort_idx]; vz = vz[sort_idx]
            else:
                vx = vy = vz = np.zeros(natoms)

            # 计算 unwrapped 坐标
            # 对斜交盒子（Lees-Edwards）：
            # x_u = x + ix*Lx + iy*xy    (x 方向有 tilt)
            # y_u = y + iy*Ly
            # z_u = z + iz*Lz
            xu = x + ix * Lx + iy * xy
            yu = y + iy * Ly
            zu = z + iz * Lz

            frame = dict(
                timestep=timestep, natoms=natoms, box=box,
                id=ids, type=types,
                x=x, y=y, z=z,
                ix=ix, iy=iy, iz=iz,
                vx=vx, vy=vy, vz=vz,
                xu=xu, yu=yu, zu=zu,
            )
            frames.append(frame)
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

    r̃_i(t) = [r_i^unwrap(t) - r_i^unwrap(0)] - γ̇ * t * y_i^unwrap(0) * x̂

    Parameters
    ----------
    frames     : list of frame dicts (sorted by time)
    shear_rate : float, γ̇ (LJ units)
    dt_frame   : float, 帧间时间间隔 (LJ units)

    Returns
    -------
    r_tilde : (N_frames, N_atoms, 3) array
        非仿射位移，轴序 [frame, atom, xyz]
    """
    N_frames = len(frames)
    N_atoms  = frames[0]['natoms']

    r_tilde = np.zeros((N_frames, N_atoms, 3))

    # 参考帧 (t=0)
    x0 = frames[0]['xu'].copy()
    y0 = frames[0]['yu'].copy()
    z0 = frames[0]['zu'].copy()

    for fi, frame in enumerate(frames):
        t = fi * dt_frame  # 从参考帧开始的时间

        dx = frame['xu'] - x0
        dy = frame['yu'] - y0
        dz = frame['zu'] - z0

        # 减去仿射贡献：Δx_affine = γ̇ * t * y_i(0)
        dx_affine = shear_rate * t * y0

        r_tilde[fi, :, 0] = dx - dx_affine
        r_tilde[fi, :, 1] = dy
        r_tilde[fi, :, 2] = dz

    return r_tilde


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        frames = read_lammps_dump(sys.argv[1], max_frames=3)
        print(f"Frames: {len(frames)}, Atoms: {frames[0]['natoms']}")
        print(f"Box: Lx={frames[0]['box']['Lx']:.4f}, "
              f"Ly={frames[0]['box']['Ly']:.4f}, "
              f"xy={frames[0]['box']['xy']:.4f}")
