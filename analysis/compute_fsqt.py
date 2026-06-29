"""
compute_fsqt.py  ——  Self Intermediate Scattering Function F_s(q,t)
=====================================================================
适用于 KA 二元 LJ 体系 + SLLOD 剪切流（Lees-Edwards 边界）

======================================================================
物理定义
======================================================================

    F_s(q, t) = (1/N) Σ_i  ⟨ exp( i q · δr̃_i(t₀, t) ) ⟩_{t₀}

非仿射位移（Yamamoto-Onuki / steady shear）：
    R_i(t) = r_i(t) - γ̇ ∫_0^t y_i(t') dt' e_x
    δr̃_i(t₀, t) = R_i(t₀+t) - R_i(t₀)

即：
    δr̃_i(t₀, t)_x = [x_i(t₀+t) - x_i(t₀)] - γ̇ ∫_{t₀}^{t₀+t} y_i(t') dt'
    δr̃_i(t₀, t)_y =  y_i(t₀+t) - y_i(t₀)
    δr̃_i(t₀, t)_z =  z_i(t₀+t) - z_i(t₀)

    其中 x_i(t) 是实验室坐标（unwrapped）:
        xu_i = x_i + ix_i·Lx + iy_i·xy_t     (从 dump 中重建)
        yu_i = y_i + iy_i·Ly
        zu_i = z_i + iz_i·Lz
    xy_t 是当前帧的 triclinic 盒倾斜量 = γ̇·t·Ly（从 dump 帧头中直接读取）

等方平均（各向同性）：
    F_s^{iso}(q, t) = (1/N) Σ_i  ⟨ sin(q|δr̃_i|) / (q|δr̃_i|) ⟩
                    = (1/N) Σ_i  ⟨ sinc(q|δr̃_i|) ⟩

方向性（仅沿 x/y/z 方向）：
    F_s^α(q, t) = (1/N) Σ_i  ⟨ cos(q·δr̃_i^α) ⟩        (α = x, y, z)

======================================================================
典型参数（KA 3D）
======================================================================
  q* ≈ 7.25  σ_bb   (big-big S(q) 第一峰)
  T  = 0.45
  type 1 = 大粒子 (80%)

======================================================================
输入：LAMMPS dump 文件格式
======================================================================
dump 列：id  type  x  y  z  ix  iy  iz  [vx  vy  vz]
（与 in.shear_template 的 DUMP_EVERY 输出一致）

box header（triclinic）：
  ITEM: BOX BOUNDS xy xz yz pp pp pp
  xlo_bound  xhi_bound  xy
  ylo_bound  yhi_bound  xz
  zlo_bound  zhi_bound  yz

======================================================================
用法示例
======================================================================
  # 单个 shear rate，自动选 q*
  python compute_fsqt.py --dump dump.shear_0.015.lammpstrj --rate 0.015

  # 多个 shear rate 对比（图上不同颜色）
  python compute_fsqt.py \\
      --dump dump.shear_0.001.lammpstrj \\
             dump.shear_0.005.lammpstrj \\
             dump.shear_0.015.lammpstrj \\
      --rate 0.001 0.005 0.015

  # 指定 q 值列表
  python compute_fsqt.py --dump dump.shear_0.015.lammpstrj \\
      --rate 0.015 --q_vals 5.0 7.25 10.0

  # 同时画方向分解（q 沿 x/y/z）
  python compute_fsqt.py --dump dump.shear_0.015.lammpstrj \\
      --rate 0.015 --directional

  # 每隔 n_skip 帧取一个时间原点（节省内存/加速）
  python compute_fsqt.py --dump dump.shear_0.015.lammpstrj \\
      --rate 0.015 --n_skip 5 --max_lag 200

  # 从 msd_data.npz 中的 r_tilde 计算（快速估算，仅 t₀=0）
  python compute_fsqt.py --npz msd_data.npz --rate 0.015
"""

import os, sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore', category=RuntimeWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 参数默认值
# ══════════════════════════════════════════════════════════════════════════════
Q_STAR_DEFAULT  = 7.25      # KA 模型 big-big S(q) 第一峰（σ_bb 单位）
TYPE_BIG        = 1         # 大粒子类型
FSQT_THRESHOLD  = 1.0 / np.e   # τ_α 定义：F_s = e^{-1} ≈ 0.368


# ══════════════════════════════════════════════════════════════════════════════
# LAMMPS dump 解析器（triclinic / Lees-Edwards 专用）
# ══════════════════════════════════════════════════════════════════════════════

def parse_dump_frame(lines, col_map):
    """
    解析一个 LAMMPS dump 帧。
    返回 dict:
        'timestep': int
        'natoms'  : int
        'box'     : dict {xlo, xhi, ylo, yhi, zlo, zhi, Lx, Ly, Lz, xy, xz, yz}
        'x','y','z','ix','iy','iz','type','id'  : numpy arrays (shape=natoms)
    """
    i = 0
    box_tric = False
    box_data = {}

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('ITEM: TIMESTEP'):
            timestep = int(lines[i+1].strip())
            i += 2

        elif line.startswith('ITEM: NUMBER OF ATOMS'):
            natoms = int(lines[i+1].strip())
            i += 2

        elif line.startswith('ITEM: BOX BOUNDS'):
            # 检查是否 triclinic
            box_tric = ('xy' in line)
            b0 = lines[i+1].split()
            b1 = lines[i+2].split()
            b2 = lines[i+3].split()

            if box_tric:
                xlo_b, xhi_b, xy = float(b0[0]), float(b0[1]), float(b0[2])
                ylo_b, yhi_b, xz = float(b1[0]), float(b1[1]), float(b1[2])
                zlo_b, zhi_b, yz = float(b2[0]), float(b2[1]), float(b2[2])
                # 从 bound 坐标还原真实盒子坐标
                xlo = xlo_b - min(0.0, xy, xz, xy+xz)
                xhi = xhi_b - max(0.0, xy, xz, xy+xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                zlo, zhi = zlo_b, zhi_b
            else:
                xy = xz = yz = 0.0
                xlo, xhi = float(b0[0]), float(b0[1])
                ylo, yhi = float(b1[0]), float(b1[1])
                zlo, zhi = float(b2[0]), float(b2[1])

            box_data = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi,
                            Lx=xhi-xlo, Ly=yhi-ylo, Lz=zhi-zlo,
                            xy=xy, xz=xz, yz=yz)
            i += 4

        elif line.startswith('ITEM: ATOMS'):
            # 解析列标题
            headers = line.split()[2:]
            if col_map is None:
                col_map = {h: k for k, h in enumerate(headers)}

            atom_lines = lines[i+1: i+1+natoms]
            raw = np.array([l.split() for l in atom_lines], dtype=float)

            frame = {'timestep': timestep, 'natoms': natoms, 'box': box_data,
                     '_col_map': col_map}

            def _get(name, dtype=float):
                if name in col_map:
                    return raw[:, col_map[name]].astype(dtype)
                return None

            frame['id']   = _get('id',   dtype=int)
            frame['type'] = _get('type', dtype=int)
            frame['x']    = _get('x')
            frame['y']    = _get('y')
            frame['z']    = _get('z')
            frame['ix']   = _get('ix',  dtype=int) if 'ix' in col_map else np.zeros(natoms, int)
            frame['iy']   = _get('iy',  dtype=int) if 'iy' in col_map else np.zeros(natoms, int)
            frame['iz']   = _get('iz',  dtype=int) if 'iz' in col_map else np.zeros(natoms, int)
            frame['xu']   = _get('xu')  # 已 unwrap 时直接用
            frame['yu']   = _get('yu')
            frame['zu']   = _get('zu')

            if frame['id'] is not None:
                order = np.argsort(frame['id'])
                for key in ('id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz', 'xu', 'yu', 'zu'):
                    if frame.get(key) is not None:
                        frame[key] = frame[key][order]

            return frame, col_map

    return None, col_map


def read_dump_lazy(dump_file, max_frames=None):
    """
    逐帧读取 LAMMPS dump 文件，返回 (frames_list, col_map)。

    内存优化：仅保留 id/type/x/y/z/ix/iy/iz 和 box，丢弃速度列。
    """
    print(f"  [dump] 读取: {dump_file}")
    frames = []
    col_map = None

    with open(dump_file, 'r') as fh:
        buffer = []
        in_frame = False

        for raw_line in fh:
            line = raw_line.rstrip('\n')
            if line.startswith('ITEM: TIMESTEP'):
                if in_frame and buffer:
                    frm, col_map = parse_dump_frame(buffer, col_map)
                    if frm is not None:
                        frames.append(frm)
                        if max_frames and len(frames) >= max_frames:
                            break
                buffer = [line]
                in_frame = True
            elif in_frame:
                buffer.append(line)

        # 最后一帧
        if in_frame and buffer and (not max_frames or len(frames) < max_frames):
            frm, col_map = parse_dump_frame(buffer, col_map)
            if frm is not None:
                frames.append(frm)

    print(f"  [dump] 读取 {len(frames)} 帧，natoms={frames[0]['natoms'] if frames else 0}")
    return frames


def stitch_dumps(frames_fine, frames_coarse, shear_rate):
    """
    拼接 fine dump（短时高分辨）和 coarse dump（长时低分辨）。

    关键问题：in.shear_template 中 fine 跑完后有 reset_timestep 0，
    导致两个 dump 的 timestep 都从 0 开始，不能直接用 timestep 区分。

    解决方案：
      - coarse Δt = 0.02 / γ̇（已知）
      - fine   Δt 从 fine dump 相邻帧的 timestep 差 × DT 推算
      - DT 从 coarse dump 推算：DT = (0.02/γ̇) / coarse_ts_step

    返回：
      frames_all : 拼接后的帧列表（fine 在前，coarse 尾部在后）
      t_phys     : 每帧的物理时间 (τ₀)，shape=(n_frames_all,)
    """
    # ── 推算 DT ──
    if len(frames_coarse) >= 2:
        coarse_ts_step = frames_coarse[1]['timestep'] - frames_coarse[0]['timestep']
        DT = (0.02 / shear_rate) / coarse_ts_step
    else:
        DT = 0.001
    print(f"  [stitch] 推算 DT = {DT:.5f} τ₀/step")

    # ── fine 帧的物理时间 ──
    fine_ts_step = (frames_fine[1]['timestep'] - frames_fine[0]['timestep']
                    if len(frames_fine) >= 2 else 10)
    dt_fine = fine_ts_step * DT
    t_fine  = np.arange(len(frames_fine)) * dt_fine
    t_end   = t_fine[-1]
    print(f"  [stitch] fine: {len(frames_fine)} 帧, Δt={dt_fine:.4f} τ₀, "
          f"覆盖 [0, {t_end:.2f}] τ₀")

    # ── coarse 帧的物理时间（从 t=0 开始，剔除与 fine 重叠部分）──
    dt_coarse  = 0.02 / shear_rate
    t_coarse   = np.arange(len(frames_coarse)) * dt_coarse
    keep_mask  = t_coarse > t_end
    frames_tail = [f for f, k in zip(frames_coarse, keep_mask) if k]
    t_tail      = t_coarse[keep_mask]
    print(f"  [stitch] coarse: 保留 {len(frames_tail)} 帧（跳过前 {keep_mask.argmax()} 帧重叠）")
    print(f"  [stitch] 拼接后: {len(frames_fine)+len(frames_tail)} 帧, "
          f"覆盖 [0, {t_tail[-1] if len(t_tail) else t_end:.1f}] τ₀")

    frames_all = frames_fine + frames_tail
    t_phys     = np.concatenate([t_fine, t_tail])
    return frames_all, t_phys


def get_unwrapped_positions(frame):
    """
    从 dump 帧还原 unwrapped（实验室）坐标。

    优先用 xu/yu/zu（若 dump 已包含）；
    否则用 wrapped x,y,z + image flags ix,iy,iz:
        xu = x + ix*Lx + iy*xy_t
        yu = y + iy*Ly
        zu = z + iz*Lz
    注意：xy_t 是当前帧的 Lees-Edwards 倾斜量，从 box_data 读取。
    """
    box = frame['box']
    Lx = box['Lx'];  Ly = box['Ly'];  Lz = box['Lz']
    xy = box['xy'];  xz = box['xz'];  yz = box['yz']

    if frame['xu'] is not None:
        return frame['xu'].copy(), frame['yu'].copy(), frame['zu'].copy()

    x  = frame['x'];  y  = frame['y'];  z  = frame['z']
    ix = frame['ix']; iy = frame['iy']; iz = frame['iz']

    xu = x + ix * Lx + iy * xy + iz * xz
    yu = y + iy * Ly + iz * yz
    zu = z + iz * Lz

    return xu, yu, zu


# ══════════════════════════════════════════════════════════════════════════════
# F_s(q,t) 计算核心
# ══════════════════════════════════════════════════════════════════════════════

def compute_fsqt_from_dump(
        frames,
        q_vals,
        shear_rate,
        particle_type=TYPE_BIG,
        n_skip=1,
        max_lag=None,
        mode='isotropic',      # 'isotropic' | 'x' | 'y' | 'z' | 'xy'
        t_phys=None,           # 非均匀时间轴（stitch 后使用），None=均匀
        verbose=True
    ):
    """
    从 dump 帧列表计算 F_s(q, t)。

    Parameters
    ----------
    frames      : list of frame dicts（来自 read_dump_lazy）
    q_vals      : list/array，波矢量大小 (σ_bb^{-1})
    shear_rate  : float，剪切速率 γ̇
    particle_type : int，1=大粒子
    n_skip      : int，时间原点采样间隔（帧数）
    max_lag     : int or None，最大延迟帧数
    mode        : 'isotropic' | 'x' | 'y' | 'z'

    Returns
    -------
    t_arr : (n_lag,) array，时间轴 [τ₀]
    Fsqt  : (n_q, n_lag) array，F_s(q, t)
    """
    n_frames   = len(frames)
    q_vals     = np.asarray(q_vals)
    n_q        = len(q_vals)

    # ── 时间轴 ──
    # 均匀模式：Δt = 0.02/γ̇（与 in.shear_template 一致）
    # 非均匀模式：由 t_phys 数组直接给出每帧的物理时间
    if t_phys is None:
        dt_frame = 0.02 / shear_rate if shear_rate > 0 else 1.0
        t_phys   = np.arange(n_frames) * dt_frame
    else:
        t_phys = np.asarray(t_phys)
        assert len(t_phys) == n_frames, "t_phys 长度必须等于帧数"

    if max_lag is None:
        max_lag = n_frames - 1
    max_lag = min(max_lag, n_frames - 1)

    # 过滤粒子类型
    types_ref = frames[0]['type']
    if types_ref is None:
        raise ValueError("dump 中没有 type 列")
    big_mask = (types_ref == particle_type)
    n_big = int(big_mask.sum())
    if n_big == 0:
        raise ValueError(f"找不到 type={particle_type} 的粒子")
    if verbose:
        print(f"  type={particle_type} 粒子: {n_big} / {frames[0]['natoms']}")

    origins = np.arange(0, n_frames - max_lag, n_skip)
    if len(origins) == 0:
        origins = np.array([0])
    if verbose:
        print(f"  时间原点数: {len(origins)}  最大延迟: {max_lag} 帧")
        print(f"  预读所有帧坐标 → (n_frames, n_big, 3)...")

    # ── 一次性读入所有帧，(n_frames, n_big, 3) ──
    all_pos = np.stack([
        np.stack(get_unwrapped_positions(f), axis=1)[big_mask]
        for f in frames
    ], axis=0)

    if shear_rate != 0.0:
        if verbose:
            print("  applying Yamamoto-Onuki nonaffine coordinate: x -= gamma_dot * integral y(t) dt")
        affine_x = np.zeros(n_big, dtype=np.float64)
        prev_y = all_pos[0, :, 1].astype(np.float64, copy=False)
        for fi in range(1, n_frames):
            curr_y = all_pos[fi, :, 1].astype(np.float64, copy=False)
            dt = float(t_phys[fi] - t_phys[fi - 1])
            affine_x += shear_rate * dt * 0.5 * (prev_y + curr_y)
            all_pos[fi, :, 0] -= affine_x
            prev_y = curr_y

    if verbose:
        print(f"  开始计算 F_s(q,t)  模式={mode}  q 数={n_q}")

    Fsqt = np.zeros((n_q, max_lag + 1))

    for lag in range(max_lag + 1):
        valid_ori = origins[origins + lag < n_frames]
        drs = all_pos[valid_ori + lag] - all_pos[valid_ori]
        dr_x = drs[..., 0]
        dr_y = drs[..., 1]
        dr_z = drs[..., 2]

        if mode == 'isotropic':
            dr  = np.sqrt(dr_x**2 + dr_y**2 + dr_z**2)
            for qi, q in enumerate(q_vals):
                qr = q * dr
                Fsqt[qi, lag] = np.where(qr < 1e-10, 1.0, np.sin(qr) / qr).mean()
        elif mode == 'x':
            for qi, q in enumerate(q_vals):
                Fsqt[qi, lag] = np.cos(q * dr_x).mean()
        elif mode == 'y':
            for qi, q in enumerate(q_vals):
                Fsqt[qi, lag] = np.cos(q * dr_y).mean()
        elif mode == 'z':
            for qi, q in enumerate(q_vals):
                Fsqt[qi, lag] = np.cos(q * dr_z).mean()
        elif mode == 'xy':
            from scipy.special import j0
            dr_xy = np.sqrt(dr_x**2 + dr_y**2)
            for qi, q in enumerate(q_vals):
                Fsqt[qi, lag] = j0(q * dr_xy).mean()

    # 时间轴：每个 lag 对应的物理时间取所有有效原点的中位数
    t_arr = np.array([
        float(np.median(t_phys[origins[origins + lag < n_frames] + lag]
                        - t_phys[origins[origins + lag < n_frames]]))
        if np.any(origins + lag < n_frames) else np.nan
        for lag in range(max_lag + 1)
    ])

    if verbose:
        print(f"  计算完成。t 范围: [{t_arr[0]:.3f}, {t_arr[-1]:.3f}] τ₀")

    return t_arr, Fsqt


# ══════════════════════════════════════════════════════════════════════════════
# 从 r_tilde npz 计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_fsqt_from_rtilde(npz_file, q_vals, shear_rate,
                              particle_type=TYPE_BIG,
                              n_skip=1, max_lag=None,
                              mode='isotropic',
                              n_origins=None, verbose=True):
    """
    从 msd_data.npz 中的 r_tilde（非仿射位移轨迹）计算 F_s(q,t)。

    r_tilde[t, i, :] 是全局 Yamamoto-Onuki 非仿射坐标。
    因此 r_tilde[t2, i] - r_tilde[t1, i] 就是 t1 到 t2 的
    非仿射位移，可以做多时间原点平均。
    """
    data      = np.load(npz_file, allow_pickle=True)
    r_tilde   = data['r_tilde']           # (n_frames, n_particles, 3)
    times     = data['times']             # (n_frames,) 物理时间
    types     = data['types']             # (n_particles,)

    big_mask  = (types == particle_type)
    n_big     = big_mask.sum()
    if verbose:
        print(f"  r_tilde shape: {r_tilde.shape}")
        print(f"  type={particle_type}: {n_big} 粒子")

    q_vals    = np.asarray(q_vals)
    n_q       = len(q_vals)
    n_frames  = r_tilde.shape[0]
    if max_lag is None:
        max_lag = n_frames // 2
    max_lag = min(int(max_lag), n_frames - 1)

    if n_origins is None:
        origin_indices = np.arange(0, n_frames - max_lag, n_skip, dtype=int)
    else:
        n_valid_origins = max(1, n_frames - max_lag)
        origin_indices = np.linspace(
            0, n_valid_origins - 1, min(int(n_origins), n_valid_origins), dtype=int
        )
        origin_indices = np.unique(origin_indices)
    if len(origin_indices) == 0:
        origin_indices = np.array([0], dtype=int)

    if verbose:
        print(f"  time origins: {len(origin_indices)}  max_lag: {max_lag} frames  mode={mode}")

    Fsqt_sum  = np.zeros((n_q, max_lag + 1))
    count_arr = np.zeros(max_lag + 1, dtype=int)

    for t0_idx in origin_indices:
        r0 = r_tilde[t0_idx, big_mask, :]   # (n_big, 3)

        for lag in range(0, max_lag + 1):
            t1_idx = t0_idx + lag
            if t1_idx >= n_frames:
                break
            r1 = r_tilde[t1_idx, big_mask, :]
            dr = r1 - r0

            if mode == 'isotropic':
                dr_norm = np.sqrt((dr**2).sum(axis=1))
                for qi, q in enumerate(q_vals):
                    qr = q * dr_norm
                    Fsqt_sum[qi, lag] += np.where(qr < 1e-10, 1.0, np.sin(qr) / qr).mean()
            elif mode == 'x':
                for qi, q in enumerate(q_vals):
                    Fsqt_sum[qi, lag] += np.cos(q * dr[:, 0]).mean()
            elif mode == 'y':
                for qi, q in enumerate(q_vals):
                    Fsqt_sum[qi, lag] += np.cos(q * dr[:, 1]).mean()
            elif mode == 'z':
                for qi, q in enumerate(q_vals):
                    Fsqt_sum[qi, lag] += np.cos(q * dr[:, 2]).mean()
            elif mode == 'xy':
                from scipy.special import j0
                dr_xy = np.sqrt(dr[:, 0]**2 + dr[:, 1]**2)
                for qi, q in enumerate(q_vals):
                    Fsqt_sum[qi, lag] += j0(q * dr_xy).mean()
            else:
                raise ValueError(f"unknown mode: {mode}")

            count_arr[lag] += 1

    valid  = count_arr > 0
    Fsqt   = np.zeros_like(Fsqt_sum)
    Fsqt[:, valid] = Fsqt_sum[:, valid] / count_arr[None, valid]

    # 时间轴
    if len(times) == n_frames:
        # 把时间轴从绝对时间转换为延迟时间
        dt = np.diff(times)
        dt_frame = float(np.median(dt))
        t_arr = np.arange(max_lag + 1) * dt_frame
    else:
        t_arr = np.linspace(0, max_lag, max_lag + 1)

    return t_arr, Fsqt


# ══════════════════════════════════════════════════════════════════════════════
# 提取 α-relaxation 时间 τ_α
# ══════════════════════════════════════════════════════════════════════════════

def extract_tau_alpha(t_arr, Fs_qt, threshold=FSQT_THRESHOLD):
    """
    通过线性插值找 F_s(q,t) = threshold 时的 t，即 τ_α。
    若 F_s 始终高于阈值（非遍历），返回 nan。
    若 F_s 从未超过阈值（超快弛豫），返回 t_arr[0]。
    """
    if Fs_qt[-1] > threshold:
        return np.nan   # 未弛豫（可能 t 范围太短）

    # 找最后一个超过阈值的点
    idx = np.where(Fs_qt >= threshold)[0]
    if len(idx) == 0:
        return t_arr[0]

    i0 = idx[-1]
    i1 = i0 + 1
    if i1 >= len(t_arr):
        return t_arr[-1]

    # 线性插值
    f0 = Fs_qt[i0];  f1 = Fs_qt[i1]
    t0 = t_arr[i0];  t1 = t_arr[i1]
    if abs(f1 - f0) < 1e-10:
        return t0
    tau = t0 + (threshold - f0) * (t1 - t0) / (f1 - f0)
    return float(tau)


# ══════════════════════════════════════════════════════════════════════════════
# 绘图函数
# ══════════════════════════════════════════════════════════════════════════════

def plot_fsqt_single(t_arr, Fsqt, q_vals, shear_rate, output_dir, mode='isotropic'):
    """
    绘制单个剪切速率下 F_s(q,t) vs t（对数 x 轴）。
    每条曲线对应一个 q 值。
    """
    os.makedirs(output_dir, exist_ok=True)
    sr_str = str(shear_rate).replace('.', 'p')
    fig, ax = plt.subplots(figsize=(7, 5))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(q_vals)-1, 1)) for i in range(len(q_vals))]

    tau_list = []
    for qi, (q, col) in enumerate(zip(q_vals, colors)):
        Fs = Fsqt[qi]
        ax.semilogx(t_arr[1:], Fs[1:], '-', color=col, lw=2.0,
                    label=rf'$q={q:.2f}$')
        # 标记 τ_α
        tau = extract_tau_alpha(t_arr, Fs)
        tau_list.append(tau)
        if not np.isnan(tau):
            ax.axvline(tau, color=col, ls='--', lw=0.8, alpha=0.5)

    ax.axhline(FSQT_THRESHOLD, color='gray', ls=':', lw=1.2,
               label=rf'$F_s = e^{{-1}} = {FSQT_THRESHOLD:.3f}$')
    ax.set_xlabel(r'$t \ [\tau_0]$', fontsize=13)
    ax.set_ylabel(r'$F_s(q, t)$', fontsize=13)
    mode_label = {'isotropic': '各向同性',
                  'x': r'$q \| \hat{x}$ (flow)',
                  'y': r'$q \| \hat{y}$ (gradient)',
                  'z': r'$q \| \hat{z}$ (vorticity)'}
    ax.set_title(rf'$F_s(q,t)$   $\dot\gamma={shear_rate}$,  T=0.45   [{mode_label.get(mode,mode)}]',
                 fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()

    out_png = os.path.join(output_dir, f'fsqt_{mode}_{sr_str}.png')
    out_pdf = os.path.join(output_dir, f'fsqt_{mode}_{sr_str}.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_png}")

    # τ_α 打印
    print(f"\n  τ_α (q*, mode={mode}):")
    for q, tau in zip(q_vals, tau_list):
        print(f"    q={q:.2f}: τ_α = {tau:.3f}" + (" (未收敛)" if np.isnan(tau) else ""))

    return tau_list


def plot_fsqt_multirate(results, output_dir):
    """
    多剪切速率比较图：
      左：F_s(q*, t) vs t (log x)
      右：τ_α vs γ̇ (log-log, 判断幂律行为)
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_fs, ax_tau = axes

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(results)-1, 1)) for i in range(len(results))]
    taus = []

    for (sr, t_arr, Fsqt, q_vals), col in zip(results, colors):
        # 找最接近 q* 的 q
        qi_star = int(np.argmin(np.abs(q_vals - Q_STAR_DEFAULT)))
        Fs = Fsqt[qi_star]
        q_used = q_vals[qi_star]

        ax_fs.semilogx(t_arr[1:], Fs[1:], '-', color=col, lw=2.5,
                       label=rf'$\dot\gamma={sr}$')
        tau = extract_tau_alpha(t_arr, Fs)
        taus.append((sr, tau))
        if not np.isnan(tau):
            ax_fs.axvline(tau, color=col, ls='--', lw=0.8, alpha=0.6)

    ax_fs.axhline(FSQT_THRESHOLD, color='k', ls=':', lw=1.0,
                  label=rf'$e^{{-1}}$')
    ax_fs.set_xlabel(r'$t \ [\tau_0]$', fontsize=13)
    ax_fs.set_ylabel(rf'$F_s(q^*={Q_STAR_DEFAULT}, t)$', fontsize=13)
    ax_fs.set_title(r'$F_s(q^*, t)$  各剪切速率对比   T=0.45', fontsize=11)
    ax_fs.set_ylim(-0.05, 1.05)
    ax_fs.legend(fontsize=9)
    ax_fs.grid(True, which='both', alpha=0.25)

    # τ_α vs γ̇  (log-log)
    valid_taus = [(sr, tau) for sr, tau in taus if not np.isnan(tau) and tau > 0]
    if len(valid_taus) >= 2:
        sr_arr  = np.array([s for s, _ in valid_taus])
        tau_arr = np.array([t for _, t in valid_taus])
        ax_tau.loglog(sr_arr, tau_arr, 'o-', color='steelblue', ms=8, lw=2)

        # 幂律拟合
        log_sr  = np.log10(sr_arr)
        log_tau = np.log10(tau_arr)
        coeffs = np.polyfit(log_sr, log_tau, 1)
        nu = coeffs[0]
        sr_fit = np.logspace(log_sr.min()-0.1, log_sr.max()+0.1, 50)
        tau_fit = 10**(np.polyval(coeffs, np.log10(sr_fit)))
        ax_tau.loglog(sr_fit, tau_fit, '--', color='red', lw=1.5,
                      label=rf'$\tau_\alpha \propto \dot\gamma^{{{nu:.2f}}}$')
        ax_tau.legend(fontsize=10)
        print(f"\n  幂律拟合: τ_α ∝ γ̇^{nu:.3f}  (期望 ≈ -1 for shear thinning)")

    ax_tau.set_xlabel(r'$\dot{\gamma}\ [\tau_0^{-1}]$', fontsize=13)
    ax_tau.set_ylabel(r'$\tau_\alpha\ [\tau_0]$', fontsize=13)
    ax_tau.set_title(r'$\alpha$-弛豫时间 vs 剪切速率', fontsize=11)
    ax_tau.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(output_dir, 'fsqt_multirate.png')
    out_pdf = os.path.join(output_dir, 'fsqt_multirate.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  → {out_png}")


def plot_fsqt_directional(t_arr_dict, Fsqt_dict, shear_rate, q_star, output_dir):
    """
    方向分解图：F_s(q*, t) 分别沿 x/y/z 方向，与各向同性比较。
    揭示剪切各向异性。
    """
    os.makedirs(output_dir, exist_ok=True)
    sr_str = str(shear_rate).replace('.', 'p')

    fig, ax = plt.subplots(figsize=(7, 5))
    style_map = {
        'isotropic': ('k',     '-',  2.5,  '各向同性 (sinc)'),
        'x':         ('red',   '--', 2.0,  r'$q \| x$ (flow)'),
        'y':         ('blue',  '-.',  2.0, r'$q \| y$ (gradient)'),
        'z':         ('green', ':',  2.0,  r'$q \| z$ (vorticity)'),
    }

    for mode, (col, ls, lw, label) in style_map.items():
        if mode not in Fsqt_dict:
            continue
        t_arr  = t_arr_dict[mode]
        Fsqt   = Fsqt_dict[mode]
        q_vals = np.asarray(Fsqt_dict[f'q_vals_{mode}'])
        qi     = int(np.argmin(np.abs(q_vals - q_star)))
        Fs     = Fsqt[qi]
        ax.semilogx(t_arr[1:], Fs[1:], ls=ls, color=col, lw=lw,
                    label=rf'{label}')

    ax.axhline(FSQT_THRESHOLD, color='gray', ls=':', lw=1.0)
    ax.set_xlabel(r'$t \ [\tau_0]$', fontsize=13)
    ax.set_ylabel(rf'$F_s(q^*={q_star:.2f}, t)$', fontsize=13)
    ax.set_title(rf'方向分解 F_s  $\dot\gamma={shear_rate}$,  T=0.45', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()

    out = os.path.join(output_dir, f'fsqt_directional_{sr_str}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 保存 / 加载结果
# ══════════════════════════════════════════════════════════════════════════════

def save_results(t_arr, Fsqt, q_vals, shear_rate, mode, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sr_str = str(shear_rate).replace('.', 'p')
    out = os.path.join(output_dir, f'fsqt_{mode}_{sr_str}.npz')
    np.savez(out, t=t_arr, Fsqt=Fsqt, q_vals=q_vals,
             shear_rate=shear_rate, mode=mode)
    print(f"  → 数据保存: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='计算 KA LJ 剪切模拟的 Self Intermediate Scattering Function F_s(q,t)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 输入源（二选一）
    p.add_argument('--dump', nargs='+', default=None,
                   help='LAMMPS coarse dump 文件（可多个，与 --rate 一一对应）')
    p.add_argument('--dump_fine', nargs='+', default=None,
                   help='LAMMPS fine dump 文件（短时高分辨，与 --dump 一一对应）')
    p.add_argument('--npz',  default=None,
                   help='msd_data.npz（含 r_tilde，快速路径）')

    # 物理参数
    p.add_argument('--rate', nargs='+', type=float, default=[0.015],
                   help='剪切速率 γ̇（与 --dump 一一对应）')
    p.add_argument('--q_vals', nargs='+', type=float,
                   default=[Q_STAR_DEFAULT],
                   help=f'波矢量大小（默认={Q_STAR_DEFAULT} σ_bb^-1）')
    p.add_argument('--q_sweep', action='store_true',
                   help='在 q 值范围内扫描：2.0~12.0，共 8 个点')
    p.add_argument('--type', type=int, default=TYPE_BIG,
                   help='粒子类型（默认=1，大粒子）')

    # 计算参数
    p.add_argument('--n_skip',  type=int, default=1,
                   help='时间原点采样间隔（帧数）。n_skip>1 可加速')
    p.add_argument('--max_lag', type=int, default=None,
                   help='最大延迟帧数。默认=全部帧数的一半')
    p.add_argument('--dt_frame', type=float, default=None,
                   help='Physical time interval between neighboring dump frames; overrides default 0.02/gamma_dot for plain dump input')
    p.add_argument('--max_frames', type=int, default=None,
                   help='Maximum number of frames to read from each --dump file')
    p.add_argument('--mode',    default='isotropic',
                   choices=['isotropic', 'x', 'y', 'z', 'xy'],
                   help='q 方向模式')
    p.add_argument('--directional', action='store_true',
                   help='同时计算 x/y/z 三个方向（画方向分解图）')

    # 输出
    p.add_argument('--output', default='fsqt_results',
                   help='输出目录')
    p.add_argument('--no_save', action='store_true',
                   help='不保存中间 npz 文件')

    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # q 值
    q_vals = np.array(args.q_vals)
    if args.q_sweep:
        q_vals = np.array([2.0, 3.5, 5.0, 6.0, 7.25, 8.5, 10.0, 12.0])
        print(f"  q sweep: {q_vals}")

    print(f"\n{'='*60}")
    print("  Self Intermediate Scattering Function  F_s(q, t)")
    print(f"  q 值     : {q_vals}")
    print(f"  粒子类型 : {args.type}")
    print(f"  模式     : {args.mode}")
    print(f"{'='*60}\n")

    # ── 路径 A：从 dump 文件计算 ──
    if args.dump is not None:
        dumps  = args.dump
        rates  = args.rate

        if len(rates) == 1 and len(dumps) > 1:
            # 自动推断
            rates_auto = []
            for d in dumps:
                # 从文件名中提取 shear rate（如 dump.shear_0.015.lammpstrj）
                base = os.path.basename(d)
                parts = base.replace('.lammpstrj','').replace('_','.'). \
                             replace('dump.shear.','').split('.')
                try:
                    sr_auto = float('.'.join(parts[-2:]))
                except Exception:
                    sr_auto = rates[0]
                rates_auto.append(sr_auto)
            rates = rates_auto
            print(f"  自动推断 shear rates: {rates}")

        if len(dumps) != len(rates):
            raise ValueError(f"--dump 和 --rate 数量不匹配 ({len(dumps)} vs {len(rates)})")

        results_multi = []

        for dump_file, sr in zip(dumps, rates):
            if not os.path.exists(dump_file):
                print(f"  ✗ 找不到 {dump_file}，跳过")
                continue

            print(f"\n[处理] γ̇={sr}  dump={dump_file}")

            frames_coarse = read_dump_lazy(dump_file, max_frames=args.max_frames)
            if len(frames_coarse) == 0:
                print("  ✗ dump 为空"); continue

            # ── fine + coarse 分别算，再拼接 ──
            fine_files = args.dump_fine or []
            fine_idx   = dumps.index(dump_file)

            if fine_idx < len(fine_files) and os.path.exists(fine_files[fine_idx]):
                print(f"\n  [fine] 计算短时段...")
                frames_fine = read_dump_lazy(fine_files[fine_idx])

                # fine 的 Δt
                coarse_ts_step = (frames_coarse[1]['timestep'] - frames_coarse[0]['timestep']
                                  if len(frames_coarse) >= 2 else 1000)
                fine_ts_step = (frames_fine[1]['timestep'] - frames_fine[0]['timestep']
                                if len(frames_fine) >= 2 else 10)
                if sr > 0:
                    DT = (0.02 / sr) / coarse_ts_step
                else:
                    DT = getattr(args, 'dt', None) or 0.005
                dt_fine = fine_ts_step * DT
                t_phys_fine = np.arange(len(frames_fine)) * dt_fine
                print(f"  fine Δt={dt_fine:.4f} τ₀, 覆盖 [0, {t_phys_fine[-1]:.2f}] τ₀")

                max_lag_fine = len(frames_fine) - 1
                t_fine, Fs_fine = compute_fsqt_from_dump(
                    frames_fine, q_vals, sr,
                    particle_type=args.type,
                    n_skip=max(1, args.n_skip // 5),
                    max_lag=max_lag_fine,
                    mode=args.mode,
                    t_phys=t_phys_fine,
                    verbose=True
                )

                print(f"\n  [coarse] 计算长时段...")
                dt_coarse = (0.02 / sr) if sr > 0 else (coarse_ts_step * DT)
                t_phys_coarse = np.arange(len(frames_coarse)) * dt_coarse
                max_lag_coarse = args.max_lag or len(frames_coarse) // 2

                t_coarse, Fs_coarse = compute_fsqt_from_dump(
                    frames_coarse, q_vals, sr,
                    particle_type=args.type,
                    n_skip=args.n_skip,
                    max_lag=max_lag_coarse,
                    mode=args.mode,
                    t_phys=t_phys_coarse,
                    verbose=True
                )

                # 拼接：fine 负责 t < fine覆盖范围，coarse 负责之后
                t_cutoff = t_phys_fine[-1]
                mask_c   = t_coarse > t_cutoff
                t_arr  = np.concatenate([t_fine,          t_coarse[mask_c]])
                Fsqt   = np.concatenate([Fs_fine,         Fs_coarse[:, mask_c]], axis=1)
                frames_t_phys = t_phys_coarse
                frames = frames_coarse  # 仅用于 directional 的 fallback

            else:
                # 仅 coarse
                if args.dt_frame is not None:
                    t_phys = np.arange(len(frames_coarse)) * args.dt_frame
                    print(f"  [time] using --dt_frame={args.dt_frame:g} tau0/frame")
                else:
                    t_phys = None
                max_lag   = args.max_lag or len(frames_coarse) // 2
                t_arr, Fsqt = compute_fsqt_from_dump(
                    frames_coarse, q_vals, sr,
                    particle_type=args.type,
                    n_skip=args.n_skip,
                    max_lag=max_lag,
                    mode=args.mode,
                    t_phys=t_phys,
                    verbose=True
                )
                frames = frames_coarse
                frames_t_phys = t_phys

            # 自动设 max_lag
            if not args.no_save:
                save_results(t_arr, Fsqt, q_vals, sr, args.mode, args.output)

            tau_list = plot_fsqt_single(t_arr, Fsqt, q_vals, sr,
                                         args.output, mode=args.mode)
            results_multi.append((sr, t_arr, Fsqt, q_vals))

            # ── 方向分解（仅用 coarse，够看各向异性趋势）──
            if args.directional:
                print(f"\n  [方向分解] γ̇={sr}")
                max_lag_d = args.max_lag or len(frames) // 2
                t_dict = {}; Fsqt_dict = {}
                for mode_d in ['isotropic', 'x', 'y', 'z']:
                    td, Fd = compute_fsqt_from_dump(
                        frames, np.array([Q_STAR_DEFAULT]), sr,
                        particle_type=args.type,
                        n_skip=args.n_skip,
                        max_lag=max_lag_d,
                        mode=mode_d,
                        t_phys=frames_t_phys,
                        verbose=False
                    )
                    t_dict[mode_d]  = td
                    Fsqt_dict[mode_d] = Fd
                    Fsqt_dict[f'q_vals_{mode_d}'] = np.array([Q_STAR_DEFAULT])
                plot_fsqt_directional(t_dict, Fsqt_dict, sr,
                                       Q_STAR_DEFAULT, args.output)

        # ── 多速率对比图 ──
        if len(results_multi) > 1:
            print("\n[绘制多速率对比图]")
            plot_fsqt_multirate(results_multi, args.output)

    # ── 路径 B：从 r_tilde npz 计算 ──
    elif args.npz is not None:
        if not os.path.exists(args.npz):
            print(f"✗ 找不到 {args.npz}"); sys.exit(1)

        sr = args.rate[0]
        print(f"\n[从 npz] {args.npz}  γ̇={sr}")

        t_arr, Fsqt = compute_fsqt_from_rtilde(
            args.npz, q_vals, sr,
            particle_type=args.type,
            n_skip=args.n_skip,
            max_lag=args.max_lag,
            mode=args.mode,
            verbose=True
        )

        if not args.no_save:
            save_results(t_arr, Fsqt, q_vals, sr, args.mode, args.output)

        plot_fsqt_single(t_arr, Fsqt, q_vals, sr, args.output, mode=args.mode)

    else:
        print("错误：请提供 --dump 或 --npz")
        p.print_help()
        sys.exit(1)

    print(f"\n✓ 完成！结果保存在 {args.output}/")
