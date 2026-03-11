"""
cage_jump_detection.py
======================
Candelier et al. (2009, 2010) 笼跳跃识别算法
严格按照论文 Appendix B 实现

关键说明（与之前版本的区别）：
  - jump_pos_xy  = 粒子发生 cage jump 时刻的真实实验室坐标 (x,y)，居中
                   这是论文 Fig.1b 散点图的数据（盒子内各处 CJ 的位置分布）
  - jump_vectors = 非仿射位移跳跃向量 Δr̃，用于各向异性分析（Fig.2c 等）
  - positions    = r_tilde（非仿射位移，非真实坐标，仅供算法内部使用）

  需要提供 --dump 参数（LAMMPS dump 文件）以读取真实粒子坐标。

用法：
    python cage_jump_detection.py \\
        --npz  msd_data.npz \\
        --dump dump.shear_0.015.lammpstrj \\
        --rate 0.015 \\
        --output figures
"""

# ===== PARAMETERS =====
LC2_DEFAULT    = 0.057
MIN_SEGMENT    = 4
MAX_PARTICLES  = 13200

import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from read_dump import read_lammps_dump
    HAS_READ_DUMP = True
except ImportError:
    HAS_READ_DUMP = False
    print("  ⚠ 未找到 read_dump.py，jump_pos_xy 将使用 r_tilde 近似（不推荐）")


# ─────────────────────────────────────────────────────────────────────────────
# 核心算法
# ─────────────────────────────────────────────────────────────────────────────

def compute_p_function(traj: np.ndarray):
    T = len(traj)
    if T < 4:
        return np.array([], dtype=int), np.array([])

    tc = np.arange(1, T - 1)
    n1 = tc + 1
    n2 = T - tc

    prefix = np.empty((T + 1, 3), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(traj, axis=0, out=prefix[1:])

    sq_norm = np.einsum('ij,ij->i', traj, traj)
    sq_prefix = np.empty(T + 1, dtype=np.float64)
    sq_prefix[0] = 0.0
    np.cumsum(sq_norm, out=sq_prefix[1:])

    c1 = prefix[tc + 1] / n1[:, None]
    c2 = (prefix[T] - prefix[tc]) / n2[:, None]

    mean_sq_S2 = (sq_prefix[T] - sq_prefix[tc]) / n2
    d1_sq = mean_sq_S2 - 2.0*np.einsum('ij,ij->i', c1, c2) + np.einsum('ij,ij->i', c1, c1)

    mean_sq_S1 = sq_prefix[tc + 1] / n1
    d2_sq = mean_sq_S1 - 2.0*np.einsum('ij,ij->i', c2, c1) + np.einsum('ij,ij->i', c2, c2)

    np.maximum(d1_sq, 0, out=d1_sq)
    np.maximum(d2_sq, 0, out=d2_sq)

    zeta = np.sqrt((tc / T) * (1.0 - tc / T))
    return tc, zeta * np.sqrt(d1_sq * d2_sq)


def find_cage_jumps_recursive(traj, lc2, time_offset=0):
    T = len(traj)
    if T < MIN_SEGMENT:
        return []
    tc_indices, p_values = compute_p_function(traj)
    if len(p_values) == 0:
        return []
    p_max_idx = np.argmax(p_values)
    p_max = p_values[p_max_idx]
    tc    = tc_indices[p_max_idx]
    if p_max < lc2:
        return []
    jump_frame = time_offset + tc
    return ([jump_frame]
            + find_cage_jumps_recursive(traj[:tc+1], lc2, time_offset)
            + find_cage_jumps_recursive(traj[tc:],   lc2, time_offset + tc))


def detect_all_cage_jumps(r_tilde, types, times, lc2, type_filter=1):
    mask = (types == type_filter)
    particle_indices = np.where(mask)[0]
    n_proc = min(len(particle_indices), MAX_PARTICLES)

    rng = np.random.default_rng(42)
    if n_proc < len(particle_indices):
        particle_indices = np.sort(rng.choice(particle_indices, size=n_proc, replace=False))

    n_frames = r_tilde.shape[0]
    print(f"  检测 type={type_filter} 粒子：{len(particle_indices)} 个，{n_frames} 帧")
    print(f"  lc² = {lc2:.4f}")

    all_pidx, all_jframes, all_jtimes = [], [], []
    all_jvec, all_jlen, all_jpos = [], [], []

    report_interval = max(1, n_proc // 20)
    n_with_jumps = total_jumps = 0

    for count, pi in enumerate(particle_indices):
        if count % report_interval == 0:
            print(f"  进度: {count}/{n_proc} ({count/n_proc*100:.0f}%)  "
                  f"跳跃: {total_jumps}", end='\r')

        traj = r_tilde[:, pi, :]
        jump_frames = sorted(set(find_cage_jumps_recursive(traj, lc2)))

        if jump_frames:
            n_with_jumps += 1
            for jf in jump_frames:
                jf = max(1, min(jf, n_frames - 2))
                jvec = r_tilde[jf+1, pi, :] - r_tilde[jf-1, pi, :]
                all_pidx.append(int(pi))
                all_jframes.append(int(jf))
                all_jtimes.append(float(times[jf]))
                all_jvec.append(jvec.copy())
                all_jlen.append(float(np.linalg.norm(jvec)))
                all_jpos.append(r_tilde[jf, pi, :].copy())  # r_tilde，非真实坐标
                total_jumps += 1

    print(f"\n  完成: {n_with_jumps}/{n_proc} 粒子有跳跃，共 {total_jumps} 个事件")

    return {
        'particle_idx': np.array(all_pidx,    dtype=int),
        'jump_frames':  np.array(all_jframes,  dtype=int),
        'jump_times':   np.array(all_jtimes),
        'jump_vectors': np.array(all_jvec)  if all_jvec  else np.zeros((0,3)),
        'jump_lengths': np.array(all_jlen),
        'positions':    np.array(all_jpos)  if all_jpos  else np.zeros((0,3)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ★ 从 dump 文件读取真实粒子坐标（论文 Fig.1b 的正确数据源）
# ─────────────────────────────────────────────────────────────────────────────

def extract_real_positions_from_dump(jumps, dump_file):
    """
    在每个 cage jump 发生的帧，读取粒子的真实实验室坐标 (x, y)，并以盒子中心为原点居中。

    关键修复：
      - dump 若提供 xu/yu（unwrapped），必须先对 Lx/Ly 取模折回盒内再居中
      - triclinic（剪切）盒：不要用 x + ix*Lx（不正确），直接用 wrapped 的 x/y
    """
    if not HAS_READ_DUMP or not os.path.exists(dump_file):
        print(f"  ✗ 无法读取 dump 文件: {dump_file}")
        return _fallback_pos(jumps)

    print(f"  读取真实坐标: {dump_file}")
    frames = read_lammps_dump(dump_file)
    n_frames = len(frames)

    box = frames[0]['box']
    Lx, Ly = box['Lx'], box['Ly']
    xlo, ylo = box['xlo'], box['ylo']
    cx = xlo + Lx / 2.0
    cy = ylo + Ly / 2.0
    print(f"  盒子 Lx={Lx:.3f}, Ly={Ly:.3f}  中心 ({cx:.3f}, {cy:.3f})")

    s = frames[0]
    has_xu = ('xu' in s) and ('yu' in s)
    has_x  = ('x'  in s) and ('y'  in s)

    particle_idx = jumps['particle_idx']
    jump_frames  = jumps['jump_frames']
    n_jumps = len(particle_idx)
    xy = np.zeros((n_jumps, 2), dtype=np.float64)

    for i, (pi, jf) in enumerate(zip(particle_idx, jump_frames)):
        fi = min(int(jf), n_frames - 1)
        f  = frames[fi]
        if has_xu:
            xu = float(f['xu'][pi]); yu = float(f['yu'][pi])
            x = (xu - xlo) % Lx + xlo - cx
            y = (yu - ylo) % Ly + ylo - cy
        elif has_x:
            x = float(f['x'][pi]) - cx
            y = float(f['y'][pi]) - cy
        else:
            x = y = 0.0
        xy[i] = [x, y]

    print(f"  jump_pos_xy: x=[{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]  "
          f"y=[{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")
    return xy

def _fallback_pos(jumps):
    """无 dump 时的近似（仅供调试，物理上不正确）。"""
    print("  ⚠ 回退到 r_tilde 中心化近似（强烈建议提供 --dump）")
    pos = jumps['positions'].copy()
    if len(pos) == 0:
        return np.zeros((0, 2))
    xy = pos[:, :2].copy()
    xy -= np.median(xy, axis=0)
    return xy


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def analyze_jump_lengths(jumps, shear_rate, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    lengths = jumps['jump_lengths']
    if len(lengths) == 0:
        print("  ⚠ 无跳跃事件"); return

    mean_lcj = np.mean(lengths)
    print(f"\n  ⟨l_cj⟩ = {mean_lcj:.4f}  (期望 ≈ 0.4 σ_bb),  N={len(lengths)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, max(2.0, np.percentile(lengths, 99)), 40)
    axes[0].hist(lengths, bins=bins, color='steelblue', edgecolor='white', density=True, alpha=0.85)
    axes[0].axvline(mean_lcj, color='red', ls='--', lw=2,
                    label=rf'$\langle l_{{cj}}\rangle={mean_lcj:.3f}$')
    axes[0].axvline(0.4, color='orange', ls=':', lw=1.5, label='Expected ≈ 0.4')
    axes[0].set_xlabel(r'$l_{cj}\ [\sigma_{bb}]$', fontsize=12)
    axes[0].set_ylabel('Prob. density', fontsize=12); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(rf'CJ length, $\dot\gamma={shear_rate}$')

    vecs = jumps['jump_vectors']
    if len(vecs) > 0:
        axes[1].scatter(vecs[:,0], vecs[:,1], s=4, alpha=0.3, c='steelblue')
        axes[1].set_xlabel(r'$\Delta\tilde{r}_x$'); axes[1].set_ylabel(r'$\Delta\tilde{r}_y$')
        lim = max(1.5, np.percentile(np.abs(vecs[:,:2]), 98))
        axes[1].set_xlim(-lim,lim); axes[1].set_ylim(-lim,lim)
        axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3)
        Pext = np.mean((vecs[:,0]*vecs[:,1])>0)
        axes[1].set_title(f'Jump vectors Δr̃  Pext={Pext:.2f}')

    plt.tight_layout()
    sr_str = str(shear_rate).replace('.','p')
    out = os.path.join(output_dir, f'cage_jumps_analysis_{sr_str}.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  → {out}")


def plot_trajectory_with_jumps(r_tilde, types, times, jumps,
                                output_dir='.', n_examples=4, shear_rate=0.015):
    os.makedirs(output_dir, exist_ok=True)
    if len(jumps['particle_idx']) == 0: return
    unique_pids = np.unique(jumps['particle_idx'])
    sel = np.random.choice(unique_pids, size=min(n_examples, len(unique_pids)), replace=False)
    fig, axes = plt.subplots(1, len(sel), figsize=(4*len(sel), 4.5))
    if len(sel) == 1: axes = [axes]
    colors = plt.cm.tab10.colors
    for ax, pi in zip(axes, sel):
        traj = r_tilde[:, pi, :]
        jfs = sorted(jumps['jump_frames'][jumps['particle_idx']==pi].tolist())
        bds = [0] + jfs + [len(traj)-1]
        for k in range(len(bds)-1):
            c = colors[k % len(colors)]
            ax.plot(traj[bds[k]:bds[k+1]+1,0], traj[bds[k]:bds[k+1]+1,1], '-', c=c, lw=1.5)
        for jf in jfs:
            ax.plot(traj[min(jf,len(traj)-1),0], traj[min(jf,len(traj)-1),1], 'k*', ms=10)
        ax.set_title(f'#{pi} ({len(jfs)} jumps)'); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.suptitle(rf'r̃ traj, $\dot\gamma={shear_rate}$')
    plt.tight_layout()
    out = os.path.join(output_dir, f'cage_jump_traj_{str(shear_rate).replace(".","p")}.png')
    fig.savefig(out, dpi=300); plt.close(fig); print(f"  → {out}")


def plot_spatial_distribution(jump_pos_xy, shear_rate, output_dir='.'):
    if len(jump_pos_xy) == 0: return
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(jump_pos_xy[:,0], jump_pos_xy[:,1], s=4, c='k', alpha=0.5, linewidths=0)
    ax.set_xlabel(r'$x / \sigma_{bb}$', fontsize=12)
    ax.set_ylabel(r'$y / \sigma_{bb}$', fontsize=12)
    ax.set_title(rf'Cage-jump positions, $\dot\gamma={shear_rate}$')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, f'cage_jump_spatial_{str(shear_rate).replace(".","p")}.png')
    fig.savefig(out, dpi=300); plt.close(fig); print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--npz',          default='msd_data.npz')
    p.add_argument('--dump',         default=None,
                   help='LAMMPS dump 文件（必须提供以获取真实坐标）')
    p.add_argument('--rate',         type=float, default=None)
    p.add_argument('--lc2',          type=float, default=None)
    p.add_argument('--output',       default='figures')
    p.add_argument('--type_filter',  type=int,   default=1)
    p.add_argument('--max_particles',type=int,   default=2000)
    args = p.parse_args()

    if not os.path.exists(args.npz):
        print(f"错误: 找不到 {args.npz}"); sys.exit(1)

    # 自动推断 dump 路径
    if args.dump is None:
        sr_guess = args.rate or 0.015
        guess = f'dump.shear_{sr_guess}.lammpstrj'
        if os.path.exists(guess):
            args.dump = guess
            print(f"  自动找到 dump: {args.dump}")
        else:
            print(f"  ⚠ 请提供 --dump 以获取真实粒子坐标（论文 Fig.1b）")

    print(f"\n{'='*55}")
    print(f"  r_tilde NPZ : {args.npz}")
    print(f"  dump 文件   : {args.dump or '未提供（使用 r_tilde 近似）'}")
    print(f"{'='*55}")

    data    = np.load(args.npz, allow_pickle=True)
    r_tilde = data['r_tilde'].astype(np.float64)
    times   = data['times']
    types   = data['types']

    shear_rate = float(args.rate or data['shear_rate'])
    lc2        = float(args.lc2  or data.get('lc2', LC2_DEFAULT))
    if lc2 <= 0: lc2 = LC2_DEFAULT
    if args.max_particles > 0: MAX_PARTICLES = args.max_particles

    print(f"  γ̇={shear_rate}  lc²={lc2:.4f}  r_tilde={r_tilde.shape}")

    # 1. 检测 cage jump（用 r_tilde）
    print("\n[1] 检测 cage jump...")
    jumps = detect_all_cage_jumps(r_tilde, types, times, lc2, args.type_filter)

    # 2. ★ 读取真实粒子坐标（论文 Fig.1b 的 x,y 来源）
    print("\n[2] 读取跳跃时刻的真实坐标...")
    if args.dump:
        jump_pos_xy = extract_real_positions_from_dump(jumps, args.dump)
    else:
        jump_pos_xy = _fallback_pos(jumps)

    # 3. 分析和绘图
    print("\n[3] 分析跳跃长度...")
    analyze_jump_lengths(jumps, shear_rate, args.output)

    print("\n[4] 绘制 r̃ 轨迹示例...")
    plot_trajectory_with_jumps(r_tilde, types, times, jumps,
                                args.output, shear_rate=shear_rate)

    print("\n[5] 绘制空间分布图...")
    plot_spatial_distribution(jump_pos_xy, shear_rate, args.output)

    # 4. 保存
    sr_str = str(shear_rate).replace('.','p')
    out_npz = os.path.join(os.path.dirname(args.npz),
                            f'cage_jumps_shearrate_{sr_str}.npz')

    box_Lx = box_Ly = np.float32(0)
    if args.dump and HAS_READ_DUMP and os.path.exists(args.dump):
        try:
            f0 = read_lammps_dump(args.dump, max_frames=1)
            box_Lx = np.float32(f0[0]['box']['Lx'])
            box_Ly = np.float32(f0[0]['box']['Ly'])
        except Exception: pass
    elif 'box_Lx' in data:
        box_Lx = data['box_Lx']; box_Ly = data['box_Ly']

    np.savez(out_npz,
             particle_idx = jumps['particle_idx'],
             jump_frames  = jumps['jump_frames'],
             jump_times   = jumps['jump_times'],
             jump_vectors = jumps['jump_vectors'],   # Δr̃，非仿射位移跳跃
             jump_lengths = jumps['jump_lengths'],
             positions    = jumps['positions'],       # r_tilde（非坐标）
             jump_pos_xy  = jump_pos_xy,              # ★ 真实 (x,y)，论文 Fig.1b 用这个
             shear_rate   = shear_rate,
             lc2          = lc2,
             types        = types,
             box_Lx       = box_Lx,
             box_Ly       = box_Ly,
             r_tilde      = r_tilde.astype(np.float32))

    print(f"\n  → 保存: {out_npz}")

    # 5. 验证
    print(f"\n{'='*55}")
    n_j = len(jumps['jump_lengths'])
    if n_j > 0:
        mean_lcj = np.mean(jumps['jump_lengths'])
        print(f"  N={n_j}  ⟨l_cj⟩={mean_lcj:.4f} (期望≈0.4)")
        Pext = np.mean((jumps['jump_vectors'][:,0]*jumps['jump_vectors'][:,1])>0)
        print(f"  Pext={Pext:.3f} (期望>0.5)")
        print(f"  jump_pos_xy: x∈[{jump_pos_xy[:,0].min():.1f},{jump_pos_xy[:,0].max():.1f}]"
              f"  y∈[{jump_pos_xy[:,1].min():.1f},{jump_pos_xy[:,1].max():.1f}]")
    print(f"{'='*55}")