"""
cage_jump_detection.py
======================
Candelier et al. (PRL 102, 2009; PRL 105, 2010) 笼跳跃识别算法
严格按照原始文献实现，已修复所有已知 bug。

Bug 修复记录（相对于旧版）：
  [BUG-1] compute_p_function：S₂ 定义错误
    原文明确写 S₂ for t₂ ∈ ]tc, T]（左开区间），tc 点只属于 S₁。
    旧代码用 traj[tc:] 把 tc 纳入 S₂，导致 c₂/d₁/d₂ 均算错。
    → 修正：n2 = T - tc - 1，c₂ 从 prefix[tc+1] 开始。

  [BUG-2] find_cage_jumps_recursive：递归右段起点错误
    旧代码：traj[tc:], offset+tc  ← tc 点被两段共享
    → 修正：traj[tc+1:], offset+tc+1

  [BUG-3] 跳跃向量计算过于粗糙
    旧代码用 r_tilde[jf+1] - r_tilde[jf-1]（单帧差，噪声大）。
    → 修正：用跳跃前后各 JVEC_WINDOW 帧的子轨迹质心差。

  [INFO]  cluster 时间阈值
    PRL 105 明确：τ_th = 2 × (cage detection precision) = 2 × MIN_SEGMENT。
    已在 cluster_cage_jumps() 中实现。

输入：
  --npz   : compute_fsqt.py 输出的 msd_data.npz（含 r_tilde, times, types）
  --dump  : LAMMPS dump 文件（用于读取真实粒子坐标，绘制 Fig.1b 类图）
  --rate  : 剪切速率（覆盖 npz 中的值）

输出：
  cage_jumps_shearrate_<rate>.npz
  figures/cage_jumps_analysis_<rate>.png
  figures/cage_jump_traj_<rate>.png
  figures/cage_jump_spatial_<rate>.png
  figures/cage_jump_clusters_<rate>.png
"""

# ===== PARAMETERS =====
LC2_DEFAULT  = 0.057   # KA 3D MD，论文 Appendix B
MIN_SEGMENT  = 4       # 最短子段帧数（cage detection precision）
JVEC_WINDOW  = 5       # 计算跳跃向量时两侧各取的帧数
MAX_PARTICLES = 13200  # type-1 粒子总数上限

import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

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
    """
    计算 Candelier 2009 公式 (2) 的 p(tc) 函数。

    原文定义：
        S₁ = {t₁ ∈ [0, tc]}    （含 tc，共 tc+1 个点）
        S₂ = {t₂ ∈ ]tc, T]}   （不含 tc，共 T-tc-1 个点）

        p(tc) = ζ(tc) · [⟨d₁(t₂)²⟩_{S₂} · ⟨d₂(t₁)²⟩_{S₁}]^{1/2}

    其中：
        d₁(t₂) = |r(t₂) - c₁|，c₁ = S₁ 质心
        d₂(t₁) = |r(t₁) - c₂|，c₂ = S₂ 质心
        ζ(tc)  = sqrt(tc/T · (1 - tc/T))

    参数
    ----
    traj : (T, D) array，粒子轨迹（r_tilde 或真实坐标均可）

    返回
    ----
    tc_indices : (T-2,) int array，有效的 tc 值（1 到 T-2）
    p_values   : (T-2,) float array，对应的 p(tc)
    """
    T = len(traj)
    if T < 4:
        return np.array([], dtype=int), np.array([])

    # tc 从 1 到 T-2（S₁ 至少 2 点，S₂ 至少 1 点）
    tc = np.arange(1, T - 1)          # shape (T-2,)

    # S₁: indices 0..tc  → n₁ = tc+1 个点
    # S₂: indices tc+1..T-1 → n₂ = T-tc-1 个点   ← [BUG-1 修复]
    n1 = tc + 1                        # shape (T-2,)
    n2 = T - tc - 1                    # shape (T-2,)   ← 旧版是 T-tc，少减了1

    # 前缀和（含位置和模长平方）
    prefix = np.empty((T + 1, traj.shape[1]), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(traj, axis=0, out=prefix[1:])

    sq_norm = np.einsum('ij,ij->i', traj, traj)
    sq_prefix = np.empty(T + 1, dtype=np.float64)
    sq_prefix[0] = 0.0
    np.cumsum(sq_norm, out=sq_prefix[1:])

    # S₁ 质心 c₁ = sum(traj[0..tc]) / (tc+1)
    c1 = prefix[tc + 1] / n1[:, None]                  # (T-2, D)

    # S₂ 质心 c₂ = sum(traj[tc+1..T-1]) / (T-tc-1)   ← [BUG-1 修复]
    valid = (n2 > 0)                                     # tc < T-1 时均成立
    c2 = np.where(
        valid[:, None],
        (prefix[T] - prefix[tc + 1]) / np.maximum(n2, 1)[:, None],
        0.0
    )                                                    # (T-2, D)

    # ⟨d₁(t₂)²⟩_{S₂} = ⟨|r_{S₂} - c₁|²⟩
    #   = ⟨r_{S₂}²⟩ - 2·c₁·c₂ + |c₁|²   （其中 ⟨r_{S₂}⟩ = c₂）
    mean_sq_S2 = np.where(
        valid,
        (sq_prefix[T] - sq_prefix[tc + 1]) / np.maximum(n2, 1),
        0.0
    )                                                    # (T-2,)  ← [BUG-1 修复]
    d1_sq = (mean_sq_S2
             - 2.0 * np.einsum('ij,ij->i', c1, c2)
             + np.einsum('ij,ij->i', c1, c1))

    # ⟨d₂(t₁)²⟩_{S₁} = ⟨|r_{S₁} - c₂|²⟩
    #   = ⟨r_{S₁}²⟩ - 2·c₂·c₁ + |c₂|²
    mean_sq_S1 = sq_prefix[tc + 1] / n1                 # (T-2,)
    d2_sq = (mean_sq_S1
             - 2.0 * np.einsum('ij,ij->i', c2, c1)
             + np.einsum('ij,ij->i', c2, c2))

    # 数值保护（浮点误差可能导致极小负值）
    np.maximum(d1_sq, 0.0, out=d1_sq)
    np.maximum(d2_sq, 0.0, out=d2_sq)

    # ζ(tc) = sqrt(tc/T · (1 - tc/T))
    zeta = np.sqrt((tc / T) * (1.0 - tc / T))

    p = zeta * np.sqrt(d1_sq * d2_sq)
    p[~valid] = 0.0                                      # n₂=0 时 p=0

    return tc, p


def find_cage_jumps_recursive(traj, lc2, time_offset=0):
    """
    递归地将轨迹分割为笼子段 + 跳跃事件。

    每次找到 p(tc) 最大值，如果大于 lc²，则记录 tc 为一个 cage jump，
    然后对 S₁ = traj[:tc+1] 和 S₂ = traj[tc+1:] 分别递归。

    [BUG-2 修复]：旧版右段是 traj[tc:]（含 tc），现改为 traj[tc+1:]。

    返回：该段内所有 cage jump 的绝对帧编号列表。
    """
    T = len(traj)
    if T < MIN_SEGMENT:
        return []

    tc_indices, p_values = compute_p_function(traj)
    if len(p_values) == 0:
        return []

    p_max_idx = np.argmax(p_values)
    p_max     = p_values[p_max_idx]
    tc        = int(tc_indices[p_max_idx])

    if p_max < lc2:
        return []

    jump_frame = time_offset + tc

    # 左段：S₁ = traj[0..tc]（含 tc，长度 tc+1）
    left  = find_cage_jumps_recursive(traj[:tc + 1], lc2, time_offset)

    # 右段：S₂ = traj[tc+1..]（不含 tc）  ← [BUG-2 修复]
    right = find_cage_jumps_recursive(traj[tc + 1:], lc2, time_offset + tc + 1)

    return [jump_frame] + left + right


def _jump_vector(r_tilde, pi, jf, n_frames, window=JVEC_WINDOW):
    """
    计算粒子 pi 在帧 jf 处的跳跃向量：跳后质心 − 跳前质心。

    [BUG-3 修复]：旧版用单帧差 r[jf+1]-r[jf-1]，噪声大。
    现改为各取 window 帧的均值，更稳健。
    """
    i0 = max(0,        jf - window)
    i1 = min(n_frames, jf)           # 跳前段 [i0, jf)
    j0 = jf + 1
    j1 = min(n_frames, jf + 1 + window)  # 跳后段 [jf+1, j1)

    if i1 <= i0 or j1 <= j0:
        # 边界情况，退化为单帧差
        jf_safe = max(1, min(jf, n_frames - 2))
        return r_tilde[jf_safe + 1, pi, :] - r_tilde[jf_safe - 1, pi, :]

    c_before = r_tilde[i0:i1, pi, :].mean(axis=0)
    c_after  = r_tilde[j0:j1, pi, :].mean(axis=0)
    return c_after - c_before


def detect_all_cage_jumps(r_tilde, types, times, lc2, type_filter=1):
    """
    对所有 type_filter 类型的粒子执行 cage jump 检测。

    参数
    ----
    r_tilde     : (n_frames, N, 3) float64，非仿射位移轨迹
    types       : (N,) int，粒子类型
    times       : (n_frames,) float，物理时间
    lc2         : float，笼子尺寸²（threshold）
    type_filter : int，只处理此类型的粒子（默认 1 = A 粒子）

    返回
    ----
    dict，键：particle_idx, jump_frames, jump_times,
              jump_vectors, jump_lengths, positions
    """
    mask = (types == type_filter)
    particle_indices = np.where(mask)[0]
    n_proc = min(len(particle_indices), MAX_PARTICLES)

    rng = np.random.default_rng(42)
    if n_proc < len(particle_indices):
        particle_indices = np.sort(
            rng.choice(particle_indices, size=n_proc, replace=False)
        )

    n_frames = r_tilde.shape[0]
    print(f"  检测 type={type_filter} 粒子：{len(particle_indices)} 个，{n_frames} 帧")
    print(f"  lc² = {lc2:.4f}  (lc = {lc2**0.5:.4f})")

    all_pidx, all_jframes, all_jtimes = [], [], []
    all_jvec, all_jlen, all_jpos     = [], [], []

    report_interval = max(1, n_proc // 20)
    n_with_jumps = total_jumps = 0

    for count, pi in enumerate(particle_indices):
        if count % report_interval == 0:
            print(f"  进度: {count}/{n_proc} ({count/n_proc*100:.0f}%)  "
                  f"跳跃事件: {total_jumps}", end='\r')

        traj = r_tilde[:, pi, :]

        # 递归分割，去重排序
        jump_frames = sorted(set(find_cage_jumps_recursive(traj, lc2)))

        if jump_frames:
            n_with_jumps += 1
            for jf in jump_frames:
                jf = max(1, min(int(jf), n_frames - 2))

                # [BUG-3 修复] 用窗口质心差计算跳跃向量
                jvec = _jump_vector(r_tilde, pi, jf, n_frames)

                all_pidx.append(int(pi))
                all_jframes.append(jf)
                all_jtimes.append(float(times[jf]))
                all_jvec.append(jvec.copy())
                all_jlen.append(float(np.linalg.norm(jvec)))
                all_jpos.append(r_tilde[jf, pi, :].copy())   # r_tilde 坐标（供内部用）
                total_jumps += 1

    print(f"\n  完成: {n_with_jumps}/{n_proc} 粒子有跳跃，共 {total_jumps} 个事件")

    return {
        'particle_idx': np.array(all_pidx,   dtype=int),
        'jump_frames':  np.array(all_jframes, dtype=int),
        'jump_times':   np.array(all_jtimes,  dtype=float),
        'jump_vectors': np.array(all_jvec)  if all_jvec  else np.zeros((0, 3)),
        'jump_lengths': np.array(all_jlen,   dtype=float),
        'positions':    np.array(all_jpos)  if all_jpos  else np.zeros((0, 3)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cluster 聚类（PRL 102 §2 + PRL 105 Table I）
# ─────────────────────────────────────────────────────────────────────────────

def cluster_cage_jumps(jumps, r_tilde, types, dt_phys,
                        tau_th_frames=None, r_neighbor=1.5):
    """
    将 cage jump 事件聚类为 cluster（空间相邻 + 时间相近）。

    聚类准则（PRL 102/105）：
      - 空间：两个 jump 的粒子互为近邻（真实坐标距离 < r_neighbor）
      - 时间：|t₁ - t₂| < τ_th

    PRL 105：τ_th = 2 × MIN_SEGMENT（帧数）= 2 × cage detection precision

    参数
    ----
    jumps        : detect_all_cage_jumps 的输出
    r_tilde      : (n_frames, N, 3)，用于取各粒子在 jump 时刻的坐标
    types        : (N,) 粒子类型
    dt_phys      : 帧间隔（物理时间单位）
    tau_th_frames: 时间阈值（帧数），默认 2 × MIN_SEGMENT
    r_neighbor   : 空间邻居半径（σ_AA 单位）

    返回
    ----
    cluster_ids : (n_jumps,) int array，每个 jump 属于哪个 cluster（-1 = 未聚类）
    n_clusters  : int
    """
    if tau_th_frames is None:
        tau_th_frames = 2 * MIN_SEGMENT

    n_jumps = len(jumps['particle_idx'])
    if n_jumps == 0:
        return np.array([], dtype=int), 0

    pidx   = jumps['particle_idx']
    jframe = jumps['jump_frames']
    pos_xy = jumps['positions'][:, :2]   # r_tilde 的 xy，仅用于近似距离

    cluster_ids = np.full(n_jumps, -1, dtype=int)
    cid = 0

    # 简单的贪心 BFS 聚类
    unvisited = set(range(n_jumps))

    for seed in range(n_jumps):
        if seed not in unvisited:
            continue

        # 找时间邻居
        dt = np.abs(jframe - jframe[seed])
        time_nbrs = np.where(dt <= tau_th_frames)[0]

        # 进一步筛选空间邻居
        cluster_members = []
        for nb in time_nbrs:
            if nb not in unvisited:
                continue
            dr = np.linalg.norm(pos_xy[nb] - pos_xy[seed])
            if dr < r_neighbor:
                cluster_members.append(nb)

        if cluster_members:
            for m in cluster_members:
                cluster_ids[m] = cid
                unvisited.discard(m)
            cid += 1

    # 单独的 jump（未与任何邻居聚在一起）保留为独立 cluster
    for i in range(n_jumps):
        if cluster_ids[i] == -1:
            cluster_ids[i] = cid
            cid += 1

    n_clusters = cid
    print(f"  → 聚类结果：{n_jumps} 个事件 → {n_clusters} 个 cluster")
    sizes = np.bincount(cluster_ids)
    print(f"     cluster 大小：均值 {sizes.mean():.1f}，最大 {sizes.max()}")

    return cluster_ids, n_clusters


# ─────────────────────────────────────────────────────────────────────────────
# 真实坐标读取（论文 Fig.1b 的数据源）
# ─────────────────────────────────────────────────────────────────────────────

def extract_real_positions_from_dump(jumps, dump_file):
    """
    在每个 cage jump 发生的帧，读取粒子的真实实验室坐标 (x, y)，
    并以盒子中心为原点居中。

    注意（triclinic 剪切盒）：
      - 每帧的 xlo 随 xy-tilt 变化，统计图用第 0 帧的 Lx/Ly 近似折叠即可。
      - 如需精确折叠，需按帧读取 xlo（对统计分布图影响很小）。
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
            # 折叠回盒内再居中
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
    """无 dump 文件时的近似（仅供调试，不用于正式分析）。"""
    print("  ⚠ 回退到 r_tilde 近似坐标（建议提供 --dump）")
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
    lc_ref   = LC2_DEFAULT ** 0.5   # ≈ 0.239，论文中约等于 0.4 σ（用不同归一化时）
    print(f"\n  ⟨l_cj⟩ = {mean_lcj:.4f}  lc = {lc_ref:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 跳跃长度分布
    bins = np.linspace(0, max(2.0, np.percentile(lengths, 99)), 40)
    axes[0].hist(lengths, bins=bins, color='steelblue', edgecolor='white',
                 density=True, alpha=0.85)
    axes[0].axvline(mean_lcj, color='red', ls='--', lw=2,
                    label=rf'$\langle l_{{cj}}\rangle={mean_lcj:.3f}$')
    axes[0].axvline(lc_ref,  color='orange', ls=':', lw=1.5,
                    label=f'$l_c={lc_ref:.3f}$')
    axes[0].set_xlabel(r'$l_{cj}$', fontsize=12)
    axes[0].set_ylabel('Prob. density', fontsize=12)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(rf'CJ length dist., $\dot\gamma={shear_rate}$')

    # 跳跃向量散点（剪切平面投影）
    vecs = jumps['jump_vectors']
    if len(vecs) > 0:
        axes[1].scatter(vecs[:, 0], vecs[:, 1], s=4, alpha=0.3, c='steelblue')
        axes[1].set_xlabel(r'$\Delta\tilde{r}_x$', fontsize=12)
        axes[1].set_ylabel(r'$\Delta\tilde{r}_y$', fontsize=12)
        lim = max(1.5, np.percentile(np.abs(vecs[:, :2]), 98))
        axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
        axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3)
        # 延伸方向（x·y > 0）的比例（期望 > 0.5，因剪切沿 45° 延伸方向）
        Pext = np.mean((vecs[:, 0] * vecs[:, 1]) > 0)
        axes[1].set_title(f'Jump vectors Δr̃   $P_{{ext}}$={Pext:.2f}')

    plt.tight_layout()
    sr_str = str(shear_rate).replace('.', 'p')
    out = os.path.join(output_dir, f'cage_jumps_analysis_{sr_str}.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  → {out}")


def plot_trajectory_with_jumps(r_tilde, types, times, jumps,
                                output_dir='.', n_examples=4, shear_rate=0.015):
    os.makedirs(output_dir, exist_ok=True)
    if len(jumps['particle_idx']) == 0:
        return
    unique_pids = np.unique(jumps['particle_idx'])
    sel = np.random.choice(unique_pids,
                           size=min(n_examples, len(unique_pids)), replace=False)
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4.5))
    if len(sel) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for ax, pi in zip(axes, sel):
        traj = r_tilde[:, pi, :]
        jfs  = sorted(jumps['jump_frames'][jumps['particle_idx'] == pi].tolist())
        bds  = [0] + jfs + [len(traj) - 1]
        for k in range(len(bds) - 1):
            c = colors[k % len(colors)]
            ax.plot(traj[bds[k]:bds[k+1]+1, 0],
                    traj[bds[k]:bds[k+1]+1, 1], '-', c=c, lw=1.5)
        for jf in jfs:
            ax.plot(traj[min(jf, len(traj)-1), 0],
                    traj[min(jf, len(traj)-1), 1], 'k*', ms=10)
        ax.set_title(f'#{pi} ({len(jfs)} jumps)')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.suptitle(rf'$\tilde{{r}}$ trajectory, $\dot\gamma={shear_rate}$')
    plt.tight_layout()
    out = os.path.join(output_dir,
                       f'cage_jump_traj_{str(shear_rate).replace(".", "p")}.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  → {out}")


def plot_spatial_distribution(jump_pos_xy, shear_rate, output_dir='.'):
    if len(jump_pos_xy) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(jump_pos_xy[:, 0], jump_pos_xy[:, 1],
               s=4, c='k', alpha=0.5, linewidths=0)
    ax.set_xlabel(r'$x / \sigma_{AA}$', fontsize=12)
    ax.set_ylabel(r'$y / \sigma_{AA}$', fontsize=12)
    ax.set_title(rf'Cage-jump positions, $\dot\gamma={shear_rate}$')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir,
                       f'cage_jump_spatial_{str(shear_rate).replace(".", "p")}.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  → {out}")


def plot_cluster_sizes(cluster_ids, shear_rate, output_dir='.'):
    if len(cluster_ids) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    sizes = np.bincount(cluster_ids)
    sizes = sizes[sizes > 0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.arange(1, sizes.max() + 2) - 0.5
    ax.hist(sizes, bins=bins, density=True, color='steelblue',
            edgecolor='white', alpha=0.85, label='data')

    # PRL 102 报告幂律 ρ(Nc) ∝ Nc^{-α}，α ∈ [3/2, 2]
    nc_vals = np.arange(1, sizes.max() + 1, dtype=float)
    for alpha, ls in [(1.5, '--'), (2.0, ':')]:
        norm = np.sum(nc_vals ** (-alpha))
        ax.plot(nc_vals, nc_vals ** (-alpha) / norm,
                ls=ls, color='red', lw=1.5, label=rf'$\alpha={alpha}$')

    ax.set_xlabel('Cluster size $N_c$', fontsize=12)
    ax.set_ylabel('Prob. density', fontsize=12)
    ax.set_yscale('log'); ax.set_xscale('log')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(rf'Cluster size dist., $\dot\gamma={shear_rate}$')
    plt.tight_layout()
    out = os.path.join(output_dir,
                       f'cage_jump_clusters_{str(shear_rate).replace(".", "p")}.png')
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Candelier cage-jump detection for SLLOD shear simulations')
    p.add_argument('--npz',           default='msd_data.npz',
                   help='r_tilde NPZ 文件（compute_fsqt.py 输出）')
    p.add_argument('--dump',          default=None,
                   help='LAMMPS dump 文件（用于读取真实坐标，论文 Fig.1b）')
    p.add_argument('--rate',          type=float, default=None,
                   help='剪切速率（覆盖 npz 中的值）')
    p.add_argument('--lc2',           type=float, default=None,
                   help=f'笼子尺寸²（默认 {LC2_DEFAULT}，KA 3D MD 论文值）')
    p.add_argument('--output',        default='figures',
                   help='图像输出目录')
    p.add_argument('--type_filter',   type=int,   default=1,
                   help='只处理此类型的粒子（1=A，2=B）')
    p.add_argument('--max_particles', type=int,   default=2000,
                   help='最多处理多少粒子（0=全部）')
    p.add_argument('--no_cluster',    action='store_true',
                   help='跳过 cluster 聚类步骤')
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

    print(f"\n{'='*60}")
    print(f"  r_tilde NPZ : {args.npz}")
    print(f"  dump 文件   : {args.dump or '未提供（使用 r_tilde 近似）'}")
    print(f"{'='*60}")

    data    = np.load(args.npz, allow_pickle=True)
    r_tilde = data['r_tilde'].astype(np.float64)
    times   = data['times']
    types   = data['types']

    shear_rate = float(args.rate if args.rate is not None
                       else data.get('shear_rate', 0.015))
    lc2        = float(args.lc2  if args.lc2  is not None
                       else float(data.get('lc2', LC2_DEFAULT)))
    if lc2 <= 0:
        lc2 = LC2_DEFAULT

    if args.max_particles > 0:
        MAX_PARTICLES = args.max_particles

    dt_phys = float(times[1] - times[0]) if len(times) > 1 else 1.0

    print(f"  γ̇={shear_rate}  lc²={lc2:.4f}  lc={lc2**0.5:.4f}")
    print(f"  r_tilde shape: {r_tilde.shape}")
    print(f"  dt_phys = {dt_phys:.4f}")

    # ── 1. Cage jump 检测 ──────────────────────────────────────────────────
    print("\n[1] 检测 cage jump...")
    jumps = detect_all_cage_jumps(r_tilde, types, times, lc2, args.type_filter)

    # ── 2. 读取真实粒子坐标 ────────────────────────────────────────────────
    print("\n[2] 读取跳跃时刻的真实坐标...")
    if args.dump:
        jump_pos_xy = extract_real_positions_from_dump(jumps, args.dump)
    else:
        jump_pos_xy = _fallback_pos(jumps)

    # ── 3. Cluster 聚类 ────────────────────────────────────────────────────
    cluster_ids = np.array([], dtype=int)
    n_clusters  = 0
    if not args.no_cluster and len(jumps['particle_idx']) > 0:
        print("\n[3] Cluster 聚类...")
        cluster_ids, n_clusters = cluster_cage_jumps(
            jumps, r_tilde, types, dt_phys,
            tau_th_frames=2 * MIN_SEGMENT
        )

    # ── 4. 可视化 ──────────────────────────────────────────────────────────
    print("\n[4] 分析跳跃长度...")
    analyze_jump_lengths(jumps, shear_rate, args.output)

    print("\n[5] 绘制 r̃ 轨迹示例...")
    plot_trajectory_with_jumps(r_tilde, types, times, jumps,
                                args.output, shear_rate=shear_rate)

    print("\n[6] 绘制空间分布图...")
    plot_spatial_distribution(jump_pos_xy, shear_rate, args.output)

    if len(cluster_ids) > 0:
        print("\n[7] 绘制 cluster 大小分布...")
        plot_cluster_sizes(cluster_ids, shear_rate, args.output)

    # ── 5. 保存 NPZ ────────────────────────────────────────────────────────
    sr_str  = str(shear_rate).replace('.', 'p')
    out_npz = os.path.join(os.path.dirname(args.npz),
                           f'cage_jumps_shearrate_{sr_str}.npz')

    box_Lx = box_Ly = np.float32(0)
    if args.dump and HAS_READ_DUMP and os.path.exists(args.dump):
        try:
            f0 = read_lammps_dump(args.dump, max_frames=1)
            box_Lx = np.float32(f0[0]['box']['Lx'])
            box_Ly = np.float32(f0[0]['box']['Ly'])
        except Exception:
            pass
    elif 'box_Lx' in data:
        box_Lx = data['box_Lx']
        box_Ly = data['box_Ly']

    save_dict = dict(
        particle_idx = jumps['particle_idx'],
        jump_frames  = jumps['jump_frames'],
        jump_times   = jumps['jump_times'],
        jump_vectors = jumps['jump_vectors'],   # Δr̃ 跳跃向量（质心差）
        jump_lengths = jumps['jump_lengths'],
        positions    = jumps['positions'],       # r_tilde 坐标（内部用）
        jump_pos_xy  = jump_pos_xy,              # ★ 真实 (x,y)，Fig.1b 用这个
        shear_rate   = shear_rate,
        lc2          = lc2,
        types        = types,
        box_Lx       = box_Lx,
        box_Ly       = box_Ly,
        r_tilde      = r_tilde.astype(np.float32),
    )
    if len(cluster_ids) > 0:
        save_dict['cluster_ids'] = cluster_ids
        save_dict['n_clusters']  = n_clusters

    np.savez(out_npz, **save_dict)
    print(f"\n  → 保存: {out_npz}")

    # ── 6. 验证摘要 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    n_j = len(jumps['jump_lengths'])
    if n_j > 0:
        mean_lcj = np.mean(jumps['jump_lengths'])
        lc_ref   = lc2 ** 0.5
        print(f"  跳跃事件数   N  = {n_j}")
        print(f"  ⟨l_cj⟩       = {mean_lcj:.4f}  (lc = {lc_ref:.4f})")
        Pext = np.mean((jumps['jump_vectors'][:, 0] *
                        jumps['jump_vectors'][:, 1]) > 0)
        print(f"  P_ext        = {Pext:.3f}  (剪切流中期望 > 0.5)")
        print(f"  jump_pos_xy  x∈[{jump_pos_xy[:,0].min():.1f},"
              f"{jump_pos_xy[:,0].max():.1f}]  "
              f"y∈[{jump_pos_xy[:,1].min():.1f},{jump_pos_xy[:,1].max():.1f}]")
        if n_clusters > 0:
            sizes = np.bincount(cluster_ids)
            print(f"  cluster 数   = {n_clusters}  "
                  f"平均大小 = {sizes.mean():.1f}  最大 = {sizes.max()}")
    else:
        print("  ⚠ 未检测到任何 cage jump，请检查 lc² 设置或轨迹长度")
    print(f"{'='*60}")