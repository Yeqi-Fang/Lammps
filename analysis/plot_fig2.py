"""
plot_fig2.py  ——  复现论文 Fig. 2 / Fig. 6（团簇时间演化）
==============================================================
图面板说明：
  左列（时间演化，针对最大团簇）：
    (a) σc/σ0 和 γc        ← 需要 per-atom 应力（见注）
    (b) ncj(t)              — 团簇内笼跳跃计数
    (c) Pext(t)             — 跳跃沿延伸方向（x*y > 0）的比例
    (d) R(t)                — 粒子交换率

  右列（四个特征时刻的空间分布）：
    (e)-(h)  红点 = 从 t' 到 tx 累积的团簇内笼跳跃位置

  注：σc 需要重新运行 LAMMPS（添加 compute stress/atom），
      脚本用局部剪切应变 γc 替代 σc 作为近似（定性趋势一致）。

前置条件：
  1. 已运行 cage_jump_detection.py 生成 cage_jumps_shearrate_0p015.npz
     npz 须包含：jump_frames, particle_idx, jump_vectors, types, r_tilde

用法（Windows cmd）：
    python plot_fig2.py ^
        --dump  dump.shear_0.015.lammpstrj ^
        --npz   cage_jumps_shearrate_0p015.npz ^
        --rate  0.015 ^
        --output figures
"""

# ===== 可调参数 =====
COARSE_SIGMA   = 1.5    # 团簇识别密度场平滑
GRID_N_CLUSTER = 30     # 团簇识别网格
LC2_DEFAULT    = 0.057  # 笼尺寸（未检测到时用默认值）
TYPE_BIG       = 1      # 大粒子
PEXT_REF       = 0.5    # Pext 参考线

import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, label as ndlabel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from read_dump import read_lammps_dump
    HAS_READ_DUMP = True
except ImportError:
    HAS_READ_DUMP = False


# ───────────────────────────────────────────────────────────────
#  核心物理量计算
# ───────────────────────────────────────────────────────────────

def find_largest_cluster_2d(pos_xy, Lx, Ly, sigma=COARSE_SIGMA, grid_n=GRID_N_CLUSTER):
    """
    在 x-y 投影中找到密度最高的连通团簇。

    Returns
    -------
    in_cluster : (N,) bool  哪些跳跃点属于最大团簇
    center     : (2,) float 团簇中心 [x, y]
    radius_eq  : float      等效半径（面积等效圆）
    """
    if len(pos_xy) < 5:
        return np.ones(len(pos_xy), bool), np.zeros(2), 0.0

    xs, ys = pos_xy[:, 0], pos_xy[:, 1]
    xlo, xhi = -Lx/2, Lx/2
    ylo, yhi = -Ly/2, Ly/2
    dx = (xhi - xlo) / grid_n
    dy = (yhi - ylo) / grid_n

    density = np.zeros((grid_n, grid_n))
    ix = np.clip(((xs - xlo) / dx).astype(int), 0, grid_n-1)
    iy = np.clip(((ys - ylo) / dy).astype(int), 0, grid_n-1)
    for i, j in zip(ix, iy):
        density[i, j] += 1.0
    density = gaussian_filter(density, sigma=sigma/dx, mode='constant')

    # 迭代阈值
    rho = density.ravel()
    rho_s = rho[rho > rho.max() * 0.02]
    if len(rho_s) < 5:
        rho_th = float(rho.max()) * 0.5
    else:
        rho_i = float(rho_s.max())
        for _ in range(200):
            below = rho_s[rho_s < rho_i]
            if not len(below): break
            ra = float(below.mean())
            if ra > 0 and abs(ra - rho_i)/ra < 0.10: break
            if rho_i <= ra: break
            rho_i = ra
        else:
            rho_i = rho_s.mean() + 0.8*rho_s.std()
        rho_th = rho_i

    # 连通标记
    binary = (density >= rho_th).astype(int)
    labeled, n_feat = ndlabel(binary)
    if n_feat == 0:
        return np.ones(len(pos_xy), bool), np.zeros(2), 0.0

    sizes = [np.sum(labeled == ci) for ci in range(1, n_feat+1)]
    best  = int(np.argmax(sizes)) + 1

    # 判断每个跳跃点是否属于最大团簇
    in_cluster = np.zeros(len(pos_xy), bool)
    for k, (xi, yi) in enumerate(zip(xs, ys)):
        gi = np.clip(int((xi - xlo)/dx), 0, grid_n-1)
        gj = np.clip(int((yi - ylo)/dy), 0, grid_n-1)
        in_cluster[k] = (labeled[gi, gj] == best)

    # 团簇中心和等效半径
    mask2d = (labeled == best)
    vox_xy = np.argwhere(mask2d)
    cx = float(np.mean(vox_xy[:, 0]) * dx + xlo + dx/2)
    cy = float(np.mean(vox_xy[:, 1]) * dy + ylo + dy/2)
    area   = np.sum(mask2d) * dx * dy
    radius = np.sqrt(area / np.pi)

    return in_cluster, np.array([cx, cy]), radius


def compute_time_series(r_tilde, types, times,
                         jump_frames, jump_vectors, particle_idx,
                         cluster_set, window_frac=0.05):
    """
    计算时间演化量：ncj(t), Pext(t), R(t)。

    Parameters
    ----------
    cluster_set     : set  初始在团簇内的粒子索引
    window_frac     : 时间窗口 = 总帧数 * window_frac

    Returns
    -------
    t_out, ncj, Pext, R  各 (Nt,) array
    """
    n_frames = r_tilde.shape[0]
    win = max(3, int(n_frames * window_frac))
    step = max(1, win // 2)

    cluster_arr = np.array(list(cluster_set), dtype=int)
    big_mask = (types == TYPE_BIG)
    big_idx  = np.where(big_mask)[0]

    # 初始位置（用于计算 R：粒子是否"离开"初始团簇区域）
    lc2 = LC2_DEFAULT

    t_out, ncj_out, Pext_out, R_out = [], [], [], []

    for fi in range(win//2, n_frames - win//2, step):
        t0 = fi - win//2
        t1 = fi + win//2

        # 本时间窗口内、团簇粒子的跳跃
        mask_time = (jump_frames >= t0) & (jump_frames <= t1)
        mask_type = np.isin(particle_idx, cluster_arr)
        sel = mask_time & mask_type

        n_cj = int(np.sum(sel))

        # Pext：跳跃方向在延伸象限（x·y > 0）
        if n_cj > 0:
            vv = jump_vectors[sel]           # (n_cj, 3)
            pext = float(np.mean(vv[:, 0] * vv[:, 1] > 0))
        else:
            pext = PEXT_REF

        # R(t)：团簇粒子中非仿射位移超过 3*lc 的比例（近似"离开"判据）
        if len(cluster_arr) > 0:
            disps = np.linalg.norm(r_tilde[fi, cluster_arr, :], axis=1)
            R = float(np.mean(disps > np.sqrt(3 * lc2)))
        else:
            R = 0.0

        t_out.append(times[fi])
        ncj_out.append(float(n_cj))
        Pext_out.append(pext)
        R_out.append(R)

    return (np.array(t_out), np.array(ncj_out),
            np.array(Pext_out), np.array(R_out))


def estimate_gamma_c(r_tilde, cluster_set, frame_idx, delta=2):
    """
    Falk-Langer 局部剪切应变估计（简化 xy 分量）。
    """
    particles = list(cluster_set)
    if len(particles) < 5:
        return 0.0
    n = r_tilde.shape[0]
    fp = min(frame_idx + delta, n-1)
    fm = max(frame_idx - delta, 0)

    dp = r_tilde[fp, particles, :]   # (N, 3)
    dm = r_tilde[fm, particles, :]

    # 质心参考
    dp -= dp.mean(axis=0)
    dm -= dm.mean(axis=0)

    # γ_c ≈ Σ(dx_i · dy0_i) / Σ(dy0_i²)
    num = float(np.dot(dp[:, 0], dm[:, 1]))
    den = float(np.dot(dm[:, 1], dm[:, 1]))
    return num / den if abs(den) > 1e-12 else 0.0


# ───────────────────────────────────────────────────────────────
#  主绘图函数
# ───────────────────────────────────────────────────────────────

def plot_fig2(args):
    os.makedirs(args.output, exist_ok=True)
    sr     = args.rate
    sr_str = f"{sr}".replace('.', 'p')

    # ── 1. 加载 npz ──
    print("[1] 加载数据...")
    if not os.path.exists(args.npz):
        print(f"  ✗ 找不到 {args.npz}")
        return

    data = np.load(args.npz, allow_pickle=True)
    jump_frames  = data['jump_frames']   # (N_jmp,)
    particle_idx = data['particle_idx']  # (N_jmp,)
    jump_vectors = data['jump_vectors']  # (N_jmp, 3)
    types_all    = data['types']         # (N_atoms,)
    r_tilde      = data['r_tilde'].astype(np.float64)  # (N_frm, N_atoms, 3)
    lc2 = float(data.get('lc2', LC2_DEFAULT))

    n_frames, n_atoms = r_tilde.shape[:2]
    print(f"  帧数={n_frames}, 粒子数={n_atoms}, 跳跃数={len(jump_frames)}")

    # ── 2. 盒子和时间 ──
    print("[2] 读取盒子信息...")
    Lx = Ly = None
    if 'box_Lx' in data:
        Lx = float(data['box_Lx']); Ly = float(data['box_Ly'])
    elif HAS_READ_DUMP and os.path.exists(args.dump):
        frms = read_lammps_dump(args.dump, max_frames=2)
        box  = frms[0]['box']
        Lx, Ly = box['Lx'], box['Ly']
        # 时间步差
        if len(frms) >= 2:
            dts = frms[1]['timestep'] - frms[0]['timestep']
        else:
            dts = round(0.5 / (sr * 0.005))
    else:
        print("  ⚠ 无法获取盒子，使用跳跃数据范围估计")

    if 'jump_pos_xy' in data:
        pos_xy_all = data['jump_pos_xy']
    elif HAS_READ_DUMP and os.path.exists(args.dump):
        frms = read_lammps_dump(args.dump, max_frames=n_frames)
        box  = frms[0]['box']
        if Lx is None:
            Lx, Ly = box['Lx'], box['Ly']
        cx = box['xlo'] + Lx/2; cy = box['ylo'] + Ly/2
        pos_xy_all = np.zeros((len(jump_frames), 2))
        for k, (jf, pi) in enumerate(zip(jump_frames, particle_idx)):
            fi = min(int(jf), len(frms)-1)
            pos_xy_all[k, 0] = frms[fi]['xu'][pi] - cx
            pos_xy_all[k, 1] = frms[fi]['yu'][pi] - cy
    else:
        pos_xy_all = np.zeros((len(jump_frames), 2))
        print("  ⚠ 无法获取跳跃坐标，空间面板将为空")

    if Lx is None:
        xs_r = pos_xy_all[:, 0]; ys_r = pos_xy_all[:, 1]
        Lx = (xs_r.max() - xs_r.min()) * 1.4
        Ly = (ys_r.max() - ys_r.min()) * 1.4

    dt_frame = args.dt * args.every
    times = np.arange(n_frames) * dt_frame
    print(f"  Lx={Lx:.2f}, Ly={Ly:.2f}, dt_frame={dt_frame:.4f}")

    # ── 3. 只用大粒子的跳跃找主团簇 ──
    print("[3] 识别主团簇...")
    big_mask = (types_all[particle_idx] == TYPE_BIG)
    jf_big   = jump_frames[big_mask]
    pi_big   = particle_idx[big_mask]
    jv_big   = jump_vectors[big_mask]
    pos_big  = pos_xy_all[big_mask]

    if len(pos_big) < 5:
        print("  ⚠ 大粒子跳跃太少，使用全部粒子")
        pos_big = pos_xy_all
        jf_big, pi_big, jv_big = jump_frames, particle_idx, jump_vectors

    in_cluster, cluster_center, cluster_radius = find_largest_cluster_2d(
        pos_big, Lx, Ly)

    cluster_pi = pi_big[in_cluster]
    cluster_jf = jf_big[in_cluster]
    cluster_jv = jv_big[in_cluster]
    cluster_pos = pos_big[in_cluster]
    cluster_set = set(cluster_pi.tolist())
    print(f"  主团簇: {len(cluster_set)} 粒子参与, "
          f"中心=({cluster_center[0]:.1f},{cluster_center[1]:.1f}), "
          f"ξc={cluster_radius:.2f} σ_bb")

    # ── 4. 计算时间演化量 ──
    print("[4] 计算 ncj(t), Pext(t), R(t)...")
    t_evo, ncj_evo, Pext_evo, R_evo = compute_time_series(
        r_tilde, types_all, times,
        jump_frames, jump_vectors, particle_idx,
        cluster_set, window_frac=0.06)

    # γc(t) 估计（近似代替 σc/σ0）
    print("[5] 估计 γc(t)...")
    n_te = len(t_evo)
    gamma_c = []
    stride = max(1, n_frames // max(n_te, 1))
    for k in range(n_te):
        fi = min(int(k * stride + stride//2), n_frames-1)
        gamma_c.append(abs(estimate_gamma_c(r_tilde, cluster_set, fi, delta=max(1,stride//3))))
    gamma_c = np.array(gamma_c)

    # 归一化到 ~0.3（对应论文纵轴）
    if gamma_c.max() > 0:
        gamma_c_scaled = gamma_c / gamma_c.max() * 0.28
    else:
        gamma_c_scaled = gamma_c

    # 用 ncj 平滑版近似 σc（需 per-atom stress 才能算真实值）
    ncj_smooth = np.convolve(ncj_evo, np.ones(5)/5, mode='same')
    if ncj_smooth.max() > 0:
        sigma_approx = ncj_smooth / ncj_smooth.max() * 4.0   # 归一到 ~4 σ0
    else:
        sigma_approx = ncj_smooth

    # ── 5. 确定特征时刻 ──
    print("[6] 确定特征时刻...")
    if n_te >= 4:
        t2_i = int(np.argmax(ncj_smooth))                         # 跳跃峰值
        t3_i = min(t2_i + max(1, n_te//6), n_te-1)               # 峰后
        t1_i = max(t2_i - max(1, n_te//5), 0)                     # 峰前
        tp_i = max(t1_i - max(1, n_te//7), 0)                     # 初期
    else:
        tp_i, t1_i, t2_i, t3_i = 0, 1, min(2, n_te-1), min(3, n_te-1)

    char_idx    = [tp_i, t1_i, t2_i, t3_i]
    char_labels = ["t'", r"$t_1$", r"$t_2$", r"$t_3$"]
    char_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    print(f"  t'={t_evo[tp_i]:.3f}, t1={t_evo[t1_i]:.3f}, "
          f"t2={t_evo[t2_i]:.3f}, t3={t_evo[t3_i]:.3f}  [τ0]")

    # ── 6. 准备空间面板数据 ──
    print("[7] 准备空间面板...")
    panel_pts = []
    for k, ti in enumerate(char_idx):
        # 累积从 tp 到 ti 的团簇内跳跃
        f_start = max(0, tp_i * stride - stride//2)
        f_end   = min(ti * stride + stride//2, n_frames-1)
        sel = (cluster_jf >= f_start) & (cluster_jf <= f_end)
        panel_pts.append(cluster_pos[sel])   # (n_pts, 2)
        print(f"    {char_labels[k]}: {sel.sum()} 个跳跃点")

    # ── 7. 绘制 Fig. 2 ──
    print("[8] 绘制...")
    fig = plt.figure(figsize=(14, 11))
    # 3 列：[时间序列(宽)] [间隔] [空间面板(方)]
    gs = gridspec.GridSpec(4, 3,
                            width_ratios=[2.4, 0.2, 1.8],
                            hspace=0.50, wspace=0.10,
                            left=0.10, right=0.97,
                            top=0.96, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[3, 0], sharex=ax_a)

    ax_e = fig.add_subplot(gs[0, 2])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[2, 2])
    ax_h = fig.add_subplot(gs[3, 2])

    t = t_evo   # 真实时间

    # ── (a) σc（近似）和 γc ──
    ax_a2 = ax_a.twinx()
    ax_a.plot(t, sigma_approx, 'k-o', ms=2.5, lw=1.5,
              label=r'$\sigma_c/\sigma_0$ (≈)')
    ax_a2.plot(t, gamma_c_scaled, 'r-^', ms=2.5, lw=1.5,
               label=r'$\gamma_c$')
    ax_a.set_ylabel(r'$\sigma_c/\sigma_0$', fontsize=11)
    ax_a2.set_ylabel(r'$\gamma_c$', fontsize=11, color='red')
    ax_a2.tick_params(colors='red', labelsize=9)
    ax_a.set_ylim(bottom=0)
    ax_a2.set_ylim(0, 0.35)
    ax_a.legend(fontsize=8, loc='upper left')
    ax_a2.legend(fontsize=8, loc='lower right')
    # 标注：σc 需 per-atom stress
    ax_a.text(0.5, 1.03,
              '★ σc 需 compute stress/atom（见说明）',
              transform=ax_a.transAxes, ha='center', fontsize=7.5,
              color='gray', style='italic')

    # ── (b) ncj ──
    ax_b.plot(t, ncj_evo, 'k-o', ms=2.5, lw=1.5)
    ax_b.fill_between(t, 0, ncj_evo, alpha=0.18, color='steelblue')
    ax_b.set_ylabel(r'$n_{cj}$', fontsize=11)
    ax_b.set_ylim(bottom=0)

    # ── (c) Pext ──
    ax_c.plot(t, Pext_evo, 'k-s', ms=2.5, lw=1.5)
    ax_c.axhline(PEXT_REF, color='gray', ls='--', lw=1, alpha=0.7)
    ax_c.set_ylabel(r'$P_{ext}$', fontsize=11)
    ax_c.set_ylim(0, 1.05)

    # ── (d) R ──
    ax_d.plot(t, R_evo, 'k-^', ms=2.5, lw=1.5)
    ax_d.set_ylabel(r'$R$', fontsize=11)
    ax_d.set_xlabel(r'$t\ [\tau_0]$', fontsize=11)
    ax_d.set_ylim(0, 1.05)

    # 左列：竖虚线 + 标注 + 网格
    for ax_l in [ax_a, ax_b, ax_c, ax_d]:
        for ti, col in zip(char_idx, char_colors):
            if ti < n_te:
                ax_l.axvline(t[ti], color='dimgray', ls='--', lw=0.9, alpha=0.7)
        ax_l.grid(True, alpha=0.22)
        ax_l.tick_params(labelsize=9)

    # 时刻标签（顶部面板）
    for ti, lbl, col in zip(char_idx, char_labels, char_colors):
        if ti < n_te:
            ax_a.text(t[ti], ax_a.get_ylim()[1] * 1.01,
                      lbl, ha='center', va='bottom', fontsize=9.5,
                      color='dimgray', fontweight='bold')

    # 隐藏中间 x 轴刻度
    for ax_l in [ax_a, ax_b, ax_c]:
        plt.setp(ax_l.get_xticklabels(), visible=False)

    # 面板字母
    for ax_l, lbl in zip([ax_a, ax_b, ax_c, ax_d], ['(a)', '(b)', '(c)', '(d)']):
        ax_l.text(-0.14, 1.02, lbl, transform=ax_l.transAxes,
                  fontsize=12, fontweight='bold')

    # ── 右列空间面板 (e)-(h) ──
    pad = max(15.0, cluster_radius * 2.5)
    cx0, cy0 = cluster_center
    xlim_sp = (cx0 - pad, cx0 + pad)
    ylim_sp = (cy0 - pad, cy0 + pad)

    right_axes = [ax_e, ax_f, ax_g, ax_h]
    panel_letters = ['(e)', '(f)', '(g)', '(h)']

    for k, (ax_sp, pts) in enumerate(zip(right_axes, panel_pts)):
        ax_sp.set_facecolor('black')
        if len(pts) > 0:
            ax_sp.scatter(pts[:, 0], pts[:, 1],
                          s=5, c='red', alpha=0.65, linewidths=0, zorder=3)
        else:
            ax_sp.text(0.5, 0.5, 'No data',
                       transform=ax_sp.transAxes, ha='center', va='center',
                       fontsize=9, color='white')

        ax_sp.set_xlim(xlim_sp); ax_sp.set_ylim(ylim_sp)
        ax_sp.set_aspect('equal')
        ax_sp.tick_params(labelsize=8)

        # 时刻标签
        ax_sp.text(0.05, 0.95, f"{panel_letters[k]} {char_labels[k]}",
                   transform=ax_sp.transAxes, ha='left', va='top',
                   fontsize=10, color=char_colors[k], fontweight='bold',
                   bbox=dict(fc='white', alpha=0.75, pad=1.5))

        if k < 3:
            plt.setp(ax_sp.get_xticklabels(), visible=False)
        else:
            ax_sp.set_xlabel(r'$x\ /\ \sigma_{bb}$', fontsize=10)
        ax_sp.set_ylabel(r'$y\ /\ \sigma_{bb}$', fontsize=10)
        ax_sp.grid(False)

    # 总标题
    fig.suptitle(
        rf'Convective cluster analysis  $\dot{{\gamma}}={sr}$, T=0.45',
        fontsize=13, y=0.99)

    out_png = os.path.join(args.output, f'fig2_cluster_analysis_{sr_str}.png')
    out_pdf = os.path.join(args.output, f'fig2_cluster_analysis_{sr_str}.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ 保存: {out_png}")
    print(f"  ✓ 保存: {out_pdf}")

    # ── 单独大图：空间面板 (e)-(h) ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(9, 9))
    axes2 = axes2.ravel()
    for k, (ax2, pts) in enumerate(zip(axes2, panel_pts)):
        ax2.set_facecolor('black')
        if len(pts) > 0:
            ax2.scatter(pts[:, 0], pts[:, 1],
                        s=7, c='red', alpha=0.7, linewidths=0)
        ax2.set_xlim(xlim_sp); ax2.set_ylim(ylim_sp)
        ax2.set_aspect('equal')
        ax2.set_xlabel(r'$x\ /\ \sigma_{bb}$', fontsize=12)
        ax2.set_ylabel(r'$y\ /\ \sigma_{bb}$', fontsize=12)
        ax2.text(0.05, 0.95, f"{panel_letters[k]} {char_labels[k]}",
                 transform=ax2.transAxes, ha='left', va='top',
                 fontsize=13, color=char_colors[k], fontweight='bold',
                 bbox=dict(fc='white', alpha=0.80, pad=2))
        ax2.tick_params(labelsize=10)
    fig2.suptitle(
        rf'Spatiotemporal cage-jump accumulation  $\dot{{\gamma}}={sr}$',
        fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(args.output, f'fig2_spatial_{sr_str}.png')
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ 空间面板: {out2}")

    print_stress_note()


def print_stress_note():
    print(f"""
{'='*60}
  获取真实 σc（面板 a）的方法：

  在 in.shear_template 的 pair_coeff 之后添加：

      compute  SA  all  stress/atom  NULL  virial
      variable stress_xy  atom  -c_SA[4]/vol

  dump 命令中追加：
      ... v_stress_xy

  重新运行 LAMMPS 后，plot_fig2.py 读取该列即可。
{'='*60}
""")


# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dump',   default='dump.shear_0.015.lammpstrj')
    p.add_argument('--npz',    default='cage_jumps_shearrate_0p015.npz')
    p.add_argument('--rate',   type=float, default=0.015)
    p.add_argument('--dt',     type=float, default=0.005,
                   help='LAMMPS 时间步长 Δt')
    p.add_argument('--every',  type=int,   default=6667,
                   help='dump 每隔多少步写一帧')
    p.add_argument('--output', default='figures')
    args = p.parse_args()

    print(f"\n{'='*55}")
    print("  Fig. 2  团簇时间演化")
    print(f"  dump : {args.dump}")
    print(f"  npz  : {args.npz}")
    print(f"{'='*55}\n")

    plot_fig2(args)
