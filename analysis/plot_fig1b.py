"""
plot_fig1b.py  ——  复现论文 Fig. 1(b) / Fig. 5(b)
======================================================
图内容：
  ● 黑色散点：在时间窗口 tχ 内积累的笼跳跃事件（投影到 x-y 剪切平面）
  ● 红色实线：最大笼跳跃团簇的轮廓边界
  ● 绿色虚线圆：位于任何团簇外部的对照区域
  ● 蓝色虚线圆：跨越不同团簇的对照区域

前置条件：
  1. 已运行 cage_jump_detection.py，生成 cage_jumps_shearrate_0p015.npz
  2. （可选）dump 文件含有 xu, yu, zu（已 unwrap 坐标）

用法（Windows cmd）：
    python plot_fig1b.py ^
        --dump  dump.shear_0.015.lammpstrj ^
        --npz   cage_jumps_shearrate_0p015.npz ^
        --rate  0.015 ^
        --output figures
"""

# ===== 可调参数 =====
COARSE_SIGMA    = 1.2    # 密度场平滑长度（σ_bb）
GRID_N          = 40     # 密度场网格数
DENSITY_TOL     = 0.08   # 迭代阈值收敛精度
CONTOUR_SMOOTH  = 1.2    # 边界平滑（格点数）
MIN_CLUSTER_FRAC= 0.005  # 最小团簇面积（占总面积比）
TYPE_BIG        = 1      # 大粒子类型

import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, label as ndlabel
from scipy.spatial import cKDTree


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from read_dump import read_lammps_dump
    HAS_READ_DUMP = True
except ImportError:
    HAS_READ_DUMP = False
    print("  ⚠ 未找到 read_dump.py，将使用 npz 中保存的坐标")


# ───────────────────────────────────────────────────────────────
def _center_positions(xs, ys, Lx, Ly):
    """
    确保坐标以盒子中心为原点。
    如果坐标中位数明显偏离 0（超过盒长的 10%），自动纠偏。
    """
    x_med = float(np.median(xs))
    y_med = float(np.median(ys))

    x_off = x_med if (Lx > 0 and abs(x_med) > Lx * 0.1) else 0.0
    y_off = y_med if (Ly > 0 and abs(y_med) > Ly * 0.1) else 0.0

    if abs(x_off) > 0 or abs(y_off) > 0:
        print(f"  ⚠ 坐标偏移检测: x中位数={x_med:.2f}, y中位数={y_med:.2f}")
        print(f"  ⚠ 自动纠偏: Δx={x_off:.2f}, Δy={y_off:.2f}")
        xs = xs - x_off
        ys = ys - y_off

    return xs, ys


def load_jump_positions(npz_file, dump_file, shear_rate):
    """
    获取所有笼跳跃事件在实验室坐标系中的 (x,y) 位置。

    优先级：
      1. npz 中的 jump_pos_xy（cage_jump_detection.py 已计算并居中）
      2. dump 文件（实时读取并居中）
      3. npz 中的 positions（r_tilde 坐标，尝试居中）
    """
    data = np.load(npz_file, allow_pickle=True)
    jump_frames  = data['jump_frames']
    particle_idx = data['particle_idx']
    types_all    = data['types']
    jump_vectors = data['jump_vectors']

    Lx = float(data['box_Lx']) if 'box_Lx' in data else 0.0
    Ly = float(data['box_Ly']) if 'box_Ly' in data else 0.0

    # ── 方法1：npz 中已保存居中的 x-y 坐标（推荐）──
    if 'jump_pos_xy' in data:
        print("  ✓ 使用 npz 中缓存的 jump_pos_xy（已居中）")
        pos = data['jump_pos_xy']              # shape (N_all, 2)
        big_mask = (types_all[particle_idx] == TYPE_BIG)
        pos_big  = pos[big_mask]
        vecs_big = jump_vectors[big_mask]
        xs, ys = pos_big[:, 0].copy(), pos_big[:, 1].copy()
        print(f"  jump_pos_xy 范围: x=[{xs.min():.2f},{xs.max():.2f}]  "
              f"y=[{ys.min():.2f},{ys.max():.2f}]")
        return xs, ys, vecs_big, data

    # ── 方法2：从 dump 文件读 ──
    if HAS_READ_DUMP and os.path.exists(dump_file):
        print("  从 dump 文件读取粒子坐标...")
        frames = read_lammps_dump(dump_file)
        n_frames = len(frames)
        box  = frames[0]['box']
        Lx, Ly = box['Lx'], box['Ly']
        cx = box['xlo'] + Lx / 2.0
        cy = box['ylo'] + Ly / 2.0
        print(f"  dump 盒子: Lx={Lx:.2f}, Ly={Ly:.2f}, 中心=({cx:.2f},{cy:.2f})")

        xs_list, ys_list, vecs_list = [], [], []
        for jf, pi, vv in zip(jump_frames, particle_idx, jump_vectors):
            if types_all[pi] != TYPE_BIG:
                continue
            fi = min(int(jf), n_frames - 1)
            x = frames[fi]['xu'][pi] - cx
            y = frames[fi]['yu'][pi] - cy
            xs_list.append(x)
            ys_list.append(y)
            vecs_list.append(vv)

        xs = np.array(xs_list)
        ys = np.array(ys_list)
        xs, ys = _center_positions(xs, ys, Lx, Ly)
        print(f"  大粒子跳跃: {len(xs)} 个，"
              f"x=[{xs.min():.2f},{xs.max():.2f}]  y=[{ys.min():.2f},{ys.max():.2f}]")
        return xs, ys, np.array(vecs_list), data

    # ── 方法3：回退到 positions（r_tilde 坐标），尝试居中 ──
    print("  ⚠ 使用 positions 回退路径（r_tilde 坐标）")
    pos = data['positions']                    # (N_all, 3)
    big_mask = (types_all[particle_idx] == TYPE_BIG)
    pos_big  = pos[big_mask]
    vecs_big = jump_vectors[big_mask]

    xs = pos_big[:, 0].copy()
    ys = pos_big[:, 1].copy()

    # 用中位数估计偏移并纠偏
    xs -= float(np.median(xs))
    ys -= float(np.median(ys))

    print(f"  纠偏后: x=[{xs.min():.2f},{xs.max():.2f}]  y=[{ys.min():.2f},{ys.max():.2f}]")
    return xs, ys, vecs_big, data



def _wrap_to_box(pos, L):
    """将坐标折回到 [0, L)；pos: (N,3), L: (3,)"""
    return np.mod(pos, L)

def compute_tchi_frames_from_rtilde(raw_data, a=0.25, max_lag=600, n_origins=16, skip_frames=200, sample_n=0):
    """
    按论文 Appendix B, Eq.(B4) 计算 χ4(Δt) 的峰值位置 tχ（单位：帧）。
    用 cKDTree.count_neighbors 计算 Q(Δt)=∑_i∑_j w(|r_i(t)-r_j(0)|)（配对数）。
    """
    if 'r_tilde' not in raw_data:
        raise ValueError("npz 中没有 r_tilde，无法按论文计算 χ4(t)")

    r_tilde = raw_data['r_tilde'].astype(np.float64)   # (F,N,3)
    F, N, _ = r_tilde.shape

    Lx = float(raw_data.get('box_Lx', 0.0))
    Ly = float(raw_data.get('box_Ly', 0.0))
    Lz = float(raw_data.get('box_Lz', Lx if Lx > 0 else Ly))  # KA 3D 通常近立方
    L  = np.array([Lx, Ly, Lz], dtype=np.float64)
    if np.any(L <= 0):
        raise ValueError(f"盒子尺寸无效: L={L}. 请确保 npz 保存了 box_Lx/box_Ly。")

    max_lag = min(int(max_lag), F - int(skip_frames) - 2)
    if max_lag < 2:
        raise ValueError("max_lag 太大或 skip_frames 太大，导致可用帧不足。")

    origins = np.linspace(int(skip_frames), F - max_lag - 1, int(n_origins), dtype=int)
    origins = np.unique(origins)

    # 可选抽样（仅用于加速，tχ 峰值位置通常很稳）
    if int(sample_n) > 0 and int(sample_n) < N:
        rng = np.random.default_rng(0)
        sel = rng.choice(N, size=int(sample_n), replace=False)
    else:
        sel = None

    # 预建 ref trees
    ref_trees = []
    for t0 in origins:
        ref = r_tilde[t0]
        if sel is not None:
            ref = ref[sel]
        ref = _wrap_to_box(ref, L)
        ref_trees.append(cKDTree(ref, boxsize=L))

    chi4 = np.zeros(max_lag + 1, dtype=np.float64)

    print(f"[tχ] 计算 χ4(Δt): a={a}, max_lag={max_lag}, origins={len(origins)}, sample={('all' if sel is None else len(sel))}")
    for lag in range(1, max_lag + 1):
        Qs = np.empty(len(origins), dtype=np.float64)
        for k, t0 in enumerate(origins):
            cur = r_tilde[t0 + lag]
            if sel is not None:
                cur = cur[sel]
            cur = _wrap_to_box(cur, L)
            cur_tree = cKDTree(cur, boxsize=L)
            Qs[k] = ref_trees[k].count_neighbors(cur_tree, r=float(a))
        chi4[lag] = Qs.var(ddof=0)

    tchi = int(np.argmax(chi4))
    return tchi, chi4, origins

def filter_events_by_time_window(raw_data, xs, ys, vecs, t0_frame, win_frames):
    """
    过滤 [t0_frame, t0_frame + win_frames) 内的笼跳跃事件。
    xs/ys/vecs 必须已经是 TYPE_BIG 过滤后的输出（load_jump_positions 的输出满足）。
    """
    jf = raw_data['jump_frames']
    pi = raw_data['particle_idx']
    types_all = raw_data['types']
    big_mask = (types_all[pi] == TYPE_BIG)

    jf_big = jf[big_mask]
    if len(jf_big) != len(xs):
        raise RuntimeError("事件数不一致：请确认 xs/ys 来自 load_jump_positions 的 TYPE_BIG 输出。")

    m = (jf_big >= int(t0_frame)) & (jf_big < int(t0_frame) + int(win_frames))
    return xs[m], ys[m], vecs[m]

# ───────────────────────────────────────────────────────────────
def compute_density_2d(xs, ys, xlim, ylim, grid_n=GRID_N, sigma=COARSE_SIGMA):
    xlo, xhi = xlim; ylo, yhi = ylim
    dx = (xhi - xlo) / grid_n
    dy = (yhi - ylo) / grid_n

    density = np.zeros((grid_n, grid_n))
    ix = np.clip(((xs - xlo) / dx).astype(int), 0, grid_n-1)
    iy = np.clip(((ys - ylo) / dy).astype(int), 0, grid_n-1)
    for i, j in zip(ix, iy):
        density[i, j] += 1.0

    sigma_vox = sigma / dx
    density = gaussian_filter(density, sigma=sigma_vox, mode='constant')

    xc = np.linspace(xlo + dx/2, xhi - dx/2, grid_n)
    yc = np.linspace(ylo + dy/2, yhi - dy/2, grid_n)

    print(f"  density min={density.min():.4f}, max={density.max():.4f}, mean={density.mean():.4f}")
    return density, xc, yc, dx, dy


def find_threshold(density, tol=DENSITY_TOL):
    rho = density.ravel()
    rho_s = rho[rho > rho.max() * 0.02]
    if len(rho_s) < 10:
        return float(np.percentile(rho, 80))

    rho_i = float(rho_s.max())
    for _ in range(300):
        below = rho_s[rho_s < rho_i]
        if len(below) == 0:
            break
        rho_avg = float(below.mean())
        if rho_avg > 0 and abs(rho_avg - rho_i) / rho_avg < tol:
            return rho_avg
        if rho_i <= rho_avg:
            break
        rho_i = rho_avg

    return float(rho_s.mean() + 0.8 * rho_s.std())


def get_cluster_contours(density, xc, yc, rho_th,
                          smooth=CONTOUR_SMOOTH,
                          min_frac=MIN_CLUSTER_FRAC):
    dens_s = gaussian_filter(density, sigma=smooth)
    total_area = (xc[-1] - xc[0]) * (yc[-1] - yc[0])

    fig_tmp, ax_tmp = plt.subplots()
    XX, YY = np.meshgrid(xc, yc, indexing='ij')
    cs = ax_tmp.contour(XX, YY, dens_s, levels=[rho_th])
    raw_paths = [p.vertices for p in cs.get_paths()]
    plt.close(fig_tmp)

    contours, centers, areas = [], [], []
    for pts in raw_paths:
        if len(pts) < 8:
            continue
        x_, y_ = pts[:, 0], pts[:, 1]

        # ★ 过滤沿盒子边界走的假轮廓
        xlo_b, xhi_b = xc[0], xc[-1]
        ylo_b, yhi_b = yc[0], yc[-1]
        margin = (xhi_b - xlo_b) / GRID_N * 1.5
        on_border = (
            (np.abs(x_ - xlo_b) < margin) | (np.abs(x_ - xhi_b) < margin) |
            (np.abs(y_ - ylo_b) < margin) | (np.abs(y_ - yhi_b) < margin)
        )
        if on_border.mean() > 0.35:
            print(f"  跳过疑似盒子边界的轮廓（{on_border.mean()*100:.0f}% 点靠近边缘）")
            continue

        area = 0.5 * abs(np.dot(x_, np.roll(y_, 1)) -
                          np.dot(y_, np.roll(x_, 1)))
        if area < total_area * min_frac:
            continue
        contours.append(pts)
        centers.append((float(np.mean(x_)), float(np.mean(y_))))
        areas.append(area)

    if areas:
        order = np.argsort(areas)[::-1]
        contours = [contours[i] for i in order]
        centers  = [centers[i]  for i in order]
        areas    = [areas[i]    for i in order]

    return contours, centers, areas


# ───────────────────────────────────────────────────────────────
def plot_fig1b(args):
    os.makedirs(args.output, exist_ok=True)
    sr = args.rate
    sr_str = f"{sr}".replace('.', 'p')

    # ── 1. 加载数据 ──
    print("[1] 加载笼跳跃数据...")
    if not os.path.exists(args.npz):
        print(f"  ✗ 找不到 {args.npz}")
        print("  请先运行: python cage_jump_detection.py ...")
        return

    xs, ys, vecs, raw_data = load_jump_positions(args.npz, args.dump, sr)
    print(f"  大粒子笼跳跃: {len(xs)} 个")

    # ── 论文定义的时间窗：tχ（由 χ4(Δt) 峰值给出）──
    tchi_frames = None
    if args.tchi_frames > 0:
        tchi_frames = int(args.tchi_frames)
        print(f"[tχ] 使用用户指定的 tχ = {tchi_frames} 帧")
    elif args.auto_tchi:
        tchi_frames, chi4, origins = compute_tchi_frames_from_rtilde(
            raw_data,
            a=args.overlap_a,
            max_lag=args.chi4_max_lag,
            n_origins=args.chi4_origins,
            skip_frames=args.chi4_skip,
            sample_n=args.chi4_sample,
        )
        print(f"[tχ] χ4(Δt) 峰值: tχ = {tchi_frames} 帧")
    else:
        print("[tχ] 未启用时间窗（会全程叠加变黑）。建议加 --auto_tchi 或 --tchi_frames。")

    # 按论文“在一个 tχ 时间间隔内累计”过滤事件
    if tchi_frames is not None:
        F = int(raw_data['r_tilde'].shape[0]) if 'r_tilde' in raw_data else int(raw_data['jump_frames'].max() + 1)
        t0 = (F - tchi_frames)//2 if args.t0_frame is None else int(args.t0_frame)  # 默认取中间一段（稳态）
        xs, ys, vecs = filter_events_by_time_window(raw_data, xs, ys, vecs, t0, tchi_frames)
        print(f"[window] 选取时间窗 [{t0}, {t0+tchi_frames})：剩余 {len(xs)} 个大粒子笼跳跃事件")


    if len(xs) < 5:
        print("  ✗ 跳跃点太少，无法绘图")
        return

    # ── 2. 获取盒子尺寸 ──
    print("[2] 获取盒子尺寸...")
    if 'box_Lx' in raw_data and float(raw_data['box_Lx']) > 0:
        Lx = float(raw_data['box_Lx'])
        Ly = float(raw_data['box_Ly'])
    elif HAS_READ_DUMP and os.path.exists(args.dump):
        frms = read_lammps_dump(args.dump, max_frames=1)
        Lx = frms[0]['box']['Lx']
        Ly = frms[0]['box']['Ly']
    else:
        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()
        Lx = x_span * 1.15
        Ly = y_span * 1.15
        print(f"  ⚠ 无法读取盒子，从点云估计 Lx={Lx:.1f}, Ly={Ly:.1f}")

    xlim = (-Lx/2, Lx/2)
    ylim = (-Ly/2, Ly/2)
    print(f"  盒子: Lx={Lx:.2f}, Ly={Ly:.2f}")
    print(f"  坐标范围: x=[{xs.min():.2f},{xs.max():.2f}]  y=[{ys.min():.2f},{ys.max():.2f}]")

    # ── 3. 计算密度场 ──
    print("[3] 计算密度场...")
    density, xc, yc, dx, dy = compute_density_2d(xs, ys, xlim, ylim)
    rho_th = find_threshold(density)
    above_frac = (density >= rho_th).mean()
    print(f"  密度阈值 ρth={rho_th:.4f},  超阈格点={above_frac*100:.1f}%")

    # ── 4. 提取团簇轮廓 ──
    print("[4] 提取团簇边界...")
    contours, centers, areas = get_cluster_contours(density, xc, yc, rho_th)
    print(f"  检测到 {len(contours)} 个有效团簇")
    for i, (ctr, a) in enumerate(zip(centers[:4], areas[:4])):
        xi = np.sqrt(a / np.pi)
        print(f"    团簇{i+1}: 中心=({ctr[0]:.1f},{ctr[1]:.1f}), "
              f"面积={a:.1f}, 等效半径={xi:.2f} σ_bb")

    # ── 5. 绘图 ──
    print("[5] 绘图...")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.scatter(xs, ys, s=2.5, c='k', alpha=0.55, linewidths=0, zorder=2,
               label='Cage jumps')

    colors_c = ['red', 'darkorange', 'darkviolet', 'navy']
    for i, (cont, col) in enumerate(zip(contours, colors_c)):
        lw = 2.2 if i == 0 else 1.4
        ax.plot(cont[:, 0], cont[:, 1],
                color=col, lw=lw, zorder=5,
                label=f'Cluster {i+1}' if i < 3 else None)

    # ── 对照圆（可选）──
    if not args.no_circles and len(contours) >= 1:
        d_smooth = gaussian_filter(density, sigma=2.0)
        min_idx  = np.unravel_index(d_smooth.argmin(), d_smooth.shape)
        gc_x = xc[min_idx[0]]; gc_y = yc[min_idx[1]]
        r_green = min(Lx, Ly) / 9
        ax.add_patch(plt.Circle((gc_x, gc_y), r_green,
                                fill=False, ls='--', ec='green', lw=1.6, zorder=6))

        if len(contours) >= 2:
            bc_x = (centers[0][0] + centers[1][0]) / 2
            bc_y = (centers[0][1] + centers[1][1]) / 2
            r_blue = min(Lx, Ly) / 8
            ax.add_patch(plt.Circle((bc_x, bc_y), r_blue,
                                    fill=False, ls='--', ec='blue', lw=1.6, zorder=6))

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r'$x\ /\ \sigma_{bb}$', fontsize=13)
    ax.set_ylabel(r'$y\ /\ \sigma_{bb}$', fontsize=13)
    ax.set_title(rf'$\dot{{\gamma}}={sr}$,  T=0.45   (KA 3D MD)', fontsize=11)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)

    legend_items = [
        mpatches.Patch(fc='k', label='Cage jumps'),
        plt.Line2D([0],[0], c='red', lw=2, label='Cluster boundary'),
    ]
    if not args.no_circles:
        legend_items += [
            plt.Line2D([0],[0], c='green', lw=1.5, ls='--', label='Outside cluster'),
            plt.Line2D([0],[0], c='blue',  lw=1.5, ls='--', label='Spanning clusters'),
        ]
    ax.legend(handles=legend_items, fontsize=9, loc='upper right', framealpha=0.8)
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    plt.tight_layout()

    out_png = os.path.join(args.output, f'fig1b_{sr_str}.png')
    out_pdf = os.path.join(args.output, f'fig1b_{sr_str}.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ 保存: {out_png}")
    print(f"  ✓ 保存: {out_pdf}")

    # ── 密度场调试图 ──
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.0))
    XX, YY = np.meshgrid(xc, yc, indexing='ij')
    im = ax2.pcolormesh(XX, YY, gaussian_filter(density, sigma=CONTOUR_SMOOTH),
                         cmap='YlOrRd', shading='auto')
    for cont, col in zip(contours[:3], ['cyan','lime','dodgerblue']):
        ax2.plot(cont[:,0], cont[:,1], color=col, lw=2)
    plt.colorbar(im, ax=ax2, label='Cage-jump density (smoothed)')
    ax2.set_xlabel(r'$x/\sigma_{bb}$', fontsize=12)
    ax2.set_ylabel(r'$y/\sigma_{bb}$', fontsize=12)
    ax2.set_title(f'Density field  ρth={rho_th:.4f}', fontsize=10)
    ax2.set_aspect('equal')
    plt.tight_layout()
    out2 = os.path.join(args.output, f'fig1b_density_{sr_str}.png')
    fig2.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ 密度场: {out2}")


# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dump',       default='dump.shear_0.015.lammpstrj')
    p.add_argument('--npz',        default='cage_jumps_shearrate_0p015.npz')
    p.add_argument('--rate',       type=float, default=0.015)
    p.add_argument('--output',     default='figures')
    p.add_argument('--no_circles', action='store_true')
    # 论文时间窗（tχ）相关参数
    p.add_argument('--auto_tchi', action='store_true', help='按论文 Eq.(B4) 自动计算 tχ（χ4 峰值）并只画一个 tχ 时间窗')
    p.add_argument('--tchi_frames', type=int, default=-1, help='手动指定 tχ（单位：帧）。>0 时优先生效')
    p.add_argument('--t0_frame', type=int, default=None, help='时间窗起点帧号 t0；默认取轨迹中间段')
    p.add_argument('--overlap_a', type=float, default=0.25, help='overlap cutoff a（3D MD 用 0.25）')
    p.add_argument('--chi4_max_lag', type=int, default=600, help='计算 χ4(Δt) 的最大 lag（帧）')
    p.add_argument('--chi4_origins', type=int, default=16, help='χ4 计算使用的 time origins 数')
    p.add_argument('--chi4_skip', type=int, default=200, help='从前 skip 帧后开始取 origins（避开瞬态）')
    p.add_argument('--chi4_sample', type=int, default=0, help='χ4 计算可选随机抽样粒子数（0=全粒子；用于加速）')
    args = p.parse_args()

    print(f"\n{'='*55}")
    print("  Fig. 1(b)  笼跳跃空间分布 + 团簇边界")
    print(f"  dump : {args.dump}")
    print(f"  npz  : {args.npz}")
    print(f"{'='*55}\n")

    plot_fig1b(args)