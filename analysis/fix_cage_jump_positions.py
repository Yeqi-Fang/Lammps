"""
fix_cage_jump_positions.py
===========================
修复 cage_jump_detection.py 中真实坐标提取的三个主要问题：

问题 1：triclinic 盒下错误使用 x + ix*Lx（应直接用包裹坐标 x）
问题 2：jump_frames 与 dump 帧数不对齐导致大量事件被 clamp
问题 3：缺乏诊断信息，难以定位原因

使用方法：
    python fix_cage_jump_positions.py \
        --npz  cage_jumps_shearrate_0p015.npz \
        --dump dump.shear_0.015.lammpstrj \
        --output figures_fixed
"""

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from read_dump import read_lammps_dump
    HAS_READ_DUMP = True
except ImportError:
    HAS_READ_DUMP = False
    print("⚠ 未找到 read_dump.py")


# ─────────────────────────────────────────────────────────────────────────────
# 诊断函数
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(npz_file, dump_file):
    """打印关键诊断信息，帮助定位 frame 对齐和坐标范围问题。"""
    print("\n" + "="*60)
    print("  [诊断] 检查 npz 与 dump 的一致性")
    print("="*60)

    data = np.load(npz_file, allow_pickle=True)
    r_tilde     = data['r_tilde']
    jump_frames = data['jump_frames']
    jump_times  = data['jump_times']
    types       = data['types']
    particle_idx= data['particle_idx']

    print(f"\n  r_tilde shape     : {r_tilde.shape}  (frames × particles × 3)")
    n_rtilde_frames = r_tilde.shape[0]
    print(f"  r_tilde 帧数      : {n_rtilde_frames}")
    print(f"  jump_frames 范围  : [{jump_frames.min()}, {jump_frames.max()}]")
    print(f"  jump_times  范围  : [{jump_times.min():.3f}, {jump_times.max():.3f}]")
    print(f"  跳跃事件总数      : {len(jump_frames)}")
    print(f"  粒子类型分布      : { {int(t): int((types==t).sum()) for t in np.unique(types)} }")

    # 检查被 clamp 的事件
    if HAS_READ_DUMP and os.path.exists(dump_file):
        print(f"\n  读取 dump 前几帧以获取帧数信息...")
        frames = read_lammps_dump(dump_file)
        n_dump_frames = len(frames)
        print(f"  dump 文件总帧数   : {n_dump_frames}")
        
        n_clamped = (jump_frames >= n_dump_frames).sum()
        pct_clamped = 100.0 * n_clamped / len(jump_frames)
        print(f"  被 clamp 的事件   : {n_clamped} / {len(jump_frames)} ({pct_clamped:.1f}%)")
        if pct_clamped > 5:
            print(f"  ⚠ 严重！超过 5% 的事件帧号超出 dump 范围，")
            print(f"    这是坐标聚集的主因之一。")
            print(f"    → 检查 r_tilde 和 dump 是否使用相同时间步长/stride")

        box = frames[0]['box']
        Lx, Ly = box['Lx'], box['Ly']
        print(f"\n  盒子尺寸          : Lx={Lx:.3f}, Ly={Ly:.3f}")
        print(f"  期望坐标范围      : x∈[{box['xlo']:.2f},{box['xlo']+Lx:.2f}]  "
              f"y∈[{box['ylo']:.2f},{box['ylo']+Ly:.2f}]")

        # 检查第一帧坐标字段
        s = frames[0]
        available_fields = [k for k in s.keys() if k not in ('box', 'step')]
        print(f"  dump 可用字段     : {available_fields}")
        has_xu = 'xu' in s
        has_ix = 'ix' in s
        has_x  = 'x'  in s
        print(f"  有 xu/yu          : {has_xu}")
        print(f"  有 x + ix         : {has_x and has_ix}")

        if has_xu:
            print("  ⚠ 使用 xu 时请注意：长时间剪切后 xu 很大，"
                  "需模 Lx 取余才能得到盒内位置。")
    else:
        print(f"  ✗ dump 文件不存在或 read_dump.py 不可用")

    # 检查已存的 jump_pos_xy
    if 'jump_pos_xy' in data:
        pos = data['jump_pos_xy']
        print(f"\n  npz 中已有 jump_pos_xy:")
        print(f"    x 范围: [{pos[:,0].min():.2f}, {pos[:,0].max():.2f}]")
        print(f"    y 范围: [{pos[:,1].min():.2f}, {pos[:,1].max():.2f}]")
        if 'box_Lx' in data:
            Lx_npz = float(data['box_Lx'])
            print(f"    盒子 Lx (npz): {Lx_npz:.3f}")
            if Lx_npz > 0:
                x_frac = (pos[:,0].max() - pos[:,0].min()) / Lx_npz
                print(f"    x 覆盖率: {x_frac*100:.1f}%  (应接近 100%)")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 核心修复：正确提取 triclinic 盒下的粒子实验室坐标
# ─────────────────────────────────────────────────────────────────────────────

def extract_real_positions_fixed(jumps, dump_file, verbose=True):
    """
    修复版：从 LAMMPS dump 文件提取每个笼跳跃时刻的粒子位置。

    关键修复：
      - 对 triclinic（剪切）盒，直接使用 LAMMPS wrapped 坐标 (x, y)
        而非 x + ix*Lx（后者对 triclinic 不正确，会造成坐标偏移）
      - 对 xu, yu（unwrapped），需先对 Lx, Ly 取模再居中
      - 对帧号超出 dump 范围的事件发出警告并跳过（不强制 clamp）
    """
    if not HAS_READ_DUMP or not os.path.exists(dump_file):
        print(f"  ✗ 无法读取 dump: {dump_file}")
        return None

    print(f"  [修复版] 读取 dump: {dump_file}")
    frames = read_lammps_dump(dump_file)
    n_frames = len(frames)
    print(f"  dump 总帧数: {n_frames}")

    box = frames[0]['box']
    Lx, Ly = box['Lx'], box['Ly']
    xlo = box['xlo']; ylo = box['ylo']
    cx = xlo + Lx / 2.0
    cy = ylo + Ly / 2.0
    print(f"  盒子: Lx={Lx:.3f} Ly={Ly:.3f}  中心=({cx:.3f},{cy:.3f})")

    s = frames[0]
    has_xu = 'xu' in s
    has_x  = 'x'  in s
    has_ix = 'ix' in s

    particle_idx = jumps['particle_idx']
    jump_frames  = jumps['jump_frames']
    n_jumps = len(particle_idx)

    xy_list = []
    valid_mask = []
    n_skipped = 0

    for i, (pi, jf) in enumerate(zip(particle_idx, jump_frames)):
        # ── 修复 1: 检查帧号范围，超出则跳过而非 clamp ──
        if int(jf) >= n_frames:
            n_skipped += 1
            valid_mask.append(False)
            continue
        valid_mask.append(True)

        fi = int(jf)
        f  = frames[fi]

        # ── 修复 2: 对 triclinic 盒使用正确坐标提取方式 ──
        if has_xu:
            # xu 是 unwrapped（累积）坐标，对 Lx/Ly 取模得到盒内位置
            xu = float(f['xu'][pi])
            yu = float(f['yu'][pi])
            # 将 unwrapped 坐标折叠回盒内，再居中
            x = (xu - xlo) % Lx + xlo - cx   # 等价于折叠后减 cx
            y = (yu - ylo) % Ly + ylo - cy
        elif has_x:
            # 直接使用 wrapped 坐标（LAMMPS 已保证在盒内）
            # 注意：不要加 ix*Lx，这对 triclinic 盒不正确！
            x = float(f['x'][pi]) - cx
            y = float(f['y'][pi]) - cy
        else:
            x = y = 0.0

        xy_list.append([x, y])

    valid_mask = np.array(valid_mask)
    xy = np.array(xy_list)

    if n_skipped > 0:
        pct = 100.0 * n_skipped / n_jumps
        print(f"  ⚠ 跳过 {n_skipped}/{n_jumps} ({pct:.1f}%) 个帧号超出范围的事件")
        print(f"    → 说明 r_tilde 帧数 ({jump_frames.max()}) > dump 帧数 ({n_frames-1})")
        print(f"    → 请检查 MSD 计算时的 stride 是否与 dump stride 一致")

    if len(xy) == 0:
        print("  ✗ 没有有效的位置数据")
        return np.zeros((0, 2)), valid_mask

    print(f"  有效事件: {len(xy)}/{n_jumps}")
    print(f"  jump_pos_xy: x=[{xy[:,0].min():.2f},{xy[:,0].max():.2f}]  "
          f"y=[{xy[:,1].min():.2f},{xy[:,1].max():.2f}]")
    
    # 验证：坐标应在 [-L/2, L/2] 内
    x_ok = (xy[:,0].min() >= -Lx/2 - 0.5) and (xy[:,0].max() <= Lx/2 + 0.5)
    y_ok = (xy[:,1].min() >= -Ly/2 - 0.5) and (xy[:,1].max() <= Ly/2 + 0.5)
    if not x_ok or not y_ok:
        print(f"  ⚠ 坐标超出盒子范围！x_ok={x_ok} y_ok={y_ok}")
        print(f"    可能的原因：dump 文件使用了 scaled 坐标或其他格式")

    return xy, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# 重新生成 Fig 1(b) / Fig 5(b) 的核心绘图逻辑
# ─────────────────────────────────────────────────────────────────────────────

def plot_fixed_fig1b(npz_file, dump_file, shear_rate, output_dir='.', type_filter=1):
    """
    用修复后的坐标重新绘制空间分布图，并进行密度分析。
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_file, allow_pickle=True)

    # 过滤粒子类型
    types_all    = data['types']
    particle_idx = data['particle_idx']
    jump_vectors = data['jump_vectors']
    jump_frames  = data['jump_frames']
    
    big_mask = (types_all[particle_idx] == type_filter)
    jumps_filtered = {
        'particle_idx': particle_idx[big_mask],
        'jump_frames':  jump_frames[big_mask],
        'jump_vectors': jump_vectors[big_mask],
    }
    print(f"  type={type_filter} 粒子跳跃数: {big_mask.sum()} / {len(particle_idx)}")

    # 提取坐标（修复版）
    result = extract_real_positions_fixed(jumps_filtered, dump_file)
    if result is None:
        print("  ✗ 无法提取位置，退出")
        return
    
    xy, valid_mask = result
    if len(xy) < 5:
        print("  ✗ 有效点太少"); return

    # 盒子尺寸
    if 'box_Lx' in data and float(data['box_Lx']) > 0:
        Lx = float(data['box_Lx']); Ly = float(data['box_Ly'])
    else:
        try:
            frames_tmp = read_lammps_dump(dump_file)
            Lx = frames_tmp[0]['box']['Lx']; Ly = frames_tmp[0]['box']['Ly']
        except Exception:
            Lx = Ly = (xy[:,0].max()-xy[:,0].min()) * 1.15

    sr_str = str(shear_rate).replace('.','p')

    # ── 图1：修复后的散点图 ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy[:,0], xy[:,1], s=3, c='k', alpha=0.5, linewidths=0)
    ax.set_xlim(-Lx/2, Lx/2); ax.set_ylim(-Ly/2, Ly/2)
    ax.set_xlabel(r'$x / \sigma_{bb}$', fontsize=13)
    ax.set_ylabel(r'$y / \sigma_{bb}$', fontsize=13)
    ax.set_title(rf'[修复] Cage-jump positions, $\dot\gamma={shear_rate}$, T=0.45', fontsize=11)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.text(0.02, 0.98, f'N={len(xy)}', transform=ax.transAxes,
            va='top', fontsize=10, color='blue')
    plt.tight_layout()
    out1 = os.path.join(output_dir, f'fig1b_fixed_{sr_str}.png')
    fig.savefig(out1, dpi=200); plt.close(fig)
    print(f"  → 散点图: {out1}")

    # ── 图2：y 方向分布（验证是否均匀）──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins_x = np.linspace(-Lx/2, Lx/2, 25)
    bins_y = np.linspace(-Ly/2, Ly/2, 25)
    axes[0].hist(xy[:,0], bins=bins_x, color='steelblue', edgecolor='white')
    axes[0].set_xlabel(r'$x / \sigma_{bb}$'); axes[0].set_ylabel('Count')
    axes[0].set_title('x 方向分布（应均匀）'); axes[0].grid(alpha=0.3)
    axes[1].hist(xy[:,1], bins=bins_y, color='coral', edgecolor='white')
    axes[1].set_xlabel(r'$y / \sigma_{bb}$')
    axes[1].set_title('y 方向分布（应均匀）'); axes[1].grid(alpha=0.3)
    plt.suptitle(f'空间均匀性检验（非均匀=坐标仍有问题）', fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(output_dir, f'fig1b_hist_{sr_str}.png')
    fig.savefig(out2, dpi=200); plt.close(fig)
    print(f"  → 均匀性检验图: {out2}")

    return xy


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--npz',    default='cage_jumps_shearrate_0p015.npz')
    p.add_argument('--dump',   default='dump.shear_0.015.lammpstrj')
    p.add_argument('--rate',   type=float, default=0.015)
    p.add_argument('--output', default='figures_fixed')
    p.add_argument('--type',   type=int, default=1, dest='type_filter')
    args = p.parse_args()

    # 步骤 1：诊断
    if os.path.exists(args.npz):
        diagnose(args.npz, args.dump)
    else:
        print(f"✗ 找不到 {args.npz}"); sys.exit(1)

    # 步骤 2：用修复版重新生成图
    print("[重新绘图]")
    xy = plot_fixed_fig1b(args.npz, args.dump, args.rate, args.output, args.type_filter)

    if xy is not None and len(xy) > 10:
        print(f"\n✓ 如果 y 方向分布仍不均匀，说明存在问题 3（帧对齐）。")
        print(f"  请将 diagnose() 输出中的帧数信息发给我进一步排查。")
