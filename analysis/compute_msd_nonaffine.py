"""
compute_msd_nonaffine.py
========================
计算 Yamamoto-Onuki 非仿射均方位移 ⟨r̃²(t)⟩

非仿射位移定义 (Yamamoto & Onuki 1998, Phys. Rev. E 58, 3515):
    r̃_i(t) = r_i(t) - r_i(0) - γ̇ · t · y_i(0) · x̂

MSD: ⟨r̃²(t)⟩ = (1/N) Σ_i |r̃_i(t)|²

从 MSD 的平台值确定 lc²（笼子尺寸），供 cage jump 检测使用。

用法：
    python compute_msd_nonaffine.py \
        --dump dump.shear_0.015.lammpstrj \
        --rate 0.015 \
        --dt_frame 33.333 \
        --output figures
"""

# ===== PARAMETERS =====
LC2_EXPECTED_3D = 0.057   # 论文 Appendix B: lc² = 0.057 for 3D MD

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from read_dump import read_lammps_dump, compute_nonaffine_displacement


def compute_msd_from_r_tilde(r_tilde, particle_types=None):
    """
    从非仿射位移数组计算 MSD。

    Parameters
    ----------
    r_tilde : (N_frames, N_atoms, 3) array
    particle_types : (N_atoms,) array or None

    Returns
    -------
    msd_total : (N_frames,) array
    msd_big   : (N_frames,) array  (type==1)
    msd_small : (N_frames,) array  (type==2)
    """
    # |r̃|² 对每个粒子
    r2 = np.sum(r_tilde**2, axis=2)  # (N_frames, N_atoms)

    msd_total = np.mean(r2, axis=1)

    if particle_types is not None:
        mask_b = (particle_types == 1)
        mask_s = (particle_types == 2)
        msd_big   = np.mean(r2[:, mask_b], axis=1) if mask_b.any() else msd_total
        msd_small = np.mean(r2[:, mask_s], axis=1) if mask_s.any() else msd_total
    else:
        msd_big   = msd_total
        msd_small = msd_total

    return msd_total, msd_big, msd_small


def find_lc2_from_msd(times, msd, lc2_expected=LC2_EXPECTED_3D):
    """
    从 MSD 曲线识别平台值 lc²。

    方法：
    1. 找 MSD 对时间对数导数最小的区间（最平）
    2. 取该区间的平均值作为 lc²

    Returns
    -------
    lc2    : float  平台值
    t_plat : float  平台中心时间
    """


    early_mask = (times < 50.0) & (msd < 2.0)
    if early_mask.sum() < 5:
        early_mask = np.ones(len(times), bool)
    times_e = times[early_mask]
    msd_e   = msd[early_mask]
    # 用对数导数找最平坦区域
    log_t   = np.log(times_e[1:] + 1e-12)
    log_msd = np.log(msd_e[1:] + 1e-30)
    d_log   = np.gradient(log_msd, log_t)

    # 找导数绝对值最小的窗口
    window   = max(3, len(d_log) // 6)
    smoothed = np.convolve(np.abs(d_log),
                            np.ones(window)/window, mode='valid')
    if len(smoothed) == 0:
        return lc2_expected, times[len(times)//2]

    plateau_center = np.argmin(smoothed) + window // 2 + 1
    plateau_center = max(1, min(plateau_center, len(msd_e)-1))

    lc2    = msd_e[plateau_center]
    t_plat = times_e[plateau_center]

    # 合理性检验
    if not (0.01 < lc2 < 1.0):
        print(f"  ⚠ 检测到 lc²={lc2:.4f}，超出合理范围 [0.01, 1.0]")
        print(f"  使用理论值 lc²={lc2_expected}")
        return lc2_expected, t_plat

    print(f"  检测到 lc²={lc2:.4f}  (理论值 {lc2_expected}，相对误差 "
          f"{abs(lc2-lc2_expected)/lc2_expected*100:.1f}%)")
    return lc2, t_plat


def plot_msd(times, msd_total, msd_big, msd_small,
             shear_rate, lc2, t_plat, output_dir='.'):
    """绘制非仿射 MSD 图。"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(times, msd_total, 'k-',  lw=2.0, label='All particles')
    ax.plot(times, msd_big,   'b--', lw=1.8, label='Big (type 1)')
    ax.plot(times, msd_small, 'r:',  lw=1.8, label='Small (type 2)')

    # 标注平台值
    ax.axhline(lc2, color='green', ls='-.', lw=1.5,
               label=rf'$l_c^2={lc2:.4f}$')
    ax.axvline(t_plat, color='gray', ls=':', lw=1, alpha=0.6)

    # 参考线
    t_range = np.array([times[1], times[-1]])
    ax.plot(t_range, 0.01 * t_range,    'gray', lw=0.8, ls='--', alpha=0.5,
            label=r'$\propto t$')
    ax.plot(t_range, 0.001 * t_range**2, 'gray', lw=0.8, ls='dotted', alpha=0.5,
            label=r'$\propto t^2$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\ [\tau_0]$', fontsize=12)
    ax.set_ylabel(r'$\langle \tilde{r}^2(t)\rangle\ [\sigma_{bb}^2]$', fontsize=12)
    ax.set_title(rf'Non-affine MSD, $\dot\gamma={shear_rate}$, T=0.45', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    plt.tight_layout()

    sr_str = f"{shear_rate}".replace('.', 'p')
    out = os.path.join(output_dir, f'msd_nonaffine_shearrate_{sr_str}.png')
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.png', '.pdf'))
    print(f"  → 保存: {out}")
    plt.close(fig)


def plot_msd_comparison(all_data, output_dir='.'):
    """
    多剪切率 MSD 对比图（复现论文图风格）。

    all_data : list of (shear_rate, times, msd_big) tuples
    """
    if not all_data:
        return
    os.makedirs(output_dir, exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_data)))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (sr, times, msd_b), color in zip(all_data, colors):
        ax.plot(times, msd_b, lw=2, color=color,
                label=rf'$\dot\gamma={sr}$')

    ax.axhline(LC2_EXPECTED_3D, color='gray', ls='--', lw=1.5,
               label=rf'$l_c^2={LC2_EXPECTED_3D}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t\ [\tau_0]$', fontsize=12)
    ax.set_ylabel(r'$\langle \tilde{r}^2(t)\rangle$ (big particles)', fontsize=12)
    ax.set_title('Non-affine MSD at multiple shear rates', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    plt.tight_layout()

    out = os.path.join(output_dir, 'msd_nonaffine_comparison.png')
    fig.savefig(out, dpi=300)
    print(f"  → 保存: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算非仿射 MSD')
    parser.add_argument('--dump',    default='dump.shear_0.015.lammpstrj',
                        help='LAMMPS dump 文件')
    parser.add_argument('--rate',    type=float, default=0.015,
                        help='剪切率 γ̇')
    parser.add_argument('--dt_frame',type=float, default=None,
                        help='帧间时间间隔 (LJ units). 不指定则从 dump_every*dt 计算')
    parser.add_argument('--dt',      type=float, default=0.005,
                        help='LAMMPS 时间步长 (默认 0.005)')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--output',  default='figures')
    parser.add_argument('--save_npz', default='msd_data.npz',
                        help='保存 MSD 数据供 cage jump 使用')
    args = parser.parse_args()

    if not os.path.exists(args.dump):
        print(f"错误：找不到文件 {args.dump}")
        print("请先运行 LAMMPS 剪切模拟，生成 dump 文件")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"非仿射 MSD 分析")
    print(f"  文件: {args.dump}")
    print(f"  γ̇  = {args.rate}")
    print(f"{'='*55}")

    # 读取 dump
    print("\n[1] 读取 dump 文件...")
    frames = read_lammps_dump(args.dump, max_frames=args.max_frames)
    if len(frames) < 3:
        print(f"帧数不足 ({len(frames)})，请检查 dump 文件")
        sys.exit(1)

    # 计算帧间时间
    if args.dt_frame is None:
        # 从 timestep 差计算
        if len(frames) >= 2:
            dt_frame = (frames[1]['timestep'] - frames[0]['timestep']) * args.dt
        else:
            dt_frame = 0.5 / args.rate  # 默认：dump_every * dt
        print(f"  帧间时间: Δt_frame = {dt_frame:.3f} τ₀")
    else:
        dt_frame = args.dt_frame

    times = np.arange(len(frames)) * dt_frame

    # 计算非仿射位移
    print("\n[2] 计算非仿射位移 r̃(t)...")
    r_tilde = compute_nonaffine_displacement(frames, args.rate, dt_frame)
    print(f"  r_tilde shape: {r_tilde.shape}  (frames × atoms × xyz)")

    # 粒子类型
    types = frames[0]['type']
    n_big   = np.sum(types == 1)
    n_small = np.sum(types == 2)
    print(f"  大粒子 (type 1): {n_big},  小粒子 (type 2): {n_small}")

    # 计算 MSD
    print("\n[3] 计算 MSD...")
    msd_total, msd_big, msd_small = compute_msd_from_r_tilde(r_tilde, types)

    # 寻找平台值 lc²
    print("\n[4] 确定笼子尺寸 lc²...")
    lc2, t_plat = find_lc2_from_msd(times[1:], msd_total[1:])

    # 绘图
    print("\n[5] 绘图...")
    os.makedirs(args.output, exist_ok=True)
    plot_msd(times, msd_total, msd_big, msd_small,
             args.rate, lc2, t_plat, args.output)

    # 保存数据
    np.savez(args.save_npz,
             times=times, shear_rate=args.rate,
             msd_total=msd_total, msd_big=msd_big, msd_small=msd_small,
             lc2=lc2, t_plat=t_plat,
             r_tilde=r_tilde.astype(np.float32),
             box_Lx=np.float32(frames[0]['box']['Lx']),
          box_Ly=np.float32(frames[0]['box']['Ly']),
          box_Lz=np.float32(frames[0]['box']['Lz']),
             types=types)
    print(f"\n  → 保存 MSD + r̃ 数据: {args.save_npz}")

    # 验证
    print(f"\n{'='*55}")
    print(f"验证检查点:")
    print(f"  lc²  = {lc2:.4f}  (论文值 0.057，允许误差 50%)")
    print(f"  N 帧 = {len(frames)}")
    if 0.02 < lc2 < 0.15:
        print(f"  ✓ lc² 在合理范围")
    else:
        print(f"  ⚠ lc² 偏离预期，可能帧数不足或剪切率偏高")
    print(f"{'='*55}")
