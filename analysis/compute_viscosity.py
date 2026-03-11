"""
compute_viscosity.py
====================
从 LAMMPS thermo 输出文件计算稳态剪切黏度

方法：η = -⟨P_xy⟩_steady / γ̇
      （丢弃前 20% 作为瞬态）

用法：
    python compute_viscosity.py [thermo_file] [shear_rate]

或者批量：
    python compute_viscosity.py --batch data/thermo.shear_*.dat
"""

# ===== PARAMETERS =====
DISCARD_FRACTION = 0.20   # 丢弃前 20% 作为瞬态
BLOCK_SIZE       = 200    # 块平均大小（步数）

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
def read_thermo(filename):
    """
    读取 LAMMPS fix ave/time 输出的热力学文件。
    格式：# 注释行（跳过），然后 step temp press pxy
    返回 dict with arrays.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue
    if len(data) == 0:
        raise ValueError(f"No data in {filename}")
    arr = np.array(data)
    result = {'step': arr[:, 0]}
    # 列名顺序：step temp press pxy（来自 in.shear_template）
    if arr.shape[1] >= 4:
        result['temp']  = arr[:, 1]
        result['press'] = arr[:, 2]
        result['pxy']   = arr[:, 3]
    return result


def compute_viscosity(thermo_file, shear_rate):
    """
    计算单个剪切率的黏度。

    Returns
    -------
    eta       : float, 稳态黏度
    eta_err   : float, 块平均误差棒
    pxy_mean  : float, ⟨P_xy⟩
    """
    d = read_thermo(thermo_file)
    pxy = d['pxy']
    n   = len(pxy)

    # 丢弃前 DISCARD_FRACTION
    start = int(n * DISCARD_FRACTION)
    pxy_ss = pxy[start:]   # 稳态数据

    pxy_mean = np.mean(pxy_ss)
    eta      = -pxy_mean / shear_rate  # η = -⟨Pxy⟩/γ̇

    # 块平均误差估计
    n_ss     = len(pxy_ss)
    n_blocks = n_ss // BLOCK_SIZE
    if n_blocks >= 2:
        block_means = np.array([
            np.mean(pxy_ss[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
            for i in range(n_blocks)
        ])
        eta_err = np.std(-block_means / shear_rate) / np.sqrt(n_blocks)
    else:
        eta_err = np.std(pxy_ss) / np.sqrt(n_ss) / shear_rate

    print(f"  γ̇={shear_rate:.4f}: η={eta:.4f} ± {eta_err:.4f}  "
          f"(⟨Pxy⟩={pxy_mean:.6f},  N_ss={n_ss})")
    return eta, eta_err, pxy_mean


def power_law(gdot, eta0, lam):
    """η = eta0 * γ̇^(-λ)"""
    return eta0 * gdot ** (-lam)


def plot_viscosity(shear_rates, etas, eta_errs, output_dir='.'):
    """绘制 η vs γ̇ 并做幂律拟合。"""
    os.makedirs(output_dir, exist_ok=True)

    sr  = np.array(shear_rates)
    eta = np.array(etas)
    err = np.array(eta_errs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(sr, eta, yerr=err, fmt='o', ms=7, color='steelblue',
                capsize=4, label='Simulation', zorder=5)

    # 幂律拟合（只用多于 1 个点时）
    if len(sr) >= 3:
        # 取剪切稀化区间（η 随 γ̇ 单调下降的部分）
        try:
            popt, pcov = curve_fit(
                power_law,
                sr, eta,
                p0=[eta[0], 0.6],
                sigma=err if np.all(err > 0) else None,
                maxfev=5000,
                bounds=([0, 0.01], [1e6, 2.0])
            )
            eta0_fit, lam_fit = popt
            perr = np.sqrt(np.diag(pcov))
            print(f"\n幂律拟合: η = {eta0_fit:.3f} × γ̇^{{-{lam_fit:.3f}±{perr[1]:.3f}}}")

            # 验证：3D KA 的 λ 应在 0.5~0.9 之间
            if 0.4 < lam_fit < 1.2:
                print(f"  ✓ λ={lam_fit:.3f} 在合理范围 (期望 0.55~0.73)")
            else:
                print(f"  ⚠ λ={lam_fit:.3f} 超出预期范围，可能需要更多剪切率数据点")

            sr_fit = np.logspace(np.log10(sr.min()), np.log10(sr.max()), 200)
            ax.plot(sr_fit, power_law(sr_fit, *popt), 'r--', lw=2,
                    label=rf'$\eta \sim \dot\gamma^{{-{lam_fit:.3f}}}$')

            # 标注 νc = λ/(D+1)
            nuc = lam_fit / 4.0  # D=3
            ax.text(0.05, 0.15,
                    rf'$\lambda={lam_fit:.3f}$, $\nu_c=\lambda/4={nuc:.3f}$',
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"  幂律拟合失败: {e}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\dot\gamma\ [\tau_0^{-1}]$', fontsize=12)
    ax.set_ylabel(r'$\eta\ [\eta_0]$', fontsize=12)
    ax.set_title('Shear Viscosity vs Shear Rate\n(3D KA Liquid, T=0.45)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    plt.tight_layout()

    out_png = os.path.join(output_dir, 'viscosity_vs_shearrate.png')
    out_pdf = os.path.join(output_dir, 'viscosity_vs_shearrate.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"  → 保存图像: {out_png}")
    plt.close(fig)

    # 保存数据
    out_csv = os.path.join(output_dir, 'viscosity_data.csv')
    np.savetxt(out_csv,
               np.column_stack([sr, eta, err]),
               header='shear_rate  eta  eta_err',
               comments='# ')
    print(f"  → 保存数据: {out_csv}")


def plot_pxy_timeseries(thermo_file, shear_rate, output_dir='.'):
    """绘制 Pxy 时间序列以检验稳态。"""
    os.makedirs(output_dir, exist_ok=True)
    d = read_thermo(thermo_file)
    pxy  = d['pxy']
    step = d['step']
    n    = len(pxy)
    start = int(n * DISCARD_FRACTION)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # 时间序列
    ax = axes[0]
    ax.plot(step, pxy, lw=0.7, color='gray', alpha=0.7)
    ax.plot(step[start:], pxy[start:], lw=0.7, color='steelblue',
            label='Steady-state region')
    ax.axhline(-shear_rate * np.mean(pxy[start:]) / shear_rate * (-1),
               color='red', ls='--', label=f'⟨Pxy⟩={np.mean(pxy[start:]):.4f}')
    ax.axvline(step[start], color='green', ls=':', label='Discard boundary')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel(r'$P_{xy}$', fontsize=11)
    ax.set_title(rf'$P_{{xy}}$ time series, $\dot\gamma={shear_rate}$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 块平均
    ax2 = axes[1]
    pxy_ss = pxy[start:]
    n_blocks = len(pxy_ss) // BLOCK_SIZE
    if n_blocks > 0:
        block_idx  = np.arange(n_blocks) * BLOCK_SIZE
        block_pxy  = np.array([np.mean(pxy_ss[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
                                for i in range(n_blocks)])
        ax2.plot(block_idx, block_pxy, 'o-', ms=4, color='darkorange')
        ax2.axhline(np.mean(pxy_ss), color='red', ls='--')
        ax2.set_xlabel('Block start (steps into steady state)', fontsize=11)
        ax2.set_ylabel(r'Block $\langle P_{xy}\rangle$', fontsize=11)
        ax2.set_title('Block-averaged Pxy', fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    sr_str = f"{shear_rate}".replace('.', 'p')
    out = os.path.join(output_dir, f'pxy_timeseries_{sr_str}.png')
    fig.savefig(out, dpi=200)
    print(f"  → 保存: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='计算剪切黏度')
    parser.add_argument('--thermo', nargs='+',
                        default=['thermo.shear_0.015.dat'],
                        help='thermo 文件列表（支持通配符）')
    parser.add_argument('--rates', nargs='+', type=float,
                        default=None,
                        help='对应的剪切率（不指定则从文件名提取）')
    parser.add_argument('--output', default='figures',
                        help='输出目录')
    args = parser.parse_args()

    # 展开通配符
    files = []
    for pat in args.thermo:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print(f"未找到文件: {args.thermo}")
        sys.exit(1)

    # 提取剪切率
    shear_rates = []
    for i, f in enumerate(files):
        if args.rates and i < len(args.rates):
            sr = args.rates[i]
        else:
            # 从文件名提取：thermo.shear_0.015.dat -> 0.015
            import re
            m = re.search(r'shear_([0-9.eE+\-]+)', f)
            if m:
                sr = float(m.group(1))
            else:
                print(f"⚠ 无法从文件名提取剪切率: {f}，请用 --rates 指定")
                sr = 0.01
        shear_rates.append(sr)

    print(f"\n{'='*50}")
    print(f"分析文件: {len(files)} 个")
    print(f"{'='*50}")

    etas, eta_errs = [], []
    for f, sr in zip(files, shear_rates):
        if not os.path.exists(f):
            print(f"文件不存在: {f}")
            continue
        print(f"\n[{f}]  γ̇ = {sr}")
        eta, eta_err, _ = compute_viscosity(f, sr)
        etas.append(eta)
        eta_errs.append(eta_err)

        # 时间序列图
        plot_pxy_timeseries(f, sr, args.output)

    if len(etas) > 0:
        plot_viscosity(shear_rates[:len(etas)], etas, eta_errs, args.output)

        print(f"\n{'='*50}")
        print("汇总结果:")
        print(f"  {'γ̇':>10}  {'η':>10}  {'η_err':>10}")
        for sr, eta, err in zip(shear_rates, etas, eta_errs):
            print(f"  {sr:>10.4f}  {eta:>10.4f}  {err:>10.4f}")
        print(f"{'='*50}")
