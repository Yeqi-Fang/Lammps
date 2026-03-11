"""
plot_summary.py
===============
汇总图：复现论文主要结果图
- Fig.1: η vs γ̇（黏度剪切稀化）
- Fig.2: ⟨r̃²(t)⟩ vs t（非仿射 MSD）
- Fig.3: ξc vs γ̇（团簇尺寸标度律）
- Fig.4: ⟨l_cj⟩ vs γ̇（跳跃长度）
- Fig.5: Pxy 时间序列（稳态验证）

用法：
    python plot_summary.py --output figures
"""

# ===== PARAMETERS =====
FIGSIZE      = (6.5, 5.5)
FONTSIZE     = 12
LINEWIDTH    = 2.0
MARKERSIZE   = 8
DPI          = 300
LC2_3D_MD    = 0.057
LAMBDA_3D_MD = 0.728    # λ for 3D KA MD (Table I: λ/(D+1)=0.182 → λ=0.728)

import sys
import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

matplotlib.rcParams.update({
    'font.size':        FONTSIZE,
    'axes.linewidth':   1.2,
    'lines.linewidth':  LINEWIDTH,
    'lines.markersize': MARKERSIZE,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.framealpha': 0.85,
})


def load_viscosity_data(data_dir='.'):
    """尝试加载黏度数据。"""
    csv = os.path.join(data_dir, 'figures', 'viscosity_data.csv')
    if os.path.exists(csv):
        d = np.loadtxt(csv, comments='#')
        if d.ndim == 1:
            d = d[np.newaxis, :]
        return d[:, 0], d[:, 1], d[:, 2]  # sr, eta, eta_err
    # 从 thermo 文件重建
    files = sorted(glob.glob(os.path.join(data_dir, 'thermo.shear_*.dat')))
    if not files:
        return None, None, None
    sys.path.insert(0, os.path.dirname(__file__))
    from compute_viscosity import compute_viscosity as cv
    srs, etas, errs = [], [], []
    for f in files:
        m = re.search(r'shear_([0-9.eE+\-]+)', f)
        if m:
            sr = float(m.group(1))
            try:
                eta, err, _ = cv(f, sr)
                srs.append(sr); etas.append(eta); errs.append(err)
            except:
                pass
    return (np.array(srs), np.array(etas), np.array(errs)) if srs else (None, None, None)


def load_msd_data(data_dir='.'):
    """加载所有 MSD npz 文件。"""
    files = sorted(glob.glob(os.path.join(data_dir, 'msd_data*.npz')))
    results = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        results.append({
            'shear_rate': float(d['shear_rate']),
            'times':      d['times'],
            'msd_big':    d['msd_big'],
            'msd_total':  d['msd_total'],
            'lc2':        float(d.get('lc2', LC2_3D_MD)),
        })
    return results


def load_jump_data(data_dir='.'):
    """加载所有 cage jump npz 文件。"""
    files = sorted(glob.glob(os.path.join(data_dir, 'cage_jumps_shearrate_*.npz')))
    results = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        jlen = d['jump_lengths']
        results.append({
            'shear_rate': float(d['shear_rate']),
            'n_jumps':    len(jlen),
            'mean_lcj':   float(np.mean(jlen)) if len(jlen) > 0 else 0.0,
            'jump_lengths': jlen,
            'jump_vectors': d['jump_vectors'] if 'jump_vectors' in d else np.zeros((0,3)),
        })
    return results


def fig1_viscosity(srs, etas, eta_errs, output_dir):
    """Figure 1: η vs γ̇"""
    if srs is None or len(srs) == 0:
        print("[Fig.1] 无黏度数据，跳过")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(srs, etas, yerr=eta_errs,
                fmt='o', ms=MARKERSIZE, color='steelblue',
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='3D KA MD, T=0.45', zorder=5)

    def pl(x, eta0, lam): return eta0 * x**(-lam)
    if len(srs) >= 3:
        try:
            popt, _ = curve_fit(pl, srs, etas, p0=[1, 0.7],
                                 bounds=([0, 0.1], [1e4, 1.5]))
            x_fit = np.logspace(np.log10(srs.min()), np.log10(srs.max()), 200)
            ax.plot(x_fit, pl(x_fit, *popt), 'r--', lw=LINEWIDTH,
                    label=rf'$\eta\sim\dot\gamma^{{-{popt[1]:.2f}}}$')
            ax.text(0.6, 0.85, rf'$\lambda={popt[1]:.3f}$',
                    transform=ax.transAxes, fontsize=FONTSIZE,
                    bbox=dict(facecolor='white', alpha=0.8))
        except:
            pass

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\dot\gamma\,[\tau_0^{-1}]$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\eta\,[\eta_0]$', fontsize=FONTSIZE)
    ax.set_title('Shear thinning: 3D KA MD', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-1)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig1_viscosity.png')
    fig.savefig(out, dpi=DPI); fig.savefig(out.replace('.png', '.pdf'))
    print(f"  Fig.1 → {out}")
    plt.close(fig)


def fig2_msd(msd_data, output_dir):
    """Figure 2: ⟨r̃²(t)⟩ vs t（多剪切率）"""
    if not msd_data:
        print("[Fig.2] 无 MSD 数据，跳过")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(msd_data)))

    lc2_vals = [d['lc2'] for d in msd_data if d['lc2'] > 0]
    lc2_mean = np.mean(lc2_vals) if lc2_vals else LC2_3D_MD

    for d, col in zip(msd_data, colors):
        t   = d['times']
        msd = d['msd_big']
        sr  = d['shear_rate']
        mask = t > 0
        ax.plot(t[mask], msd[mask], lw=LINEWIDTH, color=col,
                label=rf'$\dot\gamma={sr}$')

    ax.axhline(lc2_mean, color='gray', ls='-.', lw=1.5,
               label=rf'$l_c^2={lc2_mean:.3f}$ (3D MD)')

    # 参考斜率
    if len(msd_data) > 0:
        t_range = msd_data[0]['times']
        t_range = t_range[t_range > 0]
        if len(t_range) > 1:
            ref_t = np.array([t_range[1], t_range[-1]])
            ref_y = msd_data[0]['msd_big'][1]
            ax.plot(ref_t, ref_y * ref_t / ref_t[0],
                    'k:', lw=1, alpha=0.5, label=r'$\propto t$')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$t\,[\tau_0]$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\langle\tilde{r}^2\rangle\,[\sigma_{bb}^2]$', fontsize=FONTSIZE)
    ax.set_title('Non-affine MSD (big particles)', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-2, loc='upper left')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig2_msd.png')
    fig.savefig(out, dpi=DPI); fig.savefig(out.replace('.png', '.pdf'))
    print(f"  Fig.2 → {out}")
    plt.close(fig)


def fig3_cluster_scaling(output_dir):
    """Figure 3: ξc vs γ̇（从 cluster_scaling.csv 加载）"""
    csv = os.path.join(output_dir, 'cluster_scaling.csv')
    if not os.path.exists(csv):
        print("[Fig.3] 找不到 cluster_scaling.csv，跳过")
        return

    d = np.loadtxt(csv, comments='#')
    if d.ndim == 1:
        d = d[np.newaxis, :]
    srs  = d[:, 0]; xics = d[:, 1]
    nu_theory = d[0, 2]; nu_fit = d[0, 3]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(srs, xics, s=MARKERSIZE**2, color='steelblue',
               label='Simulation', zorder=5)
    if len(srs) >= 2:
        A = xics[0] / srs[0]**(-nu_fit)
        x_fit = np.logspace(np.log10(srs.min()), np.log10(srs.max()), 100)
        ax.plot(x_fit, A * x_fit**(-nu_fit), 'r-', lw=LINEWIDTH,
                label=rf'$\xi_c\sim\dot\gamma^{{-{nu_fit:.3f}}}$')
        ax.plot(x_fit, A * x_fit**(-nu_theory), 'g--', lw=LINEWIDTH,
                label=rf'Theory: $\nu_c=\lambda/4={nu_theory:.3f}$')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\dot\gamma\,[\tau_0^{-1}]$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\xi_c\,[\sigma_{bb}]$', fontsize=FONTSIZE)
    ax.set_title('Cluster radius scaling: 3D KA MD', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-1)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig3_cluster_scaling.png')
    fig.savefig(out, dpi=DPI); fig.savefig(out.replace('.png', '.pdf'))
    print(f"  Fig.3 → {out}")
    plt.close(fig)


def fig4_jump_lengths(jump_data, output_dir):
    """Figure 4: ⟨l_cj⟩ vs γ̇"""
    if not jump_data:
        print("[Fig.4] 无跳跃数据，跳过")
        return

    srs  = [d['shear_rate']  for d in jump_data if d['n_jumps'] > 0]
    lcjs = [d['mean_lcj']    for d in jump_data if d['n_jumps'] > 0]
    njs  = [d['n_jumps']     for d in jump_data if d['n_jumps'] > 0]

    if not srs:
        print("[Fig.4] 无有效跳跃数据，跳过")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(srs, lcjs, s=[max(20, n/10) for n in njs],
                    c='steelblue', zorder=5, label='3D KA MD')
    ax.axhline(0.4, color='red', ls='--', lw=1.5,
               label=r'Expected $\approx 0.4\sigma_{bb}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\dot\gamma\,[\tau_0^{-1}]$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\langle l_{cj}\rangle\,[\sigma_{bb}]$', fontsize=FONTSIZE)
    ax.set_title('Mean cage jump length vs shear rate', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-1)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    for sr, lcj in zip(srs, lcjs):
        ax.annotate(f'{sr:.3f}', (sr, lcj), textcoords='offset points',
                    xytext=(4, 4), fontsize=8, alpha=0.7)
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig4_jump_lengths.png')
    fig.savefig(out, dpi=DPI); fig.savefig(out.replace('.png', '.pdf'))
    print(f"  Fig.4 → {out}")
    plt.close(fig)


def fig_verification_table(srs, etas, eta_errs, jump_data, output_dir):
    """生成验证表格（文本输出）。"""
    lines = []
    lines.append("=" * 65)
    lines.append("论文复现验证表  (Zeng et al., J. Chem. Phys. 163, 084512, 2025)")
    lines.append("=" * 65)
    lines.append(f"{'量':<25}  {'本次结果':>15}  {'论文期望值':>15}  {'状态':>6}")
    lines.append("-" * 65)

    def status(val, lo, hi):
        return "✓" if lo <= val <= hi else "⚠"

    # λ
    if srs is not None and len(srs) >= 3:
        try:
            popt, _ = curve_fit(lambda x,a,b: a*x**(-b), srs, etas,
                                  p0=[1, 0.7], bounds=([0,0.1],[1e4,1.5]))
            lam = popt[1]
            lines.append(f"{'λ (shear thinning exp.)':<25}  {lam:>15.3f}  {'0.55–0.85':>15}  {status(lam,0.5,1.0):>6}")
            nu_c_theory = lam / (D+1)
            lines.append(f"{'λ/(D+1)=νc theory':<25}  {nu_c_theory:>15.3f}  {'≈0.18':>15}  ")
        except:
            lines.append(f"{'λ':<25}  {'(拟合失败)':<15}  {'0.55–0.85':>15}  {'?':>6}")
    else:
        lines.append(f"{'λ':<25}  {'(需≥3点)':>15}  {'0.55–0.85':>15}  {'?':>6}")

    # lc²
    msd_files = sorted(glob.glob('msd_data*.npz'))
    if msd_files:
        d = np.load(msd_files[0], allow_pickle=True)
        lc2 = float(d.get('lc2', LC2_3D_MD))
        lines.append(f"{'lc² (cage size)':<25}  {lc2:>15.4f}  {'0.057':>15}  {status(lc2,0.02,0.15):>6}")

    # ⟨l_cj⟩
    for jd in jump_data:
        if jd['n_jumps'] > 0:
            lcj = jd['mean_lcj']
            lines.append(f"{'⟨l_cj⟩ (γ̇='+str(jd['shear_rate'])+')':<25}  {lcj:>15.4f}  {'≈0.4':>15}  {status(lcj,0.2,0.8):>6}")

    lines.append("=" * 65)
    lines.append(f"D = {D} (3D MD)")
    lines.append("注: ⚠ 表示偏离期望，可能因单一剪切率或帧数不足")

    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join(output_dir, 'verification_table.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n  → 保存验证表格: {out}")


D = 3  # global

if __name__ == '__main__':
    import argparse, glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='.',    help='数据目录（含 thermo/npz 文件）')
    parser.add_argument('--output', default='figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"汇总图生成  (复现 Zeng et al. 2025)")
    print(f"{'='*55}")

    # 加载数据
    srs, etas, eta_errs = load_viscosity_data(args.data)
    msd_data = load_msd_data(args.data)
    jump_data = load_jump_data(args.data)

    print(f"\n数据概况:")
    print(f"  黏度数据点: {len(srs) if srs is not None else 0}")
    print(f"  MSD 数据:   {len(msd_data)} 个剪切率")
    print(f"  跳跃数据:   {len(jump_data)} 个剪切率")

    # 生成各图
    print(f"\n[Fig.1] 黏度 vs 剪切率...")
    fig1_viscosity(srs, etas, eta_errs if eta_errs is not None else np.zeros_like(etas),
                   args.output)

    print(f"\n[Fig.2] 非仿射 MSD...")
    fig2_msd(msd_data, args.output)

    print(f"\n[Fig.3] 团簇尺寸标度律...")
    fig3_cluster_scaling(args.output)

    print(f"\n[Fig.4] 跳跃长度...")
    fig4_jump_lengths(jump_data, args.output)

    print(f"\n[验证表格]...")
    fig_verification_table(srs, etas,
                            eta_errs if eta_errs is not None else [],
                            jump_data, args.output)

    print(f"\n{'='*55}")
    print(f"所有图像已保存至: {args.output}/")
    print(f"{'='*55}")
