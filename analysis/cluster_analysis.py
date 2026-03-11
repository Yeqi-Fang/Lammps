"""
cluster_analysis.py
===================
笼跳跃团簇识别与尺寸标度分析
复现论文 Fig.3c 和标度律 νc = λ/(D+1)

方法（论文 Appendix B，"Coarse sieve" + "Undersize sieve"）：
  1. 对每个时间窗口内发生笼跳跃的粒子，计算空间密度场
  2. 用指数核 φ(r) ∝ exp(-r/d) 卷积（d = g(r) 第一极小值处 ≈ σ_bb*1.2）
  3. 迭代确定密度阈值 ρth（coarse sieve）
  4. 识别团簇，过滤过小的团簇（undersize sieve）
  5. 计算等效半径 ξc = (3Vc/4π)^(1/3)
  6. 拟合 ξc ~ γ̇^{-νc}，验证 νc ≈ λ/(D+1)

用法：
    python cluster_analysis.py \
        --jumps cage_jumps_shearrate_0p015.npz \
        --frames dump.shear_0.015.lammpstrj \
        --rate 0.015 \
        --lambda_val 0.728 \
        --output figures
"""

# ===== PARAMETERS =====
D             = 3        # 维度
COARSE_D      = 1.2      # 粗化长度（σ_bb 单位，约为 g(r) 第一极小值）
ITER_TOL      = 0.10     # 迭代收敛阈值 C = 10%（论文设置）
C_PRIME       = 2        # 团簇最小尺寸因子（论文选 C'=2）
STZ_RADIUS    = 3.0      # STZ 特征尺寸（论文约 ~3d0）

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import argparse
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from read_dump import read_lammps_dump


def exponential_coarsegrain_3d(positions: np.ndarray,
                                 box: dict,
                                 coarse_d: float,
                                 grid_n: int = 30) -> Tuple:
    """
    用指数核 φ(r) ∝ exp(-r/d) 卷积粒子位置，得到空间密度场。
    （论文 MD 条件下的 coarse sieve 方法）

    Returns
    -------
    density : (grid_n, grid_n, grid_n) float array
    edges   : tuple of (xedges, yedges, zedges) for grid
    """
    Lx = box['Lx']; Ly = box['Ly']; Lz = box['Lz']

    # 使用高斯滤波近似指数核（计算效率高）
    # σ_gauss ≈ d（等效宽度）
    dx = Lx / grid_n
    dy = Ly / grid_n
    dz = Lz / grid_n

    density = np.zeros((grid_n, grid_n, grid_n))

    # 归一化坐标到 [0, grid_n)
    px = np.mod(positions[:, 0] - box['xlo'], Lx) / dx
    py = np.mod(positions[:, 1] - box['ylo'], Ly) / dy
    pz = np.mod(positions[:, 2] - box['zlo'], Lz) / dz

    # 累计每格点的粒子数
    for i in range(len(px)):
        ix = int(px[i]) % grid_n
        iy = int(py[i]) % grid_n
        iz = int(pz[i]) % grid_n
        density[ix, iy, iz] += 1.0

    # 高斯平滑（sigma = coarse_d / dx）
    sigma_vox = coarse_d / dx
    density = gaussian_filter(density, sigma=sigma_vox, mode='wrap')

    xedges = np.linspace(box['xlo'], box['xhi'], grid_n + 1)
    yedges = np.linspace(box['ylo'], box['yhi'], grid_n + 1)
    zedges = np.linspace(box['zlo'], box['zhi'], grid_n + 1)

    return density, (xedges, yedges, zedges)


def find_threshold_iterative(density: np.ndarray, tol: float = ITER_TOL) -> float:
    """
    迭代确定密度阈值 ρth（论文 Appendix B coarse sieve）。

    算法：
      ρ0 = max(density)
      ρ_{i+1} = mean(density[density < ρ_i])
      重复直到 |ρ_avg - ρ_i| / ρ_avg < tol
    """
    rho = density.ravel()
    rho_i = float(rho.max())

    for _ in range(100):
        mask = rho < rho_i
        if not mask.any():
            break
        rho_avg = float(rho[mask].mean())
        if rho_i > 0 and abs(rho_avg - rho_i) / rho_avg < tol:
            return rho_avg
        rho_i = rho_avg

    return rho_i


def identify_clusters_3d(density: np.ndarray,
                          edges: tuple,
                          rho_th: float,
                          min_volume_voxels: int = 4) -> List[Dict]:
    """
    从密度场识别团簇（connected components above threshold）。

    Returns
    -------
    clusters : list of dict
        {'volume': float, 'center': ndarray(3), 'n_voxels': int}
    """
    from scipy.ndimage import label

    binary = (density >= rho_th).astype(int)
    labeled, n_features = label(binary)

    xedges, yedges, zedges = edges
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = zedges[1] - zedges[0]
    voxel_vol = dx * dy * dz

    clusters = []
    for ci in range(1, n_features + 1):
        vox = np.argwhere(labeled == ci)
        n_vox = len(vox)
        if n_vox < min_volume_voxels:
            continue

        vol = n_vox * voxel_vol
        cx = np.mean(xedges[vox[:, 0]])
        cy = np.mean(yedges[vox[:, 1]])
        cz = np.mean(zedges[vox[:, 2]])

        clusters.append({
            'volume': vol,
            'n_voxels': n_vox,
            'center': np.array([cx, cy, cz]),
        })

    return clusters


def compute_cluster_radius(clusters: List[Dict]) -> float:
    """ξc = (3⟨Vc⟩/4π)^{1/3}"""
    if not clusters:
        return 0.0
    vols = np.array([c['volume'] for c in clusters])
    mean_vol = np.mean(vols)
    xi_c = (3 * mean_vol / (4 * np.pi)) ** (1.0 / 3.0)
    return xi_c


def analyze_clusters_single_rate(jumps_file: str,
                                   dump_file: str,
                                   shear_rate: float,
                                   output_dir: str = '.') -> Dict:
    """
    对单个剪切率进行团簇分析。

    Returns
    -------
    dict: {'shear_rate': ..., 'xi_c': ..., 'n_clusters': ...}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载 cage jump 数据
    data = np.load(jumps_file, allow_pickle=True)
    jump_frames = data['jump_frames']
    jump_times  = data['jump_times']
    types_all   = data['types']
    r_tilde     = data['r_tilde'].astype(np.float64)

    # 读取 dump 获取盒子信息
    print(f"  读取 dump 文件: {dump_file}")
    frames_obj = read_lammps_dump(dump_file, max_frames=3)
    if not frames_obj:
        print(f"  ⚠ 无法读取 dump 文件 {dump_file}")
        return {'shear_rate': shear_rate, 'xi_c': 0.0}
    box = frames_obj[0]['box']

    # 定义时间窗口（用整段轨迹内所有跳跃）
    # 获取跳跃粒子的实验室坐标（非仿射空间中的位置 r̃ 本身用于聚类）
    if len(jump_frames) == 0:
        print(f"  ⚠ 无笼跳跃事件，跳过团簇分析")
        return {'shear_rate': shear_rate, 'xi_c': 0.0}

    # 使用跳跃时刻的非仿射坐标
    positions = data['positions']  # (N_jumps, 3)
    if len(positions) == 0:
        return {'shear_rate': shear_rate, 'xi_c': 0.0}

    print(f"  N 跳跃 = {len(positions)}")

    # 近似盒子用于密度计算
    r_range = np.ptp(positions, axis=0)
    pseudo_box = {
        'xlo': positions[:, 0].min() - 1,
        'xhi': positions[:, 0].max() + 1,
        'ylo': positions[:, 1].min() - 1,
        'yhi': positions[:, 1].max() + 1,
        'zlo': positions[:, 2].min() - 1,
        'zhi': positions[:, 2].max() + 1,
        'Lx': r_range[0] + 2,
        'Ly': r_range[1] + 2,
        'Lz': r_range[2] + 2,
    }

    # 计算密度场
    grid_n = 20  # 粗分辨率
    density, edges = exponential_coarsegrain_3d(
        positions, pseudo_box, COARSE_D, grid_n)

    # 迭代确定阈值
    rho_th = find_threshold_iterative(density, ITER_TOL)
    print(f"  密度阈值 ρth = {rho_th:.4f}")

    # STZ 体积下限 Vc,ll = C' * Vstz
    Vstz = (4/3) * np.pi * (STZ_RADIUS / 2)**3
    Vc_ll = C_PRIME * Vstz
    dx = (pseudo_box['xhi'] - pseudo_box['xlo']) / grid_n
    min_vox = max(2, int(Vc_ll / dx**3))

    # 识别团簇
    clusters = identify_clusters_3d(density, edges, rho_th, min_vox)
    print(f"  识别到团簇: {len(clusters)} 个")

    if len(clusters) == 0:
        print(f"  ⚠ 未找到团簇，降低阈值重试...")
        rho_th *= 0.5
        clusters = identify_clusters_3d(density, edges, rho_th, 2)
        print(f"  重试后团簇数: {len(clusters)}")

    xi_c = compute_cluster_radius(clusters)
    print(f"  ξc = {xi_c:.3f} σ_bb")

    return {
        'shear_rate': shear_rate,
        'xi_c': xi_c,
        'n_clusters': len(clusters),
        'rho_th': rho_th,
    }


def fit_and_plot_scaling(shear_rates, xi_cs, lambda_val,
                          output_dir='.', D=3):
    """
    拟合 ξc ~ γ̇^{-νc} 并验证标度律 νc = λ/(D+1)。
    """
    os.makedirs(output_dir, exist_ok=True)

    sr  = np.array(shear_rates)
    xic = np.array(xi_cs)

    mask = xic > 0
    sr_fit  = sr[mask]
    xic_fit = xic[mask]

    if len(sr_fit) < 2:
        print(f"  ⚠ 有效数据点不足（{len(sr_fit)} 点），无法拟合标度律")
        print(f"  需要至少 2 个不同剪切率的结果")
        return None

    def power_law_neg(gdot, A, nu):
        return A * gdot**(-nu)

    try:
        popt, pcov = curve_fit(power_law_neg, sr_fit, xic_fit,
                                p0=[5.0, 0.18],
                                bounds=([0.1, 0.01], [100, 1.0]))
        A_fit, nu_c_fit = popt
        perr = np.sqrt(np.diag(pcov))

        nu_c_theory = lambda_val / (D + 1)

        print(f"\n  标度律拟合: ξc = {A_fit:.3f} × γ̇^{{-{nu_c_fit:.3f}±{perr[1]:.3f}}}")
        print(f"  理论值:     νc = λ/(D+1) = {lambda_val:.3f}/{D+1} = {nu_c_theory:.3f}")
        rel_err = abs(nu_c_fit - nu_c_theory) / nu_c_theory * 100
        print(f"  相对误差:   {rel_err:.1f}%")

        if rel_err < 20:
            print(f"  ✓ νc 与理论值吻合（误差 < 20%）")
        else:
            print(f"  ⚠ νc 偏差较大，可能需要更多剪切率数据")

    except Exception as e:
        print(f"  拟合失败: {e}")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sr_fit, xic_fit, s=80, color='steelblue', zorder=5,
               label='Simulation data')

    sr_plot = np.logspace(np.log10(sr_fit.min()), np.log10(sr_fit.max()), 100)
    ax.plot(sr_plot, power_law_neg(sr_plot, *popt), 'r-', lw=2,
            label=rf'$\xi_c \sim \dot\gamma^{{-{nu_c_fit:.3f}}}$')

    # 理论预测线
    ax.plot(sr_plot, power_law_neg(sr_plot, A_fit, nu_c_theory), 'g--', lw=2,
            label=rf'Theory: $\nu_c=\lambda/4={nu_c_theory:.3f}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\dot\gamma\ [\tau_0^{-1}]$', fontsize=12)
    ax.set_ylabel(r'$\xi_c\ [\sigma_{bb}]$', fontsize=12)
    ax.set_title('Convective cluster radius vs shear rate\n'
                 '(3D KA MD, T=0.45)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls=':', alpha=0.4)

    # 标注标度律
    ax.text(0.05, 0.92,
            rf'$\nu_c={nu_c_fit:.3f}\pm{perr[1]:.3f}$' + '\n' +
            rf'$\lambda/4={nu_c_theory:.3f}$',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.85))

    plt.tight_layout()
    out_png = os.path.join(output_dir, 'cluster_scaling.png')
    out_pdf = os.path.join(output_dir, 'cluster_scaling.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"  → 保存: {out_png}")
    plt.close(fig)

    # 保存数据表格
    out_csv = os.path.join(output_dir, 'cluster_scaling.csv')
    with open(out_csv, 'w') as f:
        f.write("# shear_rate  xi_c  lambda/(D+1)  nu_c_fit\n")
        for sr_i, xic_i in zip(sr_fit, xic_fit):
            f.write(f"{sr_i:.6f}  {xic_i:.6f}  {nu_c_theory:.6f}  {nu_c_fit:.6f}\n")
    print(f"  → 保存表格: {out_csv}")

    return nu_c_fit


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='团簇分析与标度律')
    parser.add_argument('--jumps',  nargs='+',
                        default=['cage_jumps_shearrate_0p015.npz'],
                        help='cage jump .npz 文件列表')
    parser.add_argument('--dumps',  nargs='+',
                        default=['dump.shear_0.015.lammpstrj'],
                        help='对应的 dump 文件列表')
    parser.add_argument('--rates',  nargs='+', type=float, default=None,
                        help='剪切率列表（不指定则从文件名提取）')
    parser.add_argument('--lambda_val', type=float, default=0.728,
                        help='剪切稀化指数 λ（从黏度拟合获取，3D KA ≈ 0.728）')
    parser.add_argument('--output', default='figures')
    args = parser.parse_args()

    import re, glob

    # 展开文件列表
    jumps_files = []
    for pat in args.jumps:
        jumps_files.extend(sorted(glob.glob(pat)))
    dumps_files = []
    for pat in args.dumps:
        dumps_files.extend(sorted(glob.glob(pat)))

    # 匹配文件和剪切率
    results = []
    for i, jf in enumerate(jumps_files):
        if not os.path.exists(jf):
            continue
        df = dumps_files[i] if i < len(dumps_files) else dumps_files[0]
        if args.rates and i < len(args.rates):
            sr = args.rates[i]
        else:
            m = re.search(r'(\d+p\d+|\d+\.\d+)', os.path.basename(jf))
            if m:
                sr = float(m.group(1).replace('p', '.'))
            else:
                sr = 0.015

        print(f"\n{'─'*50}")
        print(f"处理: {jf}  (γ̇={sr})")
        res = analyze_clusters_single_rate(jf, df, sr, args.output)
        results.append(res)

    if results:
        srs  = [r['shear_rate'] for r in results if r['xi_c'] > 0]
        xics = [r['xi_c']       for r in results if r['xi_c'] > 0]

        if len(srs) >= 2:
            print(f"\n{'='*50}")
            print(f"标度律拟合 (λ={args.lambda_val})")
            fit_and_plot_scaling(srs, xics, args.lambda_val,
                                  args.output, D=D)
        else:
            print(f"\n注意：只有 1 个剪切率结果，无法拟合标度律。")
            print(f"  γ̇={srs[0]:.4f}: ξc={xics[0]:.3f}")
            print(f"  理论期望 νc=λ/(D+1)={args.lambda_val/(D+1):.3f}")
            print(f"  请运行更多剪切率后再做标度律分析。")
