# Zeng et al. (2025) 复现分析指南

## 论文信息

> **Connecting shear thinning and dynamic heterogeneity in supercooled liquids by localized elasticity**  
> Ke-Qi Zeng, Dong-Xu Yu, Zhe Wang  
> *J. Chem. Phys.* **163**, 084512 (2025)  
> DOI: 10.1063/5.0282802

---

## 你已运行的命令

```bash
# 1. 平衡化
mpirun -np 6 lmp -in in.equilibrate

# 2. 剪切生产跑（γ̇ = 0.015）
mpirun -np 6 lmp -in in.shear_template \
    -var SHEAR_RATE 0.015 -var DT 0.005 -var SEED 12345 \
    -log log.shear_0.015
```

---

## 预期输出文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `thermo.shear_0.015.dat` | step, temp, press, pxy | 黏度计算 |
| `dump.shear_0.015.lammpstrj` | 原子轨迹（~100帧） | MSD, 笼跳跃 |
| `visc_blockave.shear_0.015.dat` | 块平均 −Pxy | 快速黏度检查 |
| `restart.equil` | 平衡构型 | 已用完 |

---

## 快速分析（一键）

```bash
# 进入数据目录（LAMMPS 输出所在目录）
cd /path/to/your/simulation/

# 运行全部分析
bash /path/to/analysis/run_analysis.sh . 0.015 0.005
```

---

## 逐步分析

### 步骤 1：计算稳态黏度 η

```bash
python3 compute_viscosity.py \
    --thermo thermo.shear_0.015.dat \
    --rates  0.015 \
    --output figures/
```

**物理原理：**
η = −⟨P_xy⟩_steady / γ̇

LAMMPS 中 `pxy` 是维里应力 xy 分量（正值 = 压缩）。在剪切流下：
- P_xy < 0（液体对剪切阻力）  
- η = −P_xy / γ̇ > 0

**单一剪切率的期望值：**
- 对 γ̇ = 0.015 τ₀⁻¹（剪切稀化区间），η ≈ 10²–10³ η₀（量级估计）
- 论文 Fig.5a 显示 3D MD 的黏度量级

---

### 步骤 2：非仿射 MSD

```bash
python3 compute_msd_nonaffine.py \
    --dump  dump.shear_0.015.lammpstrj \
    --rate  0.015 \
    --dt    0.005 \
    --output figures/ \
    --save_npz msd_data.npz
```

**物理原理（Yamamoto-Onuki 1998）：**
$$\tilde{r}_i(t) = r_i(t) - r_i(0) - \dot\gamma \cdot t \cdot y_i(0)\, \hat{x}$$

减去仿射漂移（线性流场贡献）后，剩余的非仿射部分描述真正的笼扩散。

**期望行为：**
- 短时：$\langle\tilde{r}^2\rangle \propto t^2$（弹道区）
- 中间：平台区，$l_c^2 \approx 0.057\,\sigma_{bb}^2$（笼内振动）
- 长时：$\langle\tilde{r}^2\rangle \propto t$（扩散）

> ⚠ **注意**：γ̇ = 0.015 时帧数约 100 帧。平台可能不明显。  
> 脚本会自动使用论文给出的 $l_c^2 = 0.057$。

---

### 步骤 3：笼跳跃检测

```bash
python3 cage_jump_detection.py \
    --npz  msd_data.npz \
    --rate 0.015 \
    --output figures/ \
    --max_particles 2000
```

**物理原理（Candelier et al. 2009, 2010；论文 Appendix B）：**

对每个粒子的非仿射轨迹 $\tilde{r}(t)$，定义分离度：

$$p(t_c) = \zeta(t_c)\left[\langle d_1^2(t_2)\rangle_{S_2}\langle d_2^2(t_1)\rangle_{S_1}\right]^{1/2}$$

其中：
- $\zeta(t_c) = \sqrt{\frac{t_c}{T}\left(1 - \frac{t_c}{T}\right)}$
- $d_k(t_i)$：时刻 $t_i$ 到子集 $S_k$ 质心的距离
- $S_1 = \{t \leq t_c\}$，$S_2 = \{t \geq t_c\}$

$p(t_c)$ 最大处定义笼跳跃，递归应用直到 $p_{\max} < l_c^2$。

**期望结果：**
- $\langle l_{cj}\rangle \approx 0.4\,\sigma_{bb}$（论文基准值）
- $P_{ext} > 0.5$（跳跃偏向延伸方向，即第1、3象限）

---

### 步骤 4：汇总图

```bash
python3 plot_summary.py --data . --output figures/
```

生成复现论文主图的对比图。

---

## 如何复现论文主要结论

### 结论 1：剪切稀化指数 λ

需要**多个剪切率**的数据。当前只有 γ̇ = 0.015。
建议运行：
```bash
for SR in 0.003 0.005 0.01 0.015 0.02 0.05 0.1; do
    mpirun -np 6 lmp -in in.shear_template \
        -var SHEAR_RATE $SR \
        -var DT $(python3 -c "print(0.003 if $SR<=0.01 else 0.001)") \
        -var SEED 12345 \
        -log log.shear_$SR
done
```
期望结果：λ/(D+1) = 0.182，即 λ ≈ **0.73**（论文 Table I，3D MD）

### 结论 2：标度律 νc = λ/(D+1)

| 体系 | λ/(D+1) | νc（模拟） | 吻合 |
|------|---------|-----------|------|
| 3D MD | 0.182 | 0.176 | ✓（误差 3%）|

从团簇半径 ξc 拟合 νc，验证与 λ/4 的一致性。

### 结论 3：笼跳跃长度

论文给出 $l_c^2 = 0.057$，跳跃长度约 $0.4\,\sigma_{bb}$。

---

## 帧数说明

对于 γ̇ = 0.015，dt = 0.005：
- `DUMP_EVERY` = round(0.5/(0.015×0.005)) = **6667 步**
- `N_prod` = round(50/(0.015×0.005)) = **666667 步**
- **约 100 帧**（生产跑总共 50 个应变单位，每 0.5 输出一帧）

100 帧对于 MSD 平台和笼跳跃检测是**偏少**的。可通过 `--max_frames` 参数控制。

---

## 物理验证核查表

```
□ 密度: N/L³ ≈ 1.2
□ 温度: ⟨T⟩ ≈ 0.45 (thermo 文件中检查)
□ Pxy < 0（在正剪切下，液体受力方向正确）
□ η = −⟨Pxy⟩/γ̇ > 0
□ lc² ≈ 0.057 (MSD 平台值)
□ ⟨l_cj⟩ ≈ 0.4 (需要多帧)
□ Pext > 0.5（论文 Fig.2c）
□ λ/(D+1) ≈ νc（需多个 γ̇）
```

---

## 文件结构

```
analysis/
├── read_dump.py              LAMMPS dump 解析器（含 Lees-Edwards 盒子处理）
├── compute_viscosity.py      黏度 η = −⟨Pxy⟩/γ̇
├── compute_msd_nonaffine.py  非仿射 MSD（Yamamoto-Onuki 定义）
├── cage_jump_detection.py    Candelier 算法笼跳跃检测
├── cluster_analysis.py       团簇识别与 ξc ~ γ̇^{-νc} 标度
├── plot_summary.py           汇总图
├── run_analysis.sh           一键运行脚本
└── README.md                 本文件
```

---

## 常见问题

**Q: dump 文件找不到？**  
A: 检查 LAMMPS 是否运行到生产跑阶段（pre-run 需要 N_pre = 20/(0.015×0.005) ≈ 266667 步）。

**Q: MSD 没有平台？**  
A: 帧数太少或 γ̇ 太高导致笼破裂太快。脚本会自动使用论文值 lc²=0.057。

**Q: 未检测到笼跳跃？**  
A: 尝试降低 `--lc2` 或增加 `--max_particles`。

**Q: 如何复现 Fig.5(a) 的黏度图？**  
A: 需要运行至少 5 个剪切率（0.003–0.1），见上方多剪切率运行命令。
