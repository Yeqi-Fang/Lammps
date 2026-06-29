# Zeng BD Strict Reproduction Notes

本文件只覆盖 Zeng et al., J. Chem. Phys. 163, 084512 (2025) 的 BD 部分。
MD、Fig. 5、Fig. 6、Fig. 7 不在本文档范围内。

## 1. 图和体系边界

| 目标 | BD/MD | 严格归属 | 文献依据 | 当前处理原则 |
| --- | --- | --- | --- | --- |
| Fig. 1(a) | BD | `phi=42.5%,45%` 的 Yukawa viscosity | Zeng Eq. (1), Appendix A; PRX Ref.33 Eq. (10) | 只用 BD Yukawa stress/force，不混入 MD 数据 |
| Fig. 1(b) | BD | `phi=45%, gamma_dot*tau0=3` 的 cage-jump 投影 | Zeng Fig. 1 caption, Appendix B | 需要 `t_chi` 窗口和 cage jumps |
| Fig. 1(c) | BD | Fig. 1(b) 红色 cluster 的 convective traceback | Zeng Sec. III A | 需要同一 cluster boundary |
| Fig. 2 | BD | Fig. 1(b) 同一 convective cluster | Zeng Fig. 2 caption, Eqs. (2)-(7) | 需要 `sigma_c, gamma_c, n_cj, P_ext, R` |
| Fig. 3 | BD | convective cluster 与 shear thinning 关系 | Zeng Eqs. (8)-(9), Fig. 3 caption | 需要多剪切率 cluster 统计 |
| Fig. 4 | BD | Fig. 2 同一 BD cluster 的 PHM/STZ 和 `g_cj`/`alpha` | Zeng Fig. 4 caption | Fig. 4 不是 MD；PHM 需 Ref.57 |

## 2. BD 物理模型

已由论文明确给出的部分：

- 体系：3D binary mixture，`Ns=4000` small particles，`Nb=16000` big particles，`N=20000`。
- 直径比：`ds/db=2/3`。
- 体积分数：`phi=42.5%` 和 `phi=45%`。
- 单位：`d0=db`，`tau0=db^2/D0`，`sigma0=kBT/d0^3`，`eta0=kBT*tau0/d0^3`。
- 相互作用：hard-sphere Yukawa potential；Yukawa 部分
  `V(r)=K exp[-z(r-dij)]/(r/dij)` for `r>=dij=(di+dj)/2`。
- Yukawa 参数：`z=4.86/db`，`K=9.69 kBT`，cutoff `r=5db`。
- 剪切：flow direction `x`，gradient direction `y`，Lees-Edwards boundary。
- PRX Ref.33 给出的 BD 运动方程：
  `r_i(t+dt)-r_i(t)=D0*f_i(t)*dt/(kBT)+sqrt(2D0*dt)G+H*r_i(t)*dt`。
- Peclet 换算：`Pe=gamma_dot*db^2/(4D0)=gamma_dot*tau0/4`。

不能当作 Zeng 原文事实的部分：

- BD integrator timestep：Zeng/PRX 主文未给出。
- 平衡步数、预剪切长度、生产长度、steady-state 丢弃规则：Zeng/PRX 主文未给出。
- 独立样本数、随机种子、误差条统计方式：Zeng/PRX 主文未给出。
- hard-core potential-free algorithm 的迭代容差、pair 顺序、残余 overlap 处理：Zeng 只引用 Ref.60 和 Ref.61；当前缺 Ref.61。
- PCCP ESI 的 `r^-36` 软核属于 Ref.40 的旧 `N=3000, phi=0.4` BD 辅助模拟，不能替代 Zeng 2025 的 hard-core potential-free 主体系。

## 3. BD 模拟目标

优先复现 Zeng BD 高剪切行为：

| `gamma_dot*tau0` | `Pe` | 用途 |
| ---: | ---: | --- |
| 1 | 0.25 | Fig. 1(a), Fig. 3, Fig. 4(b-d) |
| 3 | 0.75 | 核心代表条件；Fig. 1(b-c), Fig. 2, Fig. 4(a) |
| 10 | 2.5 | Fig. 1(a), Fig. 3, Fig. 4(b-d) |
| 20 | 5.0 | Fig. 1(a), Fig. 3, Fig. 4(b-d) |

低剪切率不是本阶段优先项。若高剪切拟合或 `xi_c` 标度点数不足，再单独决定是否补中低剪切率。

## 4. 原始输出数据要求

BD 原始轨迹至少需要包含：

- `times`。
- 粒子 `id/type`，其中 type 必须能区分 big/small。
- lab-frame 或可还原 lab-frame 的 `x,y,z`。
- Lees-Edwards 下的 image/unwrap 信息，或直接保存可用于 nonaffine reconstruction 的坐标。
- box length 和 shear strain/tilt 信息。
- 可重算 Yukawa `dV/dr` 的坐标；若直接输出 force/stress，则必须说明只包含 Yukawa contribution 还是包含 hard-core collision contribution。

严格要求：

- Fig. 1(a) 和 `sigma_c(t)` 优先使用 Yukawa differentiable force/stress。
- hard-core collision stress 不混入 BD `sigma_c`，除非后续 Ref.61 或作者 protocol 明确要求。
- cage jumps、`chi4`、cluster 计算默认使用所有粒子；如果只用 big particles，必须另列为假设，因为 Zeng Appendix B 的 `N` 和 Fig. 1(b) caption 未说明只筛 big particles。

## 5. BD 分析定义

### Fig. 1(a): viscosity

- 公式：`eta=-<sum_i r_ix f_iy>/(V gamma_dot)`。
- 来源：Zeng Eq. (1)；PRX Ref.33 Eq. (10)。
- 解释：`f_iy` 是 deterministic interparticle force；按 PRX/Zeng 上下文优先理解为 Yukawa contribution。
- 验收：`eta>0`，高剪切区出现 shear thinning，拟合 `eta~gamma_dot^-lambda`。

### Fig. 1(b-c): cage jumps and convective traceback

- cage jump：Candelier algorithm，轨迹使用 Yamamoto-Onuki nonaffine displacement。
- `lc^2=0.05` for BD。
- `t_chi`：`chi4(t)` 最大值对应时间。
- overlap function cutoff：`a=0.2` for BD。
- steady shear overlap 需使用 Zeng Appendix B 的 shear-adjusted distance。
- Fig. 1(c) 的 boundary 必须是 Fig. 1(b) 红色 cluster 的 affine traceback/transport，不是重新选择的 cluster。

### Fig. 2: cluster time evolution

- `sigma_i,xy(t)`：Zeng Eq. (2)，由 `dV/dr * r_ij,x*r_ij,y/|r_ij|` 和粒子体积 `Omega_alpha_i` 得到。
- `Omega_b/Omega_s=(db/ds)^D`，`Nb*Omega_b+Ns*Omega_s=V`，来源 Zeng Eq. (3)。
- `sigma_c(t)`：Zeng Eq. (4)，对 cluster 内粒子按体积加权平均。
- `gamma_c(t)`：Zeng Eqs. (5)-(7)，Falk-Langer local strain。
- `n_cj(t)`：cluster 内 `[t-Delta t/2,t+Delta t/2]` 的 cage-jump 数。
- `P_ext(t)`：cage-jump displacement 在 shear plane 的投影落在第一或第三象限的比例。
- `R(t)=1-N_res(t)/N_res(t0)`。
- `t', t0, t1, t2, t3`：原文没有给数值。必须人工从图/原始数据指定，或标为数据驱动判定；不能默认为论文事实。

### Fig. 3: cluster statistics and shear thinning

- `eta_c=Pc_bar*sigma_c,y/(2*gamma_dot)`，来源 Zeng Eq. (8)。
- `xi_c=(3Vc/4pi)^(1/3)`，来源 Zeng Fig. 3 discussion。
- 需要多剪切率 cluster 统计，不能只用 `gamma_dot*tau0=3`。

### Fig. 4: PHM and cage-jump correlations

- Fig. 4(a)：必须是 Fig. 2 同一 BD cluster。
- PHM：隔离 cluster，固定 cluster 外粒子，以 `t0` inherent structure 为参考。实现需 Ref.57 Richard et al. PRL。
- Fig. 4(b)：orientation-averaged `g_cj(r)`。
- Fig. 4(c)：anisotropic factor `alpha(r)`，`xi_alpha` 是 `alpha(r)` 突然变陡/截断位置。
- Fig. 4(d)：比较 `xi_alpha` 和 `xi_c`，并做 power-law fit。

## 6. Cluster demarcation

Zeng Appendix B 的 BD cluster boundary 分两步：

1. Coarse sieve：
   - BD 直接把盒子分成边长 `d` 的 blocks，统计 cage-jump density。
   - `d=L/10` for `gamma_dot*tau0=1,3,10`。
   - `d=L/20` for `gamma_dot*tau0=20`。
   - threshold `rho_th` 由迭代平均得到，tolerance `C=10%`。
2. Undersize sieve：
   - 对 coarse cluster 再分小盒，3D 系统 `d'=L/30`。
   - 每个 grid point 画半径 `sqrt(2)d'/2` 的 sphere，比较局域 cage-jump density 和 `rho_th`。
   - cluster 下限体积 `Vc,ll=C'Vstz`，`C'=2`。

重要限制：

- `L/30` 是 undersize sieve 的细网格，不是 `gamma_dot*tau0=3` 的 coarse sieve。
- 若只做 coarse sieve，需要标为不完整复现。

## 7. 缺失材料清单

必须补充或至少单独登记：

- Ref.61 Heyes-Melrose 1993：hard-core potential-free algorithm 细节。
- Ref.57 Richard et al. PRL：PHM/STZ 定义和计算细节。
- Candelier PRL 105 supplemental：cage-jump procedure 细节校验。
- 若要做到完全一致，最好向作者索取 BD integrator timestep、run length、steady-state discard、sample count 和 Fig. 1(b) cluster/window metadata。

## 8. 验收标准

- 所有 BD 参数在 `configs/BD/bd_parameter_audit.csv` 中有来源标签。
- `phi=42.5%` 可复核 `D_LS≈0.06D0`。
- `phi=45%, gamma_dot*tau0=3` 对应 `Pe=0.75`。
- Fig. 1(a) 高剪切 BD 黏度为正并 shear-thinning。
- Fig. 1(b)/Fig. 2/Fig. 4 使用同一个 BD cluster。
- 所有论文未给出的参数必须在输出或记录中标为 `缺失` 或 `推断`。
- 若缺少 PHM、hard-core algorithm、`t_chi` 窗口或 per-particle Yukawa stress，不生成伪严格结果。
