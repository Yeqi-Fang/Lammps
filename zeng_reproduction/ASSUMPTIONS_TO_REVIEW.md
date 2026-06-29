# BD Assumptions And Missing Inputs

本文件只记录 Zeng BD 复现中的假设和缺失项。MD 相关内容不在本阶段范围内。

## 必须补充的文献或信息

1. Ref.61 Heyes-Melrose 1993：Zeng 用它作为 hard-core potential-free algorithm 的来源之一。缺少它时，不能声称 hard-core overlap correction 的迭代容差、pair 顺序、残余 overlap 处理完全复现原文。
2. Ref.57 Richard et al. PRL：Fig. 4(a) 的 PHM/STZ 计算依赖该方法。缺少它时，不生成替代 PHM 图。
3. Candelier PRL 105 supplemental：Zeng Appendix B 给了 cage-jump 主公式，但 Ref.37 主文说明算法细节在补充材料中。
4. 作者 protocol 或原始 metadata：BD integrator timestep、平衡长度、生产长度、steady-state 丢弃规则、随机种子/样本数、Fig. 1(b) 的精确 `t_chi` 窗口和 cluster ID。

## 明确缺失，不能写成论文事实

1. BD integrator timestep。Zeng Appendix A 的 `Delta t` 是分析时间间隔，不等同于 BD 积分步长。
2. BD 平衡步数、预剪切长度、生产长度和 steady-state 判断标准。
3. 独立 run 数、随机种子和误差条统计方式。
4. Fig. 1(b) 红色 cluster 的原始 ID、精确边界数据和 `t_chi` 窗口起始时间。
5. Fig. 2 的 `t'`、`t0`、`t1`、`t2`、`t3` 数值。
6. Fig. 4(a) PHM 的完整实现细节。

## 当前采用的保守默认

1. BD `sigma_c` 和 viscosity 优先使用 Yukawa differentiable force/stress。hard-core collision stress 不混入，除非 Ref.61 或作者 protocol 明确要求。
2. cage jumps、`chi4`、cluster demarcation 默认使用所有粒子。若后续只筛 big particles，必须标为额外假设，因为 Zeng Appendix B 未说明只用 big particles。
3. `gamma_dot*tau0=3` 的 BD coarse sieve 使用 `d=L/10`，不是 `L/30`。`L/30` 属于 undersize sieve。
4. PCCP ESI 的 `r^-36` soft-core BD 是旧 `N=3000, phi=0.4` 辅助体系，只能作为背景参考，不能替代 Zeng 2025 BD 主体系。
5. 若没有 PHM 输入、per-particle Yukawa stress/force、`t_chi` 窗口或 cluster metadata，后续分析必须停止并报告缺失，不得输出“严格复现”结果。

## 需要特别核对的实现点

1. 现有 BD 配置或脚本若使用 `type_filter=1`，应视为 big-particle-only 假设；严格复现前需要改为 all-particle 或找到文献依据。
2. 现有 BD cluster 若直接用 `30^3` 网格作为 coarse sieve，不符合 Zeng Appendix B 对 `gamma_dot*tau0=3` 的 `L/10` coarse block 要求。
3. `chi4(t)` 需要使用 Zeng Appendix B 的 shear-adjusted overlap distance，并保留 `beta*V/N^2` 的定义。若只用未归一化 overlap variance，只能标为 proxy。
4. `t'`、`t0`、`t1`、`t2`、`t3` 不能由脚本静默启发式决定；若使用数据驱动规则，metadata 必须明确写出规则。
