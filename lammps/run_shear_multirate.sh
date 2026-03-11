#!/bin/bash
# =============================================================================
# run_shear_multirate.sh
# 批量运行多剪切率 SLLOD 剪切模拟
# 参考: Zeng et al., J. Chem. Phys. 163, 084512 (2025)  Table I / Fig. 5(a)
#
# 使用方法:
#   bash run_shear_multirate.sh [可选: lmp 可执行文件路径]
#
# 依赖文件 (同一目录):
#   restart.equil          - 平衡态 restart (由 in.equilibrate 生成)
#   in.shear_template      - LAMMPS 模板 (接受 -var SHEAR_RATE / DT / SEED)
#
# 输出文件 (每个剪切率一套):
#   dump.shear_<rate>.lammpstrj
#   dump.fine_<rate>.lammpstrj
#   thermo.shear_<rate>.dat
#   visc_blockave.shear_<rate>.dat
#   log.shear_<rate>
# =============================================================================

set -euo pipefail

# ── 可执行文件 ──────────────────────────────────────────────────────────────
LMP="${1:-lmp}"          # 默认调用 lmp；MPI 版可传入 'mpirun -np 8 lmp'
TEMPLATE="in.shear_template"
RESTART="restart.equil"
SEED=12345

# ── 检查依赖 ─────────────────────────────────────────────────────────────────
if [[ ! -f "$RESTART" ]]; then
    echo "[ERROR] 找不到 $RESTART，请先运行 in.equilibrate"
    exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "[ERROR] 找不到 $TEMPLATE"
    exit 1
fi

# ── 剪切率列表 (与论文 Fig. 5a 对应) ──────────────────────────────────────
#   γ̇  /τ₀   DT /τ₀      说明
#   ─────────────────────────────────────────────────────────────────────────
#   0.003    0.003      最慢，τ_α 最长，Wi ≈ 1.5-3 (接近 onset)
#   0.005    0.003      中慢
#   0.015    0.001      中速 (已有数据)
#   0.030    0.001      中快
#   0.060    0.001      最快
# ─────────────────────────────────────────────────────────────────────────
RATES=(0.003  0.005  0.015  0.030  0.060)
DTS=(  0.003  0.003  0.001  0.001  0.001)

echo "========================================================"
echo "  多剪切率 SLLOD 批量模拟"
echo "  LAMMPS 命令: $LMP"
echo "  剪切率: ${RATES[*]}"
echo "========================================================"

for i in "${!RATES[@]}"; do
    RATE="${RATES[$i]}"
    DT="${DTS[$i]}"
    LOGFILE="log.shear_${RATE}"

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  开始: γ̇ = ${RATE}  DT = ${DT}"
    echo "  预期输出: dump.shear_${RATE}.lammpstrj"
    echo "  日志: ${LOGFILE}"
    echo "────────────────────────────────────────────────────────"

    # 检查是否已存在（跳过已完成的率）
    if [[ -f "dump.shear_${RATE}.lammpstrj" ]]; then
        echo "  [跳过] dump.shear_${RATE}.lammpstrj 已存在"
        continue
    fi

    # 运行 LAMMPS
    $LMP -in "$TEMPLATE" \
         -var SHEAR_RATE "$RATE" \
         -var DT         "$DT" \
         -var SEED        "$SEED" \
         -log             "$LOGFILE" \
         -screen          none

    # 检查输出
    if [[ ! -f "dump.shear_${RATE}.lammpstrj" ]]; then
        echo "[ERROR] γ̇=${RATE} 运行失败，检查 ${LOGFILE}"
        exit 1
    fi

    # 统计帧数
    N_COARSE=$(grep -c "^ITEM: TIMESTEP" "dump.shear_${RATE}.lammpstrj" || true)
    N_FINE=$(grep -c "^ITEM: TIMESTEP" "dump.fine_${RATE}.lammpstrj" || true)
    echo "  ✓ 完成: coarse ${N_COARSE} 帧  fine ${N_FINE} 帧"
done

echo ""
echo "========================================================"
echo "  所有剪切率运行完毕"
echo "========================================================"
echo ""
echo "下一步：运行 F_s(q,t) 分析"
echo ""
echo "python compute_fsqt.py \\"
echo "    --dump  dump.shear_0.003.lammpstrj \\"
echo "            dump.shear_0.005.lammpstrj \\"
echo "            dump.shear_0.015.lammpstrj \\"
echo "            dump.shear_0.030.lammpstrj \\"
echo "            dump.shear_0.060.lammpstrj \\"
echo "    --dump_fine \\"
echo "            dump.fine_0.003.lammpstrj \\"
echo "            dump.fine_0.005.lammpstrj \\"
echo "            dump.fine_0.015.lammpstrj \\"
echo "            dump.fine_0.030.lammpstrj \\"
echo "            dump.fine_0.060.lammpstrj \\"
echo "    --rate 0.003 0.005 0.015 0.030 0.060 \\"
echo "    --n_skip 5 --multiplot"
