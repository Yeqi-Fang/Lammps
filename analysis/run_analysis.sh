#!/bin/bash
# ==========================================================
# run_analysis.sh
# 一键完整分析脚本
# 针对已完成的模拟：γ̇=0.015, dt=0.005
#
# 用法：
#   bash run_analysis.sh [data_dir] [shear_rate] [dt]
#   bash run_analysis.sh .       0.015      0.005
#
# 产出（figures/ 目录）：
#   - pxy_timeseries_0p015.png     Pxy 时间序列 + 稳态检验
#   - msd_nonaffine_shearrate_0p015.png  非仿射 MSD
#   - cage_jumps_analysis_0p015.png      跳跃长度分布
#   - cage_jump_trajectories_0p015.png   典型轨迹
#   - cage_jump_spatial_0p015.png        空间分布
#   - fig1_viscosity.png  fig2_msd.png   汇总图
#   - verification_table.txt             验证表格
# ==========================================================

DATA_DIR="${1:-.}"
SHEAR_RATE="${2:-0.015}"
DT="${3:-0.005}"

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="${SCRIPT_DIR}"

# 文件名
SR_STR="${SHEAR_RATE//./_}"
THERMO_FILE="${DATA_DIR}/thermo.shear_${SHEAR_RATE}.dat"
DUMP_FILE="${DATA_DIR}/dump.shear_${SHEAR_RATE}.lammpstrj"
OUTPUT_DIR="${DATA_DIR}/figures"
MSD_NPZ="${DATA_DIR}/msd_data_${SR_STR}.npz"
JUMPS_NPZ="${DATA_DIR}/cage_jumps_shearrate_$(echo $SHEAR_RATE | tr '.' 'p').npz"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Zeng et al. 2025 分析流程"
echo "  数据目录: ${DATA_DIR}"
echo "  剪切率:   ${SHEAR_RATE}"
echo "  时间步:   ${DT}"
echo "=================================================="

# ─────────────────────────────
# 步骤 1：黏度分析
# ─────────────────────────────
echo ""
echo "[步骤 1] 计算稳态黏度..."
if [ -f "${THERMO_FILE}" ]; then
    python3 "${ANALYSIS_DIR}/compute_viscosity.py" \
        --thermo "${THERMO_FILE}" \
        --rates  "${SHEAR_RATE}" \
        --output "${OUTPUT_DIR}"
    echo "  ✓ 完成"
else
    echo "  ⚠ 找不到 ${THERMO_FILE}，跳过黏度分析"
    echo "    请确认 LAMMPS 已生成 thermo.shear_${SHEAR_RATE}.dat"
fi

# ─────────────────────────────
# 步骤 2：非仿射 MSD
# ─────────────────────────────
echo ""
echo "[步骤 2] 计算非仿射 MSD..."
if [ -f "${DUMP_FILE}" ]; then
    # 计算帧间时间（dump_every * dt）
    # dump_every = round(0.5 / (shear_rate * dt))
    python3 - <<PYEOF
import subprocess, sys, os
shear_rate = ${SHEAR_RATE}
dt = ${DT}
dump_every = round(0.5 / (shear_rate * dt))
dt_frame = dump_every * dt
print(f"  dump_every={dump_every} 步, dt_frame={dt_frame:.4f} τ₀")

cmd = [
    "python3", "${ANALYSIS_DIR}/compute_msd_nonaffine.py",
    "--dump",     "${DUMP_FILE}",
    "--rate",     str(shear_rate),
    "--dt",       str(dt),
    "--dt_frame", str(dt_frame),
    "--output",   "${OUTPUT_DIR}",
    "--save_npz", "${MSD_NPZ}",
]
ret = subprocess.run(cmd)
sys.exit(ret.returncode)
PYEOF
    echo "  ✓ 完成"
else
    echo "  ⚠ 找不到 ${DUMP_FILE}，跳过 MSD 分析"
    echo "    请确认 LAMMPS 已生成 dump.shear_${SHEAR_RATE}.lammpstrj"
fi

# ─────────────────────────────
# 步骤 3：笼跳跃检测
# ─────────────────────────────
echo ""
echo "[步骤 3] 检测笼跳跃事件..."
if [ -f "${MSD_NPZ}" ]; then
    python3 "${ANALYSIS_DIR}/cage_jump_detection.py" \
        --npz    "${MSD_NPZ}" \
        --rate   "${SHEAR_RATE}" \
        --output "${OUTPUT_DIR}" \
        --max_particles 2000
    echo "  ✓ 完成"
else
    echo "  ⚠ 找不到 ${MSD_NPZ}，跳过笼跳跃分析（需先完成步骤 2）"
fi

# ─────────────────────────────
# 步骤 4：汇总图
# ─────────────────────────────
echo ""
echo "[步骤 4] 生成汇总图..."
python3 "${ANALYSIS_DIR}/plot_summary.py" \
    --data   "${DATA_DIR}" \
    --output "${OUTPUT_DIR}"
echo "  ✓ 完成"

# ─────────────────────────────
echo ""
echo "=================================================="
echo "分析完成！图像保存于: ${OUTPUT_DIR}/"
echo ""
echo "主要输出文件："
ls "${OUTPUT_DIR}"/*.png 2>/dev/null | head -20
echo "=================================================="
