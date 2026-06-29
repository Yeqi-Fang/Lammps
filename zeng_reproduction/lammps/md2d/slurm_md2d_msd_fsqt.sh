#!/bin/bash
#SBATCH -J zeng_msdfs
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o slurm_md2d_msd_fsqt_%j.out
#SBATCH -e slurm_md2d_msd_fsqt_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_GROUP="${RUN_GROUP:-md2d_prod_slim_20260629_1412}"

"${PYTHON_BIN}" zeng_reproduction/scripts/md2d_msd_fsqt.py \
  --repo-root "${REPO_ROOT}" \
  --run-group "${RUN_GROUP}" \
  --sample-dt "${SAMPLE_DT:-5.0}" \
  --max-origins "${MAX_ORIGINS:-120}" \
  --max-log-lags "${MAX_LOG_LAGS:-150}" \
  --sq-frames "${SQ_FRAMES:-64}"
