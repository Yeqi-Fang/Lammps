#!/bin/bash -l
#SBATCH -J shear2_highrate
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH --array=1-4%4
#SBATCH -o shear2_highrate.%A_%a.out
#SBATCH -e shear2_highrate.%A_%a.err
#SBATCH --no-requeue

set -euo pipefail

rates=(0.005 0.015 0.030 0.060)
dts=(0.003 0.001 0.001 0.001)
dump_evs=(1333 1333 667 333)

i=$((SLURM_ARRAY_TASK_ID - 1))
gamma="${rates[$i]}"
dt_shear="${dts[$i]}"
dump_every="${dump_evs[$i]}"
label="${gamma/./p}"

root="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_dir="${root}/shear2_runs/gdot_${label}"

mkdir -p "${run_dir}"

cd "${run_dir}"

module load compilers/intel/oneapi-2023/config
module load soft/lammps/lammps-22Dec2022

export OMP_NUM_THREADS=1
export I_MPI_PIN=0

echo "gamma=${gamma}"
echo "dt_shear=${dt_shear}"
echo "dump_every=${dump_every}"
echo "run_dir=${run_dir}"
echo "start=$(date)"

mpirun -np "${SLURM_NTASKS:-56}" lmp_oneapi \
  -in "${root}/shear2.lmp" \
  -var gamma "${gamma}" \
  -var dt_shear "${dt_shear}" \
  -var dump_every "${dump_every}" \
  -log "log.shear2_${label}" \
  -screen none

echo "done=$(date)"
