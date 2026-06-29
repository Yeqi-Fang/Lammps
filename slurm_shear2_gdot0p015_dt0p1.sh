#!/bin/bash -l
#SBATCH -J shear2_0p015_dt0p1
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -o shear2_0p015_dt0p1.%j.out
#SBATCH -e shear2_0p015_dt0p1.%j.err
#SBATCH --no-requeue

set -euo pipefail

gamma="0.015"
dt_shear="0.001"
dump_every="100"

root="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_dir="${root}/shear2_runs/gdot_0p015_dt0p1"

mkdir -p "${run_dir}"

cd "${run_dir}"

module load compilers/intel/oneapi-2023/config
module load soft/lammps/lammps-22Dec2022

export OMP_NUM_THREADS=1
export I_MPI_PIN=0

echo "gamma=${gamma}"
echo "dt_shear=${dt_shear}"
echo "dump_every=${dump_every}"
echo "dump interval = 0.1 tau0"
echo "run_dir=${run_dir}"
echo "start=$(date)"

mpirun -np "${SLURM_NTASKS:-56}" lmp_oneapi \
  -in "${root}/shear2.lmp" \
  -var gamma "${gamma}" \
  -var dt_shear "${dt_shear}" \
  -var dump_every "${dump_every}" \
  -log "log.shear2_0p015_dt0p1" \
  -screen none

echo "done=$(date)"
