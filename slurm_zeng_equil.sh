#!/bin/bash -l
#SBATCH -J zeng_ka_equil
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -o zeng_equil.%j.out
#SBATCH -e zeng_equil.%j.err
#SBATCH --no-requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${ROOT}/lammps"
NP="${SLURM_NTASKS:-56}"

cd "${WORKDIR}"

module load compilers/intel/oneapi-2023/config
module load soft/lammps/lammps-22Dec2022

export OMP_NUM_THREADS=1
export I_MPI_PIN=0

echo "=========================================="
echo "Zeng KA 3D equilibration"
echo "root      = ${ROOT}"
echo "workdir   = ${WORKDIR}"
echo "hostname  = $(hostname)"
echo "job id    = ${SLURM_JOB_ID:-NA}"
echo "mpi ranks = ${NP}"
echo "date      = $(date)"
echo "=========================================="

if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
    rm -f restart.equil equil.data log.equilibrate msd_equilibration.dat
fi

mpirun -np "${NP}" lmp_oneapi -in in.equilibrate -log log.equilibrate -screen none

test -s restart.equil
test -s equil.data

if grep -qi "Dangerous builds" log.equilibrate; then
    grep -i "Dangerous builds" log.equilibrate || true
fi

echo "Equilibration finished: $(date)"
