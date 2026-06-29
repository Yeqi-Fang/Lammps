#!/bin/bash -l
#SBATCH -J zeng_ka_shear
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH --array=1-4
#SBATCH -o zeng_shear.%A_%a.out
#SBATCH -e zeng_shear.%A_%a.err
#SBATCH --no-requeue

set -euo pipefail

RATES=(0.005 0.015 0.030 0.060)
DTS=(0.003 0.001 0.001 0.001)

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
IDX=$((TASK_ID - 1))
if (( IDX < 0 || IDX >= ${#RATES[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 1-${#RATES[@]}" >&2
    exit 2
fi

SHEAR_RATE="${RATES[$IDX]}"
DT="${DTS[$IDX]}"
BASE_SEED="${BASE_SEED:-24681357}"
SEED=$((BASE_SEED + TASK_ID * 1009))

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${ROOT}/lammps"
NP="${SLURM_NTASKS:-56}"

cd "${WORKDIR}"

module load compilers/intel/oneapi-2023/config
module load soft/lammps/lammps-22Dec2022

export OMP_NUM_THREADS=1
export I_MPI_PIN=0

echo "=========================================="
echo "Zeng KA 3D shear high-rate run"
echo "root       = ${ROOT}"
echo "workdir    = ${WORKDIR}"
echo "hostname   = $(hostname)"
echo "job id     = ${SLURM_JOB_ID:-NA}"
echo "array task = ${TASK_ID}"
echo "mpi ranks  = ${NP}"
echo "gdot       = ${SHEAR_RATE}"
echo "dt         = ${DT}"
echo "seed       = ${SEED}"
echo "date       = $(date)"
echo "=========================================="

if [[ ! -s restart.equil ]]; then
    echo "Missing ${WORKDIR}/restart.equil; run slurm_zeng_equil.sh first." >&2
    exit 3
fi

if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
    rm -f \
        "dump.fine_${SHEAR_RATE}.lammpstrj" \
        "dump.shear_${SHEAR_RATE}.lammpstrj" \
        "thermo.shear_${SHEAR_RATE}.dat" \
        "visc_blockave.shear_${SHEAR_RATE}.dat" \
        "log.shear_${SHEAR_RATE}"
fi

mpirun -np "${NP}" lmp_oneapi \
    -in in.shear_template \
    -var SHEAR_RATE "${SHEAR_RATE}" \
    -var DT "${DT}" \
    -var SEED "${SEED}" \
    -log "log.shear_${SHEAR_RATE}" \
    -screen none

test -s "thermo.shear_${SHEAR_RATE}.dat"
test -s "dump.shear_${SHEAR_RATE}.lammpstrj"
test -s "visc_blockave.shear_${SHEAR_RATE}.dat"

if grep -qi "Dangerous builds" "log.shear_${SHEAR_RATE}"; then
    grep -i "Dangerous builds" "log.shear_${SHEAR_RATE}" || true
fi

echo "Shear run finished: gdot=${SHEAR_RATE}, date=$(date)"
