#!/bin/bash -l
#SBATCH -J zeng_md3d_0015_hi
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -o zeng_md3d_0015_hi.%j.out
#SBATCH -e zeng_md3d_0015_hi.%j.err
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
echo "Zeng KA 3D MD high-resolution run"
echo "target     = Fig.5(b) / Fig.6 left main trajectory"
echo "root       = ${ROOT}"
echo "workdir    = ${WORKDIR}"
echo "hostname   = $(hostname)"
echo "job id     = ${SLURM_JOB_ID:-NA}"
echo "mpi ranks  = ${NP}"
echo "gdot       = 0.015"
echo "dt         = 0.001"
echo "Delta_t    = 0.1 tau0"
echo "date       = $(date)"
echo "=========================================="

if [[ ! -s restart.equil ]]; then
    echo "Missing ${WORKDIR}/restart.equil; run slurm_zeng_equil.sh first." >&2
    exit 3
fi

if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
    rm -f \
        dump.md3d_gdot0p015_dt0p1_stress.lammpstrj.gz \
        thermo.md3d_gdot0p015_dt0p1.dat \
        visc_blockave.md3d_gdot0p015_dt0p1.dat \
        restart.md3d_gdot0p015_dt0p1.final \
        log.md3d_gdot0p015_dt0p1
fi

mpirun -np "${NP}" lmp_oneapi \
    -in in.shear_md3d_gdot0p015_dt0p1_stress \
    -log log.md3d_gdot0p015_dt0p1 \
    -screen none

test -s thermo.md3d_gdot0p015_dt0p1.dat
test -s visc_blockave.md3d_gdot0p015_dt0p1.dat
test -s dump.md3d_gdot0p015_dt0p1_stress.lammpstrj.gz

if grep -qi "Dangerous builds" log.md3d_gdot0p015_dt0p1; then
    grep -i "Dangerous builds" log.md3d_gdot0p015_dt0p1 || true
fi

echo "High-resolution run finished: date=$(date)"
