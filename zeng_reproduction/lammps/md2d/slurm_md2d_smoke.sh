#!/bin/bash
#SBATCH -J zeng_md2d_smoke
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o slurm_md2d_smoke_%j.out
#SBATCH -e slurm_md2d_smoke_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/zeng_reproduction/lammps/md2d"

if ! command -v module >/dev/null 2>&1; then
  # The server exposes Environment Modules here in batch shells.
  source /apps/management/modules/v5.2.0/init/bash 2>/dev/null || true
fi
module load compilers/intel/oneapi-2023/config
module load soft/lammps/lammps-22Dec2022

export OMP_NUM_THREADS=1
NP="${SLURM_NTASKS:-4}"
LAMMPS_BIN="${LAMMPS_BIN:-lmp_oneapi}"
MPI_BIN="${MPI_BIN:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_GROUP="smoke_${JOB_ID}"
RAW_BASE="${REPO_ROOT}/zeng_reproduction/data/MD/2D/raw"
META_BASE="${REPO_ROOT}/zeng_reproduction/data/MD/2D/metadata"
RUN_ROOT="${RAW_BASE}/${RUN_GROUP}"
META_ROOT="${META_BASE}/${RUN_GROUP}"
mkdir -p "${RUN_ROOT}" "${META_ROOT}"

TABLE_FILE="${SCRIPT_DIR}/softcore_r12.table"
if [ ! -s "${TABLE_FILE}" ]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/make_softcore_table.py" --output "${TABLE_FILE}"
fi

INIT_DATA="${META_ROOT}/md2d_initial_smoke.data"
INIT_MANIFEST="${META_ROOT}/md2d_initial_smoke.json"
"${PYTHON_BIN}" "${REPO_ROOT}/zeng_reproduction/scripts/make_md2d_initial_data.py" \
  --output "${INIT_DATA}" \
  --manifest "${INIT_MANIFEST}" \
  --nb "${SMOKE_NB:-1000}" \
  --ns "${SMOKE_NS:-1000}" \
  --rho "${RHO:-0.8}" \
  --seed "${SEED:-20260701}"

DT="${DT:-0.005}"
TEMP="${TEMP:-0.526}"
TDAMP="${TDAMP:-0.5}"
THERMO_EVERY="${THERMO_EVERY:-100}"

EQUIL_TAG="md2d_${RUN_GROUP}_equil"
"${MPI_BIN}" -np "${NP}" "${LAMMPS_BIN}" \
  -in "${SCRIPT_DIR}/in.md2d_equilibrate" \
  -var DATA_FILE "${INIT_DATA}" \
  -var TABLE_FILE "${TABLE_FILE}" \
  -var OUT_DIR "${RUN_ROOT}" \
  -var RUN_TAG "${EQUIL_TAG}" \
  -var DT "${DT}" \
  -var TEMP_INIT "${TEMP_INIT:-1.0}" \
  -var TEMP "${TEMP}" \
  -var TDAMP "${TDAMP}" \
  -var SEED "${SEED:-20260701}" \
  -var HOT_STEPS "${SMOKE_HOT_STEPS:-1000}" \
  -var COOL_STEPS "${SMOKE_COOL_STEPS:-1000}" \
  -var EQ_STEPS "${SMOKE_EQ_STEPS:-2000}" \
  -var THERMO_EVERY "${THERMO_EVERY}" \
  -log "${RUN_ROOT}/${EQUIL_TAG}.log"

SHEAR_TAG="md2d_${RUN_GROUP}_gdot0p001"
"${MPI_BIN}" -np "${NP}" "${LAMMPS_BIN}" \
  -in "${SCRIPT_DIR}/in.md2d_shear" \
  -var RESTART_FILE "${RUN_ROOT}/${EQUIL_TAG}.restart" \
  -var TABLE_FILE "${TABLE_FILE}" \
  -var OUT_DIR "${RUN_ROOT}" \
  -var RUN_TAG "${SHEAR_TAG}" \
  -var SHEAR_RATE "${SMOKE_SHEAR_RATE:-0.001}" \
  -var DT "${DT}" \
  -var TEMP "${TEMP}" \
  -var TDAMP "${TDAMP}" \
  -var PRE_STRAIN "${SMOKE_PRE_STRAIN:-0.02}" \
  -var PROD_STRAIN "${SMOKE_PROD_STRAIN:-0.05}" \
  -var DUMP_EVERY "${SMOKE_DUMP_EVERY:-10}" \
  -var THERMO_EVERY "${THERMO_EVERY}" \
  -var VISC_EVERY "${SMOKE_VISC_EVERY:-100}" \
  -log "${RUN_ROOT}/${SHEAR_TAG}.log"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "${META_ROOT}/manifest.json" <<EOF
{
  "run_group": "${RUN_GROUP}",
  "mode": "smoke",
  "git_commit": "${GIT_COMMIT}",
  "np": ${NP},
  "dt": ${DT},
  "temperature": ${TEMP},
  "tdamp": ${TDAMP},
  "equil_tag": "${EQUIL_TAG}",
  "shear_tag": "${SHEAR_TAG}",
  "raw_dir": "${RUN_ROOT}",
  "metadata_dir": "${META_ROOT}"
}
EOF

grep -H "Loop time of" "${RUN_ROOT}/${EQUIL_TAG}.log" "${RUN_ROOT}/${SHEAR_TAG}.log" || true
test -s "${RUN_ROOT}/${SHEAR_TAG}.lammpstrj.gz"
test -s "${RUN_ROOT}/${SHEAR_TAG}.thermo.dat"
test -s "${RUN_ROOT}/${SHEAR_TAG}.visc_block.dat"

echo "MD2D_SMOKE_FINISHED ${RUN_GROUP}"
