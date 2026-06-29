#!/bin/bash
#SBATCH -J zeng_md2d
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -t 72:00:00
#SBATCH -o slurm_md2d_%A_%a.out
#SBATCH -e slurm_md2d_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/zeng_reproduction/lammps/md2d"

if ! command -v module >/dev/null 2>&1; then
  source /apps/management/modules/v5.2.0/init/bash 2>/dev/null || true
fi
module load compilers/intel/oneapi-2023/config
module load soft/lammps/lammps-22Dec2022

export OMP_NUM_THREADS=1
NP="${SLURM_NTASKS:-56}"
LAMMPS_BIN="${LAMMPS_BIN:-lmp_oneapi}"
MPI_BIN="${MPI_BIN:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${MODE:-shear}"
RUN_GROUP="${RUN_GROUP:-md2d_prod_20260629}"
EQUIL_RUN_GROUP="${EQUIL_RUN_GROUP:-${RUN_GROUP}}"
RAW_BASE="${REPO_ROOT}/zeng_reproduction/data/MD/2D/raw"
META_BASE="${REPO_ROOT}/zeng_reproduction/data/MD/2D/metadata"
GROUP_RAW="${RAW_BASE}/${RUN_GROUP}"
GROUP_META="${META_BASE}/${RUN_GROUP}"
EQUIL_RAW="${RAW_BASE}/${EQUIL_RUN_GROUP}/equil"
EQUIL_META="${META_BASE}/${EQUIL_RUN_GROUP}/equil"
mkdir -p "${GROUP_RAW}" "${GROUP_META}" "${EQUIL_RAW}" "${EQUIL_META}"

TABLE_FILE="${SCRIPT_DIR}/softcore_r12.table"
if [ ! -s "${TABLE_FILE}" ]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/make_softcore_table.py" --output "${TABLE_FILE}"
fi

DT="${DT:-0.005}"
TEMP="${TEMP:-0.526}"
TEMP_INIT="${TEMP_INIT:-1.0}"
TDAMP="${TDAMP:-0.5}"
THERMO_EVERY="${THERMO_EVERY:-1000}"
VISC_EVERY="${VISC_EVERY:-1000}"
SEED="${SEED:-20260701}"
EQUIL_TAG="md2d_${EQUIL_RUN_GROUP}_equil"
EQUIL_RESTART="${EQUIL_RAW}/${EQUIL_TAG}.restart"

if [ "${MODE}" = "equil" ]; then
  INIT_DATA="${EQUIL_META}/md2d_initial.data"
  INIT_MANIFEST="${EQUIL_META}/md2d_initial.json"
  "${PYTHON_BIN}" "${REPO_ROOT}/zeng_reproduction/scripts/make_md2d_initial_data.py" \
    --output "${INIT_DATA}" \
    --manifest "${INIT_MANIFEST}" \
    --nb "${NB:-10000}" \
    --ns "${NS:-10000}" \
    --rho "${RHO:-0.8}" \
    --seed "${SEED}"

  "${MPI_BIN}" -np "${NP}" "${LAMMPS_BIN}" \
    -in "${SCRIPT_DIR}/in.md2d_equilibrate" \
    -var DATA_FILE "${INIT_DATA}" \
    -var TABLE_FILE "${TABLE_FILE}" \
    -var OUT_DIR "${EQUIL_RAW}" \
    -var RUN_TAG "${EQUIL_TAG}" \
    -var DT "${DT}" \
    -var TEMP_INIT "${TEMP_INIT}" \
    -var TEMP "${TEMP}" \
    -var TDAMP "${TDAMP}" \
    -var SEED "${SEED}" \
    -var HOT_STEPS "${HOT_STEPS:-100000}" \
    -var COOL_STEPS "${COOL_STEPS:-200000}" \
    -var EQ_STEPS "${EQ_STEPS:-500000}" \
    -var THERMO_EVERY "${THERMO_EVERY}" \
    -log "${EQUIL_RAW}/${EQUIL_TAG}.log"

  GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${EQUIL_META}/manifest.json" <<EOF
{
  "run_group": "${RUN_GROUP}",
  "equil_run_group": "${EQUIL_RUN_GROUP}",
  "mode": "equil",
  "git_commit": "${GIT_COMMIT}",
  "np": ${NP},
  "nb": ${NB:-10000},
  "ns": ${NS:-10000},
  "rho": ${RHO:-0.8},
  "dt": ${DT},
  "temperature": ${TEMP},
  "temp_init": ${TEMP_INIT},
  "tdamp": ${TDAMP},
  "hot_steps": ${HOT_STEPS:-100000},
  "cool_steps": ${COOL_STEPS:-200000},
  "eq_steps": ${EQ_STEPS:-500000},
  "restart": "${EQUIL_RESTART}"
}
EOF
  grep -H "Loop time of" "${EQUIL_RAW}/${EQUIL_TAG}.log" || true
  test -s "${EQUIL_RESTART}"
  echo "MD2D_EQUIL_JOB_FINISHED ${RUN_GROUP}"
  exit 0
fi

if [ "${MODE}" != "shear" ]; then
  echo "ERROR: MODE must be 'equil' or 'shear'." >&2
  exit 2
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: MODE=shear requires --array=1-4." >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  1) SHEAR_RATE="0.0005"; DUMP_DT="5.0" ;;
  2) SHEAR_RATE="0.001";  DUMP_DT="0.5" ;;
  3) SHEAR_RATE="0.005";  DUMP_DT="1.0" ;;
  4) SHEAR_RATE="0.01";   DUMP_DT="1.0" ;;
  *)
    echo "ERROR: unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; use 1-4." >&2
    exit 2
    ;;
esac

if [ ! -s "${EQUIL_RESTART}" ]; then
  echo "ERROR: Missing equilibration restart: ${EQUIL_RESTART}" >&2
  echo "Submit first: sbatch --export=ALL,MODE=equil,RUN_GROUP=${RUN_GROUP} ${BASH_SOURCE[0]}" >&2
  exit 2
fi

DUMP_EVERY="$("${PYTHON_BIN}" -c 'import sys; print(int(round(float(sys.argv[1]) / float(sys.argv[2]))))' "${DUMP_DT}" "${DT}")"
SR_TAG="$("${PYTHON_BIN}" -c 'import sys; print(sys.argv[1].replace(".", "p").replace("-", "m"))' "${SHEAR_RATE}")"
RUN_TAG="md2d_${RUN_GROUP}_gdot${SR_TAG}"
RUN_RAW="${GROUP_RAW}/gdot${SR_TAG}"
RUN_META="${GROUP_META}/gdot${SR_TAG}"
mkdir -p "${RUN_RAW}" "${RUN_META}"

"${MPI_BIN}" -np "${NP}" "${LAMMPS_BIN}" \
  -in "${SCRIPT_DIR}/in.md2d_shear" \
  -var RESTART_FILE "${EQUIL_RESTART}" \
  -var TABLE_FILE "${TABLE_FILE}" \
  -var OUT_DIR "${RUN_RAW}" \
  -var RUN_TAG "${RUN_TAG}" \
  -var SHEAR_RATE "${SHEAR_RATE}" \
  -var DT "${DT}" \
  -var TEMP "${TEMP}" \
  -var TDAMP "${TDAMP}" \
  -var PRE_STRAIN "${PRE_STRAIN:-20}" \
  -var PROD_STRAIN "${PROD_STRAIN:-50}" \
  -var DUMP_EVERY "${DUMP_EVERY}" \
  -var THERMO_EVERY "${THERMO_EVERY}" \
  -var VISC_EVERY "${VISC_EVERY}" \
  -log "${RUN_RAW}/${RUN_TAG}.log"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "${RUN_META}/manifest.json" <<EOF
{
  "run_group": "${RUN_GROUP}",
  "equil_run_group": "${EQUIL_RUN_GROUP}",
  "mode": "shear",
  "git_commit": "${GIT_COMMIT}",
  "array_task_id": ${SLURM_ARRAY_TASK_ID},
  "np": ${NP},
  "shear_rate": ${SHEAR_RATE},
  "dt": ${DT},
  "dump_dt": ${DUMP_DT},
  "dump_every": ${DUMP_EVERY},
  "dump_columns": "id type x y ix iy xu yu",
  "dump_modify": "sort id",
  "temperature": ${TEMP},
  "tdamp": ${TDAMP},
  "pre_strain": ${PRE_STRAIN:-20},
  "prod_strain": ${PROD_STRAIN:-50},
  "equil_restart": "${EQUIL_RESTART}",
  "raw_dir": "${RUN_RAW}"
}
EOF

grep -H "Loop time of" "${RUN_RAW}/${RUN_TAG}.log" || true
test -s "${RUN_RAW}/${RUN_TAG}.lammpstrj.gz"
test -s "${RUN_RAW}/${RUN_TAG}.thermo.dat"
test -s "${RUN_RAW}/${RUN_TAG}.visc_block.dat"

echo "MD2D_SHEAR_JOB_FINISHED ${RUN_GROUP} ${SHEAR_RATE}"
