#!/bin/bash
# ===========================================================================
# Bash Script: Run SLLOD Shear Simulations for All Shear Rates
# Reference: Zeng et al., J. Chem. Phys. 163, 084512 (2025)
#
# Usage:
#   ./run_all_shearrates.sh
#
# Prerequisite: restart.equil must exist (run in.equilibrate first)
#
# Timestep selection rule (per prompt spec):
#   gamma_dot <= 0.01 : dt = 0.003  (damp = 0.3 tau_0)
#   gamma_dot >  0.01 : dt = 0.001  (damp = 0.1 tau_0)
#
# This ensures that:
#   - Low shear rates: steps are feasible (large dt)
#   - High shear rates: numerical stability (small dt)
# ===========================================================================

# ===== PARAMETERS =====
NPROC=4                                    # MPI processes per job
LAMMPS_EXEC="lmp"                          # LAMMPS executable name (adjust if needed)
INPUT_SCRIPT="in.shear_template"           # Template script to run
RESTART_FILE="restart.equil"               # Equilibrated restart

# Base random seed (each shear rate gets a unique seed)
BASE_SEED=12345

# All shear rates to simulate (in tau_0^{-1})
# Sorted from low to high for logical ordering
SHEAR_RATES="0.001 0.003 0.005 0.01 0.015 0.02 0.05 0.1"

# Output directories (create if not exist)
DATA_DIR="../data"
LOG_DIR="logs"
mkdir -p ${DATA_DIR} ${LOG_DIR}

# ===== PREFLIGHT CHECKS =====
echo "============================================="
echo "KA Binary LJ Shear Simulations"
echo "============================================="

# Check restart file exists
if [ ! -f "${RESTART_FILE}" ]; then
    echo "ERROR: Restart file '${RESTART_FILE}' not found."
    echo "       Please run the equilibration script first:"
    echo "       mpirun -np ${NPROC} ${LAMMPS_EXEC} -in in.equilibrate"
    exit 1
fi

# Check LAMMPS executable
if ! command -v "${LAMMPS_EXEC}" &>/dev/null; then
    echo "WARNING: '${LAMMPS_EXEC}' not found in PATH."
    echo "         Adjust LAMMPS_EXEC variable in this script."
fi

echo "Restart file   : ${RESTART_FILE}"
echo "MPI processes  : ${NPROC}"
echo "LAMMPS binary  : ${LAMMPS_EXEC}"
echo "Shear rates    : ${SHEAR_RATES}"
echo "============================================="

# ===== EXPECTED SIMULATION SIZES =====
# Print estimated step counts for each shear rate before starting
echo ""
echo "Estimated run lengths:"
printf "  %-12s %-8s %-14s %-14s %-14s\n" "gamma_dot" "dt" "pre_steps" "prod_steps" "dump_every"
printf "  %-12s %-8s %-14s %-14s %-14s\n" "---------" "--" "---------" "----------" "----------"
for SR in ${SHEAR_RATES}; do
    # Select timestep
    if awk "BEGIN { exit !(${SR} > 0.01) }"; then
        DT=0.001
    else
        DT=0.003
    fi
    PRE=$(python3  -c "print(round(20.0/(${SR}*${DT})))")
    PROD=$(python3 -c "print(round(50.0/(${SR}*${DT})))")
    DUMP=$(python3 -c "print(round(0.5/(${SR}*${DT})))")
    printf "  %-12s %-8s %-14s %-14s %-14s\n" ${SR} ${DT} ${PRE} ${PROD} ${DUMP}
done
echo ""

# ===== MAIN LOOP: RUN EACH SHEAR RATE =====
SEED=${BASE_SEED}
JOB_COUNT=0
FAILED_JOBS=()
SUCCESSFUL_JOBS=()

for SR in ${SHEAR_RATES}; do

    # ---- Select timestep based on shear rate ----
    # Rule: gamma_dot <= 0.01 -> dt=0.003; gamma_dot > 0.01 -> dt=0.001
    # awk is used for floating-point comparison (bash can't do this natively)
    if awk "BEGIN { exit !(${SR} > 0.01) }"; then
        DT=0.001
    else
        DT=0.003
    fi

    # ---- File names ----
    LOG_FILE="${LOG_DIR}/log.shear_${SR}"
    STDOUT_FILE="${LOG_DIR}/stdout.shear_${SR}"

    echo "----------------------------------------------"
    echo "Starting shear rate: gamma_dot = ${SR}"
    echo "  Timestep dt       = ${DT}"
    echo "  Thermostat damp   = $(python3 -c "print(100.0*${DT})")"
    echo "  Pre-run steps     = $(python3 -c "print(round(20.0/(${SR}*${DT})))")"
    echo "  Prod  steps       = $(python3 -c "print(round(50.0/(${SR}*${DT})))")"
    echo "  Dump  every       = $(python3 -c "print(round(0.5/(${SR}*${DT})))")"
    echo "  Log file          = ${LOG_FILE}"
    echo "  Start time        = $(date '+%Y-%m-%d %H:%M:%S')"

    # ---- Run LAMMPS ----
    mpirun -np ${NPROC} ${LAMMPS_EXEC} \
        -in  ${INPUT_SCRIPT} \
        -var SHEAR_RATE ${SR} \
        -var DT         ${DT} \
        -var SEED       ${SEED} \
        -log ${LOG_FILE} \
        > ${STDOUT_FILE} 2>&1

    EXIT_CODE=$?
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "  Status: COMPLETED (exit code 0) at ${END_TIME}"
        SUCCESSFUL_JOBS+=("${SR}")

        # Move output files to data directory
        mv -f dump.shear_${SR}.lammpstrj    ${DATA_DIR}/ 2>/dev/null && \
            echo "  Moved: dump.shear_${SR}.lammpstrj -> ${DATA_DIR}/"
        mv -f thermo.shear_${SR}.dat        ${DATA_DIR}/ 2>/dev/null && \
            echo "  Moved: thermo.shear_${SR}.dat -> ${DATA_DIR}/"
        mv -f visc_blockave.shear_${SR}.dat ${DATA_DIR}/ 2>/dev/null && \
            echo "  Moved: visc_blockave.shear_${SR}.dat -> ${DATA_DIR}/"
    else
        echo "  Status: FAILED (exit code ${EXIT_CODE}) at ${END_TIME}"
        echo "  Check log: ${LOG_FILE}"
        FAILED_JOBS+=("${SR}")
    fi

    # Increment seed to avoid correlation between independent runs
    SEED=$((SEED + 1111))
    JOB_COUNT=$((JOB_COUNT + 1))

done

# ===== SUMMARY =====
echo ""
echo "============================================="
echo "SIMULATION SUMMARY"
echo "============================================="
echo "Total jobs     : ${JOB_COUNT}"
echo "Successful     : ${#SUCCESSFUL_JOBS[@]}"
echo "Failed         : ${#FAILED_JOBS[@]}"

if [ ${#SUCCESSFUL_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "Successful shear rates:"
    for SR in "${SUCCESSFUL_JOBS[@]}"; do
        echo "  gamma_dot = ${SR}"
    done
fi

if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "FAILED shear rates (check logs):"
    for SR in "${FAILED_JOBS[@]}"; do
        echo "  gamma_dot = ${SR}  ->  ${LOG_DIR}/log.shear_${SR}"
    done
    exit 1
fi

echo ""
echo "All simulations completed successfully."
echo "Output files are in: ${DATA_DIR}/"
echo "Log files are in   : ${LOG_DIR}/"
echo ""
echo "Next step: run Python analysis scripts in ../analysis/"
echo "  python3 compute_viscosity.py"
