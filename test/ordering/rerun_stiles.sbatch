#!/bin/bash
#SBATCH --job-name=stiles-corrected
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-stiles-corrected-%j.out

# CPU-only by default. sbatch command-line partition/account/GPU overrides
# can reuse an allocation configuration already known to work for the user.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the NextLA.jl repository root}"
[[ -f test/ordering/stiles_direction_check.jl ]] || { echo 'Pull the corrected ordering code before submitting.' >&2; exit 1; }
if ! command -v module >/dev/null; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        if [[ -f "$init" ]]; then source "$init"; break; fi
    done
fi
module load "${JULIA_MODULE:-julia}"
export MATRIX_DIR="${MATRIX_DIR:-$HOME/orcd/scratch/nextla-matrices}"
export LEGACY_MATRIX_DIR="${LEGACY_MATRIX_DIR:-$HOME/orcd/scratch/nextla-legacy-matrices}"
# Do not inherit the old all-method sweep folder: its .finished markers would
# skip the required corrected runs. This dedicated folder can itself resume.
export ORDERING_SWEEP_OUTPUT="${ORDERING_STILES_OUTPUT:-$PWD/benchmark-results/ordering-stiles-corrected}"
export STILES_CORES="${SLURM_CPUS_PER_TASK:?CPU allocation required}"
export OMP_NUM_THREADS="$STILES_CORES"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export ORDERING_BLAS_THREADS=1 ORDERING_METHODS=stiles
export ORDERING_MATRIX_TIMEOUT="${ORDERING_MATRIX_TIMEOUT:-600}"
export ORDERING_REPEATS="${ORDERING_REPEATS:-3}"
unset ORDERING_SWEEP_MATRICES
mkdir -p "$ORDERING_SWEEP_OUTPUT"
trap 'code=$?; echo "[$(date -Is)] Finished with exit code $code; output: $ORDERING_SWEEP_OUTPUT"' EXIT
{
    echo "Job=$SLURM_JOB_ID node=$(hostname) cores=$STILES_CORES"
    echo "Commit=$(git rev-parse HEAD)"
    echo "Started=$(date -Is)"
    echo "Matrices=$MATRIX_DIR; legacy=$LEGACY_MATRIX_DIR"
    echo "Per-matrix timeout=$ORDERING_MATRIX_TIMEOUT; repetitions=$ORDERING_REPEATS"
} | tee "$ORDERING_SWEEP_OUTPUT/batch-$SLURM_JOB_ID.txt"
# Setup was performed in the previous allocation. Opt in only if needed.
if [[ "${ORDERING_SETUP:-0}" == 1 ]]; then
    timeout --signal=TERM --kill-after=10s 1200 julia --startup-file=no test/ordering/setup.jl --stiles
fi
# Fail early rather than starting an overnight sweep with a bad convention.
timeout --signal=TERM --kill-after=10s 300 \
    julia --startup-file=no --project=test/ordering test/ordering/stiles_direction_check.jl
bash test/ordering/sweep.sh
