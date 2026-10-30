#!/bin/bash
#SBATCH --job-name=nextla-catalog
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --partition=mit_normal_gpu
#SBATCH --time=04:00:00
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --output=slurm-catalog-%j.out

# Submit from the full repository: sbatch test/catalog/run.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
[[ -f src/cholesky_tree_subdiv.jl ]] || { echo "Submit from the full NextLA.jl repository." >&2; exit 1; }
command -v timeout >/dev/null || { echo "GNU timeout is required." >&2; exit 1; }
if ! command -v module >/dev/null; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        if [[ -f "$init" ]]; then source "$init"; break; fi
    done
fi
module load "${JULIA_MODULE:-julia}"

export MATRIX_DIR="${MATRIX_DIR:-$HOME/orcd/scratch/nextla-matrices}"
export RESULTS_DIR="${RESULTS_DIR:-$PWD/benchmark-results/catalog-$SLURM_JOB_ID}"
export STILES_CORES="${SLURM_CPUS_PER_TASK:?CPU allocation is required}"
export OMP_NUM_THREADS="$STILES_CORES"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export REQUIRE_CUDA=1 GKSwstype=100
export GPU_TIMEOUT_SECONDS="${GPU_TIMEOUT_SECONDS:-600}"
export STILES_TIMEOUT_SECONDS="${STILES_TIMEOUT_SECONDS:-300}"

# Bound preparation as well as workers. Leave time before Slurm's hard cutoff.
job_end="${SLURM_JOB_END_TIME:?Slurm must supply the allocation end time}"
stage() {
    local limit="$1"; shift
    local remaining=$(( job_end - $(date +%s) - 120 ))
    (( remaining > 0 )) || { echo "Allocation time exhausted before: $*" >&2; exit 1; }
    (( limit <= remaining )) || limit="$remaining"
    echo "[$(date -Is)] Running (limit ${limit}s): $*"
    timeout --signal=TERM --kill-after=10s "${limit}s" "$@"
}
trap 'code=$?; echo "[$(date -Is)] Job finished with exit code $code. Results: $RESULTS_DIR"' EXIT
echo "Node: $(hostname); job: $SLURM_JOB_ID; sTiles cores: $STILES_CORES"
echo "Matrices: $MATRIX_DIR; results: $RESULTS_DIR"
julia --version

# Set CATALOG_SETUP=0 only when this environment is already prepared (e.g. resume).
if [[ "${CATALOG_SETUP:-1}" == 1 ]]; then
    stage 1800 julia --startup-file=no test/catalog/setup.jl --plots
fi
stage 600 julia --startup-file=no test/catalog/download.jl --check-only
stage 900 julia --startup-file=no --project=test/catalog test/catalog/integration_tests.jl
stage 300 julia --startup-file=no --project=test/catalog test/catalog/stiles_smoke.jl

budget="${SWEEP_TIMEOUT_SECONDS:-10800}"
[[ "$budget" =~ ^[1-9][0-9]*$ ]] || { echo "SWEEP_TIMEOUT_SECONDS must be a positive integer." >&2; exit 1; }
remaining=$(( job_end - $(date +%s) - 300 ))
(( remaining > 0 )) || { echo "No allocation time left for the sweep." >&2; exit 1; }
(( budget <= remaining )) || budget="$remaining"
export SWEEP_TIMEOUT_SECONDS="$budget"
# The supervisor checkpoints each completed method and bounds each worker.
stage "$((budget + 60))" julia --startup-file=no --project=test/catalog test/catalog/run.jl
stage 180 julia --startup-file=no --project=test/catalog test/catalog/plot.jl "$RESULTS_DIR/results.csv"
