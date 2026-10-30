#!/bin/bash
# Run within the existing H200 / 14-CPU allocation, from the repository root.
set -euo pipefail
[[ -f src/cholesky_tree_subdiv.jl ]] || { echo 'Run from the NextLA repository root.' >&2; exit 1; }
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 14 ]] || { echo 'This sweep requires your 14-CPU allocation.' >&2; exit 1; }
module load julia
export MATRIX_DIR="${MATRIX_DIR:-$HOME/orcd/scratch/nextla-matrices}"
export LEGACY_MATRIX_DIR="${LEGACY_MATRIX_DIR:-$HOME/orcd/scratch/nextla-legacy-matrices}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export BENCH_REPEATS=5 GPU_TIMEOUT_SECONDS=900 STILES_TIMEOUT_SECONDS=180
export GPU_MEMORY_COPIES=4
result_root="${EXTENDED_RESULTS_DIR:-$PWD/benchmark-results/extended-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$result_root"
job_end="${SLURM_JOB_END_TIME:?Run inside your active allocation}"
stage() {
    local cap="$1"; shift
    local left=$((job_end-$(date +%s)-120))
    (( left > 0 )) || { echo 'Allocation reserve reached; completed checkpoints are retained.'; exit 0; }
    (( cap <= left )) || cap="$left"
    echo "[$(date -Is)] $* (cap ${cap}s)"
    timeout --signal=TERM --kill-after=10s "${cap}s" "$@"
}
{
    echo "Job=$SLURM_JOB_ID; allocated CPUs=$SLURM_CPUS_PER_TASK; node=$(hostname)"
    nvidia-smi -L
    lscpu
    taskset -pc $$
} > "$result_root/hardware.txt" 2>&1
echo "Results: $result_root"
# Alternating core order reduces a simple monotonic order/time confound.
# All cases use identical files, five repeats, OMP threads = requested sTiles cores.
export MATRIX_MANIFEST="$PWD/test/catalog/matrices.csv"
export REAL_MATRICES="${CORE_MATRICES:-inla_graph_sem_n5000,inla_graph_spacetime,inla_graph_animal1}"
export METHODS=stiles
for cores in ${CORE_COUNTS:-14 1 8 2 4}; do
    [[ "$cores" =~ ^[1-9][0-9]*$ && "$cores" -le "$SLURM_CPUS_PER_TASK" ]] || { echo "Invalid core count: $cores" >&2; exit 1; }
    export STILES_CORES="$cores" OMP_NUM_THREADS="$cores" SWEEP_TIMEOUT_SECONDS=240
    export RESULTS_DIR="$result_root/cores-$cores"
    stage 270 julia --startup-file=no --project=test/catalog test/catalog/run.jl
done
# The old file names are searched automatically before downloading anything.
stage 900 julia --startup-file=no test/catalog/legacy_inputs.jl || echo 'Input preparation stopped; benchmark any completed files and record missing ones.'
export MATRIX_DIR="$LEGACY_MATRIX_DIR"
export MATRIX_MANIFEST="$PWD/test/catalog/legacy-matrices.csv"
unset REAL_MATRICES
export STILES_CORES=14 OMP_NUM_THREADS=14 STILES_TIMEOUT_SECONDS=300
export METHODS=all RESULTS_DIR="$result_root/legacy-14"
left=$((job_end-$(date +%s)-300))
(( left > 0 )) || exit 0
export SWEEP_TIMEOUT_SECONDS="$left"
stage "$((left+60))" julia --startup-file=no --project=test/catalog test/catalog/run.jl
echo "Saved extended experiment: $result_root"
