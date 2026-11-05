#!/bin/bash
#SBATCH --job-name=nextla-ordering
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-ordering-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the NextLA.jl repository root}"
[[ -f test/ordering/report.jl ]] || { echo 'Submit from the repository root.' >&2; exit 1; }
if ! command -v module >/dev/null; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        if [[ -f "$init" ]]; then source "$init"; break; fi
    done
fi
module load "${JULIA_MODULE:-julia}"
export MATRIX_DIR="${MATRIX_DIR:-$HOME/orcd/scratch/nextla-matrices}"
export ORDERING_MATRICES="${ORDERING_MATRICES:-bcsstk15}"
export ORDERING_OUTPUT="${ORDERING_OUTPUT:-$PWD/benchmark-results/ordering-$SLURM_JOB_ID}"
export STILES_CORES="${SLURM_CPUS_PER_TASK:?CPU allocation required}"
export OMP_NUM_THREADS="$STILES_CORES"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export ORDERING_BLAS_THREADS=1
echo "Node: $(hostname); CPU cores: $STILES_CORES; matrices: $ORDERING_MATRICES"
echo "Output: $ORDERING_OUTPUT"
# No GPU allocation is needed for this report. Setup is done once in the job;
# set ORDERING_SETUP=0 after the environment is prepared.
if [[ "${ORDERING_SETUP:-1}" == 1 ]]; then
    timeout --signal=TERM --kill-after=10s 1200 julia --startup-file=no test/ordering/setup.jl --stiles
fi
timeout --signal=TERM --kill-after=10s 300 julia --startup-file=no --project=test/ordering test/ordering/runtests.jl
timeout --signal=TERM --kill-after=10s 1800 julia --startup-file=no --project=test/ordering test/ordering/report.jl
