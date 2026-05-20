#!/usr/bin/env bash
# Run Path (s) full 2.5D sweeps with cuSOLVERMp baselines on NP ranks (default 4).
# 1) Rebuild benches   2) Build vendor METRICS table   3) run_full25d_sweep   4) run_derived_sweep
#
# Env (optional):
#   NP=4                    MPI / GPU rank count
#   CONDA_PREFIX            OpenMPI + NCCL + libcusolverMp prefix
#   CUDA_HOME               CUDA toolkit (default /usr/local/cuda)
#   MPI_INCLUDE             mpi.h directory (default $CONDA_PREFIX/include)
#   SIZES N_MAX             forwarded via run_* scripts
#   Derived sweep: TeX-derived grid uses auto M from each bench (GPU memory × NEXTLA_FASTMEM_FRAC / σ).
#   VENDOR_OUT              path for NEXTLA_VENDOR_METRICS_TABLE (default logs/vendor_metrics_<job>.txt)
#   RUN_ALL_FIVE=1          also run run_all_5variants.sh (older scqr3_2p5d_variants + h/g/q; long)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

NP="${NP:-4}"
: "${CONDA_PREFIX:=${HOME}/miniforge3}"
: "${CUDA_HOME:=/usr/local/cuda}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MPI_INCLUDE="${MPI_INCLUDE:-${CONDA_PREFIX}/include}"

LOGDIR="${HERE}/../logs"
mkdir -p "$LOGDIR"
JOB_TAG="${SLURM_JOB_ID:-manual}"
export VENDOR_OUT="${VENDOR_OUT:-$LOGDIR/vendor_metrics_${JOB_TAG}.txt}"

echo "=== build ($HERE) ==="
bash ./build.sh

echo "=== vendor table -> $VENDOR_OUT ==="
if [[ -x "$HERE/cusolverMp_geqrf_bench" ]]; then
  bash ./capture_vendor_table.sh "$VENDOR_OUT" || echo "warning: capture_vendor_table.sh failed (continuing)"
else
  echo "warning: cusolverMp_geqrf_bench missing; skip vendor table"
fi
if [[ -f "$VENDOR_OUT" ]] && grep -q '^[0-9]' "$VENDOR_OUT"; then
  export NEXTLA_VENDOR_METRICS_TABLE="$VENDOR_OUT"
  echo "export NEXTLA_VENDOR_METRICS_TABLE=$VENDOR_OUT"
fi

echo "=== run_full25d_sweep.sh $NP ==="
bash ./run_full25d_sweep.sh "$NP"

bash ./run_derived_sweep.sh "$NP"

if [[ "${RUN_ALL_FIVE:-0}" == "1" ]]; then
  echo "=== run_all_5variants.sh (RUN_ALL_FIVE=1) ==="
  export NP
  bash ./run_all_5variants.sh
fi

echo "=== DONE run_2p5d_vs_vendor_4gpu.sh ==="
