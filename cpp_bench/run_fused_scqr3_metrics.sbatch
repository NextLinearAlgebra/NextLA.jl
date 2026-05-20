#!/bin/bash
#SBATCH --job-name=fused-scqr3
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
# Example: adjust paths, conda OpenMPI, and np flags for your cluster.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH="${CONDA_PREFIX:-}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

python3 fused_scqr3_metrics.py \
  --mpirun "mpirun --oversubscribe" \
  --np-vendor 4 \
  --np-ours 4 \
  --vendor-bin "${ROOT}/cusolverMp_geqrf_bench" \
  --ours-bin "${ROOT}/scqr3_full25d_bench" \
  --N 4096 \
  --px-vendor 2 \
  --py-vendor 2 \
  --ours-extra "--px=2 --py=2 --pz=1 --passes=2 --layout=slab"
