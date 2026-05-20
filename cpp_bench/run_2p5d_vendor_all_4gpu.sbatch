#!/bin/bash
#SBATCH --job-name=2p5d-vendor-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
# Default partition: override if your site uses another queue, e.g.
#   sbatch --partition=full-node --time=08:00:00 ...
#SBATCH --partition=large
#SBATCH --output=/home/ftome_local/comparative-bench/NextLA.jl/logs/2p5d_vendor_4gpu_%j.out
#SBATCH --error=/home/ftome_local/comparative-bench/NextLA.jl/logs/2p5d_vendor_4gpu_%j.err
#
# Full Path (s) 2.5D sweeps vs libcusolverMp (FP64 + FP32 baselines inside the sweep scripts),
# plus TeX-derived grid sweep (run_derived_sweep).  See run_2p5d_vs_vendor_4gpu.sh for env vars.
#
# Submit from anywhere:
#   sbatch /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench/run_2p5d_vendor_all_4gpu.sbatch
#
# Chain after another job:
#   sbatch --dependency=afterany:PRIOR_JOBID ...

set -euo pipefail
export NP="${NP:-4}"
export SLURM_EXPORT_ENV=ALL

ROOT="/home/ftome_local/comparative-bench/NextLA.jl/cpp_bench"
cd "$ROOT"

# Prefer conda on the node; override with #SBATCH --export=ALL,CONDA_PREFIX=/path
if [[ -z "${CONDA_PREFIX:-}" && -d "${HOME}/miniforge3" ]]; then
  export CONDA_PREFIX="${HOME}/miniforge3"
fi

bash ./run_2p5d_vs_vendor_4gpu.sh

echo "Slurm job ${SLURM_JOB_ID:-?} finished at $(date -Is)"
