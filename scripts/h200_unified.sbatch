#!/bin/bash
#SBATCH -J nextla_uni
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 00:55:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run the unified single-process H200 benchmark. One Julia invocation, all
# configs swept inside — JIT compile is amortized once instead of paid per-config.

set -euo pipefail
cd /home/ftome_local/comparative-bench/NextLA.jl
mkdir -p logs

export PATH="/tmp/julia-1.11.5/bin:$PATH"
if command -v module >/dev/null 2>&1; then
  module load cuda/12.8.0-y5ulor7 >/dev/null 2>&1 || true
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
if [ "${ALLOW_NON_H200:-0}" != "1" ] && [ -n "${gpu_name}" ] && [[ "${gpu_name}" != *H200* ]]; then
  echo "Expected H200, found: ${gpu_name}" >&2
  exit 1
fi

# Force line-buffered stdout so we see progress live.
exec stdbuf -oL -eL julia --project=bench scripts/h200_unified.jl
