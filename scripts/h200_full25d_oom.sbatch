#!/bin/bash
#SBATCH --job-name=nextla_full25d
#SBATCH --partition=full-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/home/ftome_local/comparative-bench/NextLA.jl/logs/full25d_oom-%j.out
#SBATCH --error=/home/ftome_local/comparative-bench/NextLA.jl/logs/full25d_oom-%j.err

# Full-2.5D × all-variants × all-precisions × LA on/off sweep on 8 H200s.
# Uses absolute paths so it can run from any CWD.

NEXTLA_DIR=/home/ftome_local/comparative-bench/NextLA.jl
mkdir -p "$NEXTLA_DIR/logs"

export NP=${NP:-8}
export NEXTLA_FASTMEM_FRAC=${NEXTLA_FASTMEM_FRAC:-0.72}
export NEXTLA_BENCH_RUNS=${NEXTLA_BENCH_RUNS:-5}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sbatch start  NP=$NP"
"$NEXTLA_DIR/cpp_bench/run_full25d_oom_sweep.sh"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sbatch end"
