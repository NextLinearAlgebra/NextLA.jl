#!/bin/bash
#SBATCH --job-name=nla_oom_np2
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/home/ftome_local/comparative-bench/NextLA.jl/logs/full25d_oom_np2-%j.out
#SBATCH --error=/home/ftome_local/comparative-bench/NextLA.jl/logs/full25d_oom_np2-%j.err

# Full BC 2.5D × 5 variants × 4 matrix modes × LA on/off × N up to OOM,
# on 2 H200 GPUs.  Auto-derives (c, Px, Py, Pz, b) from compute_derived_schedule
# at every N.  cuSolverMp baseline at every N >= 4096 with MB matched to b.

NEXTLA_DIR=/home/ftome_local/comparative-bench/NextLA.jl
mkdir -p "$NEXTLA_DIR/logs"

export NP=2
export NEXTLA_FASTMEM_FRAC=${NEXTLA_FASTMEM_FRAC:-0.72}
export NEXTLA_BENCH_RUNS=${NEXTLA_BENCH_RUNS:-5}
# Skip Givens entirely: its single-block tournament kernel is Θ(b·log m)
# sequential and consumes hours/cell at N>=4096, blowing the 12h budget.
# Run Givens as a separate small sweep if needed.
export GIVENS_CAP=0

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sbatch start  NP=$NP"
echo "  host=$(hostname)"
echo "  gres=$SLURM_GPUS_ON_NODE  jobid=$SLURM_JOB_ID"
"$NEXTLA_DIR/cpp_bench/run_full25d_oom_sweep.sh"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sbatch end"
