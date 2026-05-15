#!/bin/bash
#SBATCH -J nextla_qr_int
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Post-Step-6 integrated H200 benchmark. Sweeps every env-knob combination that
# matters for the final speedup story:
#
#   1. FP64 default-M  + force_c1=auto (let c emerge from PM/N²)
#   2. FP64 default-M  + force_c1=1    (single-device clean — skip fanout)
#   3. FP64 OLD smem-only M + force_c1=1  (Step 3 A/B vs the original M proxy)
#   4. FP64 SYRK OFF       (Step 4 A/B vs cuBLAS GEMM in the Gram)
#   5. FP32 TF32=0    (apples-to-apples FP32 ALU vs cuSOLVER TF32)
#   6. FP32 TF32=1    (apples-to-apples both using TF32)
#
# Each run is BENCH_SIZES=2000,4000,8000 — N=1000 is too noisy, N=16000 too slow
# for the dev partition's 1 h limit when combined with all 6 runs.

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

echo "=========================================================="
echo " H200 integrated NextLA benchmark   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  host=$(hostname)  gpu=${gpu_name}"
echo "  julia=$(julia --version)"
echo "=========================================================="

SIZES="${BENCH_SIZES:-2000,4000,8000}"
# Trim the b-sweep to 3 values per N to keep all 6 configs inside the 1 h limit.
# Each cell (N, b, ortho, type) is ~(nwarm + nrun) = 6 calls of geqrf_2p5d!, and
# the first call per N×b×type warms the JIT (~20-90 s) — wide sweeps blow the budget.
export BENCH_BS="${BENCH_BS:-128,256,512}"

run_config () {
  local label="$1"; shift
  echo
  echo "###############################################################"
  echo " ${label}"
  for kv in "$@"; do
    echo "   ${kv}"
    export "${kv}"
  done
  echo "###############################################################"
  BENCH_SIZES="${SIZES}" BENCH_ORTHO=fast julia --project=bench scripts/bench_gpu_vs_cusolver.jl
  # Reset overrides for next config.
  for kv in "$@"; do
    key="${kv%%=*}"
    unset "${key}"
  done
}

run_config "Config 1  FP64 :fast  default-M  c=auto  (2.5D engaged)" \
  BENCH_T=f64  BENCH_C=auto  NEXTLA_FORCE_C1=0

run_config "Config 2  FP64 :fast  default-M  FORCE_C1=1 (single-device clean)" \
  BENCH_T=f64  BENCH_C=auto  NEXTLA_FORCE_C1=1

run_config "Config 3  FP64 :fast  OLD smem-only M  FORCE_C1=1 (Step 3 A/B baseline)" \
  BENCH_T=f64  BENCH_C=auto  NEXTLA_FORCE_C1=1  NEXTLA_M_BYTES=233472

run_config "Config 4  FP64 :fast  SYRK OFF  (Step 4 A/B vs cuBLAS GEMM)" \
  BENCH_T=f64  BENCH_C=auto  NEXTLA_FORCE_C1=1  NEXTLA_USE_SYRK=0

run_config "Config 5  FP32 :fast  TF32 OFF  (FP32 ALU)" \
  BENCH_T=f32  BENCH_C=auto  NEXTLA_FORCE_C1=1  NEXTLA_TF32=0

run_config "Config 6  FP32 :fast  TF32 ON   (apples-to-apples vs cuSOLVER)" \
  BENCH_T=f32  BENCH_C=auto  NEXTLA_FORCE_C1=1  NEXTLA_TF32=1

echo
echo "### Phase D — large-N :safe orthogonality gate ###"
julia --project=bench -e '
  using CUDA, NextLA, KernelAbstractions, LinearAlgebra
  N = 4096
  A0 = CUDA.randn(Float64, N, N)
  R  = CUDA.zeros(Float64, N, N); tau = CUDA.zeros(Float64, N)
  A  = copy(A0)
  NextLA.geqrf_2p5d!(N, N, A, R, tau; ortho=:safe)
  CUDA.synchronize()
  G  = A'"'"' * A - I
  println("  N=", N, " FP64 :safe  |Q\x27Q - I|_F = ", norm(G),
          "  residual = ", norm(A0 - A * R) / norm(A0))
'

echo
echo "=========================================================="
echo " Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="
