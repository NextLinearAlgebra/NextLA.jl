#!/bin/bash
#SBATCH -J nextla_qr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H200 single-GPU benchmark harness for NextLA.geqrf_2p5d! vs CUDA.CUSOLVER.geqrf!.
# Phase A: regime probe (c, b, b_min, b_max for each N).
# Phase B: CPU correctness (validate_ortho_modes.jl).
# Phase C: GPU benchmark sweep (FP64 + FP32 + optional TF32).
# Phase D: large-N orthogonality gate (N=4096, ortho=:safe).
#
# Knobs forwarded as env (defaults pick a sensible H200 run):
#   BENCH_SIZES   matrix sizes, comma-separated (default 1000,2000,4000,8000)
#   BENCH_BS      override the b-sweep, comma-separated
#   BENCH_T       precisions: f64,f32,c64,c32   (default f64,f32)
#   BENCH_ORTHO   ortho modes: fast,safe        (default fast,safe)
#   BENCH_C       c override (int or "auto")    (default 1)
#   BENCH_PHASES  comma-separated phases (default A,B,C,D)
#   NEXTLA_TF32   enable TF32 in cuBLAS (Step 5)
#   NEXTLA_M_BYTES  fast-memory budget per SM in bytes (Step 3)
#   NEXTLA_P_OVERRIDE  override SM count (Step 3)
#   NEXTLA_FORCE_C1  "1" forces c=1 even when params.c>1 (Step 2)
#   NEXTLA_USE_SYRK  "1"/"0" toggles SYRK vs GEMM for Gram (Step 4)

set -euo pipefail

cd /home/ftome_local/comparative-bench/NextLA.jl
mkdir -p logs

# Pin our local Julia 1.11.5 (system /tmp/julia-1.12.4 is missing LazyArtifacts).
export PATH="/tmp/julia-1.11.5/bin:$PATH"

# Load CUDA toolkit module (12.8.0 visible to nvcc / driver shim).
if command -v module >/dev/null 2>&1; then
  module load cuda/12.8.0-y5ulor7 >/dev/null 2>&1 || true
fi

# H200 gate check (skip with ALLOW_NON_H200=1).
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
if [ "${ALLOW_NON_H200:-0}" != "1" ] && [ -n "${gpu_name}" ] && [[ "${gpu_name}" != *H200* ]]; then
  echo "Expected H200, found: ${gpu_name}" >&2
  echo "Set ALLOW_NON_H200=1 to override this check." >&2
  exit 1
fi

# Pretty banner.
echo "=========================================================="
echo " H200 NextLA QR benchmark"
echo "  host:     $(hostname)"
echo "  date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  gpu:      ${gpu_name}"
echo "  driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
echo "  julia:    $(julia --version)"
echo "  cuda:     $(nvcc --version 2>/dev/null | tail -n1 || echo n/a)"
echo "=========================================================="

# Env reflection so logs are self-contained.
for v in BENCH_SIZES BENCH_BS BENCH_T BENCH_ORTHO BENCH_C BENCH_PHASES \
         NEXTLA_TF32 NEXTLA_M_BYTES NEXTLA_P_OVERRIDE NEXTLA_FORCE_C1 \
         NEXTLA_USE_SYRK NEXTLA_PEAK_FP64 NEXTLA_PEAK_FP32; do
  printf '  %-22s = %s\n' "$v" "${!v:-}"
done
echo "----------------------------------------------------------"

phases="${BENCH_PHASES:-A,B,C,D}"
sizes="${BENCH_SIZES:-1000,2000,4000,8000}"

# Phase A — regime probe
if [[ ",${phases}," == *,A,* ]]; then
  echo
  echo "### Phase A — regime probe ###"
  julia --project=bench -e "
    using CUDA, NextLA, KernelAbstractions
    be = CUDABackend()
    for N in (1000, 2000, 4000, 8000, 16000)
      for T in (Float64, Float32)
        p = compute_params(be, T, N)
        println(\"T=\$T  N=\$N  c=\$(p.c)  b=\$(p.b)  b_min=\$(p.b_min)  b_max=\$(p.b_max)  Px=\$(p.Px)  M=\$(p.M)  P=\$(p.P)  TILE=\$(p.TILE_DIM)\")
      end
    end
  "
fi

# Phase B — CPU correctness
if [[ ",${phases}," == *,B,* ]]; then
  echo
  echo "### Phase B — CPU correctness (validate_ortho_modes.jl) ###"
  julia --project=bench -t 4 scripts/validate_ortho_modes.jl
fi

# Phase C — GPU benchmark sweep
if [[ ",${phases}," == *,C,* ]]; then
  echo
  echo "### Phase C — GPU bench sweep (sizes=${sizes}) ###"
  BENCH_SIZES="${sizes}" julia --project=bench scripts/bench_gpu_vs_cusolver.jl
fi

# Phase D — large-N orthogonality gate
if [[ ",${phases}," == *,D,* ]]; then
  echo
  echo "### Phase D — large-N orthogonality gate (N=4096, :safe) ###"
  julia --project=bench -e "
    using CUDA, NextLA, KernelAbstractions, LinearAlgebra
    N = 4096
    A0 = CUDA.randn(Float64, N, N)
    R  = CUDA.zeros(Float64, N, N); tau = CUDA.zeros(Float64, N)
    A  = copy(A0)
    NextLA.geqrf_2p5d!(N, N, A, R, tau; ortho=:safe)
    CUDA.synchronize()
    G  = A' * A - I
    println(\"  N=\$N FP64 :safe   |Q'Q - I|_F = \$(norm(G))   residual=\$(norm(A0 - A * R) / norm(A0))\")
  "
fi

echo
echo "=========================================================="
echo " Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="
