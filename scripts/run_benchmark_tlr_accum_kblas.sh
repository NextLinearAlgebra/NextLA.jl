#!/usr/bin/env bash
# Run KBLAS first, then feed its persisted factors/results to the Julia NextLA benchmark.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
kblas_root=${KBLAS_ROOT:-/home/alecarraro/Documents/dev/kblas-gpu}
magma_root=${MAGMA_ROOT:-/home/alecarraro/Documents/dev/magma/install}
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
julia_project=${JULIA_PROJECT:-/home/alecarraro/Documents/dev/gpuenv}
build_dir=${KBLAS_BENCH_BUILD_DIR:-"${script_dir}/.kblas"}
exe="${build_dir}/kblas_tlr_accum_bench"
source="${script_dir}/kblas_tlr_accum_bench.cu"
smoke_args=()
[[ "${1:-}" == "--smoke" ]] && smoke_args=(--smoke)

[[ -f "${kblas_root}/lib/libkblas-gpu.a" ]] || { echo "KBLAS_ROOT is invalid: ${kblas_root}" >&2; exit 1; }
[[ -f "${magma_root}/lib/libmagma.so" ]] || { echo "MAGMA_ROOT is invalid: ${magma_root}" >&2; exit 1; }
[[ -x "${cuda_root}/bin/nvcc" ]] || { echo "CUDA_ROOT is invalid: ${cuda_root}" >&2; exit 1; }
mkdir -p "${build_dir}"

# Every run gets a fresh directory so Julia cannot accidentally consume stale
# records.  A caller-supplied directory must likewise not already exist.
if [[ -n "${KBLAS_BENCH_RESULTS_DIR:-}" ]]; then
  results_dir=${KBLAS_BENCH_RESULTS_DIR}
  [[ ! -e "${results_dir}" ]] || { echo "KBLAS_BENCH_RESULTS_DIR already exists: ${results_dir}" >&2; exit 1; }
else
  results_dir="${build_dir}/results/run-$(date +%Y%m%d-%H%M%S)-$$"
fi
mkdir -p "${results_dir}"

# Compile only this small benchmark executable when the source changed.  KBLAS
# itself is linked from its existing configured archive and is never rebuilt.
if [[ ! -x "${exe}" || "${source}" -nt "${exe}" ]]; then
  "${cuda_root}/bin/nvcc" -std=c++17 -O3 \
    -I"${kblas_root}/include" -I"${kblas_root}/src" -I"${magma_root}/include" \
    "${source}" "${kblas_root}/lib/libkblas-gpu.a" \
    -L"${magma_root}/lib" -L"${cuda_root}/lib64" \
    -Xlinker -rpath -Xlinker "${magma_root}/lib:${cuda_root}/lib64" \
    -lmagma -lcusparse -lcublas -lcudart -lopenblas -lgomp -o "${exe}"
fi

LD_LIBRARY_PATH="${magma_root}/lib:${cuda_root}/lib64:${LD_LIBRARY_PATH:-}" \
  "${exe}" --output "${results_dir}" "${smoke_args[@]}"
export KBLAS_ROOT="${kblas_root}" MAGMA_ROOT="${magma_root}" CUDA_ROOT="${cuda_root}"
echo "KBLAS: ${kblas_root}"
echo "MAGMA: ${magma_root}"
echo "CUDA compiler: ${cuda_root}/bin/nvcc"
julia --project="${julia_project}" "${script_dir}/benchmark_tlr_accum_kblas.jl" \
  --results "${results_dir}" "${smoke_args[@]}"
