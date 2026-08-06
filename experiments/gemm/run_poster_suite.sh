#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="$(realpath -m "$1")"
if [[ -e "${output_root}" ]]; then
    echo "refusing to overwrite existing output: ${output_root}" >&2
    exit 1
fi

mkdir -p "${output_root}/nextla" "${output_root}/comparisons"
julia_bin="${JULIA:-julia}"
kblas_csv="${NEXTLA_KBLAS_CSV:-${repo_root}/experiments/results/gemm/kblas/constant_rank_fp32_rank_b16_b8.csv}"

env \
    NEXTLA_MEMORY_N=16384 \
    NEXTLA_MEMORY_PRECISION=fp16 \
    NEXTLA_MEMORY_TILE_DIVISORS=16,8 \
    NEXTLA_MEMORY_DISTRIBUTIONS=uniform,skewed \
    NEXTLA_MEMORY_LAYOUTS=compressed_dense,dense_compressed,compressed_compressed \
    NEXTLA_MEMORY_RANK_BANDS=32:16 \
    NEXTLA_MEMORY_WORKSPACE_LEVELS=1,2,4,8,16,32,64 \
    NEXTLA_MEMORY_WARMUP=1 \
    NEXTLA_MEMORY_REPS=4 \
    NEXTLA_MEMORY_ANALYSIS_REPS=1 \
    NEXTLA_MEMORY_OUTPUT="${output_root}/nextla/memory_pareto_fp16_n16384_rank_b32_b16.csv" \
    "${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/gemm/run_memory_pareto.jl"

env \
    NEXTLA_TUNING_SIZES=4096,8192,16384,32768,65536 \
    NEXTLA_TUNING_PRECISIONS=bf16,fp16,fp32,tf32 \
    NEXTLA_TUNING_TILE_DIVISORS=16,8 \
    NEXTLA_TUNING_RANK_BANDS=32:16 \
    NEXTLA_TUNING_DISTRIBUTIONS=skewed \
    NEXTLA_TUNING_LAYOUTS=compressed_dense,dense_compressed,compressed_compressed \
    NEXTLA_TUNING_WORKSPACE_LEVELS=1,2,4,8,16,32,64 \
    NEXTLA_TUNING_MIXED_STRIPES=1,2,4,8,16 \
    NEXTLA_TUNING_WARMUP=1 \
    NEXTLA_TUNING_REPS=4 \
    NEXTLA_TUNING_ANALYSIS_REPS=1 \
    NEXTLA_TUNING_OUTPUT="${output_root}/nextla/precision_scaling_skewed_rank_b32_b16_workspace_sweep.csv" \
    "${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/gemm/run_workspace_tuning.jl"

python3 "${repo_root}/experiments/gemm/select_workspace_winners.py" \
    "${output_root}/nextla/precision_scaling_skewed_rank_b32_b16_workspace_sweep.csv" \
    --expected-mixed-stripes 1,2,4,8,16 \
    --output "${output_root}/nextla/precision_scaling_skewed_rank_b32_b16_best_workspace.csv"

env \
    NEXTLA_TUNING_SIZES=4096,8192,16384,32768,65536 \
    NEXTLA_TUNING_PRECISIONS=fp32 \
    NEXTLA_TUNING_TILE_DIVISORS=16,8 \
    NEXTLA_TUNING_RANK_BANDS=16:16,8:8 \
    NEXTLA_TUNING_DISTRIBUTIONS=uniform \
    NEXTLA_TUNING_LAYOUTS=compressed_compressed \
    NEXTLA_TUNING_WORKSPACE_LEVELS=1,2,4,8,16,32,64 \
    NEXTLA_TUNING_WARMUP=1 \
    NEXTLA_TUNING_REPS=4 \
    NEXTLA_TUNING_ANALYSIS_REPS=1 \
    NEXTLA_TUNING_OUTPUT="${output_root}/nextla/constant_rank_fp32_workspace_sweep.csv" \
    "${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/gemm/run_workspace_tuning.jl"

python3 "${repo_root}/experiments/gemm/select_workspace_winners.py" \
    "${output_root}/nextla/constant_rank_fp32_workspace_sweep.csv" \
    --output "${output_root}/nextla/constant_rank_fp32_best_workspace.csv"

python3 "${repo_root}/experiments/gemm/build_kblas_comparison.py" \
    "${output_root}/nextla/constant_rank_fp32_best_workspace.csv" \
    "${kblas_csv}" \
    --allow-unconfirmed \
    --output "${output_root}/comparisons/constant_rank_fp32_best_nextla_vs_kblas.csv"

echo "Completed poster experiment suite: ${output_root}"
echo "Generate figures with:"
echo "  python3 ${repo_root}/experiments/gemm/build_poster_figures.py --results-dir ${output_root} --kblas ${kblas_csv}"
