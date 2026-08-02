#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
julia_bin="${JULIA:-julia}"

"${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/gemm/precision_sweep.jl"
"${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/gemm/memory_sweep.jl"

if [[ "${NEXTLA_RUN_ABLATION:-0}" == "1" ]]; then
    "${julia_bin}" --project="${repo_root}/experiments" \
        "${repo_root}/experiments/gemm/rank_bucketing_ablation.jl"
fi
