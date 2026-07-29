#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
julia_bin="${JULIA:-julia}"

"${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/compressed_dense.jl"
"${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/dense_compressed.jl"
"${julia_bin}" --project="${repo_root}/experiments" \
    "${repo_root}/experiments/rows_per_run.jl"
