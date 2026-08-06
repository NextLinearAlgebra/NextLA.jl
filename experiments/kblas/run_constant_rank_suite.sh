#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 KBLAS_BENCHMARK_EXECUTABLE OUTPUT.csv" >&2
    exit 2
fi

benchmark="$1"
output="$2"
if [[ ! -x "${benchmark}" ]]; then
    echo "benchmark executable is missing or not executable: ${benchmark}" >&2
    exit 1
fi
if [[ -e "${output}" ]]; then
    echo "refusing to overwrite existing output: ${output}" >&2
    exit 1
fi

mkdir -p "$(dirname "${output}")"
printf '%s\n' \
    'mode,storage_type,N,q,b,rank,rank_B,rank_C,final_rank,rank_over_b,warmups,repetitions,tlr_median_ms,tlr_min_ms,dense_median_ms,dense_min_ms,executed_flops,dense_flops,flop_ratio_ceiling' \
    > "${output}"

for n in 4096 8192 16384 32768 65536; do
    for q in 8 16; do
        b=$((n / q))
        for rank_divisor in 16 8; do
            rank=$((b / rank_divisor))
            "${benchmark}" lld "${n}" "${n}" "${n}" "${b}" \
                "${rank}" "${rank}" "${rank}" 3 10 >> "${output}"
        done
    done
done

echo "Completed KBLAS constant-rank suite: ${output}"
