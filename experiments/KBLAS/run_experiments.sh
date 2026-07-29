#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
kblas_root=${KBLAS_ROOT:-"$(cd -- "$script_dir/../../../kblas-gpu" 2>/dev/null && pwd)"}
magma_root=${MAGMA_ROOT:-/usr/local/magma}
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
build_dir=${KBLAS_BENCH_BUILD_DIR:-"$script_dir/build"}
output_dir=${KBLAS_RESULTS_DIR:-"$script_dir/results"}
source_file="$script_dir/benchmark_tlr_dense.cu"
library="$kblas_root/lib/libkblas-gpu.a"

# ── Strong-scaling configuration ────────────────────────────────────────────
# These sizes, tile size, and repetition counts mirror the Julia strong-scaling
# campaign as closely as KBLAS permits.
STRONG_SIZES=(1024 2048 4096 8192 16384 32768)
STRONG_TILE_SIZE=512
STRONG_RANK_A=64
STRONG_RANK_B=128
STRONG_OUTPUT_RANK=128

# KBLAS has no FP16 TLR entry point; float/double are the available closest
# comparison modes to the Julia FP32/FP64 paths.
WARMUP=${KBLAS_WARMUP:-1}
REPS=${KBLAS_REPS:-3}
PRECISIONS=${KBLAS_PRECISIONS:-float,double}

[[ -x "$cuda_root/bin/nvcc" ]] || { echo "invalid CUDA_ROOT: $cuda_root" >&2; exit 1; }
[[ -f "$library" ]] || { echo "KBLAS library not found: $library" >&2; exit 1; }
mkdir -p "$build_dir" "$output_dir"

compile() {
    local precision=$1
    local macro=()
    local executable="$build_dir/benchmark_tlr_dense_$precision"
    [[ "$precision" == float ]] && macro=(-DBENCH_FLOAT)
    if [[ ! -x "$executable" || "$source_file" -nt "$executable" || "$library" -nt "$executable" ]]; then
        "$cuda_root/bin/nvcc" -std=c++17 -O3 "${macro[@]}" \
            -I"$kblas_root/include" -I"$kblas_root/src" -I"$magma_root/include" \
            "$source_file" "$library" \
            -L"$magma_root/lib" -L"$cuda_root/lib64" \
            -Xlinker -rpath -Xlinker "$magma_root/lib:$cuda_root/lib64" \
            -lmagma -lcusparse -lcublas -lcurand -lcudart -lopenblas -lgomp -o "$executable"
    fi
    echo "$executable"
}

run_case() {
    local executable=$1 mode=$2 m=$3 k=$4 n=$5 b=$6 rank_A=$7 rank_B=$8 rank_C=$9
    "$executable" "$mode" "$m" "$k" "$n" "$b" "$rank_A" "$rank_B" "$rank_C" "$WARMUP" "$REPS"
}

IFS=',' read -r -a selected_precisions <<< "$PRECISIONS"
header="mode,precision,m,k,n,tile_size,rank_A,rank_B,rank_C,final_rank,dense_ms,tlr_ms,speedup,dense_gflops,tlr_gflops"
for precision in "${selected_precisions[@]}"; do
    case "$precision" in
        float|double) executable=$(compile "$precision") ;;
        *) echo "unsupported precision: $precision" >&2; exit 2 ;;
    esac

    for mode in lld lll; do
        result="$output_dir/strong_scaling_${mode}_${precision}.csv"
        echo "$header" > "$result"
        for n in "${STRONG_SIZES[@]}"; do
            run_case "$executable" "$mode" "$n" "$n" "$n" \
                "$STRONG_TILE_SIZE" "$STRONG_RANK_A" "$STRONG_RANK_B" \
                "$STRONG_OUTPUT_RANK" >> "$result"
        done
    done
done

echo "KBLAS results written to $output_dir"
