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

# ── Experiment configuration ─────────────────────────────────────────────────
STRONG_SIZES=(4096 8192 16384 32768)
STRONG_TILE_SIZE=512
STRONG_RANK=64

RANK_MATRIX_SIZE=16384
RANK_TILE_SIZE=512
RANK_VALUES=(8 16 32 64 128 256)

TILE_MATRIX_SIZE=16384
TILE_SIZES=(128 256 512 1024 2048)
RANK_TILE_NUMERATOR=1
RANK_TILE_DENOMINATOR=8

SHAPE_TILE_SIZE=512
SHAPE_CASES=(
    "square 16384 16384 16384"
    "tall 32768 8192 8192"
    "wide 8192 32768 8192"
    "large_k 8192 8192 32768"
    "small_k 16384 16384 4096"
)
SHAPE_RANK=32

WARMUP=${KBLAS_WARMUP:-3}
REPS=${KBLAS_REPS:-10}
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
            -I"$kblas_root/include" -I"$kblas_root/src" "$source_file" "$library" \
            -L"$magma_root/lib" -L"$cuda_root/lib64" \
            -Xlinker -rpath -Xlinker "$magma_root/lib:$cuda_root/lib64" \
            -lmagma -lcusparse -lcublas -lcudart -lopenblas -lgomp -o "$executable"
    fi
    echo "$executable"
}

run_case() {
    local executable=$1 experiment=$2 m=$3 k=$4 n=$5 b=$6 r=$7 precision=$8
    local result="$output_dir/${experiment}_${precision}.csv"
    if [[ ! -f "$result" ]]; then
        echo "precision,m,k,n,tile_size,rank,dense_ms,tlr_ms,speedup,dense_gflops,tlr_gflops" > "$result"
    fi
    "$executable" "$m" "$k" "$n" "$b" "$r" "$WARMUP" "$REPS" >> "$result"
}

IFS=',' read -r -a selected_precisions <<< "$PRECISIONS"
for precision in "${selected_precisions[@]}"; do
    case "$precision" in
        float|double) executable=$(compile "$precision") ;;
        *) echo "unsupported precision: $precision" >&2; exit 2 ;;
    esac

    for n in "${STRONG_SIZES[@]}"; do
        run_case "$executable" strong_scaling "$n" "$n" "$n" "$STRONG_TILE_SIZE" "$STRONG_RANK" "$precision"
    done
    for r in "${RANK_VALUES[@]}"; do
        run_case "$executable" rank_sweep "$RANK_MATRIX_SIZE" "$RANK_MATRIX_SIZE" "$RANK_MATRIX_SIZE" "$RANK_TILE_SIZE" "$r" "$precision"
    done
    for b in "${TILE_SIZES[@]}"; do
        r=$((b * RANK_TILE_NUMERATOR / RANK_TILE_DENOMINATOR))
        run_case "$executable" tile_size_sweep "$TILE_MATRIX_SIZE" "$TILE_MATRIX_SIZE" "$TILE_MATRIX_SIZE" "$b" "$r" "$precision"
    done
    for shape in "${SHAPE_CASES[@]}"; do
        read -r name m k n <<< "$shape"
        run_case "$executable" "matrix_shape_${name}" "$m" "$k" "$n" "$SHAPE_TILE_SIZE" "$SHAPE_RANK" "$precision"
    done
done

echo "KBLAS results written to $output_dir"
