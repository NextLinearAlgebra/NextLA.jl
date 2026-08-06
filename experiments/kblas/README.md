# KBLAS constant-rank comparison

`benchmark_tlr_tlr.cu` is a standalone KBLAS TLR GEMM benchmark. The poster
uses its FP32 `lld` mode: TLR x TLR with dense output. The executable reports
both median and minimum times after the requested warmups and repetitions.

## Dependencies

Build KBLAS with TLR, SVD, and MAGMA support. Set paths appropriate to the
target system:

```bash
export KBLAS_ROOT=/path/to/kblas-gpu
export CUDA_ROOT=/path/to/cuda
export MAGMA_ROOT=/path/to/magma
```

KBLAS should be configured with:

```make
_SUPPORT_TLR_=TRUE
_SUPPORT_SVD_=TRUE
_USE_MAGMA_=TRUE
```

Compile the benchmark against the local KBLAS, MAGMA, CUDA, cuBLAS, cuRAND,
and BLAS installations. Define `BENCH_FLOAT` for the poster's FP32 suite:

```bash
nvcc -O3 -DBENCH_FLOAT \
    -I"$KBLAS_ROOT/include" -I"$MAGMA_ROOT/include" \
    experiments/kblas/benchmark_tlr_tlr.cu \
    "$KBLAS_ROOT/lib/libkblas-gpu.a" \
    -L"$MAGMA_ROOT/lib" -lmagma \
    -lcublas -lcurand -lcudart -lopenblas \
    -o kblas_tlr_tlr_benchmark
```

Exact link flags vary between systems and KBLAS builds.

## Run the poster grid

The suite covers `N=4096,...,65536`, `b=N/8,N/16`, and constant ranks
`r=b/16,b/8`, using three warmups and ten measurements:

```bash
bash experiments/kblas/run_constant_rank_suite.sh \
    ./kblas_tlr_tlr_benchmark \
    experiments/results/gemm/kblas/my_constant_rank_fp32.csv
```

The output path must not exist. The checked-in reference result is:

```text
experiments/results/gemm/kblas/constant_rank_fp32_rank_b16_b8.csv
```

KBLAS does not expose the variable-rank distribution used by NextLA. In the
skewed-rank comparison it is therefore padded uniformly to `r=b/16`; the
controlled constant-rank comparison gives both libraries identical tile work.
