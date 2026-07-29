# Compressed FTLR dense-output experiment

From the repository root:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/compressed_dense.jl
```

This runs only `CompressedFTLRMatrix × CompressedFTLRMatrix → dense`, using a
reusable symbolic analysis and no experimental pipeline. The default grid is:

- matrix sizes `4096, 8192, 16384, 32768`;
- tile sizes `N/16, N/8, N/4`;
- constant ranks `b/32`, `b/16`, and `b/8`;
- uniform and low-rank-skewed distributions over `[b/32,b/8]`;
- FP16 storage with FP32 compute, FP32 storage with TF32 compute, and full FP32;
- one warmup and three measured repetitions.

Results are appended to
`experiments/results/compressed_dense_v2.csv`. Completed case
IDs are skipped when restarting. Environment variables prefixed
`NEXTLA_DENSE_` can override the defaults; see
`compressed_dense.jl`.

The fixed-rank padded baseline uses tile-row-major A, tile-column-major B,
maximum reusable numerical workspace, and no symbolic phase:

```bash
julia --project=experiments experiments/padded_dense.jl
```

It benchmarks only ranks `b/32`, `b/16`, and `b/8`, and writes to
`experiments/results/padded_dense_v2.csv`.

`grouped_gemm.jl` and `KBLAS/` are independent primitive/baseline
microbenchmarks and are intentionally retained.

The matching dense-left experiment uses the same sizes, tiles, rank profiles,
and precision modes. It reports transient scheduling, symbolic analysis, and
execution with reused two-stage grouped-GEMM descriptors:

```bash
julia --project=experiments experiments/dense_compressed.jl
```

Results are appended to `experiments/results/dense_compressed_v2.csv`.

The scheduler sweep uses `N = 2^11,…,2^15`, `b=N/8`, `r=b/8`, and rows/run
from one through eight:

```bash
julia --project=experiments experiments/rows_per_run.jl
```

Results are appended to `experiments/results/rows_per_run_v2.csv`.

Run all four sequentially with:

```bash
bash experiments/run_all.sh
```
