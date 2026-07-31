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

## Execution-rank bucketing ablation

The focused bucketing experiment compares identical logical rank maps under
`:exact`, `:q8`, `:q16`, and `:pow2` execution capacities:

```bash
julia --project=experiments experiments/rank_bucketing.jl
```

Its defaults are deliberately small enough for a local CUDA GPU:

- matrix sizes `4096,8192`;
- a `16 × 16` tile grid;
- ranks sampled uniformly from `1:64`;
- FP16 storage with FP32 accumulation;
- four output rows per run;
- one warmup and ten measured numerical executions.

Summary results are written to `experiments/results/rank_bucketing.csv`.
Per-stage grouped shapes, group sizes, and ordinary-GEMM fallback members are
written to `experiments/results/rank_bucketing_groups.csv`. The latter is
important for interpreting `:exact`: exact rank stacks can create unaligned
intermediate subviews in the current lowering, so its fallback count must not
be confused with the effect of shape count alone. Compare `:q8` with `:q16`
and `:pow2` to isolate the aligned shape-bucketing trade-off.

The configuration uses `NEXTLA_BUCKET_*` environment variables. For example:

```bash
NEXTLA_BUCKET_SIZES=4096 \
NEXTLA_BUCKET_REPS=10 \
NEXTLA_BUCKET_POLICIES=exact,q8,q16,pow2 \
julia --project=experiments experiments/rank_bucketing.jl
```

For the H100 sweep, retain the fixed `16 × 16` grid and expand the sizes; this
keeps the rank distribution and row-run policy comparable while changing only
the problem scale:

```bash
NEXTLA_BUCKET_SIZES=4096,8192,16384,32768 \
NEXTLA_BUCKET_TILE_DIVISOR=16 \
NEXTLA_BUCKET_MIN_RANK=1 NEXTLA_BUCKET_MAX_RANK=64 \
NEXTLA_BUCKET_ROWS=4 NEXTLA_BUCKET_WARMUP=1 NEXTLA_BUCKET_REPS=10 \
julia --project=experiments experiments/rank_bucketing.jl
```

## Poster plots

Generate the strong-scaling figure, workspace-sensitivity figure, and combined
poster panel from the checked-in CSV files with:

```bash
python3 -m pip install --upgrade -r experiments/requirements-plot.txt
python3 experiments/plot_poster.py
```

The plotting script requires Matplotlib and writes PDF, SVG, and 300-DPI PNG
versions to `experiments/figures/`. Its defaults select the `16 × 16` skewed-rank
compressed experiment for the central plot and FP16/FP32-accumulate workspace
results at `N = 4096, 16384, 65536` for the supporting plot. Run
`python3 experiments/plot_poster.py --help` for selection and output options.
Performance ratios use the best observed timings: `dense_min_ms /
analyzed_min_ms` for strong scaling and `numeric_min_ms` for the workspace
sweep.

Run all four sequentially with:

```bash
bash experiments/run_all.sh
```

## Stage-2 fusion/layout model

Before implementing a different intermediate layout, run the CPU-only
roofline filter:

```bash
julia --project=experiments experiments/stage2_layout_tradeoff.jl
```

It compares today's FoldRight pipeline with (1) changing Stage 2 alone, which
requires permutations of both `S` and `T`, and (2) a coupled redesign that
also fuses Stage 1 across scheduled output rows and therefore emits `S` in the
layout consumed by the new Stage 2.  The coupled version pays only the `T`
permutation needed to retain today's one-wide-GEMM-per-row Stage 3.  The model
also prints the correctly mirrored FoldLeft opportunity: fuse Stage 2 across
`j` for fixed `(i,k)` to reuse `U`, then permute `T` for its current Stage 3.

Set `LAYOUT_Q`, `LAYOUT_BM`, `LAYOUT_BN`, `LAYOUT_RMAX`, `LAYOUT_ROWS`,
`LAYOUT_ELEMENT_BYTES`, `LAYOUT_PEAK_TFLOPS`, and `LAYOUT_BW_TBPS` to match a
target.  This is a metadata/roofline screen; a predicted win still requires a
GPU prototype and measurement.
