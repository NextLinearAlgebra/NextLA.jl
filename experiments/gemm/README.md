# Dense-output GEMM experiment suite

These runners cover the three compressed operand layouts with a dense output:

- `compressed_dense`: compressed A × dense B;
- `dense_compressed`: dense A × compressed B;
- `compressed_compressed`: compressed A × compressed B.

Every compressed timing uses a reusable symbolic analysis and reusable numerical
workspace. Operand generation, workspace allocation, symbolic analysis, and
output clearing are excluded from `numeric_*_ms`. The CSV retains analysis time
separately and stores every numerical sample as a semicolon-separated field.

## Setup and dry run

From the repository root:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/gemm/precision_sweep.jl --list
julia --project=experiments experiments/gemm/memory_sweep.jl --list
julia --project=experiments experiments/gemm/rank_bucketing_ablation.jl --list
```

BF16 and TF32 cases require an NVIDIA SM80 or newer GPU. The runner checks this
before creating an output file.

## 1. Size and precision sweep

```bash
julia --project=experiments experiments/gemm/precision_sweep.jl
```

Defaults:

- `N = 4096, 8192, 16384, 32768, 65536`;
- BF16/FP32 accumulation, FP16/FP32 accumulation, FP32, and TF32;
- tile sizes `N/16` and `N/8`;
- uniform and low-rank-skewed exact ranks over `[b/16,b/8]`;
- all three operand layouts plus one dense baseline per `(N, precision)`;
- q8 physical execution-rank padding;
- four tile rows per compressed×compressed workspace run and one tile-wide
  stripe for either mixed layout;
- one warmup and three measured executions.

The default matrix has 260 CSV rows: 20 dense baselines and 240 compressed
cases. Common overrides are:

```bash
NEXTLA_PRECISION_SIZES=4096,8192 \
NEXTLA_PRECISION_PRECISIONS=bf16,fp16 \
NEXTLA_PRECISION_REPS=5 \
julia --project=experiments experiments/gemm/precision_sweep.jl
```

Set `NEXTLA_PRECISION_MIN_RANK_DIVISOR=8` and
`NEXTLA_PRECISION_MAX_RANK_DIVISOR=4` to run the higher `[b/8,b/4]` interval.
Other controls are `NEXTLA_PRECISION_TILE_DIVISORS`,
`NEXTLA_PRECISION_DISTRIBUTIONS`, `NEXTLA_PRECISION_LAYOUTS`,
`NEXTLA_PRECISION_ANALYSIS_REPS`, `NEXTLA_PRECISION_ROWS`,
`NEXTLA_PRECISION_MIXED_STRIPES`, `NEXTLA_PRECISION_EXECUTION_POLICY`,
`NEXTLA_PRECISION_FILL`, `NEXTLA_PRECISION_SEED`, and
`NEXTLA_PRECISION_FILTER`.

## 2. Fixed-size speedup versus memory ratio

```bash
julia --project=experiments experiments/gemm/memory_sweep.jl
```

Defaults:

- fixed `N = 16384` and FP16 storage with FP32 accumulation;
- tile sizes `N/16` and `N/8`;
- uniform and skewed ranks;
- rank bands `[b/64,b/32]`, `[b/32,b/16]`, `[b/16,b/8]`, and
  `[b/8,b/4]`;
- all three operand layouts, one warmup, and five measured executions.

Change the fixed point or rank bands with, for example:

```bash
NEXTLA_MEMORY_N=32768 \
NEXTLA_MEMORY_PRECISION=tf32 \
NEXTLA_MEMORY_RANK_BANDS=32:16,16:8,8:4 \
julia --project=experiments experiments/gemm/memory_sweep.jl
```

The plotted x coordinate should be the recorded `memory_ratio`, not a rank-based
estimate. It is computed from actual allocated factor and workspace lengths:

```text
(A_storage_bytes + B_storage_bytes + workspace_bytes)
-----------------------------------------------------
                  2 * N^2 * sizeof(T)
```

For a dense operand, its storage contribution is `N^2*sizeof(T)`; for a
compressed operand, it is the physical outer-plus-inner factor storage,
including execution-rank padding. Dense output C, rank metadata, and backend
library internals are intentionally excluded, matching the requested A+B
operand comparison. The component byte counts and denominator are all present
in the CSV so the definition can be audited or recomputed.

The remaining controls use the `NEXTLA_MEMORY_` prefix and parallel those of
the precision sweep: `TILE_DIVISORS`, `DISTRIBUTIONS`, `LAYOUTS`, `WARMUP`,
`REPS`, `ANALYSIS_REPS`, `ROWS`, `MIXED_STRIPES`, `EXECUTION_POLICY`, `FILL`,
`SEED`, and `FILTER`.

## 3. Recommended bar-plot ablation

```bash
julia --project=experiments experiments/gemm/rank_bucketing_ablation.jl
```

This is the most useful compact third figure: compare `exact`, `q8`, `q16`, and
`pow2` execution-rank policies for the same exact rank maps. Plot numerical time
as grouped bars (uniform versus skewed), annotate each bar with
`memory_ratio` or `padding_waste_pct`, and mark `has_fallback=true`. This makes
the launch-shape regularity versus padded work/storage trade-off visible. Exact
rank should not be presented as a pure no-padding baseline without the fallback
annotation, because arbitrary exact ranks can trigger ordinary-GEMM alignment
fallbacks in the current lowering.

The defaults use `N=16384`, `b=N/16`, ranks `[b/32,b/8]`, FP16, four rows per
run, and ten repetitions. Controls use the `NEXTLA_ABLATION_` prefix:
`N`, `TILE_DIVISOR`, `DISTRIBUTIONS`, `POLICIES`, `RANK_BAND`, `PRECISION`,
`WARMUP`, `REPS`, `ANALYSIS_REPS`, `ROWS`, `FILL`, and `SEED`.

## Output safety and CSV conventions

Each invocation creates a new timestamp-and-PID-named file under
`experiments/results/gemm/`. It never appends to the checked-in legacy results.
An explicit output can be selected with `NEXTLA_PRECISION_OUTPUT`,
`NEXTLA_MEMORY_OUTPUT`, or `NEXTLA_ABLATION_OUTPUT`; the runner aborts if that
path already exists.

Dense baselines are first-class `record_kind=baseline` rows. Compressed rows
refer to them through `baseline_case_id` and also contain the joined dense
median/minimum and derived speedups. Use median timings for the primary plot;
`*_min_ms` is retained for best-case diagnostics. Precision names mean:

- `bf16`: BF16 storage, FP32 accumulation;
- `fp16`: FP16 storage, FP32 accumulation;
- `fp32`: FP32 storage and FP32 compute;
- `tf32`: FP32 storage with TF32 tensor-core compute.

Synthetic factors are random by default and setup is untimed. `FILL=constant`
is available for faster setup and `FILL=zeros` for plumbing tests; GEMM shapes
and launch work do not depend on the values.

Before combining or plotting one or more result files, validate their baseline
joins and recomputed memory ratios/speedups with:

```bash
python3 experiments/gemm/check_results.py experiments/results/gemm/*.csv
```

Run the first two sweeps sequentially (and optionally the ablation) with:

```bash
bash experiments/gemm/run_sweeps.sh
NEXTLA_RUN_ABLATION=1 bash experiments/gemm/run_sweeps.sh
```
