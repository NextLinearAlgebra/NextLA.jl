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
julia --project=experiments experiments/gemm/workspace_tuning_sweep.jl --list
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
- one compressed×compressed work unit (maximum useful workspace) and one
  tile-wide stripe for either mixed layout;
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
`NEXTLA_PRECISION_ANALYSIS_REPS`, `NEXTLA_PRECISION_RUNS`,
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
- one fixed rank band `[b/16,b/8]`;
- run targets `1,2,4,8,16,32,64` for compressed×compressed: `runs=1` is
  maximum workspace and increasing runs progressively lowers the budget,
  including column-blocked schedules above the tile-grid dimension;
- the same generic levels are interpreted as stripe counts for mixed layouts
  and capped at their full-width tile-grid dimension;
- all three operand layouts, one warmup, and five measured executions.

Change the fixed point or rank bands with, for example:

```bash
NEXTLA_MEMORY_N=32768 \
NEXTLA_MEMORY_PRECISION=tf32 \
NEXTLA_MEMORY_RANK_BANDS=16:8 \
NEXTLA_MEMORY_WORKSPACE_LEVELS=1,2,4,8,16,32,64 \
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
`REPS`, `ANALYSIS_REPS`, `RANK_BANDS`, `WORKSPACE_LEVELS`,
`EXECUTION_POLICY`, `FILL`, `SEED`, and `FILTER`.

## Workspace-tuned scaling and KBLAS comparison

Workspace is an implementation tuning parameter. The publication workflow has
three deliberately separate passes:

1. `workspace_tuning_sweep.jl` records every candidate while reusing the same
   A, B, and C allocations for all workspace levels of one logical case;
2. `select_workspace_winners.py` selects the lowest numerical median for every
   complete `(N, precision, layout, q, rank, distribution, policy)` key;
3. `workspace_confirmation_sweep.jl` reconstructs and independently remeasures
   only the selected configurations.

The raw tuning measurements are never overwritten or discarded. The selector
also accepts `--max-memory-ratio RATIO` to tune under an explicit storage cap.
By default, compressed×compressed tests run targets
`NEXTLA_TUNING_WORKSPACE_LEVELS=1,2,4,8,16,32,64`; these move from maximum
workspace toward the column-blocked floor. Mixed layouts use
`NEXTLA_TUNING_MIXED_STRIPES=all`, meaning every stripe count from 1 through
`q`. An explicit list or range such as `1:64` is accepted for either setting.
Winner selection checks these default grids, catching interrupted files. When
using custom grids, give the selector matching `--expected-runs` and/or
`--expected-mixed-stripes` values; use `--allow-incomplete` only for an
intentional subset.

Run the proportional-rank scaling tuner with:

```bash
NEXTLA_TUNING_SIZES=4096,8192,16384,32768,65536 \
NEXTLA_TUNING_PRECISIONS=bf16,fp16,fp32,tf32 \
NEXTLA_TUNING_TILE_DIVISORS=16,8 \
NEXTLA_TUNING_RANK_BANDS=32:16 \
NEXTLA_TUNING_DISTRIBUTIONS=uniform,skewed \
NEXTLA_TUNING_LAYOUTS=compressed_dense,dense_compressed,compressed_compressed \
NEXTLA_TUNING_WORKSPACE_LEVELS=1,2,4,8,16,32,64 \
NEXTLA_TUNING_MIXED_STRIPES=all \
NEXTLA_TUNING_WARMUP=1 \
NEXTLA_TUNING_REPS=5 \
NEXTLA_TUNING_ANALYSIS_REPS=1 \
julia --project=experiments experiments/gemm/workspace_tuning_sweep.jl
```

This full grid contains 2,500 rows and is intentionally a substantial tuning
run. Use `NEXTLA_TUNING_FILTER` or smaller size/precision lists for staged runs.
Select and independently confirm its winners with:

```bash
python3 experiments/gemm/select_workspace_winners.py RAW_TUNING.csv

NEXTLA_CONFIRM_WARMUP=3 \
NEXTLA_CONFIRM_REPS=10 \
NEXTLA_CONFIRM_ANALYSIS_REPS=3 \
julia --project=experiments \
    experiments/gemm/workspace_confirmation_sweep.jl WINNERS.csv
```

Generate scaling time, speedup, and achieved-ceiling figures from the confirmed
CSV using `plot_results.py` exactly as for a precision sweep. Each speedup plot
retains the case-specific dashed FLOP-ratio ceiling.

The checked-in KBLAS results use FP32 compressed×compressed multiplication at
constant ratios `r/b = 1/16` and `1/8`. Produce exactly the corresponding
NextLA tuning grid with:

```bash
NEXTLA_TUNING_SIZES=4096,8192,16384,32768,65536 \
NEXTLA_TUNING_PRECISIONS=fp32 \
NEXTLA_TUNING_TILE_DIVISORS=16,8 \
NEXTLA_TUNING_RANK_BANDS=16:16,8:8 \
NEXTLA_TUNING_DISTRIBUTIONS=uniform \
NEXTLA_TUNING_LAYOUTS=compressed_compressed \
NEXTLA_TUNING_WORKSPACE_LEVELS=1,2,4,8,16,32,64 \
NEXTLA_TUNING_WARMUP=1 \
NEXTLA_TUNING_REPS=5 \
NEXTLA_TUNING_ANALYSIS_REPS=1 \
julia --project=experiments experiments/gemm/workspace_tuning_sweep.jl
```

After selection and confirmation, join and plot the matched cases:

```bash
python3 experiments/gemm/compare_kblas.py \
    CONFIRMED.csv \
    experiments/results/gemm/kblas_tlr_tlr_fp32_fixed_rank_b16_b8.csv

source ../.plenv/bin/activate
python3 experiments/gemm/plot_kblas_comparison.py JOINED.csv \
    --formats png,pdf,svg
```

For absolute constant ranks instead of ranks proportional to `b`, set
`NEXTLA_TUNING_RANK_BANDS=` and, for example,
`NEXTLA_TUNING_FIXED_RANKS=32,48,64`.

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

The defaults use `N=16384`, `b=N/16`, ranks `[b/32,b/8]`, FP16, one target
run, and ten repetitions. Controls use the `NEXTLA_ABLATION_` prefix:
`N`, `TILE_DIVISOR`, `DISTRIBUTIONS`, `POLICIES`, `RANK_BAND`, `PRECISION`,
`WARMUP`, `REPS`, `ANALYSIS_REPS`, `RUNS`, `FILL`, and `SEED`.

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

Generate the exploratory plot set for every result CSV with:

```bash
source ../.plenv/bin/activate
python3 experiments/gemm/plot_results.py
```

Each run gets its own directory under `experiments/figures/gemm/`, so results
with different ranks or sweep settings are never mixed. To plot selected files
or emit publication formats:

```bash
python3 experiments/gemm/plot_results.py \
    experiments/results/gemm/precision_sweep__*.csv \
    experiments/results/gemm/memory_sweep__*.csv \
    --formats png,pdf,svg
```

The plotting pass creates speedup, numerical-time, and achieved-ceiling views.
Every speedup figure overlays a dashed `dense_flops/executed_flops` curve. The
executed FLOPs include actual execution-rank padding and workspace-dependent
fold selection; the dashed curve is therefore specific to each rank map, tile
size, precision case, and workspace schedule.

Run the first two sweeps sequentially (and optionally the ablation) with:

```bash
bash experiments/gemm/run_sweeps.sh
NEXTLA_RUN_ABLATION=1 bash experiments/gemm/run_sweeps.sh
```
