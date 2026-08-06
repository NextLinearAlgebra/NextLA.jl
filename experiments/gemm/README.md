# Dense-output TLR GEMM poster suite

All compressed timings exclude operand generation, workspace allocation,
symbolic analysis, and output clearing. Symbolic-analysis time is stored in a
separate CSV column. Poster figures use median numerical time.

## Stored measurement campaigns

### 1. Memory Pareto

`run_memory_pareto.jl` measures `N=16384`, FP16 storage with FP32
accumulation, tile sizes `b=N/8,N/16`, uniform and skewed ranks in
`[b/32,b/16]`, and all three operand layouts. Workspace targets are
`1,2,4,8,16,32,64`; increasing the target lowers workspace for TLR x TLR.

Canonical result:

```text
results/gemm/nextla/memory_pareto_fp16_n16384_rank_b32_b16.csv
```

### 2. Skewed multiprecision scaling

`run_workspace_tuning.jl` measures `N=4096,...,65536`, BF16, FP16, TF32, and
FP32, both tile sizes, all three layouts, and skewed ranks in `[b/32,b/16]`.
Every logical case is measured over the workspace grid before
`select_workspace_winners.py` chooses its lowest median time.

Canonical results:

```text
results/gemm/nextla/precision_scaling_skewed_rank_b32_b16_workspace_sweep.csv
results/gemm/nextla/precision_scaling_skewed_rank_b32_b16_best_workspace.csv
```

### 3. NextLA constant-rank tuning

The same tuner measures FP32 TLR x TLR at `r/b=1/16,1/8`, `b=N/8,N/16`,
and all five sizes. The selected fastest workspace is used in Figure 4.

Canonical results:

```text
results/gemm/nextla/constant_rank_fp32_workspace_sweep.csv
results/gemm/nextla/constant_rank_fp32_best_workspace.csv
```

### 4. KBLAS constant-rank benchmark

The KBLAS campaign has the same sizes, tile divisors, and rank ratios as the
NextLA constant-rank campaign. See `../kblas/README.md`.

## One-command NextLA reproduction

The following runs campaigns 1--3 with one warmup and four repetitions,
selects winners, and constructs the KBLAS comparison CSV:

```bash
bash experiments/gemm/run_poster_suite.sh OUTPUT_DIRECTORY
```

The output directory must not already exist.

## Individual tools

`common.jl`, `compressed_dense_support.jl`, and `operand_generation.jl` are
shared implementation modules required by both Julia benchmark harnesses.

List a benchmark grid without running it:

```bash
julia --project=experiments experiments/gemm/run_memory_pareto.jl --list
julia --project=experiments experiments/gemm/run_workspace_tuning.jl --list
```

Select fastest workspace configurations:

```bash
python3 experiments/gemm/select_workspace_winners.py RAW.csv \
    --expected-mixed-stripes 1,2,4,8,16
```

Join selected constant-rank cases with KBLAS:

```bash
python3 experiments/gemm/build_kblas_comparison.py \
    NEXTLA_BEST.csv KBLAS.csv --allow-unconfirmed
```

Validate raw or selected NextLA CSVs:

```bash
python3 experiments/gemm/validate_results.py FILE.csv [FILE2.csv ...]
```

Regenerate all final figures and numerical highlights:

```bash
.plenv/bin/python experiments/gemm/build_poster_figures.py
```

The plotter overwrites only deterministic figure artifacts. Benchmark and
derivation tools refuse to overwrite explicit CSV outputs.

## CSV definitions

The primary time is `numeric_median_ms`. The memory ratio is measured from
actual allocations:

```text
A_storage_bytes + B_storage_bytes + workspace_bytes
---------------------------------------------------
              storage(A+B)_dense
```

Dense output storage, symbolic metadata, and backend-library internals are not
included. Precision labels mean BF16/FP16 storage with FP32 accumulation,
native FP32, or TF32 Tensor Core execution from FP32 storage.
