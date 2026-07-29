# Compressed Dense-Output GEMM: Symbolic Analysis and Alignment

## Purpose

This document summarizes the compressed dense-output GEMM implementation in
NextLA, with emphasis on:

- the distinction between symbolic analysis and numerical execution;
- the metadata prepared by the symbolic pass;
- exact ranks versus padded execution ranks;
- grouped-GEMM pointer and dimension requirements;
- the dense row-run alignment problem and its fix;
- the expected performance behavior and current limitations.

The two principal products discussed here are:

```text
CompressedFTLR × CompressedFTLR → dense
dense          × CompressedFTLR → dense
```

The mirrored `CompressedFTLR × dense → dense` implementation uses the same
two-stage mixed-product ideas, but partitions the dense operand by columns.

## Why a symbolic pass exists

A compressed GEMM is not a single regular matrix multiplication. Each
compressed tile is represented by two low-rank factors, and its rank can differ
from the ranks of neighboring tiles. Before launching numerical kernels, the
implementation must determine:

1. which tiles contain numerical work;
2. which low-rank association or fold to use;
3. how much intermediate workspace each run needs;
4. how output rows or columns are divided into runs;
5. the dimensions and leading dimensions of every GEMM task;
6. which tasks can share one grouped-GEMM group;
7. the device pointers for every A, B, and C task operand;
8. which tasks are safe for grouped cuBLAS and which require ordinary GEMM.

Most of this work depends on matrix geometry, rank metadata, storage layout,
transpose modes, compute policy, and workspace capacity. It does not depend on
the numerical values stored in the factors. Rebuilding it on every multiplication
is therefore unnecessary when the same matrix layout is used repeatedly.

The explicit symbolic API separates this reusable work:

```julia
analysis = analyze_compressed_gemm(
    C, A, B;
    workspace,
    transA = 'N',
    transB = 'N',
    compute,
)

gemm!(
    C, A, B;
    alpha,
    beta,
    workspace,
    transA = 'N',
    transB = 'N',
    compute,
    analysis,
)
```

The first call builds persistent scheduling and grouped-GEMM metadata. The
second call performs numerical work using that metadata.

## Symbolic versus numerical responsibilities

### Symbolic analysis

The symbolic pass currently:

- validates dimensions, backend compatibility, transpose modes, and precision;
- constructs logical operands for the requested transpose modes;
- snapshots exact and execution ranks;
- constructs rank prefixes and workspace requirements;
- selects the row-run or column-run partition;
- builds the two- or three-stage GEMM task graph;
- groups tasks with identical GEMM signatures;
- separates unsafe tasks into ordinary-GEMM fallback lists;
- creates host pointer tables;
- pins the host pointer tables;
- uploads persistent pointer tables to the GPU;
- retains the matrices and numerical workspace referenced by those pointers.

For `CompressedFTLR × CompressedFTLR`, the analysis type is
`CompressedGemmAnalysis`. For a mixed dense/compressed product, it is
`CompressedMixedGemmAnalysis`.

### Numerical execution

An analyzed numerical call:

- verifies that the analysis is still compatible with its operands and workspace;
- applies `beta` to output regions with no active low-rank contribution;
- submits prepared grouped stages in dependency order;
- substitutes the requested `alpha` and `beta` into the terminal stage;
- synchronizes only according to normal CUDA stream ordering.

It does not rebuild the row schedule or re-upload pointer tables.

Factor values may change in place between analyzed calls. The ranks, layouts,
storage identities, transpose modes, compute mode, and workspace capacity may
not change without rebuilding the analysis.

## Exact ranks and execution ranks

Each compressed tile has an exact numerical rank:

```text
r_exact
```

Tensor-core-friendly execution and safe packed storage use a padded rank:

```text
r_exec = 0                         if r_exact = 0
r_exec = round_up(r_exact, 8)      otherwise
```

For example:

```text
exact ranks:      [0, 1, 7, 8, 9]
execution ranks:  [0, 8, 8, 8, 16]
```

Factor storage reserves `r_exec` columns. Columns beyond `r_exact` are filled
with zero. Public factor views expose only exact-rank columns, while internal
GEMM views use the padded execution width.

This padding provides:

- execution dimensions compatible with the intended tensor-core kernels;
- packed factor offsets that remain aligned;
- regular rank classes for grouped GEMM;
- correct execution-FLOP accounting.

It is important to report both:

```text
exact ideal FLOPs
executed padded FLOPs
```

Their difference measures padding overhead rather than useful numerical work.

## Numerical lowering

### Compressed × compressed

For

```text
C = Acompressed × Bcompressed
```

the implementation selects a fold according to ranks, layouts, and workspace.
A row run generally uses three stages:

1. Contract the inner factors of A with the outer factors of B.
2. Fold one family of outer/inner factors into an intermediate.
3. Apply the terminal factor stack and update the dense output.

Tasks with identical transpose flags, dimensions, leading dimensions, scalar
values, and storage types are submitted through grouped GEMM.

### Dense × compressed

For a compressed tile

```text
B[k,j] = U[k,j] * V[k,j]'
```

the mixed product is evaluated in two stages:

```text
T[k,j] = A[:,k] * U[k,j]
C[:,j] = concatenated(T[:,j]) * stacked(V[:,j])'
```

With limited workspace, A and C are divided into dense row runs. For each run:

1. Stage 1 computes all dense-block-times-outer-factor products.
2. Stage 2 applies one stacked inner-factor product per output tile column.

The workspace needed by a run of height `h` is:

```text
h × total_execution_rank
```

This produces the initial capacity:

```julia
raw_height = floor(workspace_elements / total_execution_rank)
```

That formula controls memory, but by itself it does not guarantee that every
dense subview is safe for grouped cuBLAS.

## Grouped-GEMM alignment requirements

The CUDA grouped adapter requires every task's A, B, and C starting pointer to
be 16-byte aligned. If a task is unsafe, it is separated from the grouped
submission and executed through ordinary GEMM.

Execution-rank padding protects compressed factor offsets, but not every task
operand is a compressed factor. Mixed GEMM tasks also reference:

- row views into the standalone dense operand;
- row views into the dense output;
- views into the numerical workspace.

These views can begin at unaligned addresses even when every rank is a multiple
of eight.

## The row-run alignment problem

Julia matrices are column-major. The byte address of a dense view beginning at
`(row, column)` is:

```text
base
+ (row - 1) * sizeof(T)
+ (column - 1) * leading_dimension * sizeof(T)
```

Consider FP16 and a raw row-run height of 25:

```text
run 1 starts at row 1
run 2 starts at row 26
```

The second run begins at a row offset of 25 elements:

```text
25 × 2 bytes = 50 bytes
```

Fifty is not divisible by sixteen, so dense A and C views in the second run are
not 16-byte aligned. The grouped adapter routes those tasks to ordinary GEMM.
With hundreds of row runs and hundreds of tasks per run, this produces hundreds
or thousands of individual kernel submissions.

This explains the characteristic benchmark pattern:

- constant-rank cases often happen to select aligned run heights;
- variable ranks change `total_execution_rank`;
- the changed total produces heights such as 21, 25, or 46;
- most subsequent row origins are unaligned;
- numerical time increases by orders of magnitude.

The slowdown is real execution behavior, not timing noise.

## Alignment-aware row-run sizing

For a 16-byte boundary, the row alignment quantum is:

```julia
alignment_rows = 16 ÷ gcd(16, sizeof(T))
```

Therefore:

```text
FP16/BF16: 8 rows
FP32:      4 rows
FP64:      2 rows
```

The scheduler selects the largest aligned run that fits:

```julia
capacity = floor(workspace_elements / total_execution_rank)

if capacity >= total_rows
    height = total_rows
elseif capacity >= alignment_rows
    height = floor(capacity / alignment_rows) * alignment_rows
else
    height = capacity
end
```

A single full-height run is safe because it starts at the aligned allocation
base. A final short run is also safe: its starting row is determined by the
preceding aligned full runs, while its own length does not affect another run.

If the workspace cannot hold one alignment quantum, the implementation
preserves the low-workspace contract and lets unsafe tasks use the existing
ordinary-GEMM fallback.

### Why this is optimal within the row-run model

Let:

```text
W = workspace elements
R = total execution rank
a = alignment quantum in rows
hcap = floor(W / R)
```

Every nonfinal run length must be a multiple of `a`, otherwise the next run has
an unaligned origin. Under the fixed budget, no run may exceed `hcap`.

The largest legal height is therefore:

```text
h = a * floor(hcap / a)
```

Every legal schedule needs at least `ceil(total_rows / h)` runs, and repeated
runs of height `h` followed by one tail attain that lower bound. The selected
height is thus optimal for minimizing run count under:

- fixed workspace;
- row-based partitioning;
- no dense-panel copy;
- 16-byte-aligned grouped operands.

It is not a proof that row partitioning is globally optimal. A future
column-batched algorithm may be faster when enough workspace is available.

## Pointer-mode management

cuBLAS grouped GEMM uses host arrays for group scalars, while ordinary fallback
GEMMs use the normal device-pointer mode expected by CUDA.jl.

The mixed analysis records whether any prepared stage contains fallback work:

```text
analysis.has_fallback
```

For a regular fallback-free analysis:

1. pointer mode is set to host once;
2. all runs and stages are submitted;
3. pointer mode is restored to device once.

If an analysis contains fallback tasks, per-stage management is retained so
grouped and ordinary GEMMs receive the correct pointer mode.

This avoids two cuBLAS state changes for every stage in the regular optimized
path.

## Analysis lifetime and staleness

Prepared pointer tables contain addresses into:

- the output matrix;
- compressed factor storage;
- the standalone dense operand;
- the numerical workspace.

An analysis is therefore bound to those storage objects. Numerical values can
change in place, but storage reallocation invalidates the pointers.

Before numerical submission, the implementation checks:

- matrix object identity;
- workspace identity and capacity;
- transpose modes;
- compute policy;
- exact-rank snapshot;
- execution-rank snapshot;
- whether the analysis has been closed.

Calling `close(analysis)` releases backend-owned prepared descriptors and their
device pointer tables.

## Understanding the benchmark columns

The primary speedup is:

```text
speedup = dense_time / analyzed_compressed_time
```

Consequently:

```text
speedup > 1    compressed execution is faster
speedup = 1    tie
speedup < 1    compressed execution is slower
speedup = 0.01 compressed execution is about 100× slower
```

A value such as `0.01` is not a 0.01% improvement.

The explicit path should be reported with separate values:

```text
analysis time
analyzed numerical time
cold total = analysis + analyzed numerical
amortization crossover
```

If another path takes `alternative_time` per call, the number of reused
executions required to recover analysis cost is:

```text
ceil(analysis_time / (alternative_time - analyzed_time))
```

when `analyzed_time < alternative_time`.

## Current symbolic-pass efficiency

The symbolic pass is reusable, but the mixed implementation still creates
prepared pointer storage independently for each run and stage.

For a 16×16 grid with 256 row runs, this can mean approximately:

```text
512 prepared stages
1,536 pinned host pointer arrays
1,536 device pointer-array allocations/uploads
```

This is why mixed analysis can take hundreds of milliseconds even when the
analyzed numerical phase is much shorter.

Flattening all run-stage pointer tables into shared host-pinned and device pools
could substantially reduce analysis time, but should have little effect on the
already analyzed numerical time. It is therefore a cold-start optimization,
not the fix for the original numerical collapse.

That pooling work is intentionally deferred until corrected alignment and
pointer-mode results establish whether symbolic cold time is important enough
for the intended reuse model.

## Expected effect of the implemented fixes

### Alignment-aware row runs

Benefits:

- keeps regular variable-rank tasks in grouped GEMM;
- removes the catastrophic ordinary-GEMM launch loop;
- respects the exact workspace budget;
- changes constant-rank cases only when their height was unaligned.

Costs:

- may leave a few rows' worth of workspace unused;
- may slightly increase the number of runs;
- cannot create an aligned grouped schedule below one alignment quantum.

Across the initial recorded cases, alignment rounding increased run count by
about 0.7% on average. The worst observed case changed a height of 21 to 16,
increasing run count by about 31%. That is still preferable to executing nearly
all tasks as individual GEMMs.

### One pointer-mode scope

Benefits:

- replaces per-stage cuBLAS pointer-mode changes with one pair per complete GEMM;
- reduces driver overhead in analyses with many runs;
- matches the intended analyzed execution model.

Cost:

- analyses containing fallback work still need the slower safe path.

## Relevant implementation files

- `compressed_ftlr/analysis.jl`  
  Explicit analysis and execution for compressed × compressed.

- `compressed_ftlr/mixed_dense.jl`  
  Two-stage mixed products, aligned row-run selection, mixed analysis, and
  fallback-aware pointer-mode handling.

- `compressed_ftlr/stages.jl`  
  Construction of compressed × compressed numerical stages.

- `gemm_grouped.jl`  
  Backend-independent grouped task preparation and fallback splitting.

- `ext/cuda/gemm.jl`  
  CUDA grouped-GEMM descriptors, alignment checks, pointer uploads, and cuBLAS
  submission.

- `experiments/compressed_dense.jl`  
  Compressed × compressed benchmark.

- `experiments/dense_compressed.jl`  
  Dense × compressed benchmark.

## Summary

The implementation separates a compressed GEMM into:

```text
symbolic phase:
    geometry + ranks + workspace
        → runs + stages + groups + persistent pointers

numerical phase:
    factor values + alpha + beta
        → prepared grouped submissions
```

Execution-rank padding and row-run alignment solve different problems:

```text
rank padding
    aligns compressed storage and tensor-core dimensions

row-run alignment
    aligns dense A/C subview origins
```

Both are required for the regular mixed product to remain entirely in grouped
GEMM. The alignment-aware scheduler is the optimal fixed-workspace solution
within the current row-run algorithm, while shared symbolic pointer pools and a
column-batched traversal remain possible future optimizations.
