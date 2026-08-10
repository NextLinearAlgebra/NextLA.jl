# CompressedFTLR dense-result lowering

This directory contains every lowering with at least one exact-rank compressed
operand and a dense destination.

## Files

- `metadata.jl` computes rank prefixes and output tile extents.
- `costs.jl` derives exact workspace and arithmetic costs for FoldRight and
  FoldLeft.
- `schedule.jl` packs the output into rectangular `DenseResultRun`s under the
  workspace budget.
- `three_stage.jl` lowers compressed × compressed.
- `two_stage.jl` lowers the two products with one dense operand as fixed-fold
  specializations.
- `analysis.jl` owns the compressed × compressed symbolic-analysis type and
  compatibility checks.
- `validation.jl` validates packed layouts, backends, and precision policy.

The shared prepared-run lifecycle and numerical executor live one level above
in `dense_result/runs.jl`.

## Scheduling model

For compressed × compressed, Stage 1 always forms

```text
S[i,k,j] = V[i,k]' W[k,j].
```

One scheduled rectangle then chooses:

```text
FoldRight: T[i,k,j] = S[i,k,j] Z[k,j]'
           C[i,J]   = [U[i,1] ... U[i,q]] T[i,J]

FoldLeft:  T[i,k,j] = U[i,k] S[i,k,j]
           C[I,j]   = T[I,j] [Z[1,j]' ... Z[q,j]']
```

The greedy scheduler takes the widest feasible column block and packs the
largest contiguous row runs admitted by the exact byte profile. Runs execute
sequentially and reuse one numerical arena. The public minimum is the smallest
single-column rectangle that can run; the maximum admits the cheapest full
region fold.

For a product with one dense operand, one side of this graph is already dense:

- compressed × dense is always FoldRight and partitions physical output
  columns;
- dense × compressed is always FoldLeft and partitions physical output rows.

Their `TwoStageCompressedPlan` is the corresponding one-sided rank prefix and
fixed fold. Both produce the same `DenseResultRunTasks` type as the full
three-stage lowering.

## Symbolic analysis

`analyze_compressed_gemm` prepares persistent grouped descriptors. An analysis
is bound to the output, operands, numerical workspace, transpose modes, compute
policy, and exact/execution rank snapshots. Factor values and numerical scalars
may change in place between calls. `close(analysis)` releases backend-owned
descriptors and pointer tables.

The one-shot API deliberately constructs, executes, and closes the same
analysis instead of maintaining a second task builder.
