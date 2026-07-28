# CUDA BCLR dense-result GEMM

This implementation adds the first executable portion of the BCLR design.
`BCLRMatrix` stores exact-rank factors in packed allocations with a prefix
offset table for each factor's own tile order. The original padded
`TLRMatrix` lowering remains unchanged.

The BCLR default is outer/`U` factors in tile-row-major order and inner/`V`
factors in tile-column-major order. Thus a BCLR used as `B = WZᵀ` has the
desired W-row and Z-column packing without duplicating either factor.

## Implemented scope

- Regular full tile grids, `BCLRMatrix × BCLRMatrix → dense C`, with all
  logical `N/T` operand combinations.
- CUDA `cublasGemmGroupedBatchedEx`: homogeneous Float16 (with Float32
  compute), Float32, and Float64 factor/output storage; TF32 is available for
  Float32 storage. The installed grouped API rejects Float16 → Float32 output,
  so BCLR keeps all three stages grouped and storage-homogeneous.
- FoldRight with A outer and B outer factors packed by tile row; FoldLeft is
  selected per row run when B inner factors are packed by tile column.
- Stage 1 folds B's row panel into N; Stage 2 is tilewise; Stage 3 groups
  wide row GEMMs.
- Exact workspace bounds: the minimum is the largest one-row run and the
  maximum holds all rows. Intermediate budgets greedily pack contiguous rows.
- Each `gemm!` builds a host-only `BCLRRankPlan` once: logical A K-prefixes,
  B row/column rank sums, and byte/FLOP prefixes. Fold feasibility and range
  costs are then O(1), and the same plan supplies Stage-1/2/3 stack offsets.

## File layout

- `container/bclr_matrix.jl`: packed exact-rank container and padded-TLR packer.
- `dense_result/ragged_schedule.jl`: rank-aware workspace profile and row packing.
- `dense_result/ragged_stages.jl`: three-stage FoldRight lowering.
- `dense_result/ragged_low_rank_terms.jl`: BCLR validation and driver.
- `src/gemm_grouped.jl` and `ext/cuda/gemm.jl`: grouped GEMM abstraction and CUDA binding. Members coalesce by shape and per-group scalars, because cuBLAS supplies `alpha`/`beta` per group.

## Deferred milestones

Tails/boundaries, buckets, AMD, and BCLR-output compression are separate
follow-on milestones. In particular, dense accumulation requires no
result-rank estimation.
