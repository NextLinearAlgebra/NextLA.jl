# CUDA CompressedFTLR dense-result GEMM

This implementation adds the first executable portion of the CompressedFTLR design.
`CompressedFTLRMatrix` stores exact-rank factors in packed allocations with a prefix
offset table for each factor's own tile order. The original padded
`PaddedFTLRMatrix` lowering remains unchanged.

The CompressedFTLR default is outer/`U` factors in tile-row-major order and inner/`V`
factors in tile-column-major order. Thus a CompressedFTLR used as `B = WZᵀ` has the
desired W-row and Z-column packing without duplicating either factor.

## Implemented scope

- Nominal tile grids with optional right, bottom, and corner tails,
  `CompressedFTLRMatrix × CompressedFTLRMatrix → dense C`, with all
  logical `N/T` operand combinations.
- CUDA `cublasGemmGroupedBatchedEx`: homogeneous Float16 (with Float32
  compute), BF16 (with Float32 compute on SM80+), Float32, and Float64 factor/output storage; TF32 is available for
  Float32 storage. The installed grouped API rejects Float16 → Float32 output,
  so CompressedFTLR keeps all three stages grouped and storage-homogeneous.
- BF16 packed factors use leading dimensions rounded up to eight BF16 values,
  preserving row/column stacks while making factor starts 16-byte aligned.
  SM75 and older devices fail before cuBLAS submission with an explicit error.
- FoldRight with A outer and B outer factors packed by tile row; FoldLeft is
  selected per row run when B inner factors are packed by tile column.
- Stage 1 folds B's row panel into N; Stage 2 is tilewise; Stage 3 groups
  wide row GEMMs.
- Exact workspace bounds: the minimum is the largest one-row run and the
  maximum holds all rows. Intermediate budgets greedily pack contiguous rows.
- Each `gemm!` builds a host-only `CompressedFTLRRankPlan` once: logical A K-prefixes,
  B row/column rank sums, and byte/FLOP prefixes. Fold feasibility and range
  costs are then O(1), and the same plan supplies Stage-1/2/3 stack offsets.

## File layout

- `container/compressed_ftlr_matrix.jl`: packed exact-rank container and padded-TLR packer.
- `dense_result/ragged_schedule.jl`: rank-aware workspace profile and row packing.
- `dense_result/ragged_stages.jl`: three-stage FoldRight lowering.
- `dense_result/ragged_low_rank_terms.jl`: CompressedFTLR validation and driver.
- `src/gemm_grouped.jl` and `ext/cuda/gemm.jl`: grouped GEMM abstraction and CUDA binding. Members coalesce by shape and per-group scalars, because cuBLAS supplies `alpha`/`beta` per group.

## Deferred milestones

Buckets, AMD, and CompressedFTLR-output compression are separate follow-on
milestones. In particular, dense accumulation requires no
result-rank estimation.
