# TLR GEMM design

The dense-output implementation has two direct paths:

```text
TLRMatrix × TLRMatrix
  -> CompressedFTLR off-diagonal product
  -> O_A D_B, D_A O_B, and D_A D_B grouped updates

CompressedFTLRMatrix × CompressedFTLRMatrix
  -> exact-rank symbolic schedule
  -> three grouped-GEMM stages per run
```

`PaddedFTLRMatrix × PaddedFTLRMatrix → PaddedFTLRMatrix` remains a separate ARA
path. It shares logical operand and factor-access machinery, but it does not
participate in the dense-output schedule above.

## Canonical operands

`LogicalTLROperand{Op}` is a zero-copy logical view of `op(A)`. It resolves `N/T` once,
before execution:

- dimensions, tile grids, and tile order follow the logical orientation;
- right and bottom panels swap under transpose;
- low-rank outer/inner factors swap, so every logical tile remains `outer * inner'`;
- dense diagonal/corner tiles retain physical storage plus their BLAS operation flag.

For `TLRMatrix`, the logical view also provides dense diagonal tile references
with the BLAS operation needed after transposition. Off-diagonal factors are
obtained from its full-grid compressed child, whose diagonal ranks are zero.

`InteriorOperand` remains a storage-addressing helper for the regular-grid
padded-output ARA implementation. It does not own storage.

## Regular low-rank core

For `A_ik = U_ik V_ik'` and `B_kj = W_kj Z_kj'`, the shared core computes:

```text
Stage 1: S_ikj = V_ik' W_kj
Stage 2: T_ikj = S_ikj Z_kj'       (FoldRight)
Stage 3: C_ij  += U_ik T_ikj
```

`FoldLeft` instead forms `U_ik S_ikj` and stacks `Z_kj` in Stage 3. Fold selection is
available only where the required reduction stack is complete and contiguous.

The exact-rank planner chooses FoldRight or FoldLeft for each rectangular output
run according to packing validity, workspace, and active-rank cost. Runs may be
split below a full tile row into contiguous column blocks, so the minimum
workspace is the true one-block floor. CPU executes each grouped submission as
an ordinary GEMM loop; CUDA submits it through cuBLAS grouped GEMMEx.

## Dense-output paths

`TLRMatrix` is represented as `O + D`, where `O` is one full-grid
`CompressedFTLRMatrix` and `D` is separate dense diagonal storage. The driver
first computes `O_A O_B` with the compressed path, applying the caller's
`beta` exactly once. It then accumulates `O_A D_B`, `D_A O_B`, and `D_A D_B`.
The two cross terms use budgeted two-stage grouped GEMMs; the diagonal product
is one heterogeneous grouped submission. Ragged boundary tile dimensions come
from the same full-grid geometry and need no boundary categories or scheduler.

## Workspace contract

Dense-output GEMM exposes exact reusable-arena bounds. For `TLRMatrix`, the
minimum is the maximum of the compressed-product minimum and the largest
single dense-diagonal cross-term intermediate. The maximum is the maximum of
the compressed-product maximum and enough aligned storage to batch every
intermediate in either cross pass. Each phase resets and reuses the same arena;
the requirements are not added together.

`gemm!` requires `workspace`, either an integer byte count or a reusable
`DenseGemmWorkspace`. A minimum-sized workspace greedily chunks cross terms;
larger workspaces reduce grouped submissions. There are no interior and
boundary arena partitions or execution streams; `gemm!` no longer takes a
workspace-policy argument.

The bound covers numerical scratch allocated by the TLR GEMM implementation.
Output storage, persistent TLR factors, host batch descriptors, and
backend-library internal allocations are outside it. Dynamic lending of the
interior slice to the auxiliary stream is deferred until profiling justifies
an event boundary.

## Precision

Operand factor/intermediate storage follows the TLR operand element type. Output
storage follows `C`. GEMM scalars use the selected compute precision. Dense
output relies on the grouped interface: a CPU loop or CUDA grouped GEMMEx.
Backends without that interface are rejected before scheduling.

## Exact-rank CompressedFTLR dense output

`CompressedFTLRMatrix` is the exact-rank companion to padded `PaddedFTLRMatrix`. Its outer and
inner factors are independently packed one-dimensional allocations with host
prefix offsets in their own tile orders. A factor span is therefore a compact
matrix view of its active rank, not a `maxrank` prefix with a zero tail.

The CPU/CUDA path is CompressedFTLR × CompressedFTLR → dense with any logical
`N/T` combination. It forms a ragged run plan and invokes the common grouped
interface for all stages (a CPU GEMM loop or `cublasGemmGroupedBatchedEx`). A row-packed A
outer factor enables FoldRight's U stack; row-packed B outer factors retain
the Stage-1 `j → N` W-panel fusion. A column-packed B inner factor additionally
enables FoldLeft. The planner chooses one valid fold for every contiguous row
run after checking exact scratch bytes; it compares non-common active-rank
flops and breaks ties toward FoldRight's wider terminal GEMM.

The CompressedFTLR workspace profile retains the public minimum/maximum contract. The
minimum is the largest one-row exact requirement, and the maximum is enough
for the best full-region fold. A budget between them greedily admits contiguous
rows while a single numerical arena is reset and reused per run. Host rank,
offset, and cuBLAS grouped-pointer metadata are outside this numerical bound.
The grouped API uses device-resident pointer tables and host group metadata.
It supports homogeneous Float16 (Float32 compute), BF16 (Float32 compute on SM80+),
Float32, and Float64 CompressedFTLR storage here. BF16 factors have leading dimensions
rounded to eight elements so every packed factor starts at a 16-byte boundary; an SM75 or
older CUDA device rejects the BF16 grouped path clearly. Current cuBLAS rejects grouped Float16 → Float32 output, so that
mixed storage signature is explicitly rejected rather than falling back to
ordinary or stream-batched GEMM.

The mixed dense/CompressedFTLR CPU/CUDA paths use the same dual packing in a
two-stage reduction. For dense × compressed, one grouped call forms every
`A_k U_kj` piece for a dense-row slab directly in output-column-major rank
stacks; a second grouped call multiplies those stacks by the zero-copy
column-packed inner-factor panels. Compressed × dense is the mirror: its first
grouped call forms every `V_ik' B_k` piece, and its second grouped call uses the
zero-copy row-packed outer-factor panels. Logical transpose swaps the factor
roles and packing orders, so both terminal panels remain contiguous for `N/T`.
The workspace budget selects the largest dense row or column slab that fits.
Smaller legacy budgets fall back to the sequential lowering.

`analyze_compressed_gemm` also accepts either mixed operand ordering. Its
`CompressedMixedGemmAnalysis` owns the prepared grouped descriptors for each
slab, validates the bound dense/compressed/output/workspace objects and rank
metadata, and permits factor values and numerical scalars to change between
calls.

## GEMM source layout

`gemm/common/` contains operand-independent precision, workspace, axis-strategy,
and dense-product helpers. `TLRMatrix` is a full-grid `CompressedFTLRMatrix`
whose diagonal ranks are zero, plus separate dense diagonal storage. Its
off-diagonal product uses `gemm/dense_result/compressed_ftlr/`; three grouped
second-pass terms add `O_A D_B`, `D_A O_B`, and `D_A D_B`. There is no
skip-diagonal or boundary-region dense-output scheduler. Finally,
`gemm/padded_result/` owns the ARA-style algorithm whose destination is a
`PaddedFTLRMatrix`; its name deliberately describes the result representation,
not the historical generic term “TLR result.”

## TLR result integration boundary

The predictable TLR-result API requires physical `TileRowMajor` storage for
`C`, `A`, and `B`. `PaddedFTLRMatrix` therefore defaults to that order. Transpose flags
change the logical order but never reinterpret the physical contract.

For a regular tile grid, the supported sampling table is:

| `transA` | `transB` | natural sample | run |
|---|---|---|---|
| `N` | `N` | right, `XΩ` | fixed output row |
| `N` | `T` | right or left | fixed column or row, selected once from active ranks |
| `T` | `T` | left, `XᵀΩ` | fixed output row |
| `T` | `N` | deferred | requires packed/reduced general-storage execution |

When both sides are natural, the implementation compares retained workspace
for the two run families using the active operand rank caps. It does not choose
per tile, which keeps the ARA batch uniform and swap-compactable.

The ARA path owns sampling, convergence, one co-range apply, truncation, and
output scatter. A full-capacity row/column panel is consumed as a view. A
rank-trimmed prefix is packed once into a contiguous run-local panel because a
strided rank slice cannot be reshaped into a valid terminal GPU GEMM operand.

Canonical execution can use a reusable `TLRGemmWorkspace`.
`tlr_gemm_minimum_workspace_bytes` provides one scheduler slot and
`tlr_gemm_maximum_workspace_bytes` provides the widest complete fixed-axis
lane. An intermediate byte budget selects the largest slot count whose
monotonic bound fits, while excess bytes are capped at the maximum.

Each row or column is scheduled as one lane. A complete full-`k` ARA pass runs
over its active prefix; converged slots are finalized as one wave and pending
members are admitted into the released suffix. Progress and rank exhaustion
are slot-local, so a late member receives its full `maxrank` budget. A mixed
full/terminal block is compacted into at most two orthogonalization groups.
Persistent slot state (`Q`, `S`, packed factor stacks) is separated from one
phase arena shared by admission, sampling, and finalization, giving
`persistent + max(admission, sampling, finalization)` storage. ARA state,
traversal outputs, host mirrors, member maps, and scatter diagnostics are
workspace-owned.

Arbitrary physical layouts, boundary tiles, `TN`, cross-lane batching,
mixed-tile cohorts, streams, and chunked-`k` reduction remain deferred.
