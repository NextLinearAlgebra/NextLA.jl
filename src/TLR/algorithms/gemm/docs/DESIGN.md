# TLR GEMM design

The GEMM implementation is organized as direct execution paths, not as a contraction
compiler. The public call graph is intentionally short:

```text
gemm! API
  -> dense-output driver
  -> row/column traversal
  -> precision-aware batched kernels
```

Canonical factor accessors and regular Stage 1 are shared by the dense-output
paths and the ARA factor-list implementation under development. Boundary and
dense-diagonal work is expressed by direct helpers for the actual operand
combination.

## Canonical operands

`LogicalTLROperand{Op}` is a zero-copy logical view of `op(A)`. It resolves `N/T` once,
before execution:

- dimensions, tile grids, and tile order follow the logical orientation;
- right and bottom panels swap under transpose;
- low-rank outer/inner factors swap, so every logical tile remains `outer * inner'`;
- dense diagonal/corner tiles retain physical storage plus their BLAS operation flag.

`InteriorOperand`, `PanelOperand`, and `CornerOperand` are storage-addressing helpers.
They return factor views through `tilefactor` and describe contiguity through the small
layout traits used by traversal selection. They do not describe algebra or own storage.

## Regular low-rank core

For `A_ik = U_ik V_ik'` and `B_kj = W_kj Z_kj'`, the shared core computes:

```text
Stage 1: S_ikj = V_ik' W_kj
Stage 2: T_ikj = S_ikj Z_kj'       (FoldRight)
Stage 3: C_ij  += U_ik T_ikj
```

`FoldLeft` instead forms `U_ik S_ikj` and stacks `Z_kj` in Stage 3. Fold selection is
available only where the required reduction stack is complete and contiguous.

Stage 1 is deliberately output-independent and independently callable. A future TLR
merge algorithm may consume its `S` factors without going through dense Stage 3.

`RegularGeometry{T}` contains only the concrete dimensions needed to size runs and
workspace. `T` is a type parameter so scratch allocation and reusable batch-vector
element types remain inferable. There is no operation descriptor or semantic IR.

## Traversals

Two traversal families cover the four effective layout combinations:

- `KAsGemmK`: row/write-once traversal. A run covers a rectangular block of output
  tiles. The complete reduction is fused into the terminal GEMM, so `beta` is applied
  by that single write.
- `KAsSerialLoop`: column/streaming traversal. Runs block contraction tiles and right
  panel positions. The destination region is scaled by `beta` once, then every terminal
  write accumulates with `beta = 1`.

Within each family Stage 1 either batches tilewise or fuses the right operand's
contiguous `j` panel into GEMM N. Run dimensions come from the regional slice
assigned by the global workspace policy.
All batch vectors are allocated once with concrete view element types and refilled with
`empty!`/`push!`.

## Dense-output paths

`execute_lowrank_gemm!` is the shared direct driver. Its arguments are the destination,
canonical operand geometries, four factor accessors, `RegularGeometry`, output tile
origin, scalars, compute mode, and workspace budget. It selects or accepts a fold and
traversal, allocates one typed workspace, and executes the runs.

The full-TLR driver calls it for the regular interior and each live low-rank boundary
term. `TLRDenseDiagMatrix` retains its tuned diagonal/off-diagonal interior kernels and
uses direct helpers for boundary combinations:

- low-rank × low-rank: the shared budgeted core;
- low-rank × dense: a budgeted two-stage row batch;
- dense × low-rank: a budgeted two-stage column batch;
- dense × dense: one direct batched GEMM.

The top-level dense driver keeps four disjoint output regions (interior, right,
bottom, corner). GPU backends use two streams: one for the interior and one
which executes right, bottom, and corner serially. CPU executes the same two
groups in order.

## Workspace contract

Dense-output GEMM exposes two exact global bounds:

- `gemm_minimum_workspace_bytes(A, B; transA, transB)` is the interior
  minimum plus the largest minimum of the serialized boundary regions.
- `gemm_maximum_workspace_bytes(A, B; transA, transB)` is the interior
  full-width requirement plus the largest full-width boundary requirement.
  Increasing the workspace beyond this value cannot enlarge a run.

Every query includes transpose-aware tails and specialized dense-boundary
kernels. Any budget between the two bounds is correct, making policies such as
a multiple of the minimum or a fraction of the maximum explicit without
claiming an unmeasured performance optimum.

`gemm!` requires `workspace`, either an integer global byte count or a reusable
`DenseGemmWorkspace`. Both modes use one typed numerical arena. The
`InteriorFirstWorkspace` policy reserves the auxiliary minimum, gives remaining
capacity to the interior up to its maximum, and assigns the rest to the
auxiliary stream:

```text
Waux,min = max(Wright,min, Wbottom,min, Wcorner,min)
Winterior = min(Winterior,max, Wglobal - Waux,min)
Waux = Wglobal - Winterior
```

Right, bottom, and corner reset and reuse the same auxiliary slice. An integer
constructs a temporary arena; passing a `DenseGemmWorkspace` reuses its device
allocation and streams across calls. Budgets below the global minimum are
rejected and capacity beyond the global maximum is unused.

The bound covers numerical scratch allocated by the TLR GEMM implementation.
Output storage, persistent TLR factors, host batch descriptors, and
backend-library internal allocations are outside it. Dynamic lending of the
interior slice to the auxiliary stream is deferred until profiling justifies
an event boundary.

## Precision

Operand factor/intermediate storage follows the TLR operand element type. Output storage
follows `C`. GEMM scalars use the selected compute precision. `precision_gemm_batched!`
is the sole backend dispatch point for ordinary GEMM, CUDA GEMMEx/TF32, and capability
validation.

## TLR result integration boundary

The predictable TLR-result API requires physical `TileRowMajor` storage for
`C`, `A`, and `B`. `TLRMatrix` therefore defaults to that order. Transpose flags
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
Arbitrary physical layouts, boundary tiles, `TN`, budgeted scheduling, and a
reusable run arena remain the general-storage/R4-R5 boundary.
