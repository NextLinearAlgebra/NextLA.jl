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
contiguous `j` panel into GEMM N. Run dimensions come directly from `max_workspace`.
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

The top-level dense driver keeps four disjoint output regions (interior, right, bottom,
corner). CPU executes them in order. GPU backends use four independent streams and one
final synchronization.

## Workspace contract

`gemm_workspace_bytes(A, B; transA, transB)` returns the smallest per-operation budget
at which every direct kernel used by dense-output `gemm!` runs at full width. It is the
maximum of the actual requirements of the eight possible region terms, including
transpose-aware tails and the specialized dense-boundary kernels. Smaller budgets remain
correct and partition budgeted work into more runs.

The bound covers promoted contraction scratch. Output storage, persistent TLR factors,
and backend-library internal allocations are outside that public budget, as before.

## Precision

Operand factor/intermediate storage follows the TLR operand element type. Output storage
follows `C`. GEMM scalars use the selected compute precision. `precision_gemm_batched!`
is the sole backend dispatch point for ordinary GEMM, CUDA GEMMEx/TF32, and capability
validation.

## TLR result integration boundary

TLR-result GEMM is intentionally absent from the public dispatch until the ARA
factor-list implementation is complete. That implementation reuses logical
operands, factor accessors, run geometry, workspace utilities, and precision
dispatch, while owning sampling, convergence, truncation, and output scatter.
