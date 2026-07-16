# TLR GEMM design

This document describes the GEMM implementation that exists today. Future compiler,
TLR-output, compression, and merge-tree work belongs in `ROADMAP.md`.

## 1. Implemented operations

Every public method computes

```text
C := alpha * op(A) * op(B) + beta * C
```

where `op` is `N` or the non-conjugating transpose `T`.

| left operand | right operand | output | status |
| --- | --- | --- | --- |
| `TLRMatrix` | `TLRMatrix` | dense | implemented |
| `TLRDenseDiagMatrix` | `TLRDenseDiagMatrix` | dense | implemented |
| `TLRMatrix` | dense matrix | dense | implemented |
| dense matrix | `TLRMatrix` | dense | implemented |
| `TLRDenseDiagMatrix` | dense matrix | dense | not implemented |
| dense matrix | `TLRDenseDiagMatrix` | dense | not implemented |
| any supported input pair | TLR | not implemented |

The two TLR operands of a TLR–TLR product currently have the same storage element
type and container kind. Mixed `TLRMatrix`/`TLRDenseDiagMatrix` products are not
public methods.

`TLRMatrix` supports rectangular matrices, rectangular nominal tiles, independent
tails on all three GEMM dimensions, all tile-order pairs, and all four `N/T`
combinations.

The non-transposed `TLRDenseDiagMatrix` path retains its established equal-tiling
requirement. A transposed dense-diagonal operand is accepted only in the square,
equal-square-tiling regime. Unsupported transposed rectangular dense-diagonal
products fail explicitly.

Complex arithmetic and conjugating transpose are not supported by this layer.

## 2. Matrix model

A fully low-rank tile is represented canonically as

```text
A[i,k] = U[i,k] * V[i,k]'
B[k,j] = W[k,j] * Z[k,j]'
```

with fixed allocated rank width `maxrank`. Inactive factor columns are expected to be
zero padded by the container/compression layer.

`TLRMatrix` stores every tile in low-rank form. `TLRDenseDiagMatrix` stores regular
diagonal tiles and the dense corner as full-rank dense tiles; its remaining tiles are
low rank. Dense tiles are not represented as low-rank factors with an identity
operand.

For a low-rank tile pair, the common algebra is

```text
S = V' * W
T = S * Z'
C += U * T
```

The lowering may instead associate the last two stages from the left:

```text
S  = V' * W
T' = U * S
C += T' * Z'
```

Both are the same contraction. The selected association is a storage-layout decision.

## 3. Source organization

```text
gemm/
├── gemm.jl                    public methods, validation, region orchestration
├── operands.jl                canonical N/T operands and factor-storage views
├── precision.jl               TLR precision defaults and signature validation
├── tlr_dense.jl               complete TLR×dense and dense×TLR kernels
├── contraction/
│   ├── domain.jl              regular/boundary (i,k,j) spans
│   ├── leaves.jl              low-rank and dense algebra leaves
│   ├── init.jl                output initialization policy
│   ├── operation.jl           ContractOp and dense output mapping
│   └── lowering.jl            scheduled contracts, execution, workspace query
├── lowering/
│   ├── strategy.jl            placement, fusion, grid, and fold traits
│   ├── schedule.jl            geometry, runs, workspaces, batch buffers
│   └── stages.jl              Stage 1/2/3 descriptors and GEMM calls
└── regions/
    ├── interior.jl
    ├── right.jl
    ├── bottom.jl
    └── corner.jl
```

The dependency direction is:

```text
physical containers
  → canonical logical operands
  → domains + leaves + ContractOp
  → scheduled contraction
  → budgeted runs and stage descriptors
  → precision-aware GEMM/GEMMEx
```

The region files compose operations for a destination region. They do not define
transpose semantics, precision policy, or generic low-rank scheduling.

## 4. Canonical operands and transpose

`LogicalTLROperand{Op}` is an internal zero-copy view of an
`AbstractTLRMatrix`. `LogicalDenseOperand{Op}` provides the corresponding view for a
standalone dense operand. Flags are normalized case-insensitively to `N` or `T`; every
other flag throws `ArgumentError`.

The logical TLR operand owns the effective interpretation of:

- matrix size;
- nominal and tail tile sizes;
- full and regular tile-grid dimensions;
- tile order;
- region coordinates;
- low-rank outer and inner factors.

For `T`, both matrix and tile axes are reversed, tile row/column order is exchanged,
and the right and bottom regions swap. The factor mapping is

```text
outer(op(A), region) = inner(A, transpose_region(region))
inner(op(A), region) = outer(A, transpose_region(region))
```

Consequently every lowering still sees `outer * inner'`; no region kernel interprets
`transA` or `transB`.

A dense diagonal or dense corner is exposed as `LogicalDenseTile{Op}`: a physical
view plus an `N/T` operation. The implementation never materializes a transposed
dense tile.

`LogicalTLROperand` remains GEMM-internal. The roadmap records the proposal to promote
this behavior to a container-level lazy transpose view when another algorithm needs it.

## 5. Contraction representation

### Domains

A `ContractDomain` is an `(i,k,j)` triple of `AxisSpan`s. Each span is either the
regular tile range or the optional boundary tile. The eight combinations partition
the complete tile-triple space:

| operation | i | k | j | output region |
| --- | --- | --- | --- | --- |
| `interior` | regular | regular | regular | interior |
| `int_by_rpanel` | regular | regular | boundary | right |
| `bpanel_by_int` | boundary | regular | regular | bottom |
| `rpanel_by_bpanel` | regular | boundary | regular | interior |
| `rpanel_by_corner` | regular | boundary | boundary | right |
| `corner_by_bpanel` | boundary | boundary | regular | bottom |
| `bpanel_by_rpanel` | boundary | regular | boundary | corner |
| `corner_by_corner` | boundary | boundary | boundary | corner |

An aligned axis has an empty boundary span, so the corresponding operations become
empty without special transpose or tail code.

The domain describes coordinates only. Whether an interior tile exists in low-rank
storage is a leaf property: `FullGrid` includes every regular tile, while `SkipDiag`
omits the dense diagonal.

### Leaves

`LowRankLeaf` pairs canonical outer and inner factor operands. Those operands may
address a two-dimensional interior, a one-dimensional right/bottom panel, or a
single corner. `DenseLeaf` carries a `LogicalDenseTile`.

The leaf pair selects one of four lowering families:

| left leaf | right leaf | lowering |
| --- | --- | --- |
| low rank | low rank | three-stage contraction |
| low rank | dense | two stages |
| dense | low rank | two stages |
| dense | dense | one GEMM |

This is why dense tiles remain distinct leaves: the lowering can call the appropriate
dense GEMM directly without creating identity factors.

### Operations and outputs

A `ContractOp` contains only:

- a `ContractDomain`;
- left and right leaves;
- a `DenseOutput`;
- an `InitPolicy`;
- `alpha`.

It contains no fold, iterator placement, run width, workspace, stream, or backend
library call.

`DenseOutput` maps operation-local tile coordinates back into the correct part of
`C`, using effective row geometry from `op(A)` and effective column geometry from
`op(B)`. Regular and tail output tiles therefore use the same scheduled code.

`InitPolicy(beta)` represents both first-writer and accumulation behavior:
`ScaleExisting(beta)` carries the caller's beta, while `AccumulateExisting(T)` is
the `beta == 1` case.

All representation and scheduled-operation types remain concrete Julia types. This
is required for inference of workspace array and batch-vector element types; the
implementation does not use a heterogeneous runtime list of operations.

## 6. Lowering low-rank contractions

Lowering a low-rank pair performs four decisions:

1. derive `ContractGeometry` from the domain and leaves;
2. choose `FoldRight` or `FoldLeft`;
3. map the reduction to `KAsGemmK` or `KAsSerialLoop`;
4. choose the budgeted run dimensions.

The resulting `ScheduledLowRankContract` owns these decisions. Execution promotes
workspace once, refills preallocated batch vectors for each run, and invokes
Stage 1/2/3 through `precision_gemm_batched!`.

Specialized scheduled types implement low-rank–dense, dense–low-rank, and
dense–dense leaf pairs. Their terminal GEMM observes the same output and compute
policy as the three-stage path.

### Reduction placement

Tile order determines the physically contiguous tile axis:

| effective order | left operand contiguous axis | right operand contiguous axis |
| --- | --- | --- |
| tile column-major | `i` | `k` |
| tile row-major | `k` | `j` |

For the left operand:

- contiguous `k` selects `KAsGemmK`: the tile reduction is fused into a GEMM K
  dimension and an output tile is written once;
- contiguous `i` selects `KAsSerialLoop`: reduction runs accumulate into the same
  output tile.

The right operand's contiguous axis controls whether Stage 1 can fuse its free
column axis. The four effective order pairs select:

| A order | B order | reduction placement | Stage-1 free-axis mapping |
| --- | --- | --- | --- |
| row-major | row-major | `KAsGemmK{:j}` | `JAsGemmN` |
| row-major | column-major | `KAsGemmK{:k}` | `FreeAsBatch` |
| column-major | column-major | `KAsSerialLoop{:k}` | `IAsGemmM` |
| column-major | row-major | `KAsSerialLoop{:j}` | `IJAsGemmMN` |

Transpose changes effective tile order before this decision.

### Fold selection

`FoldRight` produces `T = S*Z'` and finishes with the left outer factor.
`FoldLeft` produces `T' = U*S` and finishes with the right inner factor. A fold is
legal only when its required reduction stack is complete and contiguous. When both
choices are legal, the scheduler prefers a write-once placement and then the smaller
terminal intermediate.

The bottom-panel–right-panel corner operation is deliberately kept as a serial,
budget-blocked reduction instead of forming an unbounded complete stack.

## 7. Workspace and initialization

`max_workspace` is a per-operation scratch budget in bytes. It controls `RowRun` or
`ColumnRun` sizes and the promoted S/T buffers. It is not a strict whole-call memory
cap:

- one element of progress is always permitted even if the requested budget is
  smaller;
- persistent operand/output storage is excluded;
- on GPU, four disjoint output regions may execute concurrently, so their scratch
  allocations can coexist.

`gemm_workspace_bytes(A, B; transA, transB)` returns the maximum full-width scratch
requirement among the eight structured contractions emitted for the logical TLR pair.
It is not the sum of concurrently executing regions and does not describe the direct
standalone-dense kernels.

Batch-vector containers are allocated with concrete view element types when workspace
is promoted, then reused with `empty!` and `push!` inside run loops.

Output initialization depends on placement:

- `KAsGemmK` writes each output tile once, so the terminal GEMM receives beta;
- `KAsSerialLoop` may write an output tile repeatedly, so its destination region is
  scaled once before the reduction and every terminal GEMM accumulates with beta one.

At the top level, each of the four output regions has a first term that owns the
caller's beta and a second term that accumulates with beta one. Empty reductions still
apply the required output scaling.

## 8. TLR–TLR region orchestration

The dense result is split into four disjoint regions:

```text
C = [ interior  right
      bottom    corner ]
```

Each region contains two boundary-cube operations listed in section 5. CPU execution
runs the regions sequentially. Non-CPU execution creates four streams, assigns one
region to each stream, and synchronizes all four at the end. Operations within one
region remain ordered because they may update the same output tiles.

For `TLRMatrix × TLRMatrix`, all tiles are low rank and the region operations use the
structured low-rank lowering directly.

For `TLRDenseDiagMatrix × TLRDenseDiagMatrix`, the low-rank off-diagonal and boundary
parts use the same representation. Regular dense-diagonal contributions are expanded
in the region entry points:

```text
A_int * B_int =
    O_A * O_B +
    D_A * D_B +
    O_A * D_B +
    D_A * O_B
```

The `O_A*O_B` term owns beta for the interior; dense-diagonal components accumulate
after it. The analogous diagonal contribution is performed before the structured
off-diagonal remainder in the right and bottom regions. Dense corners lower through
`DenseLeaf`, not through an identity-factor approximation.

## 9. Products with one standalone dense operand

`TLRMatrix × dense` and `dense × TLRMatrix` deliberately use direct two-stage tile
kernels rather than `ContractOp`.

For a left TLR tile:

```text
T = V' * B_block
C_block += U * T
```

For a right TLR tile:

```text
T = A_block * U
C_block += T * V'
```

The standalone dense operand is viewed through `LogicalDenseOperand`, so `N/T`
changes view coordinates and GEMM flags without copying data. `max_workspace` blocks
dense columns or rows respectively. The intermediate retains the TLR operand storage
type. Beta is applied once to the complete dense output before tile accumulation.

These paths require the dense operand, TLR factors, and output to use the same backend,
and the standalone dense operand currently has the same element type as the TLR
operand.

## 10. Precision policy

TLR GEMM supports operand storage types `Float16`, `Float32`, and `Float64`.
Operand storage is inferred from the inputs and output storage from `C`; the caller
selects only `compute`.

Defaults are:

| operand storage | default compute |
| --- | --- |
| `Float16` | `Float32` |
| `Float32` | `Float32` |
| `Float64` | `Float64` |

The central invariant is that S and T use operand storage. Stages that produce another
intermediate therefore validate

```text
Tin × Tin → Tin
```

while only the terminal stage may validate

```text
Tin × Tin → Tout
```

This permits, subject to backend support, FP16 operands with FP32 accumulation and
either FP16 or FP32 output. FP32 and FP64 use their backend-supported same-precision
signatures. `alpha` and `beta` are converted to the compute type.

`TF32()` is a CUDA-only compute mode for FP32 operands and FP32 output. Selecting it
on another backend fails validation before scheduling.

All scheduled calls go through `precision_gemm!` or
`precision_gemm_batched!`. Those functions validate the actual operand,
destination, backend, and compute-mode signature before calling native GEMM or
GEMMEx.

On CPU, the generic policy currently accepts same-type FP32 and FP64 GEMM. GPU
mixed-precision availability is determined by the CUDA/AMDGPU backend extensions.
Other backends reject signatures they do not advertise.

## 11. Validation and degeneracies

TLR–TLR entry points validate once, after logical transpose canonicalization:

- effective inner matrix dimensions agree;
- `C` has the effective output shape;
- effective contraction-axis nominal tile sizes agree;
- precision and backend signatures are supported.

Standalone-dense paths additionally validate effective dimensions and a common backend.

A zero-rank `TLRMatrix × TLRMatrix` product scales `C` by beta and returns. Empty
boundary domains are no-ops unless they own initialization of a nonempty output
region. The lowering avoids constructing zero-sized runs.

`N/T` are non-conjugating operations. A future complex implementation must add a
separate adjoint/conjugation-aware factor mapping rather than treating `C` as `T`.

## 12. Extension boundaries

The implemented contraction representation ends at a dense output mapping. Milestones
4–6 in `ROADMAP.md` add:

- output sinks that can receive dense tiles or low-rank updates;
- bounded TLR accumulation and recompression;
- contraction and compression workspace accounting;
- merge-tree planning;
- the remaining TLR-output product families.

Those features must preserve the current separation:

- operands canonicalize storage and transpose;
- leaves describe algebraic values;
- domains describe iteration spaces;
- lowering selects association, placement, runs, and library calls;
- output policy owns materialization or compression;
- memory budget and approximation budget remain distinct.
