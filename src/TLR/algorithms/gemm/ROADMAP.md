# TLR algebra roadmap

This roadmap tracks the evolution of the TLR multiplication code from a dense-output
GEMM implementation into a small compiler for tiled dense and low-rank contractions.
`DESIGN.md` documents implemented behavior; this file records future milestones and
their acceptance gates.

Status: `[x]` done, `[>]` active, `[ ]` planned.

| status | milestone | deliverable | depends on | acceptance gate |
| --- | --- | --- | --- | --- |
| `[x]` | 1. Canonical operands | One zero-copy, whole-matrix logical operand for `N/T`, including panels, corners, dense diagonal tiles, effective geometry, and output targeting. | Current interior operand and staged GEMM. | Both TLR containers match dense references for supported boundary transpose cases on CPU and representative GPU cases. |
| `[x]` | 2. Precision policy | Central GEMM/GEMMEx dispatch that infers operand/output storage, keeps intermediate factors operand-typed, and accepts an explicit compute mode. | Canonical operands. | Supported mixed-precision combinations have backend capability tests, and every lowering preserves the intermediate-type invariant. |
| `[ ]` | 3. Contraction IR | Dense/low-rank leaves, orientation domains, contraction descriptors, and dense/low-rank update production. | Canonical operands and precision policy. | Existing dense-output terms lower through the IR without performance or correctness regressions. |
| `[ ]` | 4. Output sinks | A shared update stream targeting either dense materialization or TLR factor accumulation. | Contraction IR. | One contraction can be lowered unchanged to dense and TLR outputs. |
| `[ ]` | 5. Bounded TLR accumulation | Streaming merge/recompression whose live contraction, factor, and compression scratch fits one workspace budget. | TLR output sink and tile-source compression. | The first TLR-output product respects rank, approximation, and memory limits. |
| `[ ]` | 6. Merge-tree planning | Balanced and k-ary merge strategies and complete coverage of TLR×TLR / TLR×dense to dense / TLR outputs. | Bounded accumulation. | All four product families select a valid plan from geometry, precision, error, and workspace constraints. |

## Architectural decisions

- Transpose is canonicalized before term generation or lowering; executors consume
  canonical `outer * inner'` low-rank factors.
- Dense tiles are distinct algebra leaves. They share the contraction IR but use
  specialised one- or two-stage lowerings rather than materialised identity factors.
- Dense and TLR results are output policies (sinks), not separate input algebras.
- Workspace bytes and approximation tolerance are independent budgets. A TLR-output
  plan accounts for contraction scratch, live candidate factors, compression scratch,
  output factors, and concurrency.
- Mixed precision distinguishes input storage, intermediate storage, GEMM compute,
  output storage, and compression/orthogonalisation precision.

Milestone 2 covers real `Float16`/`Float32`/`Float64` operands, the valid
same- and mixed-output combinations, and CUDA TF32. Operand storage comes from `A`
and `B`, output storage comes from `C`, and only the compute mode is selected by the
caller. GEMM scalars use compute precision, while intermediate factors retain operand
precision. Compression precision remains part of the later TLR-output milestones.

## Container-level lazy transpose proposal

The Milestone 1 `LogicalTLROperand{Op}` remains an internal, zero-copy GEMM view for
now. When a second algorithm needs the same whole-matrix transpose semantics, promote
the abstraction from `algorithms/gemm` into the container layer rather than adding
another algorithm-specific wrapper.

The proposed container view is a sibling wrapper, conceptually
`TLROpView{Op,A<:AbstractTLRMatrix}`, not a subtype of the concrete `TLRMatrix` or
`TLRDenseDiagMatrix` types. It should:

- provide `Base.transpose(A::AbstractTLRMatrix)` as a lazy `:T` view and unwrap a
  double transpose;
- retain an internal `:N` view so lowering receives one canonical operand family;
- centralize axis reversal for dimensions, nominal/tail tiles, and tile grids;
- swap right/bottom regions, outer/inner factors, tile order, and region coordinates
  without moving factor storage;
- preserve dense diagonal/corner tiles as physical views carrying an `N/T` operation,
  avoiding transposed copies and strided `PermutedDimsArray` inputs to batched GEMM;
- extend later to a distinct `:C` adjoint operation when complex arithmetic is added.

This promotion is architectural reuse, not a new lowering: the existing canonical
operand behavior and tests become the acceptance baseline. Defer it until transpose
is consumed outside GEMM, since moving the wrapper earlier would provide no runtime
or kernel benefit.

## Progress rule

A milestone becomes `[x]` only when its acceptance gate is covered by tests and its
implemented behavior is reflected in `DESIGN.md`. At most one milestone should be
`[>]` while implementation is active.
