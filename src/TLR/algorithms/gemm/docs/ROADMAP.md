# TLR algebra roadmap

This roadmap tracks the TLR multiplication code. `DESIGN.md` documents the current
direct execution architecture; this file retains milestone history and future gates.
The compiler-style contraction IR introduced for Milestone 3 was removed before M5:
it made the public path harder to follow without adding behavior that direct,
output-oriented drivers could not provide.

Status: `[x]` done, `[>]` active, `[ ]` planned.

Current dense-output coverage: `TLR×TLR`, `TLR×dense`, and `dense×TLR` for the
fully low-rank `TLRMatrix`; standalone-dense products with `TLRDenseDiagMatrix` and all
TLR-output products remain planned.

| status | milestone | deliverable | depends on | acceptance gate |
| --- | --- | --- | --- | --- |
| `[x]` | 1. Canonical operands | One zero-copy, whole-matrix logical operand for `N/T`, including panels, corners, dense diagonal tiles, effective geometry, and output targeting. | Current interior operand and staged GEMM. | Both TLR containers match dense references for supported boundary transpose cases on CPU and representative GPU cases. |
| `[x]` | 2. Precision policy | Central GEMM/GEMMEx dispatch that infers operand/output storage, keeps intermediate factors operand-typed, and accepts an explicit compute mode. | Canonical operands. | Supported mixed-precision combinations have backend capability tests, and every lowering preserves the intermediate-type invariant. |
| `[x]` | 3. Direct budgeted contractions | Canonical factor accessors, concrete regular geometry, row/column traversal, and explicit budgeted boundary kernels. | Canonical operands and precision policy. | Every term honours `max_workspace`, reuses concrete batch vectors, and selects a legal fold/traversal. |
| `[~]` | 4. TLR-output fallback | Row-family regular-grid accumulation into bounded dense slabs followed by recompression. | Direct regular contraction core. | The supported M4 subset is correct on CPU/CUDA and provides the differential oracle for future work. |
| `[ ]` | 5. Bounded TLR accumulation (deferred) | Numerically robust factor-space merge/recompression whose live scratch fits one workspace budget. | Direct Stage 1 and tile-source compression. | The first production TLR-output product respects rank, approximation, and memory limits. |
| `[ ]` | 6. Merge-tree planning | Balanced and k-ary merge strategies and complete coverage of TLR×TLR / TLR×dense to dense / TLR outputs. | Bounded accumulation. | All four product families select a valid plan from geometry, precision, error, and workspace constraints. |

Status legend also uses `[~]` for *partially delivered* (a usable subset ships and is
tested, but the full acceptance gate is intentionally deferred).

## Milestone 4 status and deferred Milestone 5

Milestone 4 shipped a **row-family, regular-grid, `beta == 0` TLR-output sink** built on
dense accumulation into a bounded slab followed by randomized-sketch recompression
(`compress_tiles!`). It is tested (CPU + CUDA) and stands as the reference oracle and the
recompression fallback for Milestone 5 — see `M5_ORTHOGONAL_MERGE.md` for the full record
of what was delivered, what was deliberately skipped, and why.

M5 is deferred. The dense row-family fallback is retained as a differential-test oracle
and possible recompression fallback. `M5_ORTHOGONAL_MERGE.md` remains a future draft, but
its numerical algorithm and its references to the former IR/sink architecture require
revision before implementation.

## Architectural decisions

- Transpose is canonicalized before term generation or lowering; executors consume
  canonical `outer * inner'` low-rank factors.
- Dense boundary tiles use specialised one- or two-stage helpers rather than
  materialised identity factors.
- Dense and TLR-output drivers own their terminal writes after shared regular Stage 1.
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

## Historical Milestone 3 record

The section below records the measurements and problems that motivated unified
budgeting. Its IR types, descriptor steps, and terminology are historical: the useful
outcomes now live in the direct regular core and explicit boundary helpers.

### Why

The interior received the full treatment — budgeted runs, preallocated batch buffers,
layout-driven placement, fold selection. The six panel and corner terms received none
of it:

- ~~`tlr_gemm_int_by_rpanel` and `tlr_gemm_bpanel_by_int` (the `TLRMatrix` variants)
  accept `budget::Int` and never read it~~ — fixed, step 4. Their dense-diagonal
  counterparts honoured it, so budget compliance depended on container type.
- ~~`tlr_gemm_bpanel_by_rpanel` (`O(q_c)`), `tlr_gemm_rpanel_by_corner` (`O(q_m)`) and
  `tlr_gemm_corner_by_bpanel` (`O(q_n)`) take no budget at all~~ — fixed, step 4 (both
  containers). Only `tlr_gemm_corner_by_corner` is legitimately unbudgeted — a corner is
  one tile in any geometry.
- ~~Every panel term rebuilds `[view(...) for i in …, k in …]` batch vectors per call~~ —
  fixed, step 4; they use `_batchvec` + `empty!`/`push!` like the interior.
- ~~`tlr_gemm_bpanel_by_rpanel` materialises a `permutedims` copy~~ — fixed, step 4 by
  laying `T` out as `[rA, p, s_n]` so Stage 3's K-stack is a pure `reshape`.
- ~~`choose_fold` is interior-only; every other term is hard-coded `FoldRight`~~ — fixed,
  step 6. Leaf layout and complete-stack traits drive the shared choice; the singleton
  bottom×right output deliberately keeps its reduction serial so it remains budgeted.

The milestone is therefore *one scheduler for eight terms*, not a vocabulary
migration. The rename table in `IR_VOCABULARY.md` is deliberately off the critical
path: new code adopts the semantic names, working tuned code is not renamed for its
own sake.

Step 0 measured the gap rather than asserting it, and step 4 closed it. On a
boundary-tiled `Float64` `TLRMatrix` pair (`b=8, r=4, nt=10`), bytes allocated per term
call at `budget=1` versus `budget=128 MiB`:

| term | before (1 / huge) | after (1 / huge) |
| --- | --- | --- |
| `_offdiag_offdiag_gemm!` | 11,224 / 744,072 | 11,224 / 744,072 |
| `rpanel_by_bpanel` | 15,176 / 68,792 | 15,176 / 68,792 |
| `int_by_rpanel` | **96,736 / 96,736** | 5,888 / 47,352 |
| `bpanel_by_int` | **115,936 / 115,936** | 7,816 / 66,552 |
| `bpanel_by_rpanel` | no budget param | 5,088 / 5,920 |
| `rpanel_by_corner` | no budget param | 1,856 / 6,272 |
| `corner_by_bpanel` | no budget param | 2,048 / 8,200 |

Byte-identical across a 128-million-fold budget range is what "ignores the budget" looks
like; the same two terms honoured it on `TLRDenseDiagMatrix`, so it was an implementation
gap, not a property of the panel geometry. Note the migrated terms also allocate ~2× less
at full budget, because the per-call batch-vector comprehensions are gone.

### The structural observation

The eight terms are the eight corners of the `(i,k,j) ∈ {regular, boundary}³` cube,
and the four regions of `C` are that cube projected onto `(i,j)`:

| term | i | k | j |
| --- | --- | --- | --- |
| `offdiag_offdiag` | `1:q_m` | `1:q_c` | `1:q_n` |
| `int_by_rpanel` | `1:q_m` | `1:q_c` | `bnd` |
| `bpanel_by_int` | `bnd` | `1:q_c` | `1:q_n` |
| `rpanel_by_bpanel` | `1:q_m` | `bnd` | `1:q_n` |
| `rpanel_by_corner` | `1:q_m` | `bnd` | `bnd` |
| `corner_by_bpanel` | `bnd` | `bnd` | `1:q_n` |
| `bpanel_by_rpanel` | `bnd` | `1:q_c` | `bnd` |
| `corner_by_corner` | `bnd` | `bnd` | `bnd` |

One parameterized span triple covers all eight; a per-term domain type would only
rename the eight functions. Diagonal skipping stays a leaf property (`GridKind`),
not a domain property — that is where it already lives. On `TLRDenseDiagMatrix` a
regular span carries two leaves (dense diagonal + low-rank off-diagonal), so those
corners expand into up to 2×2 leaf-pair contractions — which is why
`tlr_gemm_int_by_int` has four components today. The `D`/`O` split falls out of leaf
selection rather than being hand-written per term.

Keeping the domain purely geometric is what makes the claim testable: the eight
domains then **partition** the full tile-triple space — every triple covered, none
twice — for both containers and for every mix of aligned and tailed axes. That is
falsifiable without reference to the code being replaced (a missing corner is a gap,
a miscounted span an overlap), which a shadow test restating the current loops would
not be. `tile_present` refines the all-regular corner afterwards.

Emptiness is derived rather than guarded: a tile-aligned axis has no tail, so its
boundary span is empty and every term pinning it drops out. This replaces the
`region_tile_count(...) == 0 && return C` guard each term opens with, and the two
agree by test. Worth noting because it is easy to get backwards: a tail on the
*contraction* axis gives A a right panel and B a bottom panel, so `rpanel_by_bpanel`
switches **on**, while `bpanel_by_rpanel` needs tails on `i` and `j` instead.

### Steps

| step | deliverable | checkpoint |
| --- | --- | --- |
| `[x]` 0 | `test/TLR/gemm_budget.jl` budget-response gate; boundary-tiled benchmark configs. | Done: 6 pass, 5 `@test_broken`. Six terms gained a perf signal they never had. |
| `[x]` 1 | Historical IR factor wrappers. | Superseded: storage mapping is now pinned in `test/TLR/gemm_core.jl`, and `PanelOperand`/`CornerOperand` live with canonical operands. |
| `[x]` 2 | Historical span/domain model. | Superseded by explicit output-region drivers and direct live-axis counts. |
| `[x]` 3 | `_interior_geom` → `geometry(domain, leaves)`; `runs`/`allocate_workspace` domain-driven; `lower_init!` replaces the inline placement branch. | Done: 267 correctness tests unchanged; geometry pinned field-by-field to the operand quantities the old code read; workspace promotion proven concretely typed. |
| `[x]` 4 | Migrate panel/corner terms: `int_by_rpanel`, `bpanel_by_int`, `bpanel_by_rpanel` (also killed the `permutedims` copy), `rpanel_by_corner`, `corner_by_bpanel` — the last two on both containers. | Done: budget gate **6 pass/5 broken → 13 pass/0 broken**. Every term but `corner_by_corner` (one tile in any geometry) now honours `max_workspace` and uses preallocated batch buffers. |
| `[x]` 5 | `ContractOp` + `DenseOutput`; interior lowers to a concrete scheduled contraction and executes through the existing Stage 1/2/3 machinery. | Done: semantic construction contains no schedule/workspace/backend state; automatic selection covers all layout pairs and budget extremes, and both folds retain dense-reference correctness on their legal layouts; promoted workspace and tuned stages are unchanged. |
| `[x]` 6 | Emit the seven boundary operations; `choose_fold` for all terms; extend `gemm_workspace_bytes` beyond the interior; update the completed architecture in `DESIGN.md`. | Done: all eight cube operations emit `ContractOp`s. Four leaf-pair lowering families cover LR×LR, LR×Dense, Dense×LR, and Dense×Dense; bottom×right retains a budgeted serial reduction. The transpose-aware workspace query is the maximum full-width requirement across all emitted operations. CPU correctness/budget/IR suites and representative CUDA suites pass. |

### Measurement: what "no regression" can actually mean here

Step 3 originally carried the gate "benchmark flat". That gate does not hold up, and the
correction matters for steps 4 and 5.

An interleaved A/B of `scripts/benchmark_gemm.jl`'s shape (HEAD vs the domain-driven
scheduler, three rounds each, alternating, `minimum` of 5 reps) measured **same-code**
run-to-run variation of ~15% median and 35% worst on the reference machine. The
old-vs-new deltas spanned −12.8%…+7.7% — entirely inside that band — and the mean flipped
sign between sessions (+3.8% one run, −4.3% the next). The benchmark therefore resolves
nothing below roughly 20%: it would pass a 15% regression without noticing, and a
single-shot run invites reading noise as signal in either direction.

What does hold the hot path:

- **Inference.** Every change in this milestone runs *once per `gemm!` call*, against
  thousands of tile GEMMs — a few hundred nanoseconds of setup cannot matter. The real risk
  is type inference of the *promotion*: scratch is `allocate(backend, eltype(geom), …)`, and
  `allocate`'s result type is inferable only when the element type is known to the compiler.
- **Allocation equality.** Bytes allocated by `_offdiag_offdiag_gemm!` must match the
  pre-refactor code exactly at several budgets. This is a sharp, noise-free gate: an
  inference regression shows up as per-run allocation, and byte-equality across budgets is
  a far stronger claim than "the benchmark looks flat".

Keep the benchmark for gross regressions and for the panel terms' first-ever signal, but
do not treat it as the acceptance gate. If a step ever needs a real performance verdict,
it needs a quieter machine and an interleaved A/B — not one run of each.

#### The `Tin` regression (caught, fixed — do not reintroduce)

Step 3 first carried the operand type as a `Tin::DataType` **field** on the geometry.
`allocate(backend, geom.Tin, …)` is then a runtime value, so inference of
`allocate_workspace` collapsed to `RowWorkspace{_A,_B,…}` with bare `Vector` batch buffers,
and the staged loops allocated **~1.4 KB per run** through dynamic dispatch:

| budget | before step 3 | with `Tin` field | with `ContractGeometry{T}` |
| --- | --- | --- | --- |
| 1 | 11,224 | 148,136 | 11,224 |
| 64 KiB | 75,936 | 90,448 | 75,936 |
| 128 MiB | 744,072 | 746,344 | 744,072 |

All 267 correctness tests passed throughout, and the benchmark could not resolve it. The
fix is `ContractGeometry{T}` — the storage type is a **type parameter**, read via `eltype`.

Two lessons worth keeping:

- **`isconcretetype(typeof(x))` is vacuous.** `typeof` of any runtime value is always
  concrete, so that assertion cannot fail — it is what let the regression through. Use
  `@inferred` or `isconcretetype(only(Base.return_types(f, argtypes)))`. `eltype(buf)` on a
  `Vector` *is* meaningful, because a Vector's declared element type can be abstract.
- **The gate was verified by injecting the bug.** With `eltype` returning a runtime value,
  correctness stayed 267/267 while the inference tests failed 7 — a gate that has never been
  seen to fail is not known to work.

### Constraints discovered while planning

- **The IR must be compile-time-shaped.** `contract_ops(A, B)` returns a concretely
  typed tuple so the term loop unrolls and span-generic stage loops specialise. A
  `Vector{Any}` of ops would put every term behind dynamic dispatch and regress the
  interior. This is a genuine NextLA-versus-MLIR divergence: MLIR's IR is a runtime
  data structure, ours is a type.
- **Fusion and `FoldLeft` are conditional capabilities.** `JAsGemmN` needs `rowpanel`
  contiguity over `j`; `FoldLeft` needs the clean `reshape(bz.data, bn, rB·q_c, q_n)`
  Z-stack. Both hold only on a full interior grid. They stay keyed on leaf/domain
  properties; a degenerate boundary span must decline them, not silently reshape.
- **Scratch backend is inconsistent today.** Panel terms allocate on
  `get_backend(A)` (the container's stored backend); `allocate_workspace` uses
  `get_backend(ops.av.data)` (the array's). Identical in production, but it means a
  backend-level allocation probe cannot see interior scratch. Step 3 should settle on
  the container backend.

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
