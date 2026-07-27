# TLR GEMM — worklog and roadmap

The target design is [algorithm.tex](algorithm.tex): a blocked adaptive
randomized range finder (ARA) applied per output tile, with the reduction kept
as an implicit factor list instead of a dense tile.

Reference: Boukaram, Turkiyyah & Keyes, *Randomized GPU algorithms for the
construction of hierarchical matrices from matrix-vector operations*, SIAM J.
Sci. Comput. 41(4):C339–C366, 2019, Algorithm 2.3.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

## Design decisions

### ARA orthogonalization

Each sample block uses `(projection · Cholesky-QR)²`, matching Algorithm 2.3.
The interleaving is required in finite precision: the first normalization can
amplify a near-null column's overlap with the existing basis, so the second
projection must happen after that amplification.

The factorization is unshifted. A shift places a data-independent floor under
the triangular diagonal used by the stopping test and can make a requested
tolerance impossible to reach. With an unshifted promoted Gram, factorization
breakdown is treated as a rank signal and the valid leading block is retained.
The supported tolerance floor is `sqrt(eps(Tgram)/2)`.

The stopping signal is the diagonal of the composite triangular factor, not
the original sample-column norm. Random combinations of a small residual range
can all have large column norms while becoming dependent within the block; the
triangular diagonal detects that dependence.

### Active packing

R3 uses a dense active prefix. After a tile converges, its run-local state is
swap-removed into a retired suffix. Subsequent GEMM, Gram, factorization, and
triangular-solve calls operate on `1:nactive`, without masked batch slots or
device pointer lists containing holes.

The canonical API stores `C`, `A`, and `B` physically tile-row-major. Sampling
side follows the logical layouts:

- `NN` has only the natural right sample and uses fixed output rows;
- `TT` has only the natural left sample and uses fixed output rows;
- `NT` admits both. A rank-derived retained-workspace estimate chooses a
  fixed-column right run or a fixed-row left run for the whole operation;
- `TN` admits neither natural contraction stack and is deferred.

The side is never selected per tile: doing so would fragment the batch and
require a second active-packing layer. Full-capacity panels are zero-copy.
When rank metadata trims a panel below its physical capacity, its active
prefix is packed once into compact run-local storage and reused for every ARA
pass; reshaping a strided rank slice is not a valid GPU GEMM operand.

### Truncation

After range finding, one co-range apply forms `Z = XᵀQ`. A thin SVD of `Z`
provides the optimal represented-matrix truncation because `Q` is orthonormal:
the singular values of `Z` are the singular values of `QZᵀ`. The reported
residual is therefore the achieved discarded energy, not an estimate.

## Workspace contract

Compression allocates its reusable numerical storage up front. The live ARA
set is:

1. persistent basis `Q` and one Gaussian block;
2. one block sample and BCGS2 coefficient buffer;
3. one promoted panel, one promoted Gram, and two triangular factors;
4. one block of convergence scalars plus status/cut vectors;
5. co-range storage, written directly into the output factor panel;
6. rank and error diagnostics.

The removed shift machinery and wide one-shot sketch buffers are not part of
the workspace. Sampler, sample-output, basis, co-range, and triangular batch
views are built with the workspace. Packed execution reuses these arrays and
only restricts operations to the current active prefix.

The remaining material allocation is final truncation: backend thin-SVD APIs
currently return fresh factor arrays and may allocate solver workspaces.
Making truncation consume caller-owned storage is part of R4.
An empty co-range returns empty factors directly and never enters LAPACK or a
GPU solver.

The factor-application layer (`ColumnRunCoupling`, `RowRightRunCoupling`,
`RowLeftRunCoupling` in `tlr_result/run_coupling.jl`) extends the same contract to the
couplings and their applies, on the run's hot per-pass sampler specifically.
No sampling pass allocates numeric storage: `S`, the per-column/row coupling
stacks, and the `H`/`T`/`G`/`W` scratch are sized once at run construction
and reused for every pass, restricted to the active prefix by view rather
than resized. No sampling pass allocates a device pointer list either for
its ragged operands: a batched GEMM whose operand cannot be expressed as a
single strided-batched view (because active-prefix packing permutes it, or
because it is a run-owned scratch buffer sharing that same call) goes
through a `BatchPtrDescriptor` built once per run at construction; a
converged member's swap-removal into the retired suffix updates only the
tiny device address table (`swap_batch_ptrs!`), never the numeric buffers a
stable-address field points into. Run output (`U`, `V`) and diagnostic
buffers (`ranks`, `err_sq`) are likewise driver-owned and reused across the
traversal's outer loop in `gemm!(C::TLRMatrix,...)`, not reallocated per
output row/column.

This contract stops at the sampler. Three things it does not cover, by
deliberate choice, each documented at its own site:

- The co-range apply (called once per run after convergence, not once per
  pass) is left entirely on the pre-existing `Vector`-of-views path. Its
  cost does not compound with pass count the way the sampler's did, and in
  `RowRightRunCoupling`/`ColumnRunCoupling` its ragged operand is batched
  over member × contraction-tile rather than member-only, which would need
  a second, `q_k`-times-larger descriptor purely to cover a call that runs
  `O(q_m)`/`O(q_n)` times total across a `gemm!` call, not once per pass.
- Within the sampler, a call whose operands are all run-owned and
  non-ragged (`ColumnRunCoupling`'s and `RowLeftRunCoupling`'s T/W-formation
  in `RowRightRunCoupling`) is left on the `Vector`-of-views path too when
  it is not reshape-representable as a true strided batch in the operands'
  current memory layout. These calls were never the source of the
  allocation this contract targets — only ragged operands force pointer
  form — so descriptor fields were not added for them.
- `Y` (the ARA basis being grown) is owned by the caller's `ARAWorkspace`,
  not the run, so a sampler call that mixes it with a ragged operand builds
  its pointer array fresh each pass (once per `apply_*_run!` call, reused
  across every GEMM within that call, not rebuilt per GEMM). Caching it
  would require threading a descriptor through `numerics/ara.jl` itself;
  R4's "reusable run workspaces" item is the natural place for that.

Separately, `ara_cholesky_pass!`'s `trsm_batched!`/`potrfBatched!` calls
allocate a transient device pointer array internally on every ARA pass,
independent of anything above — a pre-existing cost this contract does not
touch. Extending `BatchPtrDescriptor` to those primitives is a candidate
follow-up, not started here.

## Cleanup before R3/R4 — 2026-07-27

The unsupported shared-basis TLR-result implementation and dense-slab
recompression fallback were removed together with their tests and obsolete
design document.

The old general orthogonalization and coordinate-pruning frameworks were also
removed. Active ARA now owns one small `ARACholeskyWorkspace` and one
`ara_cholesky_pass!` primitive that performs only promoted copy, batched Gram,
unshifted batched Cholesky, upper-factor copy, and batched triangular solve.
ARA calls the primitive twice with projection interleaved. Shift buffers,
factor-output construction, rank-policy code, and the pre-Gram column-norm
buffers are gone.

This cleanup deliberately leaves dense-output GEMM, ARA compression, the R1/R2
factor-list primitives, and the R3 packed column-family implementation intact.

Focused verification:

- CPU: numerical primitives 4/4; ARA bookkeeping 25/25, basis 24/24,
  truncation 34/34, and interleaving 6/6; compression 110/110; dense-output
  GEMM 66/66; R1 23/23, R2 20/20, and packed R3 40/40.
- CUDA through `../gpuenv`: numerical primitive 1/1; ARA bookkeeping 2/2,
  basis 10/10, and truncation 9/9; R1 2/2, R2 3/3, packed R3 40/40; focused
  packed compression reconstruction passed.
- Whole TLR suite intentionally not run unless focused failures require it

## R3 canonical TLR-result integration — 2026-07-27

Added

```julia
gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
      alpha=true, beta=false, transA='N', transB='N',
      tol=0, rel=false, eps_rel=nothing,
      r_required=10, block=32, compute=nothing)
```

for regular tile grids with physical `TileRowMajor` storage. The implementation
supports right and left sampling, swap-compacts converged members, performs one
co-range apply and batched truncation per run, and scatters directly into the
canonical row-major output. `beta*C` participates as an additional factor pair.

For `NT`, both natural sampling sides exist. The chooser uses the active rank
caps from the operand metadata and compares the retained right/left run
workspace. This is an operation-level decision, so every run has one uniform
batch shape. The more complicated `TN` case, arbitrary physical orders,
boundary tiles, and reduction sub-panelling remain explicitly outside this
contract.

Focused verification:

- CPU: canonical default order, `NN`, `NN` with nonzero beta, both `NT` rank
  choices, `TT` with and without beta, chooser behavior, and rejection
  boundaries: 19/19.
- CUDA through `../gpuenv`: `NN`, right-selected `NT`, and `TT`
  reconstruction and rank-cap checks: 6/6.
- The GPU test also verifies compact active-rank panel packing; full-rank
  canonical panels remain views.

## Dense-output global workspace arena — 2026-07-27

Dense-output GEMM now exposes exact global minimum and maximum queries under a
two-stream execution model:

```text
global = interior + max(right, bottom, corner)
```

The interior occupies one stream. Right, bottom, and corner execute serially
on the second stream and reset/reuse one auxiliary arena slice. The static
`InteriorFirstWorkspace` policy reserves the auxiliary minimum, grows the
interior to full width, then grows the auxiliary slice. This is an explicit
temporary policy, not a claimed saturation model.

The required `workspace` keyword accepts either a global byte count or a
reusable `DenseGemmWorkspace`. Integer mode allocates one typed arena for the
call; object mode reuses the same numerical allocation and two streams.
Budgets below the global minimum fail, while bytes beyond the global maximum
cannot change the schedule. The old `DEFAULT_GEMM_BUDGET` and
`max_workspace` interface were removed.

The bounds now include the previously omitted full-batch dense-diagonal
intermediates as well as the regular low-rank and boundary terms. Focused
tests cover correctness at both endpoints, a large tall/skinny min/max gap,
all transpose combinations, rejection below the minimum, the regional
sum/max identities, and repeated use of one arena. Dynamic cross-stream
lending is intentionally deferred until profiling establishes a profitable
event boundary.

Focused verification:

- CPU workspace bounds, split identities, endpoint correctness, and arena
  reuse: 19/19; direct-term budget compliance: 13/13.
- CPU dense-output GEMM: dense-diagonal 4/4, fully low-rank 7/7, fold
  selection 7/7, logical operands 17/17, complete-grid transpose 13/13,
  boundary transpose 6/6, and one-dense-operand paths 12/12.
- CUDA through `../gpuenv`: a boundary-tiled fully low-rank product passed
  twice using the same two-stream `DenseGemmWorkspace` and unchanged device
  storage; dense-diagonal boundary and TLR×dense arena paths also matched
  their dense references.

## R3 performance closure — 2026-07-27

An audit of `tlr_result/run_coupling.jl` found the R3 sampling couplers rebuilding
`Vector`-of-views batch-pointer lists from scratch on every ARA sampling
pass — on CUDA/AMDGPU each such call allocates and frees a fresh device
pointer array (`_unsafe_batch_strided`/`_device_batch_strided`), several
times per pass, contradicting the workspace contract above (which the
basis-growth side, `ARAWorkspace`, already honoured). This entry closes that
gap on the run's hot per-pass sampler; the scoping choices and the residuals
left in place are documented inline where they occur and summarized in the
"Workspace contract" section above.

Three pieces landed:

1. Reshape fixes in `RowRightRunCoupling`/`RowLeftRunCoupling` that made two
   already-strided-eligible calls (the final reduction and, for
   `RowLeftRunCoupling`, the G-formation) fully strided with no pointer
   array at all — no new infrastructure, just removing a redundant `Aouter`
   field and a no-op `_batch_views` wrapper.
2. `BatchPtrDescriptor`/`swap_batch_ptrs!` (`src/gemm_batched.jl`) and
   `precision_gemm_batched_ptrs!` (`src/gemm_precision.jl`): a build-once,
   swap-maintained device pointer table, plugging onto the pre-existing
   `gemm_batched_ptrs!`/`gemmEx_batched_ptrs!` entry points. CPU never
   constructs one (`_build_batch_ptrs` has no CPU method), so CPU call sites
   keep the original `Vector`-of-views path unconditionally.
3. Wiring the descriptor into all three `RunCoupling` structs' hot sampler,
   per the scoping documented in "Workspace contract" above (co-range and
   non-ragged same-call operands left as-is; `Y`'s pointer array is the one
   per-run-call residual, built once and reused across the GEMMs within a
   single `apply_*_run!` call rather than rebuilt per GEMM).

Also hoisted `gemm!(C::TLRMatrix,...)`'s per-output-row/column `U`, `V`,
`rr`, `ee`, and `_store_tlr_run!`'s `slots_dev` scratch, previously
reallocated on every iteration of the traversal despite being loop-invariant
in shape.

Focused verification (added two new permanent regression tests along the
way: `RangeFind packed row run (R3)`, a fixed-row analogue of the existing
fixed-column swap-heavy fixture, since neither the pre-existing row-run
tests nor the end-to-end `gemm!` tests used deliberately graded ranks to
force real active-prefix swaps for `RowRightRunCoupling`/
`RowLeftRunCoupling` specifically):

- CPU: full TLR suite through `gemm_tlr_r3.jl`, 674/674 including the new
  swap-heavy row-run fixture (with and without beta) for both sampling
  sides.
- CUDA through `../gpuenv`: same suite, 674/674, including the new
  `BatchPtrDescriptor` unit tests (build, single-slot swap, block swap,
  CPU-throws) in `test/gemm_batched.jl`.

Not done: extending `BatchPtrDescriptor` to `ara_cholesky_pass!`'s
`trsm_batched!`/`potrfBatched!` calls (the residual noted in "Workspace
contract"); the allocation-regression test itself (`test/TLR/gemm_r3_alloc.jl`,
tracked separately below).

## R4 scheduler scope — 2026-07-27

R4 treats three scheduling choices as independent:

1. admission scope: one fixed-axis lane, several independent fixed-axis
   lanes, or an arbitrary cross-axis tile pool;
2. execution scope: the existing lane-local couplings or a new mixed-tile
   coupling;
3. reduction granularity: one full-`k` contraction or accumulating
   `k`-chunks.

The first implementation is deliberately
`SingleLane + LaneLocal + FullK`. It preserves the sharing encoded by the
existing couplings: a fixed-column `ColumnRunCoupling` forms one shared
`H`, while fixed-row couplings retain their corresponding shared operand.
Cross-axis admission is not a larger instance of this scheduler: it loses
that sharing and requires a new arbitrary-tile coupling. Reduction chunking
is orthogonal and remains off by default because `c` accumulating terminal
GEMMs move approximately `(2c-1) * b_m * s` sample elements instead of
`b_m * s`, where `c` is the number of chunks.

### R4a — rolling admission within one lane

The lane owns a fixed-capacity slot arena and a pending queue. At an ARA
convergence boundary, members that finish are swap-compacted as in R3, but
filling the released slots requires a new admission primitive rather than
another swap:

```text
admit_member!(slot, pending_member)
    install/update that member's coupling descriptors
    form its one-time S core over the full contraction range
    clear Q, rank/error, convergence, and sample-count state
    make the slot active for the next complete ARA pass
```

Stable-address descriptor fields may be updated in place, but no numerical
buffer is allocated during admission. A slot is not eligible for admission
until its retired member has completed co-range application, truncation, and
output scatter.

Sampling parameters are cohort-global. A newly admitted member starts with
fresh basis and convergence state, but joins the lane's current pass/block
schedule; it does not replay earlier sketch widths or run a private narrower
schedule. This is correct because its future random blocks still define a
valid range finder, but can oversample an easy late arrival relative to an
independent run. Phase-one profiling must therefore report wasted sampled
columns/FLOPs per admitted member in addition to occupancy and active width.

Retirement is wave-batched, not member-at-a-time. After one complete
full-`k` sampling pass and its convergence test, all members retired by that
test form one retirement wave. The scheduler batches co-range application
over the wave using the existing lane coupling, then truncates and scatters
the wave before recycling its slots. This bounds co-range launch growth by
the number of convergence boundaries rather than the number of output tiles.
The initial implementation should retain a packed retirement-wave descriptor
so that rank variation does not force one launch per member.

R4a measurements are: active and pending width per pass; underfilled tail
passes and time after pending-lane exhaustion; admission-time `S` cost;
retirement-wave sizes and co-range launch time; sampled columns/FLOPs per
member relative to a standalone run; rank/pass distribution; and time split
among contraction, orthogonalization, and finalization.

### R4a.5 — multi-lane batch growth

Before implementing shared cross-lane slot ownership, recover cross-lane
occupancy by growing a single batched-GEMM call's batch count rather than by
running several lane schedulers on independent streams. `dense_result/low_rank_terms.jl`'s
`execute_dense_stage3!(::KAsSerialLoop, ::FoldRight, run::ColumnRun, ...)`
already does exactly this for the dense-output path: for one fixed
contraction tile `k`, it folds every row-panel × column-panel pair in the run
into one `precision_gemm_batched!` call (the shared operand's pointer is
simply pushed once per pairing, not broadcast) rather than issuing one launch
per pairing. The same mechanism — one bigger pointer-batched call spanning
several lanes' pending/active members — applies here: it needs no stream
primitive, no shared free-list, and no cross-stream ordering, only a larger
batch built from whichever lanes have members ready. It gives up the
same-axis sharing those members would have had in a purely lane-local batch
(each entry becomes an independent multiply-accumulate, matching how
`RowRightRunCoupling`'s already-ragged `Bouter` is handled today), which is
the same structural cost cross-axis admission always pays — this section is
about the mechanism for combining lanes, not about avoiding that cost.

Growing the batch this way is also why a shared, phase-reset arena (see "R4
arena — reusable run/ARA workspace" below) works cleanly here where
independent streams would not: one bigger single-launch batch keeps every
buffer's lifetime sequential and deterministic, so an arena can be sized to
the *max* of what's concurrently live and reset between phases. Independent
streams need their buffers simultaneously resident by construction, which
would force summing arenas instead of maxing them — the two ideas are in
tension, not complementary, so this plan drops streams in favor of batch
growth wherever the two would otherwise compete for the same lane.

R4b (a shared arena with several lane-local cohorts) is gated on profiling
showing a material utilization gap after R4a.5. R4c (one arbitrary mixed-tile
cohort) is gated separately because it needs a new coupling and forfeits
fixed-axis sharing. Full-`k` reduction remains the baseline at every stage;
`ChunkedKReduction(kappa)` is introduced only if measurements show that
full-`k` transient storage, rather than lane tails, is the occupancy limit.

### R4 arena — reusable run/ARA workspace

Completed before scheduling. `ARARunArena` separates whole-run persistent
storage (`Q`, `S`, and packed factor stacks) from rewound phase storage.
Constructor-only `S` packing is released before sampling; after
`ara_build_basis_packed!` completes, sampling/orthogonalization scratch is
released and the same phase storage is reused for co-range application and
truncation. The bound is therefore

```text
persistent + max(S-construction, sampling, finalization)
```

rather than the sum of all three phases. `ara_run_workspace_bytes` computes
the three typed components analytically and arena exhaustion remains a hard
error.

The canonical driver also hoists truncation ranks/errors, member maps, output
panels, and global scatter diagnostics. `TLRGemmWorkspace` owns all of this
numerical storage and can be reused across complete `gemm!` calls;
`tlr_gemm_workspace_bytes` is its allocation-free exact byte query. Passing
an integer validates the available byte count and constructs temporary
storage; omitting `workspace` retains the convenience path. Workspace objects
are tied to one backend, element/rank type, operation geometry, sampling
family, block width, and active rank caps and reject incompatible reuse.

This gives R4a's `admit_member!` the required allocation-free numerical
foundation, including trimmed-rank factor packing used to form a newly
admitted member's `S`. Pointer descriptors and backend-library solver
workspace remain metadata/library storage outside this numerical contract,
as already scoped in "Workspace contract." The scheduler itself is still
untouched.

Focused verification:

- CPU canonical TLR-result GEMM: 19/19; reusable workspace byte accounting,
  repeated backing-storage identity, numerical reuse, undersized integer
  rejection, and incompatible-operation rejection: 7/7.
- CUDA through `../gpuenv`: canonical right/left sampling families 6/6;
  reusable workspaces for `NN`, both `NT` choices, and `TT`, each used twice
  with exact byte accounting: 12/12; hot-pass allocation regression 4/4 and
  traversal-count arena-reuse regression 1/1.

## Roadmap

- [x] **A0 — convergence bookkeeping.** Per-member sample counts, running
  maximum, consecutive-small count, detected rank, and breakdown masking.
- [x] **A1 — batched ARA core.** Reusable workspace and interleaved two-pass
  projection/normalization.
- [x] **A2 — final truncation.** Batched thin SVD, optimal rank selection,
  achieved residual, and zeroed factor tails.
- [x] **A3 — compression integration.** Dense-tile sampler using ARA rather
  than a wide one-shot sketch.
- [x] **R1 — factor-list sampler.** Coupling prologue plus fused right/left
  application for one output tile.
- [x] **R2 — one-tile range finder.** ARA over the implicit factor list.
- [x] **R3 — canonical TLR-result GEMM.** Fixed-column and fixed-row
  packed-active runs, right and left sampling, operation-level rank-based side
  selection, output scatter, and the regular-grid row-major `gemm!` contract.
- [ ] **R4 — scheduler and arena.** Rank-metadata workspace estimates,
  budget-bounded run growth, reduction sub-panelling, reusable run workspaces,
  and caller-owned truncation scratch where supported.
- [ ] **R5 — general-storage integration.** Boundary regions, arbitrary
  physical layout descriptors, the packed/reduced `TN` path, and measured
  scheduling across those cases.

## Deferred

Ragged per-row/per-column ranks; non-regular boundary tiles; arbitrary-storage
and `TN` packing/reduction; two-sided/BLR² output; alternative batched QR
implementations; and a measured run-level fold cost model.
