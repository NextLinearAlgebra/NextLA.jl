# TLR GEMM — plan: finish accumulation + ragged ranks

Tracks the TLR×TLR→TLR work: the row-basis path (`row_basis/`, algorithm in
[alg.md](alg.md), architecture in [DESIGN.md](DESIGN.md)) and the planned
storage-contract change to per-row/per-column maxrank ("ragged ranks").

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

## State (2026-07-24)

Shipped and green (row-basis 47/47, output 22/22, full TLR 558/558, CPU+CUDA):

- Stages 1–3 numerically complete and unit-tested
  ([basis.jl](../row_basis/basis.jl), [coefficients.jl](../row_basis/coefficients.jl),
  [merge.jl](../row_basis/merge.jl), driver in [driver.jl](../row_basis/driver.jl)).
- Rectangular `(bm, k, bn)` tiles; β≠0; CUDA enabled (all non-transposed layouts
  route to row-basis; transposed → M4 dense sink, β=0 only).
- β=0 path batched per row: coefficients (`_accumulate_row_block!`, 3 grouped-batch
  GEMM calls/row) + single batched prune. GPU crossover vs CPU ≈ b=128;
  b=1024,r=8: row-basis 571 ms vs dense 1575 ms (2.8×).
- Workspaces hoisted (3.1× fewer allocs); host-mirror scalar reads; merge maxrank
  cap fix; merge `_finish!`/`_scatter_tile!` dedup.
- **Known envelope limit (measured):** shared basis rank `t = min(b, K·rA)`
  saturates once `rA ≳ b/K`; saturated rows do dense-sized work *plus* basis
  overhead — up to 1.9× slower than the M4 dense sink. Unsaturated rows win
  1.4–2.8×. This drives Phase A1.
- Planner in [workspace.jl](../row_basis/workspace.jl) still dead code and stale
  (models per-tile shapes, not the batched ones) — absorbed into Phase B2.

## Ragged-rank contract analysis (summary, 2026-07-24)

Proposal: per-**row** maxrank for row-major factor storage, per-**column** for
col-major (rank varies along the storage-panel axis only).

- **Key property:** panels stay rectangular and zero-copy (`[b, mr_i, K]`), so
  every panel fusion survives: Stage-1 `:j` fuse, Stage-3 GEMM-K/FoldLeft stacks,
  M5 `Ubar`/`Zstack`. What breaks is batching *across* rows of A / cols of B —
  those become **grouped** batches (uniform within a group).
- **Migration is incremental:** padded factor columns are zero, so one can store
  per-row ranks but compute any run padded to the run-max rank — no batching
  changes at all — then tighten: rank-bucketed runs → exact grouped GEMM.
- **Grouped GEMM:** CUDA.jl ≥5.11 wraps `cublas[SD]gemmGroupedBatched`
  (FP32/FP64 only; other compute modes → one batched call per group). AMD:
  loop of batched calls over a stream pool (house pattern exists in
  compress.jl). CPU: already a heterogeneous loop.
- **M5 payoff:** `S_i = min(b, K·mr_i)` — saturation becomes per-row; only
  genuinely high-rank rows fall back to dense.
- **Two-pass C:** capacity from host-side `ranks(A)`/`ranks(B)`:
  `mr_i(C) = min(max_j Σ_k min(rA[i,k], rB[k,j]), min(bm,bn), rmax)`; loose
  bound → cap hard, optionally compact after fill; β≠0 needs a capacity policy
  (pre-sized C or realloc).
- **Compression v1:** sketch at uniform width `min(max_i mr_i, tm, tn)` into
  dedicated scratch (output-aliasing breaks when `mr_i < S`), per-slab cap
  array in `prune_randqb_columns!`, ragged scatter.
- **Costs:** offset-table plumbing through container/operands/kernels, loss of
  strided-batched on affected paths, more+smaller launches if grouping is done
  naively (hedged by run-max padding and rank bucketing).

## Phase A — finish accumulation on the uniform contract

Order chosen to *shrink the surface* before the ragged migration.

- [x] **A1 — Per-row saturation guard.** Shipped (2026-07-24). Mechanics:
  - pre-guard: `S_full < θ·b` ⇒ no row can saturate, guard disarmed;
  - armed guard caps the sketch at `S = ⌈θ·b⌉` (a wider basis would be discarded
    anyway) and passes `tguard` so `build_row_basis!` returns right after rank
    detection for saturated rows (skips rotations + residual);
  - saturated rows (`t ≥ θ·b`) run through the M4 dense sink via synthesized
    single-row `RowRun`s (`_m4_row!`); after `SAT_STREAK_CUTOFF = 2` consecutive
    saturated rows the rest of the matrix routes dense without probing;
  - fallback exists for β = 0 + row family only; otherwise rows stay on the
    (correct) row-basis path. `sat_threshold` exposed on `gemm!` (θ ≥ 2 ⇒ pure
    row-basis, θ = 0 ⇒ pure dense);
  - `basis.residual_sq` now folded into per-tile residuals (diagnostic add,
    read once per row, only when `eps_basis > 0`).
  *Gate (CUDA, nt=16, b=512):* r=8 **359 vs 496 ms** (row-basis wins 1.4×);
  r=128 (deep saturation) **5445 vs 5479** (ties/wins); boundary band
  (t ≈ θ·b: r=16 +13%, r=32 +9.5%) carries the per-row probe premium — the
  latency-bound eigh/CholQR2 of two probes; irreducible without batched basis
  builds (C2) and tunable away via θ when the rank regime is known.
  Tests: guard testset (θ default/2.0/0.0 routes, column-family fall-through,
  mixed-rank routing); suites 55/55 + 22/22 CPU+CUDA.

- [x] **A2 — Stage-2 unification + β≠0 cleanup.** Shipped (2026-07-24):
  `coefficients.jl`/`CoefficientWorkspace` deleted — β≠0 now takes its whole-row
  coefficients from the same batched `_accumulate_row_block!` as β=0 (α folded
  there; merge called with α=1) and loops only the merge. One merge workspace per
  row (old factors padded to `maxrank` width — zero-padded columns leave the
  algebra unchanged and make the tile dims uniform, batching-ready). Merge lost
  the `rho0` shortcut/sync: the prune kernel zero-pads `chol.V`, so an absorbed
  residual vanishes through the second-pass algebra on its own; tail writes are
  sized by the one remaining `rho` read. Replacement unit test for the batched
  Stage 2 (both `t≤rA`/`t>rA` branches). Suites 61/61 + 22/22.
  **Gate result (honest miss):** β≠0 CUDA at b=256,r=8: 929 ms vs β=0 212 ms
  (4.4×) — and the pre-A2 baseline measured 924 ms, i.e. allocs/packs/syncs were
  *not* the bottleneck: the serialized per-tile merge compute (2 CholQR2 passes +
  ~8 small latency-bound kernels per tile, pipeline-draining) is. Closing the
  gap needs the **batched β≠0 merge** (per-slab active-columns prune + cross-tile
  batched CholQR2) — tracked in C2, now the named owner of the ≤2× gate.

- [~] **A3 — Copy/launch removal bundle.**
  - [x] Reuse one Ω across rows (shipped 2026-07-24): draw it once per `gemm!`
    call, then reuse it for every row-specific `Ubar` basis build.
  - [ ] Store `P` transposed in the basis workspace (kills the per-row `Pc`
    compaction).
  - [ ] Add a shared-Q variant of `prune_orthogonal_columns!` (kills the `Qm`
    materialization).
  - [ ] Skip redundant per-tile fills in the β=0 scatter (C already zeroed).
  - [ ] Hoist packed B panels out of the row loop for non-preferred layouts.
  *Gate:* warmed allocation count strictly down; no regression in the b-sweep.

## Phase B — ragged ranks (per-row / per-col maxrank)

- [ ] **B1 — Container + addressing.** Flat factor buffer + per-row
  `offsets`/`mr` tables on `TLRMatrix` (interior first; boundary regions keep
  global maxrank initially); thread through `InteriorOperand`
  (`rankdim` → per-row); uniform ranks remain the degenerate case so the
  current behavior is the bit-comparable oracle. Paths not yet migrated must
  throw on non-uniform, never silently mis-read.
  *Gate:* uniform-ragged equivalence on the full suite.

- [ ] **B2 — Run-max-padded compute + budget/arena (absorbs old P2).** M4 + M5
  consume ragged storage padded to run-max rank (no grouped GEMM yet). Scratch
  moves from fixed 5-d tensors to one liveness-carved arena with offset tables
  (alg.md §7.1) — built once, for the ragged layout. Wire `max_workspace`:
  jblock/q chunking of `Rstack`/`Sbuf` (today unbounded, ~1 GB at
  nt=64,b=512,r=64), rewrite the planner model to the batched shapes.
  *Gate:* mixed-rank suite green; budget-compliance test (tiny budget runs
  correctly, never exceeds); planner no longer dead code.

- [ ] **B3 — Compression + two-pass C.** Uniform-width sketch into scratch,
  per-slab caps in the prune kernels, ragged scatter; `gemm!` pass-1 capacity
  inference from `ranks(A)`/`ranks(B)` + allocation, pass-2 fill; document the
  β≠0-into-undersized-C policy. *Gate:* compress→gemm round-trip with skewed
  ranks; measured memory footprint vs global maxrank on a skewed case.

- [ ] **B4 — Measured tightenings.** Native grouped GEMM (CUDA FP32/64;
  stream-pool fallback elsewhere); rank-bucketed run scheduling; packed-`Ubar`
  (Σ ranks < b < K·mr_i regime). Each lands only with a reproducible win.

## Phase C — acceptance + backlog

- [ ] **C1 — Perf acceptance (alg.md §M5.5).** Four layout pairs, skewed +
  uniform ranks, tiny/full budgets, CPU+CUDA, vs the dense baseline; warmed
  allocations; no >15% regression unexplained. Add the merge `rho < rC` test
  (near-parallel `Uold`/`Q`).
- [x] **C2a — Batched β≠0 row merge. Shipped (2026-07-24), gate met:**
  β≠0 CUDA b=256: nt=16 **871 → 350 ms** (ratio vs β=0: 4.27× → **1.70×** ≤ 2×);
  nt=8: 147 → 67 ms (1.65×); per-tile 3.40 → 1.37 ms; profiler events per call
  305k → 111k, host-API share 46% → 39%. Implementation `merge_row_block!` +
  `BatchedMergeWorkspace` in row_basis/merge.jl; per-tile `merge_row_basis_tile!`
  retained as the unit-tested reference. New tests: batched-vs-per-tile equality
  on identical inputs (ranks, reconstructions, energies; capped + uncapped) and
  the explicit zero-tail invariant after CholQR2 compress. Suites 85/85 + 22/22.
  Original scope note:
  Full-width batching at `rcap = maxrank`, exploiting the zero-tail invariant:
  `prune_cholqr_coordinates!` zero-pads both factor tails per slab, so batching
  every merge at `[bm, rcap, g=qn]` and pruning at the uniform active width
  `t + rcap` is *algebraically equivalent* to per-tile `t + ρ_j` (the energy
  prune drops exact-zero columns first under any cap). Consequences: **no host
  `ρ` read at all** (0 syncs in the merge body, vs 3/tile), two batched
  `mixed_cholqr2_compress!` calls per row (the CholQR2 machinery is already
  slab-batched), full-width batched tail GEMMs, one batched final prune, one
  rank/error vector read per row. `merge_row_basis_tile!` is retained as the
  unit-tested reference; new tests: batched-vs-per-tile equality on identical
  inputs, explicit zero-tail invariant, plus the existing β end-to-end suite.
  *Gate:* β≠0 ≤ 2× β=0 at b=256 (1.3–1.5× is target, not promise — β≠0 does
  real extra work); profile launch count + DtoH copies before/after.
  **C2b (deferred until profiling demands):** per-slab active-count prune —
  becomes B3 infrastructure for ragged compression regardless. For the ragged
  future, run C2a per equal-C-capacity bucket (row-major C: one bucket per
  output row; col-major C: bucket output columns by capacity).

- [ ] **C2 — Backlog (measured, unscheduled):** basis build batched across rows
  or two-slot pipeline (§7.3; basis ≈28% of GPU time, latency-bound eigh);
  coefficient-aware `gamma`; B-side/right-basis dual (explicit non-goal for now).
