# TLR GEMM — row-basis TODO

Tracks remaining work on the global row-basis TLR×TLR→TLR path
(`_row_basis_gemm!` and friends). The algorithm is specified in
[alg.md](alg.md); the architecture in [DESIGN.md](DESIGN.md). Stages 1–3 are
numerically complete and unit-tested. All items below concern the **executor**,
which is currently a naive sequential loop that ignores §7 of `alg.md`.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

## State (2026-07)

- Stage 1 `build_row_basis!` — [row_basis.jl](../row_basis.jl) — done, tested.
- Stage 2 `accumulate_row_coefficients!` — [coefficient_accumulation.jl](../coefficient_accumulation.jl) — done (both `t≤rA` / `t>rA`), tested.
- Stage 3 `merge_row_basis_tile!` — [orthogonal_merge.jl](../orthogonal_merge.jl) — done (β, empty tiles, residual reorth), tested.
- Driver `_row_basis_gemm!` — [row_basis_gemm.jl](../row_basis_gemm.jl) — correct, **CPU-only**, allocation-heavy, no budget/concurrency.
- Planner `row_basis_workspace_plan` — [row_basis_workspace.jl](../row_basis_workspace.jl) — **written but unused (dead code)**.

## Bugs / regressions found (routing CPU through row-basis)

- [x] **Bug A — merge `invalid maxrank`.** `merge_row_basis_tile!` asserted
  `maxrank <= t + rC`, but the driver passes `C.maxrank` as a *cap*. When the basis
  rank `t` is below the cap (e.g. a single contraction tile, β=0), it threw. Fixed:
  effective cap is `min(maxrank, active)` per prune (orthogonal_merge.jl).
- [x] **Bug B — rectangular / non-square tiles.** Generalized the row-basis path to
  distinct `(bm, k, bn)` dims (bm = A/C row tile, k = contraction, bn = B/C col tile):
  Stage 2 sizes `M` and the `Zstack` by `bn`; the merge sizes its right factor
  (`Vmerge`,`Vtmp`) by `bn` and its left factor by `bm`. New `bn` keyword on
  `CoefficientWorkspace` / `OrthogonalMergeWorkspace` (defaults to the square case).
  Covered by a new `row_basis_gemm.jl` rectangular test (β=0 and β≠0).
- [x] **Stale `gemm_tlr_output.jl` tests.** col×row and β≠0 no longer throw (row-basis
  supports both); dropped those two `@test_throws`, kept the regular-grid guard.

## Plan

- [x] **P1 — Hoist workspaces out of the i/j loops.** Done: `RowBasisWorkspace`,
  `omega`, `gamma` once before the row loop; `CoefficientWorkspace` and (β=0) merge
  scratch once per row. **750 KB → 244 KB on an 8×8 grid (3.1×).** 43/43 green.

- [x] **P3 — Remove per-tile host syncs.** Done: reusable 1-element host mirrors in
  the merge workspace; scalars move via `copyto!` (bulk, no per-call alloc, single D2H
  on CUDA). `Array(values)` per-row eigenvalue copy retained (host rank loop, O(qm)).

- [x] **P4 — Enable CUDA.** Dropped the CPU-only gate; all non-transposed layouts now
  route to row-basis on both backends (transposed → M4). Fixed a CUDA-only bug: the
  compressed `basis.P` is a strided row-subset view, which CUBLAS rejects in the batched
  coefficient GEMMs — now compacted into a contiguous `t×(K*rA)` block per row. CUSOLVER
  eigh already present. Added CUDA-inclusive β≠0 and (via gemm_tlr_output) rectangular
  coverage. Row-basis subset 47/47, output subset 22/22 on CPU+CUDA. **Note:** still
  correct-but-slow on GPU (per-tile host syncs + skinny GEMMs) until P5.

- [ ] **P2 — Wire the budget/planner.** Thread `max_workspace` into the driver; use
  `row_basis_workspace_plan`/`workspace_fits` to choose `q` (Stage-2 depth) and reject
  over-budget runs. Stops silently overrunning `max_workspace` (alg.md §7.4).

- [x] **P5 — Concurrency (β=0 path).** Two batched pieces:
  - **Batched row merge** — every tile in a row shares Q, so the whole row is pruned in
    one `prune_orthogonal_columns!` call with two D2H copies instead of 2·qn.
  - **Batched coefficient accumulation** (`_accumulate_row_block!`) — the per-(k,j) S/R
    GEMMs and the terminal Zstack GEMM run as three batched calls per row (batch dims
    K·qn, K·qn, qn) instead of qn sequential per-tile calls.

  **Measured (nt=16, sweep b, CPU vs CUDA):** batching wins in the launch/sync-bound
  regime and GPU crosses over to winning around b≈128:
  `b=16: 4.4/37.5ms · b=64: 57/87 · b=128: 257/242 · b=256: 1047/856` (CPU/CUDA ms).
  The b=16 CUDA case dropped from 81 ms (pre-P5) to 37.5 ms (2.2×). At very large b
  (≥512) each per-tile GEMM already saturates the device (fp64-compute-bound), so
  batching is ~neutral there — expected. β≠0 stays on the per-tile path (future work).
  Row concurrency `h` and depth `q` budgeting deferred to P2.

- [ ] **P6 — Perf acceptance (alg.md §M5.5).** Benchmark all four layout pairs and
  tiny/full budgets vs the M4 dense sink; capture warmed allocations; enforce
  "no row/Z-panel copy on preferred layout, no >15% regression".

- [ ] **Tests / follow-ups.** Add a merge test forcing residual rank `rho < rC`
  (near-parallel `Uold`/`Q`). Coefficient-aware `gamma` (alg.md §3) is deferred —
  driver currently hard-codes `gamma = 1` (row_basis_gemm.jl:53-54).
