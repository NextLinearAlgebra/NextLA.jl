# TLR GEMM — roadmap to the adaptive one-sided algorithm

Target design: [algorithm.tex](algorithm.tex) — product-aware weights, a
*certified* shared basis built by adaptive extension, per-tile routing
(R1/R2/R3) executed group-wise, and a factor-list fallback that never
materializes a dense tile. Architecture notes: [DESIGN.md](DESIGN.md).
The older [alg.md](alg.md) describes the **currently implemented** design and
is superseded by `algorithm.tex`; keep it until M3 lands, then delete.

Scope for now: **one-sided** (row or column basis, chosen a priori per
`algorithm.tex` §A Priori Side Selection). Two-sided/BLR² is out of scope
until the one-sided path is complete.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

## What survives, what goes

The current implementation (all green: 85/85 row-basis, 22/22 output,
596/596 full TLR, CPU+CUDA) maps onto the new design as follows.

**Keep as-is — these are the building blocks the new algorithm needs:**
- `mixed_cholqr2_factor!` / `mixed_cholqr2_compress!` and the batched CholQR2
  workspace — used by every phase.
- `prune_orthogonal_columns!` / `prune_cholqr_coordinates!` and the
  **zero-tail invariant** (padded columns are exactly zero) — this invariant
  is load-bearing for all group padding in the new scheduler.
- `_accumulate_row_block!` — *is* Phase 3, including both parenthesizations
  (`t≤rA` → `T_{iℓ}` precompute; else `S_{iℓj}`) and batching over `j`.
- `merge_row_block!` + `BatchedMergeWorkspace` (C2a) — *is* Phase 5.
- Operand accessors (`rowpanel`/`colpanel`/`tilefactor`), precision policy,
  `precision_gemm_batched!`, arena/`_take` pattern, test helpers and the
  batched-vs-reference oracle pattern.

**Rewrite:**
- `build_row_basis!` — replace eigenvalue-based rank selection with the
  certificate `δ_i = E_i^w − Σ c_ℓ²‖P_{iℓ}‖²` plus adaptive BCGS2 extension.
  Keeps the blockwise sketch, CholQR2 and the workspace arena.
- `_row_basis_gemm!` — restructure from "loop tiles" to
  route → partition → batch per group.

**Scrapped (done 2026-07-25):**
- [x] `row_basis/workspace.jl` planner (`RowBasisWorkspaceShape/Plan`,
  `workspace_fits`) + `test/TLR/row_basis_workspace.jl` — dead code modelling
  per-tile shapes that no longer exist.
- [x] The A1 saturation guard: `_m4_row_context`, `_m4_row!`,
  `SAT_STREAK_CUTOFF`, the `sat_threshold` kwarg on `gemm!`/`_row_basis_gemm!`,
  the `tguard` kwarg on `build_row_basis!`, and the guard testset.
  **R3 subsumes it** — routing to a factor-list compressor beats routing to a
  dense-slab sink (`4sρ(b_m+b_n)` vs forming a `b_m×b_n` tile), and it is a
  per-tile decision rather than per-row.
  *Known regression until M2 lands:* saturated rows are no longer diverted, so
  they run the (correct but slower) row-basis path — up to ~1.9× slower than
  the dense sink in the deeply saturated regime. This is the gap R3 closes.

**Still to scrap, but not yet (they have live dependents):**
- `tlr_output.jl` (dense-slab sink) and `_compress_run_into_factors!` — still
  the only route for transposed operands; delete at M2 once R3 covers them.
- `_row_basis_eigh!` and its CUSOLVER extension method — `build_row_basis!`
  still uses the eigensolve for rank selection; delete at M1 when the
  certificate + adaptive extension replaces it. That also removes a per-row
  host sync (`Array(values)`) and the only CUSOLVER dependency in this path.

## Milestones

- [ ] **M1 — Phase 0 weights + certified basis.**
  Compute `c_ℓ = ‖B_{ℓ,:}‖_F` once per GEMM from `‖V^B_{ℓj}‖_F` (the
  orthonormal-`U` invariant makes this exact), and `ρ_i`, `γ_j`, `E_i^w`
  from rank metadata. Rewrite `build_row_basis!` per Alg. 3: weighted
  blockwise sketch, CholQR2, `P_{iℓ} = Q_iᵀU^A_{iℓ}` (unweighted), certificate
  `δ_i`, then extend-by-BCGS2 until `δ_i ≤ ε²E_i^w`. `gamma` is already a
  parameter of the current kernel, so Phase 0 mostly fills in real values.
  *Gate:* certificate is a true upper bound on the achieved basis error
  (compare against explicit `‖Ū − QP‖_F`); rank vs the current eigh-based
  selection measured on structured and random inputs — **expect some rank
  regression**, since extension does not rotate to the optimal subspace;
  `@inferred` clean; CPU+CUDA.

- [ ] **M2 — R3: factor-list tile compressor.**
  Implement `FactorQB` (Alg. 2) and `DirectTile` (Alg. 8) — two-pass sketch
  over `{(U^A_{iℓ}, R^A_{iℓj})} ∪ {(U^C_{ij}, βV^C_{ij})}`, batched over a
  tile group, never forming the tile. Then delete the A1 guard and route
  saturated/uneconomical tiles here instead.
  *Gate:* accuracy matches the dense-slab sink on the cases it replaces;
  faster than it in the saturated regime (where dense currently ties/wins);
  transposed operands covered so `tlr_output.jl` can be deleted.

- [ ] **M3 — Routing + scheduling.**
  `PartitionRow` (Alg. 4) on the host; global `γ`-sort once per GEMM;
  three contiguous segments per row; min-group fold into R1; padding to
  group max; batched execution per group (Alg. 1). Keep Phase 3 and Phase 5
  batched across the whole row as today.
  *Gate:* **launch count and D2H copies per call must not exceed the current
  batched implementation by more than the number of non-empty groups** — this
  is the regression the whole scheduling section exists to prevent (measure
  with `CUDA.@profile`, as in the C2a gate); no accuracy change vs M2.

- [ ] **M4 — R2: coefficient-space sketch.**
  `CompressTile` R2 branch: `FactorQB` on the `M_{ij}` factor list
  `{(αV^B_{ℓj}, W_{iℓj}ᵀ)}`, small SVD, then one rotation through `Q_i`.
  *Gate:* on inputs where `γ_j ≪ t_i`, R2 beats R1 in time at equal accuracy;
  the metadata bound `r̂ = min(t_i,b_n,γ_j)` is never violated by the
  achieved rank.

- [ ] **M5 — Side selection + column basis.**
  Implement `R_side = (b_n/b_m)(r_B/r_A)(t̄/ū)` and the dual construction
  (mirror of M1 with `d_ℓ = ‖A_{:,ℓ}‖_F`), including the shifted/normalized
  CholQR2 the dual's conditioning requires. Optional sample-probe to confirm
  the a priori choice with the certificate.
  *Gate:* on a wide output (`b_n > b_m`) the dual is selected and is faster;
  round-trip accuracy equals the primal; conditioning test with block norms
  spanning several orders of magnitude.

- [ ] **M6 — Acceptance.**
  Four layout pairs, structured vs random inputs, tiny/full budgets, CPU+CUDA;
  warmed allocations; no unexplained regression >15% against the pre-rewrite
  baseline. Retire `alg.md`.

## Deferred (unchanged in intent, re-sequenced after M6)

- **Ragged ranks (per-row/per-col maxrank).** Note that `algorithm.tex` is
  already written in terms of true per-tile ranks `r_{A,iℓ}` and their sums
  `ρ_i`, `γ_j` — so the new algorithm *assumes* ragged ranks conceptually and
  merely tolerates the padded uniform contract. Migration plan (container +
  offset tables → run-max-padded compute + arena/budget → compression with
  per-slab caps + two-pass C allocation → grouped GEMM) is unchanged; the
  earlier analysis is preserved in git history.
- **Basis-build batching across rows / two-slot pipeline** — basis build was
  ~28% of GPU time and is latency-bound; M1 removes the eigensolve, so
  re-measure before scheduling this.
- **Two-sided (BLR²) construction** — requires a container that can retain
  `Q_i Γ_{ij} G_jᵀ`; out of scope until the one-sided path is complete.
- **Boundary / non-regular-grid tiling** — still rejected by
  `_validate_tlr_output`; a whole matrix class remains unsupported.
- **Transposed operands with `β≠0`** — currently throws (M4 sink is β=0
  only); M2 should remove the restriction.
