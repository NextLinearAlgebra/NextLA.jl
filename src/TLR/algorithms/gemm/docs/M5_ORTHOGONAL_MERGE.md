# Milestone 5 — incremental orthogonal-update TLR output

> **Deferred draft — do not implement as written.** The numerical merge/truncation
> algorithm needs revision and validation before implementation. References below to
> `ContractOp`, scheduled contractions, `DenseOutput`, `SlabOutput`, and a generic output
> sink describe the former architecture; M5 must instead consume only canonical factor
> operands, `RegularGeometry`/runs, output-independent Stage 1, and precision/workspace
> utilities. This file is retained as design history and a source of requirements.

This is the implementation prompt for the TLR×TLR→TLR *production* algorithm: a
factor-space incremental orthogonal merge that replaces Milestone 4's dense-accumulate +
recompress sink. It is self-contained — it records what M4 delivered, what was skipped and
why, the merge algorithm and its derivation, exactly what to reuse vs replace in the
codebase, the mandatory numerical guards, a step plan with gates, and the open decisions.

**Do not start from a blank sheet.** Most of the scaffolding (entry point, validation,
lowering, Stages 1/2, budgeted runs, test oracle) already exists from M4 and is reused.

---

## 1. What Milestone 4 delivered (and is kept)

Public API, in `gemm.jl`:

```julia
gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
      alpha=true, beta=false, tol=0.0, rel=false,
      max_workspace, transA='N', transB='N', compute=nothing)
```

Scope actually implemented and **tested (CPU + CUDA, 24 new tests, 397 regression green)**:

- **Regular grid only** (no boundary/tail tiles): validated in `_validate_tlr_output`
  (`gemm.jl`). One interior contraction per output tile; the seven boundary operations are
  empty by construction.
- **`beta == 0`** only (output factors overwritten).
- **Row-family layouts only** — every tile-order pair *except* `A` tile-column-major ×
  `B` tile-row-major. That lone pair is the column family (`KAsSerialLoop`) and throws an
  `ArgumentError`. (Recall only col×row is stuck; col×col is rescued to `FoldLeft`.)

Mechanism (dense accumulate → recompress):

- `SlabOutput` (`contraction/operation.jl`) — a run-local dense target answering the same
  `output_tile` / `output_rowblock` accessors as `DenseOutput`, so **Stage 3 is reused
  unchanged**. `stage3`'s signature was widened to `Union{DenseOutput,SlabOutput}`
  (`lowering/stages.jl`).
- `contraction/sink.jl` — `TLROutputWorkspace` (bounded: slab + staged-GEMM workspace +
  temp factor buffers `Uc`/`Vc` + compression `accum` arena + reusable `tiles`/`p0s`/`q0s`
  batch, all sized to the widest run block), `_tlr_gemm_rowfamily!` (run loop: stages fill
  the slab with `alpha·Σ_k A_ik B_kj`, β=0), and `_compress_run_into_factors!` (wrap the
  slab tiles as `DenseTiles`, run `compress_tiles!`, scatter compacted factors + ranks +
  residuals into `C`'s interior slots).
- `_run_tlr_output!` (`gemm.jl`) dispatches on placement: `KAsGemmK` → row family;
  `KAsSerialLoop` → the deferred-column-family `ArgumentError`.

Verified invariants worth relying on:

- **Factor padding is zero to `maxrank`.** `_prune_rank_kernel`
  (`compression/kernels.jl`) clears columns `k+1 … size(U,2)=maxrank` ("padded downstream
  GEMMs cannot observe stale factors"). So both `compress!`- and `gemm!`-produced TLR
  matrices satisfy `U_full · V_full' == U[:,1:r]·V[:,1:r]'`. The merge relies on this.
- **Workspace inference** is pinned with `@inferred` in `test/TLR/gemm_tlr_output.jl`
  (the `Tin`-regression discipline).
- **Rank overflow** is honest: a tile whose true rank exceeds `maxrank(C)` keeps `maxrank`
  columns and reports `residuals(C) > tol`.

Tests: `test/TLR/gemm_tlr_output.jl` (correctness vs `alpha·A·B`, overflow, inference,
validation), included from `runtests.jl`. Driver `assert_tlr_output_matches_dense` in
`test/TLR/helpers.jl`.

**This dense path is retained as (a) the differential-test oracle for the merge and (b) the
randQB recompression fallback the merge needs on rank overflow. Do not delete or extend
it.**

## 2. What was skipped (re-homed onto the merge path)

| skipped in M4 | why | where it lands |
| --- | --- | --- |
| `beta != 0` | mechanical on the dense path (seed slab with `beta·U_old·V_old'`); *cleaner* on the merge (native factor-space) | M5 step 2, falls out of the merge with `rC>0` |
| column family (col×row) | `KAsSerialLoop`: a tile completes only after the full k-loop; needs depth-first `_column_block` + family-aware arena lifetime. Transpose is self-dual, so it does **not** escape | M5 step 2 (depth-first blocking) |
| `TLRDenseDiagMatrix` output | `SkipDiag` breaks the `FoldLeft` rescue, so even col×col is column-family there; also dense-diagonal tiles are a separate leaf | later (after full-LR merge lands) |
| `gemm_workspace_bytes` TLR variant | budget-sizing convenience | M5 step 3 |
| per-run `carve_tile_workspace` host churn | bounded, non-hot-path (dwarfed by compression compute; one call at default budget) | only if it ever profiles hot |

## 3. Why the merge (the competitiveness argument)

Per output tile, changed cost (tile square `b`, accumulated panel rank `s = q_c·r`):

```
dense accumulate + randQB :  ~ b²·(s + rmax)     (materialize b×b tile, then sketch it)
incremental orthogonal    :  ~ b·s²
ratio (merge / dense)     :  ~ s/b
```

In the regime TLR targets, `s ≪ b`, so the merge is `~b/s` cheaper **and** never allocates
the `b×b` temporary. It also fuses accumulation and recompression into one factor-space
pass.

The honest caveat: the merge's **column-energy truncation is not SVD-optimal** — when the
merged right factor has correlated columns (which happens exactly during accumulation) it
**over-retains rank** vs randQB's randomized-SVD-quality truncation. So the merge wins
flops/memory but can keep higher rank for the same tolerance. Hence the SVD/randQB escape
(= the retained M4 dense path) is a required component, not optional.

## 4. The algorithm

### 4.1 Model and invariant

Every output tile is kept **left-orthonormal**:

```
C_ij = U_ij V_ij',   U_ij' U_ij = I,   rC := rank(C_ij)   (rC = 0 for an empty/first-write tile)
```

Inputs `A`, `B` are ordinary TLR matrices; their `U` factors need **not** be orthonormal
(the merge orthonormalizes what it needs). The output invariant `U' U = I` is what makes
truncation exact and SVD-free.

### 4.2 Panel update in factored form

For an output row `i` and a contraction panel `K = {k_1,…,k_q}` (`s = Σ_k rank(A_ik)`):

- **Shared left factor** (this is literally the FoldRight U-stack; shared across all `j` in
  the row, so it and its Gram are computed once per row):
  ```
  W_i = [ U^A_{i,k_1} | … | U^A_{i,k_q} ]           (b_m × s)
  ```
- **Per-tile right factor**, built from the existing Stage 1 + a Stage-2 variant:
  ```
  S_ikj = (V^A_ik)' · W^B_kj                          (rA × rB)   — Stage 1 (reused as-is)
  H_ij  = [ …, Z^B_kj · S_ikj', … ] stacked over k    (b_n × s)   — Stage-2 variant
  ```
  Note `Z_kj S_ikj' = (S_ikj Z_kj')' = T_ikj'`, i.e. `H` is the **transpose of the current
  Stage-2 `T`**. Produce `H` directly with a `('N','T')` GEMM (`Z · S'`) rather than the
  current `('N','T')` `S · Z'`; reuse Stage 1's `S`.

Then `ΔC_ij = W_i H_ij'` exactly, and the target is `C_ij ← beta·U V' + alpha·W_i H_ij'`.

**Fold the scalars into the factors** (keeps the merge formula scalar-free):

```
V ← beta · V_old      (b_n × rC)
H ← alpha · H_ij       (b_n × s)
```

### 4.3 Per-tile orthogonal merge

Given orthonormal `U` (b_m×rC), `V` (b_n×rC), stacked `W = W_i` (b_m×s), `H` (b_n×s):

```
B      = U' W                          (rC × s)      component of W inside span(U)
W_perp = W - U B                       (b_m × s)     component of W outside span(U);   U'W_perp = 0 exactly
                                                     (shifted CholQR2, in high precision)
W_perp ≈ Q_perp R_perp                 Q_perp (b_m×s'), R_perp (s'×s), Q_perp'Q_perp ≈ I, Q_perp ⊥ U
Q      = [ U | Q_perp ]                (b_m × (rC+s'))         orthonormal
Vbar   = [ V + H B' | H R_perp' ]      (b_n × (rC+s'))
```

Identity (verified): `Q Vbar' = U V' + W H'`. (Uses `Q_perp R_perp = W - U B`; the
`U B H'` cross-terms cancel.)

**SVD-free truncation** (exact because `Q` is orthonormal):

```
e_l  = ‖Vbar[:,l]‖²                     column energies
drop the smallest e_l while Σ_dropped e_l ≤ tol²   (rel: ≤ tol²·Σ_l e_l),  capped at maxrank(C)
error² = Σ_dropped e_l                  (EXACT, not a bound)
U_ij ← Q[:,keep]   (still orthonormal),   V_ij ← Vbar[:,keep]
```

`beta = 0` / first write (`rC = 0`) degenerates cleanly: `B` empty, `W_perp = W`,
`Q = Q_perp = orthonormalize(W)`, `Vbar = H R_perp'`, then truncate. This is the factor-space
analog of randQB **without** the random sketch — exact orthogonalization of the `s`-wide
factor, cheaper than randQB and with no range-capture error (but energy-, not SVD-optimal).

## 5. Mandatory numerical guards

These are not optional; the scheme is delicate exactly here.

1. **Do not reuse `G_i = W_i'W_i` for the residual Gram by default.** `G_perp = G_i − B'B`
   is a catastrophic-cancellation structure — identical in class to the `‖A‖²−‖V‖²` EI-floor
   bug already recorded in project memory (`tlr-ei-cancellation-floor`) — and it is worst
   exactly when the update lies mostly inside `span(U)` (the common accumulation case).
   Instead form `W_perp` explicitly (needed anyway for `Q_perp`) and compute its Gram
   **fresh, in high precision**. The `G_i`-sharing optimization is a *benchmark-it-later*
   option, gated behind the shifted-Cholesky escalation **and** a re-orthogonality residual
   check — not the default.
2. **Orthogonality drift.** `U' Q_perp = 0` only up to round-off in `W − U B`, and it drifts
   over streaming merges; a rank-deficient `W_perp` (panel already in `span(U)`) also makes
   `Q_perp` ill-conditioned. The energy=error identity **relies** on `Q` being orthonormal,
   so add a re-orthogonalization safeguard (a block-CGS2 re-projection step, or the SVD
   re-baseline) and a cheap `‖U'Q_perp‖` check.
3. **Column-energy truncation over-retains.** Needs the **SVD/randQB escape** when a tile's
   rank exceeds a threshold (route it through the retained M4 dense path, or a small core
   SVD). Measure the typical inflation vs SVD in step 0 to size the threshold.
4. **Reuse `cholqr2!`** from `compression/algorithm.jl` for the batched shifted CholQR of
   `W_perp` — it already has the FKNYY shift, POTRF-failure escalation, and the two-pass
   refinement. Do not hand-roll a new CholQR.
5. **KA CPU `@synchronize` is literal-only** (project memory `ka-cpu-synchronize-literal`):
   any new reduction kernel (column energies, truncation) must keep barriers out of loops/
   helpers/macros on the CPU backend — mirror the compression reduction kernels.

## 6. What to reuse vs replace

**Reuse unchanged:**

- `gemm!(C::TLRMatrix, …)` entry, `_validate_tlr_output`, the `_run_tlr_output!` dispatch
  shape (`gemm.jl`).
- Lowering: `contract_domains`/`interior_leaves`/`geometry`/`runs`/`_row_block`
  (`contraction/*`, `lowering/schedule.jl`) — to get budgeted runs and the `W_i` U-stack.
- **Stage 1** (`S = V'W`) verbatim (`lowering/stages.jl`).
- The bounded-workspace *pattern* (preallocate at max group width, refill per run) and the
  reuse discipline from M4 step A.
- Test oracle `assert_tlr_output_matches_dense` and `reconstruct_tlr`.
- `cholqr2!`, `_compress_accum_type` (`compression/`).

**Add / vary:**

- **Stage-2 variant** producing `H = Z·S'` (b_n × s) instead of `T = S·Z'`.
- The **batched merge primitive** (§4.3) and its workspace (W-stack, `B`, `W_perp`,
  `G_perp`, CholQR high-precision scratch, `Vbar`, energies), sized per run/group.

**Replace (drop for the merge path):**

- `SlabOutput` + the dense `Stage 3` + `_compress_run_into_factors!` (randQB into a `b×b`
  slab). The merge works entirely in factor space — **no dense slab, no `b×b` temporary.**

## 7. Step plan (with gates)

| step | deliverable | gate |
| --- | --- | --- |
| **0** | Standalone **batched merge primitive** (no GEMM plumbing): `(U,V,W,H) → truncated (U',V')`. Guards decided here (fresh `W_perp` Gram; escalation; re-orth check; exact energy truncation). | Merge identity `Q Vbar' = UV' + WH'` to round-off; energy = reconstruction error **exactly**; `‖U'U − I‖` bounded; **rank inflation vs SVD measured** on synthetic correlated/uncorrelated tiles. |
| **1** | Wire as the TLR-output sink (**row family**): Stage 1 + Stage-2H build `S`, `H`; `W_i` is the U-stack; batched merge writes factors + ranks + resid. `beta = 0`. | A/B vs the retained M4 dense path on the same oracle: same `tol` met; **rank compared**; **no `b×b` temp allocated** (allocation + `@inferred` gate); interleaved A/B flop/time delta recorded. |
| **2** | `beta != 0` (native: `V←beta·V`, `H←alpha·H`, `rC>0`); **column family** (col×row) via depth-first `_column_block` + family-aware arena; **SVD/randQB escape** on rank overflow; streaming-stability guard. | All regular-grid layouts + `beta`; drift bounded over a multi-panel accumulation; overflow tiles routed through the escape and reported in `resid`. |
| **3** | `gemm_workspace_bytes` TLR-merge variant; `DESIGN.md` merge-sink section; `ROADMAP.md` M5 → `[x]`. | Docs reflect behavior; both sinks pass the same oracle; budget query matches promoted workspace. |

Boundary tiles and `TLRDenseDiagMatrix` output remain out of scope until the full-LR merge
lands (Milestone 6 territory).

## 8. Tests

- **Merge primitive (step 0):** synthetic `(U orthonormal, V, W, H)`; assert the identity to
  round-off, `error² == Σ_dropped e_l` exactly, `U'U≈I`; sweep correlated vs uncorrelated
  `H` columns and tabulate rank vs a reference SVD truncation.
- **Sink (step 1+):** extend `gemm_tlr_output.jl` — `reconstruct(C_after) ≈ alpha·A·B (+
  beta·reconstruct(C_before))` at full capacity; overflow → `resid > tol`; inference
  `@inferred`; **CUDA cases** (host/device boundaries — note the M4 `copyto!` scalar-fallback
  bug that only CUDA caught). A/B equal-accuracy comparison against the dense reference path.

## 9. Open decisions to make explicitly

- **`G_i` sharing**: default to fresh `W_perp` Gram (stability); benchmark shared-`G_i` as an
  opt-in once correctness is locked.
- **Re-orthogonalization cadence**: every merge vs on-drift-threshold vs SVD re-baseline.
- **SVD-escape threshold**: rank fraction of `maxrank` that triggers the randQB/SVD fallback;
  size it from step 0's inflation measurement.
- **Default sink**: does the merge become the default TLR-output path with dense as opt-in
  fallback, or vice versa? (Recommendation: merge default once step 1 A/B is favorable;
  dense reachable for the SVD-escape and as a correctness reference.)

## 10. Environment / workflow notes

- Tests run under the `gpuenv` project: `julia --project=../gpuenv …` (CPU + working CUDA;
  AMDGPU/Metal extensions are pre-existing-broken and auto-skipped by `available_backends()`).
- Fast iteration: a scratch runner that `using NextLA` + includes
  `test/{gpu_backends,backend_test_helpers}.jl` and `test/TLR/helpers.jl`, then the target
  test — see project memory `nextla-test-running`.
- **Always exercise CUDA**, not just CPU: the one M4 bug (host `SubArray`→device `copyto!`
  scalar fallback) was invisible on CPU.
