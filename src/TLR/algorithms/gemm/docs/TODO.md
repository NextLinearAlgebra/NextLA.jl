# TLR GEMM — worklog and roadmap

The design is [algorithm.tex](algorithm.tex): a blocked **adaptive randomized
range finder (ARA)** applied per output tile, where the contribution of the
reduction index is kept as an implicit *factor list* rather than a dense tile.

This file is the hand-off document. It records **why** the current design was
chosen — including several directions that were tried and abandoned with
evidence — so the next person does not re-derive them. Read
[algorithm.tex](algorithm.tex) first for the algorithm; read this for the
rationale, the traps, and what is left.

Reference: Boukaram, Turkiyyah & Keyes, *Randomized GPU algorithms for the
construction of hierarchical matrices from matrix-vector operations*, SIAM J.
Sci. Comput. 41(4):C339–C366, 2019. Algorithm 2.3 is the batched ARA loop we
are implementing; §2.3.1 is the Cholesky-QR argument.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Worklog — decisions and the evidence behind them

### 1. Shifted CholQR2 is not rank-revealing (and that turned out to be fine)

**Observed.** `mixed_cholqr2_compress!` was measured on panels of prescribed
rank (`Y = U·diag([1×40, g×24])·Wᵀ`, `b=256, s=64`, 5 seeds, fp64). At
`g ≤ 1e-12` it returned a nearly correct *rank* (41–43 vs. the true 40) but
`‖QᵀQ − I‖ = 1.0`. The basis, not the count, is what breaks.

**Mechanism.** The shift (Fukaya et al., shifted CholeskyQR) exists to guarantee
`potrf` completes — [cholqr2.jl](../../../numerics/cholqr2.jl) even doubles it
until every batch member succeeds. That is correct for robustness, but it means
breakdown can no longer *signal* deficiency: the factorization returns silently
and `Q = Y R₁⁻¹` amplifies noise through the `O(δ^{-1/2})` trailing block of
`R₁⁻¹`. Because `κ(R₁)` is global, the damage is **not** confined to the
deficient columns.

**Two fixes were tried and rejected on measurement:**

| Route | Result at true rank 40 |
|---|---|
| Threshold the unpivoted Cholesky diagonal | 51–54 at *every* gap; `‖QᵀQ−I‖` up to 2.7e-2 |
| Gram eigendecomposition with `τ = √u` | 46–52 (noise eigenvalues sit at `u·λ_max`) |
| Gram eigendecomposition with `τ = √(C·s·u)`, `C≈10` | **40**, matching the SVD oracle exactly |

The unpivoted-diagonal route fails because Cholesky does not order its diagonal
by significance, and pivoting is unavailable at batch scale (data-dependent).
The calibration constant `C` is *not* cosmetic — it is the difference between
46–52 and 40.

**Why this ultimately did not gate the design.** That experiment was
*monolithic*: one wide panel, deficiency spread across all columns by a Gaussian
`W`, and severity pushed to `κ = 1e14`. None of that holds in blocked ARA, where
(a) blocks are small and already BCGS2-projected, (b) deficiency appears only at
convergence and sits at the *stopping tolerance*, so `κ(Y_Δ) ≈ 1/ε` rather than
`1/u`, and (c) since `R` is triangular, column `j` of `Q_Δ = Y_Δ R⁻¹` depends
only on the leading `j×j` block of `R` — contamination flows *forward only*, so
if the small pivots appear at positions `br+1…bs`, columns `1…br` are clean.

So **`RangeFind` needs no rank-revealing orthogonalizer inside its loop.** The
monolithic result still applies to any wide one-shot sketch — see item 3.

Harness: `rankprobe.jl` / `rankprobe_gpu.jl` (scratch). Worth promoting to
`test/TLR/` as a characterization test for the wide-panel paths and to pin the
`ε ≳ √u_hi` boundary, but it is **not** a gate on the ARA loop.

### 2. The shift is removed from ARA (derived, not tuned)

`_cholqr_shift_kernel!` adds `coeff · max_i G[i,i] · multiplier`, and
`max_i G[i,i]` is the largest squared column norm. So every triangular diagonal
entry is bounded below by `√(coeff·multiplier)` **relative to the panel scale,
independent of the data** — at `b_m=256, s=32` in fp64, `√coeff = 3.4e-6`.

Since `R[j,j]` is the residual of column `j` against the basis and the earlier
columns of the block, that floor sits directly under the stopping test
`R[j,j]/R_max < ε_rel`. **A tolerance below `3.4e-6` is unsatisfiable for any
input**: the loop never converges, every tile returns at `maxrank`, nothing
errors. That is the trap A0 was built to expose.

**The policy follows from two theorems, with nothing to tune.**

- *Yamamoto, Nakatsukasa, Yanagisawa & Fukaya*, ETNA 44:306–326, 2015:
  CholeskyQR2 attains `‖QᵀQ − I‖ = O(u)` provided `κ(Y) ≤ u^{-1/2}`.
- *Fukaya, Nakatsukasa, Yanagisawa & Yamamoto*, SIAM J. Sci. Comput. 42(1),
  2020: the shift `11(ms + s(s+1))·u·‖Y‖²` provably prevents breakdown. That is
  the constant `_cholqr_shift_coeff` implements — the shift was never a magic
  number, and neither is the floor it implies.

Comparing the tolerance range each policy *provably* supports settles it on
paper:

| Policy | Proven condition | Smallest usable `ε_rel` (fp64 Gram) |
| --- | --- | --- |
| Keep the Fukaya shift | `ε_rel > √coeff` | 3.4e-6 |
| **No shift** | `κ(Y_Δ) ≤ u^{-1/2}`, and `κ(Y_Δ) ≈ 1/ε_rel` at the stopping block | **1.05e-8** |

The shift costs ~320× of usable range and buys nothing, because **breakdown and
loss of the CholeskyQR2 guarantee coincide**: `potrf` fails exactly when `G` is
numerically singular, i.e. `κ(Y_Δ) ≳ u^{-1/2}`, which is precisely the boundary
of the first theorem. So the unshifted loop is self-certifying — success
certifies `O(u)` orthogonality, failure certifies rank deficiency.

**Breakdown delimits the pass; it does not destroy it.** `potrf` returns
`info = k` meaning the leading minor of order `k` is not positive definite, so
columns `1..k-1` were validly factored. Because `R` is upper triangular, column
`j` of `Y R⁻¹` depends only on the leading `j×j` block of `R`, so the valid
prefix is untouched by the failed tail — contamination is forward-only.
Measured on a rank-10 operator sampled at width 16, `cond(Y) = 1.7e12`,
`info = 12`, and `diag(R1) = [3.14, …, 1.64, 0.062, 8.2e-7, 1.2e-14, …]`: the
rank is legible in the valid prefix. `ara_mask_breakdown!` zeroes columns
`k..width` of the basis block and their `dR` entries; the zeros then read as
"no new content", which is what they are.

Finally, breakdown implies **convergence**, and this too is derived rather than
assumed: `Y_Δ = (I-QQᵀ)XΩ_Δ` with Gaussian `Ω_Δ` puts the samples in general
position, so `rank(Y_Δ) = rank((I-QQᵀ)X)` almost surely. A block that yields
only `k-1 < width` independent directions has therefore captured the entire
residual range, and the member is done.

`cholqr2_relative_shift_floor` survives as the guard for paths that *do* shift —
the wide one-shot panels of item 3.

### 3. `compress!` (dense → TLR) moved to ARA

[compress.jl](../../compression/compress.jl) used to sketch **one-shot at full
`maxrank`** — its docstring: *"`maxrank` is both the output capacity and the
sketch capacity."* For a tile of true rank `r ≪ maxrank` that is a monolithic,
wide, rank-deficient panel with no blocking and no BCGS2: precisely the setup of
item 1, where `κ(R₁)` pollutes the basis before the greedy V-column-norm prune
ever runs. That prune had no guarantees there.

Now `compress_tiles!` runs the ARA loop, so each tile stops at its own rank and
the orthonormalizer never sees a wide deficient panel. `TileSource` already
*was* the black-box sampler interface — `_sketch!`/`_cosketch!` are
`ApplyRight`/`ApplyLeft` — so the same loop serves both this path and the GEMM
output.

The reported residual is now exact rather than indicative: with `Q` orthonormal,
`‖A − QQᵀA‖² = ‖A‖² − ‖Z‖²` by Pythagoras, and truncation adds `Σ_{k>r} σ_k²`,
so the total is `‖A‖² − Σ_{k≤r} σ_k²`. That is the randQB_EI indicator and it
keeps its cancellation floor (see [[tlr-ei-cancellation-floor]] reasoning:
both terms are `O(‖A‖²)` while the difference may be far smaller), so it is
clamped below the rounding floor of the subtraction.

`Qbuf` cannot alias the output `U` the way the co-range aliases `V`: the final
lift `U = Q·W` would read and write one array. It is a separate working-precision
buffer, accounted in `compress_arena_elems(...).work` so the budget stays honest.

### 4. The exact residual was removed from the hot path

`algorithm.tex` previously verified the Frobenius residual when convergence was
marginal or the rank cap was hit. Both branches were dropped:

- **Rank cap.** Saturation is already visible in `ranks(C)` at zero cost — the
  same convention `compress!` documents. And the *truncation* error is exactly
  `Σ_{k>r} λ_k` from the eigenvalues already computed for the final rotation, so
  no `E_X` is needed to report it. Only the *capture* error is unknown, and
  under a cap the truncation term dominates.
- **Marginal stop.** Economically dominated. `E_X` costs `O(b·q_k²·r²)`; one
  more sketch block costs `O(b·q_k·r·Δs)`. The ratio is `q_k·r/Δs ≈ 32` at
  `q_k=16, r=32, Δs=16` — **it is ~32× cheaper to draw another block than to
  check whether you needed to.** Verification only pays if it usually says
  "stop", but this branch is reached only when the stop looked doubtful.

Also, the sampling stop is not merely a heuristic: with *fresh* Gaussian blocks
it is the Halko–Martinsson–Tropp a posteriori estimator, failure probability
decaying like `10^{-p}` in the consecutive-small count. Raising `p` costs one
block and buys an order of magnitude — far cheaper than `E_X`.

`FactorEnergy`/`ResidualEnergy` should exist only behind a `verify=true` debug
flag used by tests.

### 5. Known accuracy limits (document, do not fight)

- **`ε ≳ √u_hi`.** At the stopping block `κ(Y_Δ) ≈ 1/ε`, and CholQR2 needs
  `κ ≲ u_hi^{-1/2}`. With a Float64 Gram that is `ε ≳ 1e-8`. The reference says
  the same (§2.3.1: *"If greater accuracy is required, quad precision may be
  needed to stabilize the double precision Cholesky QR"*). Notably this is the
  **same `√u` barrier** as the Gram-route resolution limit in item 1, reached
  from the opposite direction.
- **ARA overshoots rank** by 1–4.5 on average (their Fig. 3(b)), growing with
  rank, and *"the batch may contain some matrices with a relatively large rank
  difference."* Their suggested remedy (§2.1) is an SVD of the small `k×k`
  factor — which is exactly the final `K = ZᵀZ` eigensolve in `algorithm.tex`.
  So the blueprint already closes their noted weakness; keep that step.

### 6. The final truncation uses `gesvda`, not a Gram

With `Q` orthonormal and `Z = P Σ Wᵀ` the thin SVD of `Z`,
`Q Zᵀ = (QW) Σ Pᵀ` — and `QW` is orthonormal, so this *is* the SVD of the
represented matrix. Truncation is therefore Eckart–Young optimal, with squared
error `Σ_{k>r} σ_k²` exact and free. Two routes reach it: eigendecompose the
Gram `K = ZᵀZ`, or factor `Z` directly.

Measured (`b_n=256`, `nb=128`, fp64, ms; and rank recovered at `τ = 1e-8` with a
true rank of `s_Q/2` and a spectral gap of `1e-10`):

| `s_Q` | gesvda | gram+syevj | rank gesvda | rank gram | true |
| --- | --- | --- | --- | --- | --- |
| 16 | 11.65 | **4.31** | **8** | 11 | 8 |
| 32 | 29.62 | 35.64 | **16** | 21 | 16 |
| 48 | **22.34** | 73.01 | — | — | — |
| 64 | **27.09** | 94.78 | 42 | 44 | 32 |

`gesvda` wins for `s_Q > 32` on speed and everywhere on rank, because it never
squares the condition number: the Gram route resolves singular values only to
`√u·σ_max`, so with a gap below that it cannot separate signal from noise and
over-retains by 30–40%. It also **saves two GEMMs** — `Z = PΣWᵀ` yields both
output factors directly (`U = Q·W[:,1:r]`, `V = P[:,1:r]·diag σ`, a column
scaling), where the Gram route needs `K = ZᵀZ`, then `V = ZW`, then `U = QW` —
and it removes the need for a batched `syevj` binding entirely.

Adopted: **one path, `gesvda`**, accepting the 2.7× penalty at `s_Q = 16`
(7 ms per 128 tiles in absolute terms) in exchange for a single code path.

Two things that are *not* hand-waved:

- **`gesvda`'s left factor is not orthonormal when returned untruncated**
  (measured `‖UᵀU−I‖ = 8e-4` at gap `1e-6`). We consume the **right** factor,
  measured clean at `~1e-14`, and truncate before use. There is a test on it.
- **`Q` carries exact zero columns** (`ara_mask_breakdown!` puts them there), so
  `QᵀQ ≠ I` and `U = QW` is not obviously orthonormal. It is, and by an
  identity rather than luck: column `j` of `Z = XᵀQ` is zero whenever column `j`
  of `Q` is, so `Z[:,j] = PΣW[j,:]ᵀ = 0` gives `σ_k W[j,k] = 0` for all `k`.
  The retained right singular vectors therefore vanish on exactly the rows
  indexing dead columns of `Q`, and `UᵀU = W ᵀ(QᵀQ)W = I`. Tested directly with
  a deliberately holed `Q`.

The Gram route was not unsafe, merely slower and looser: `Z` comes out of the
ARA loop, whose tail already sits at `ε_rel ≥ √u`, so the Gram's `√u` resolution
would have been *exactly* sufficient by construction — the same `√u` bound
appearing for the third time (items 2 and 5).

Backends: CUDA `gesvdaStridedBatched`; AMD `rocsolver_?gesvdj_strided_batched`
(batched one-sided Jacobi, the closest rocSOLVER equivalent — **written but
never executed**, since the AMDGPU extension does not precompile here; the
generic method keeps ROCArrays correct meanwhile); CPU a LAPACK loop.

### 7. `potrf` success is not a usability certificate

Two bugs surfaced when `compress!` moved onto ARA, both caught by the **Float32
CUDA** roundtrip, and both worth knowing before touching this code.

**`potrf` can succeed on a pivot you must not divide by.** Measured on a rank-4
Float32 panel of width 12: `info = 0`, yet pivots 5–12 sat at `~5e-7` against a
leading `6.6`. The triangular solve then returns garbage — and on an exactly
zero panel, `NaN` from `0/0`. Breakdown detection alone therefore under-reports;
the guard is the CholeskyQR2 validity condition applied per column, `R[j,j] ≥
√u·R_max`. That is the same `√u` as items 2 and 5, not a new constant.

**The cut must come from the first CholQR pass.** `dR` is the composite `R₂R₁`
diagonal, and once pass 1's solve has produced garbage columns, pass 2's Gram is
built from them, so a contaminated `R₂` drags the composite below threshold on
columns that were fine. Measured: a rank-20 operator sampled at width 8 gave
`potrf` status 5 — four genuine new directions — while the composite-based cut
fired at column 1 and lost them, reporting rank 16 instead of 20.
`_ara_cut_kernel!` reads `R1` directly for this reason.

Everything from the cut is zeroed, which is not over-eager: `R` is upper
triangular, so column `j` of `Y R⁻¹` is built from the leading `j×j` block, and
once a pivot is invalid every later column inherits it.

### 8. Batched factorization survey (measured on this machine)

| API | Batched | Notes |
| --- | --- | --- |
| `cublasXgeqrfBatched` | yes | **no batched `orgqr` exists** — Q must be formed by hand |
| `gesvdaStridedBatched` | yes | tall-skinny; `U` is **not** orthonormal unless truncated (`‖QᵀQ−I‖ = 8e-4` at gap 1e-6) |
| `gesvdjBatched` | yes | hard cap `m,n ≤ 32` |
| `syevjBatched` | yes | no size cap; used for the final `K` eigensolve |
| `Xgesvdr!` / `Xgesvdp!` | no | single-matrix only |

Cost, `b=256`, `nb=128`, fp64, ms/batch — `s=16`: gesvda 16.2 / syrk+syevj 7.1;
`s=32`: 35.5 / 32.5; `s=64`: 27.1 / 95.7. Crossover at `s≈32`. Householder QR
(`geqrf_batched!`: 10.6 ms at `s=64`) is the fastest and would make the existing
row-norm pruning valid unchanged, but needs a batched `orgqr` built from the
repo's `larfb`/`larfg` kernels. Parked.

---

## Current state

`row_basis/*` is the **currently shipped** TLR-output path (`gemm.jl:299`) and
stays until the ARA path is green. The uncommitted M1 shared-basis work was
reverted — the design it served no longer exists.

- [x] **A0 — convergence bookkeeping.** [numerics/ara.jl](../../../numerics/ara.jl),
  tests [test/TLR/ara.jl](../../../../../test/TLR/ara.jl), 40/40 CPU + 12/12 CUDA.
  Provides `cholqr2_relative_shift_floor`, `ara_column_norms_sq!`,
  `ara_block_norms!` (shift-corrected `dR`), `ARAConvergenceState`,
  `ara_reset!`, `ara_update_convergence!`.

  Two design points worth knowing:
  - **`dR`, not column norms.** At convergence the block is a set of random
    combinations of the few remaining directions, so every *column* still has
    `O(1)` norm while the *block* is rank-deficient. `R[j,j]` is the residual
    against both the basis and the earlier columns of the same block, so it
    collapses exactly when nothing new is left. There is a regression test
    asserting column norms are blind in precisely this case.
  - **Convergence is judged per block, not mid-block.** A late significant
    column resets the consecutive-small run. This is the false-early-stop guard;
    do not "optimize" it into a short-circuit.

  `ara_column_norms_sq!` must be called on the panel **after BCGS2, before the
  Gram** — the factorization overwrites the quantity it reproduces.

## Next

- [x] **A1 — batched ARA core.** `ARAWorkspace`, `ara_build_basis!`,
  `ara_stopping_floor`, `ara_mask_breakdown!` in
  [numerics/ara.jl](../../../numerics/ara.jl). 24/24 CPU + 20/20 CUDA.
  `mixed_cholqr2_factor!` gained `shift_coeff` / `escalate` / `status`, and
  `coeff == 0` now means genuinely unshifted (it previously floored at
  `eps(RT)` *absolutely*, which is a large relative shift for a small-norm
  panel).

  The loop rejects `eps_rel < √u_hi` with an `ArgumentError` rather than running
  to `maxrank` — the failure mode of item 2 is now impossible to hit silently.

  `block` is a keyword and is a **performance knob only**: there is a test
  asserting the recovered rank and the achieved accuracy are unchanged across
  `block ∈ {2, 5, 16}`. The reference uses 32 (warp size).

  Deliberately *not* done here: `Z = XᵀQ` is left to A2 as a single apply after
  the loop (the reference does the same, its line 32), rather than accumulated
  per pass as `algorithm.tex` draws it. Same total work, fewer launches.

- [x] **A2 — final truncation.** `batched_thin_svd!` (backend hook) and
  `ara_truncate!` in [numerics/ara.jl](../../../numerics/ara.jl); CUDA override
  in `ext/NextLACUDAExt.jl`, AMD in `ext/amdgpu/svd.jl`. 31/31 CPU + 10/10 CUDA.
  Rationale and the measured comparison are worklog item 6. Gates: the achieved
  error equals the Eckart–Young optimum to round-off, the reported `err_sq` is
  the achieved error rather than a bound, ragged ranks stay in one batched call
  with surplus columns exactly zero, and the device path is asserted to stay on
  device (the generic fallback would pass every numerical check while silently
  costing a host round trip per tile).

- [x] **A3 — `compress!` on ARA.** `compress_tiles!` now runs `ara_build_basis!`
  plus `ara_truncate!` against a dense-tile sampler; the one-shot
  `maxrank`-width sketch, `mixed_cholqr2_basis!` and `prune_randqb_columns!` are
  off this path. Worklog items 3 and 7. All pre-existing compress tests green
  (56 + 15 + 12 CPU, 4 CUDA, 12 packed-batch) plus a new testset asserting the
  recovered rank equals the true rank rather than the capacity, and that the
  sampling block width changes neither rank nor accuracy.

- [ ] **R1 — factor-list sampler.** `S`/`H`/`T` prologue and the fused
  `ApplyRight`/`ApplyLeft` over the zero-copy `rowpanel`/`colpanel` views in
  [operands.jl](../operands.jl). β=0 and β≠0 (the old tile rides inside the
  sketch as one further factor pair). Gate: matches a dense reference; dependent
  launches per tile are `O(1)` in `q_k`.

- [ ] **R2 — `RangeFind` on one tile.** A1 + R1. No exact residual (item 4).

- [ ] **R3 — batch across a run.** Tiles converge at different pass counts. The
  reference solves this with per-member sample counts (`batchSetSamples`), which
  A0 already implements: a converged member has `samples == 0` and is skipped.
  What remains is **packing active members contiguously** so a converged member
  does not keep occupying a batch slot. `H_ℓj` hoisted per column.

- [ ] **R4 — scheduler + arena.** `w_ij(s)` from rank metadata, budget-bounded
  run growth, sub-panel fallback for long reductions. Gate: a tiny budget runs
  correctly and never exceeds it.

- [ ] **R5 — integration.** `gemm!` TLR-output dispatch, boundary regions, all
  layout pairs.

- [ ] **R6 — acceptance, then deletion** of `row_basis/`, `tlr_output.jl`,
  `docs/alg.md`.

## Deferred

Ragged per-row/per-column ranks · boundary (non-regular-grid) tiles ·
transposed operands with β≠0 · two-sided/BLR² output · batched `orgqr` from
`larfb`/`larfg` (item 6) · treating shift escalation as a convergence signal
(item 2).
