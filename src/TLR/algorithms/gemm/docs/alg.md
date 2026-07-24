# Global Row-Basis TLR GEMM — preferred M5 design

This document specifies the deferred M5 algorithm for

```text
C ← α A B + β C
```

when all three operands are regular-grid TLR matrices.  It is deliberately an
explicit execution design, not an IR or a generic contraction framework.

The primary target is the layout pair

```text
logical A: TileRowMajor       logical B: TileColMajor
```

For that pair, both factor panels used by the algorithm are zero-copy views of
the stored factors.  Other layout pairs remain supported through direct
fallbacks described in §6; this is a fast-path preference, not a public API
restriction.  Layout always means the *logical* layout after `transA` and
`transB` have been canonicalized.

---

## 1. Representation and invariants

For one tile row/column, write

```text
A[i,k] = U_A[i,k] V_A[i,k]ᵀ       U_A, V_A: b × rA
B[k,j] = W_B[k,j] Z_B[k,j]ᵀ       W_B, Z_B: b × rB
C[i,j] = U_C[i,j] V_C[i,j]ᵀ       U_C, V_C: b × rC
```

`rA`, `rB`, and `rC` below denote the concrete padded rank capacities used by
the kernels.  Unused factor columns are zero.  This makes all panel views and
batches rectangular and concretely typed.

The load-bearing output invariant is:

```text
U_C[i,j]ᵀ U_C[i,j] ≈ I.
```

Stage 3 restores this invariant after every update.  Consequently, norms of
the represented tile and coordinate-energy truncation are computed from its
right factor without reconstructing a dense tile.

For one output row `i`, M5 finds a shared left basis `Q[i]` and block
coefficients `P[i,k]`:

```text
U_A[i,k] ≈ Q[i] P[i,k],       Q[i]ᵀ Q[i] ≈ I,
P[i,k]: t[i] × rA.
```

Thus the product contribution to output tile `(i,j)` has the form

```text
ΔC[i,j] ≈ Q[i] M[i,j]ᵀ,
M[i,j] = Σₖ Z_B[k,j] (P[i,k] (V_A[i,k]ᵀ W_B[k,j]))ᵀ.     # b × t[i]
```

Only `M[i,j]`, not a dense `b × b` product tile, is accumulated.

---

## 2. Numerical policy

* Factor GEMMs use the established GEMM precision policy.
* Gram matrices, panel covariance, eigenvalues, rank decisions, and squared
  norms use the promoted/high-precision policy already used by mixed CholQR2.
* `mixed_cholqr2_factor!` is used for panel orthogonalization; narrow residual
  compression uses the same mixed-precision CholQR2 machinery.
* Rank pruning is fused into the merge path where possible.  Do not decompose
  hot paths into generic callbacks merely to share a few lines of code.

The Stage-1 sketch has capacity `S ≤ b`; it is not a promised output rank.
`tmax ≤ S` and the tile cap is `rmax`.

---

## 3. Stage 1 — standalone shared-row-basis builder

This stage is independently callable and testable.  It is the architectural
extension point M5 needs; dense output owns no part of it.

For logical row-major `A`, `rowpanel(A.U, i)` is already the storage

```text
[U_A[i,1]  …  U_A[i,K]]  with shape b × (K*rA).
```

No `W` assembly and no streamed loop of small sketch GEMMs is needed.

```text
FUNCTION build_row_basis!(A_row_panel, gamma[1:K], eps_basis, S, workspace)
    # A_row_panel is a b × rA × K zero-copy view for the preferred path.
    Ubar ← reshape(A_row_panel, b, K*rA)              # zero-copy

    # gamma is a per-k importance weight.  gamma = 1 is the first correctness
    # baseline; a coefficient-aware estimator may supply nonuniform values.
    Ω ← random_normal(K*rA, S)
    Ωgamma ← copy(Ω)
    for k = 1:K:
        scale rows ((k-1)*rA+1 : k*rA) of Ωgamma by gamma[k]

    Y ← Ubar * Ωgamma                                # b × S, one GEMM
    Q0, _ ← mixed_cholqr2_factor!(Y)                 # Q0: b × S

    Pfull ← Q0ᵀ * Ubar                               # S × (K*rA), one GEMM

    # Preserve unweighted Pfull.  Compute the weighted covariance either by
    # scaling a reusable scratch copy then SYRK, or by a proven faster fused
    # kernel.  It is mathematically:
    Ksmall ← Σₖ gamma[k]^2 Pfull[:,block(k)] Pfull[:,block(k)]ᵀ
                                                            # S × S, high precision

    R, lambda ← eigh(Ksmall)                         # descending eigenvalues
    t ← choose_rank(lambda, eps_basis, tmax)
    Q ← Q0 * R[:,1:t]                                # b × t
    P ← R[:,1:t]ᵀ * Pfull                            # t × (K*rA), one GEMM
    Pblocks ← reshape(P, t, rA, K)                   # zero-copy block view

    # Do this explicitly, rather than infer it from I - DᵀD.  It is both a
    # diagnostic and the basis-error measurement used during validation.
    Ebar ← copy(Ubar)
    Ebar ← Ebar - Q * P                              # one GEMM into the copy

    return Q, Pblocks, Ebar, t
END
```

The panel contains padded zero columns; these are intentionally included.
They make GPU batches rectangular and do not change the result.

The preferred implementation is approximately four panel-wide GEMM/SYRK
operations plus CholQR2 and the small `eigh`.  The rotations and residual GEMM
are retained because they avoid a second pass over `k` and provide a robust
error diagnostic.

---

## 4. Stage 2 — accumulate right coefficients

Stage 2 produces `M[i,j]` in the shared basis.  It must never form a dense
product tile and must not compress inside the `k` loop.

For every `(i,k)`, choose the cheaper exact association for

```text
R[i,k,j] = P[i,k] V_A[i,k]ᵀ W_B[k,j],       # t × rB.
```

```text
IF t[i] ≤ rA:
    T[i,k] ← V_A[i,k] * P[i,k]ᵀ             # b × t; reusable over j
    R[i,k,j] ← T[i,k]ᵀ * W_B[k,j]           # t × rB
ELSE:
    S[i,k,j] ← V_A[i,k]ᵀ * W_B[k,j]         # rA × rB
    R[i,k,j] ← P[i,k] * S[i,k,j]            # t × rB
END
```

The first branch is normally preferred after successful row compression: it
pre-compresses `A`'s right factor once and reuses it across output columns.
The second avoids materializing a wider `b × t` factor when `t > rA`.

### Preferred terminal accumulation: logical B column-major

For fixed `(i,j)`, the `Z_B[k,j]` factors are a zero-copy stack:

```text
Zstack ← [Z_B[1,j] … Z_B[K,j]]               # b × (K*rB), zero-copy
Rstack ← [R[i,1,j]ᵀ; …; R[i,K,j]ᵀ]           # (K*rB) × t

M[i,j] ← Zstack * Rstack                     # b × t, one terminal GEMM
```

This is the TLR analogue of the dense fused path.  `Rstack` is a concrete,
reusable workspace batch; form it in `k` panels when the full stack does not
fit the workspace budget, and accumulate into `M[i,j]`.

### Logical B row-major fallback

`Z_B[k,j]` is not a zero-copy k-stack in this layout.  Preserve the same
algebra with one of the following budgeted implementations:

```text
M[i,j] ← 0
for each k-panel Kp:
    form R[i,k,j] for k ∈ Kp in a concrete batch
    M[i,j] += Σ_{k∈Kp} Z_B[k,j] * R[i,k,j]ᵀ
```

Use batched GEMM/reduction or pack a bounded `Z` panel; do not degrade this to
scalar tile loops.  The numerical result is the same up to ordinary GEMM
rounding order.

Unlike dense output, a global *left*-basis M5 is intentionally asymmetric.
The usual dense FoldLeft/FoldRight choice is not a second, equivalent M5 merge:
expanding `S Zᵀ` to `rA × b` before applying `P` is normally wasteful because
the output coefficient has only `t` columns.  A B-side/right-basis M5 would be
a separate future algorithm, with its own orthogonality and merge design.

---

## 5. Stage 3 — merge once, prune once

For each output tile, merge the full product coefficient and old tile into an
orthogonal coordinate system, then prune exactly once.

```text
FUNCTION merge_tile!(C[i,j], Q, M, alpha, beta, eps_tile, rmax)
    U ← U_C[i,j] ;  V ← V_C[i,j]              # U is orthonormal

    IF t == 0:
        V_C[i,j] ← beta * V                   # product is zero
        return
    END

    IF beta == 0 OR rC == 0:
        Qmerge ← Q
        Vmerge ← alpha * M
    ELSE:
        D    ← Qᵀ * U                          # t × rC
        Ures ← U - Q * D                       # b × rC
        Qres, Rres ← mixed_cholqr2_compress!(Ures, eps_residual)
        # Qres: b × rho, Rres: rho × rC, Ures ≈ Qres Rres

        # Reorthogonalize the narrow residual against Q when required by the
        # CholQR2 diagnostic, while updating D/Rres to preserve the factor.
        reorthogonalize_against!(Qres, Rres, Q, D)

        Qmerge ← [Q | Qres]
        Vmerge ← [alpha*M + beta*V*Dᵀ | beta*V*Rresᵀ]
    END

    # Qmerge is orthonormal, so ||Vmerge[:,l]||² is the exact energy of the
    # corresponding coordinate.  Fused rank pruning drops the least energetic
    # coordinates subject to eps_tile and rmax.
    keep ← prune_orthogonal_columns!(Vmerge, eps_tile, rmax)
    U_C[i,j] ← Qmerge[:,keep]
    V_C[i,j] ← Vmerge[:,keep]
END
```

An optional small-Gram rotation before the final prune is permitted when it
measurably lowers ranks.  It must remain inside this merge kernel, not become a
generic output-sink abstraction.

---

## 6. Layout and overflow policy

1. **Preferred M5:** logical `A` row-major, logical `B` column-major.  Stage 1
   and the terminal Stage-2 GEMM are both zero-copy panel paths.
2. **Supported B fallback:** logical `A` row-major, logical `B` row-major.
   Keep the zero-copy Stage 1, then use budgeted packed/batched Stage 2.
3. **Supported A fallback:** logical `A` column-major.  Pack one bounded
   `Ubar` row panel and run the same left-basis algorithm.  Do not introduce a
   right-basis dual merely to avoid this pack.
4. **No compression:** if `t` reaches the configured dense threshold or the
   requested rank/error cannot be represented, route that row/tile to the
   existing dense/M4 fallback.  Never silently exceed workspace or `rmax`.

The choice is made from canonical logical operands.  Physical storage order
alone is insufficient because transpose flags exchange logical row and column
major order.

---

## 7. Workspace, saturation, and scheduling policy

Workspace is a budget for useful GEMM dimensions and independent work; it does
not itself create GPU occupancy.  A single sufficiently large GEMM can saturate
the device with little user scratch, while a large allocation feeding only tiny
GEMMs cannot.  Consequently M5 selects a plan from actual live buffers and
backend-calibrated work shapes rather than consuming every byte it is given.

### 7.1 Liveness-carved default

The initial implementation owns one concrete workspace arena and carves typed
views from it according to live ranges.  The peak is the maximum of phase
requirements, not their sum:

```text
workspace_single = max(bytes_basis_build,
                       bytes_coefficient_accumulation,
                       bytes_merge)
```

For `h` active rows, `j` output columns, and a Stage-2 depth panel `q`, the
first sizing model, in factor elements, is:

```text
persistent per active row:  Q = b*t,  P = K*t*rA

basis build:                Y/Q0 = b*S,  Pfull = S*K*rA,
                            residual = b*K*rA,
                            weighted P/covariance = S*K*rA + S*S (promoted)

coefficient accumulation:  Tpanel = h*q*b*t                  (t <= rA)
                         or Sbuf   = h*j*q*rA*rB               (t > rA)
                            Rstack = h*j*q*rB*t
                            M      = h*j*b*t

merge:                      b*(rC + t + rho) per concurrent tile
```

Promoted Gram/covariance storage is added with its true element size.  `Pfull`
is compacted or reused as persistent `P`; the explicit residual copy is freed
after its diagnostic norm is recorded; and `S` in the `t > rA` branch is
overwritten by `R` as soon as `R = P*S` completes.  In the usual `t <= rA`
branch, `S[i,k,j]` is not allocated at all.

`Q`, `P`, and unfinished `M` are genuinely persistent for their row/tile
lifetimes and must not be aliased.  All other roles are arena aliases.  CUDA
stream order makes reuse safe within one stream without a host synchronization.

### 7.2 Depth versus row/column concurrency

`K` depth and row concurrency enlarge different dimensions:

```text
row basis:        (b × K*rA) * (K*rA × S)
terminal Stage 2: (b × q*rB) * (q*rB × t)
independent jobs: h active rows × j output columns
```

Use full `K` for the zero-copy Stage-1 row-basis GEMMs whenever it fits: this is
the purpose of the row-major A fast path.  For Stage 2, choose `q` only up to
the terminal GEMM's saturation knee.  Once `q*rB` is large enough, spend
additional budget on `h` and output-column concurrency rather than making an
already saturated inner dimension deeper.  This is particularly important when
`t` is small and the terminal GEMM is skinny.

The planner enumerates feasible `(h, jblock, q)` triples, computes their exact
liveness peak, and selects the smallest candidate predicted or measured to
reach near-peak throughput.  It must consider the actual backend, precision,
tile shape, ranks, and SM count; no universal byte count can imply occupancy
for cuBLAS/cuBLASLt kernels.

### 7.3 Pipeline is optional and measured

The alternative is a two-slot pipeline: while slot A builds the basis for row
`i+1`, slot B accumulates/merges row `i`.  It requires events, separate stream
handles, and distinct live arenas:

```text
workspace_pipeline_2 = bytes_basis_build(slot A)
                     + bytes_accumulate_merge(slot B)
```

Do not enable it by default.  It can improve throughput when small CholQR2,
eigensolve, or panel kernels leave the GPU idle, but concurrent large GEMMs
normally compete for the same device resources.  Add it only after benchmarking
one versus two slots on the target CPU/GPU regimes and retain it only for a
reproducible win.

### 7.4 Workspace query contract

The existing `gemm_workspace_bytes` full-width meaning is preserved.  M5 should
eventually expose a separate saturation recommendation (or a backward-compatible
`target = :saturating` query) that reports the selected `(h, jblock, q)` and:

```text
minimum:    correct execution with narrower runs
saturating: smallest near-peak-throughput plan
full-width: all relevant direct work fits at once
```

Giving a larger budget than the selected saturation plan must not cause needless
scratch retention.

---

## 8. Implementation plan and gates

### M5.1 — standalone Stage 1

Implement only `build_row_basis!` for a contiguous logical row-major `A` panel.
Use `gamma = 1` first, then the coefficient-aware weights.  Tests:

* zero-copy panel reshape and padded/zero rank handling;
* equality with a streamed reference calculation;
* weighted covariance uses `gamma²`, while returned `P` is unweighted;
* reconstruction and explicit residual norms;
* rank choice, mixed precision, `@inferred` workspaces, CPU and CUDA.

### M5.2 — preferred coefficient path

Implement the `t ≤ rA` and `t > rA` contractions and the column-major `Zstack`
terminal GEMM.  Verify against dense tile products before merging.  Benchmark
aligned and tailed grids, small tiles, and occupancy-saturating large/batched
cases.

### M5.3 — merge and pruning

Implement the one-merge/one-prune tile driver, including `beta`, empty tiles,
rank caps, and residual reorthogonalization.  Verify tile accuracy, output
orthogonality, exact coordinate energies, and repeated-update rank behaviour.

### M5.4 — fallbacks and integration

Add B-row-major packed/batched accumulation, then bounded packing for
column-major A.  Integrate only after the preferred path is correct.  Preserve
the existing M4/dense route for overflow and unsupported conditions.

### M5.5 — performance acceptance

Capture warmed allocations and timings for all four logical layout pairs,
rank-asymmetric cases, aligned/tail grids, tiny/full workspace budgets, CPU,
and available CUDA.  The preferred layout must use no row-panel or Z-panel
copy.  Any reproducible slowdown of 15% or more against the established dense
or M4 baseline requires investigation before promotion.

---

## 9. Explicit non-goals

* No compiler-style semantic IR, generic sink callback, or scheduled-operation
  hierarchy.
* No per-`k` recompression of `C`.
* No streamed small-GEMM Stage 1 when the logical A row panel is contiguous.
* No B-side/right-basis M5 hidden behind a FoldLeft/FoldRight switch.
* No materialization of dense `b × b` product tiles on the M5 path.
