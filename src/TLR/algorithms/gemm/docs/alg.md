# Global Row-Basis TLR GEMM

This document describes the row-basis TLR-output path implemented in
`src/TLR/algorithms/gemm/row_basis/`, together with the immediate extension
boundary for panel-specific rank capacities.  It is an execution design for

```text
C <- alpha * A * B + beta * C
```

where `A`, `B`, and `C` are regular-grid `TLRMatrix` objects.  It does not
materialize dense output tiles on the row-basis path.  The fallback M4 path
does materialize a bounded dense slab and recompresses it.

The public TLR-output GEMM selects this path for non-transposed operands.  The
preferred physical layout is

```text
A: TileRowMajor                 B: TileColMajor.
```

For that pair, A's row panels and B's column panels are zero-copy.  Other
orders are correct through bounded packing helpers; they are not the primary
performance target.  The terms "row-major" and "column-major" below refer to
the logical operands after `N/T` canonicalisation.

---

## 1. Representation and invariants

For an output row `i`, contraction index `k`, and output column `j`, write

```text
A[i,k] = U[i,k] V[i,k]'          U, V: bm x rA
B[k,j] = W[k,j] Z[k,j]'          W: bk x rB,  Z: bn x rB
C[i,j] = Uc[i,j] Vc[i,j]'        Uc: bm x rC, Vc: bn x rC.
```

`rA`, `rB`, and `rC` are factor *capacities*, not the effective numerical
ranks in `ranks(C)`.  Factors are zero-padded through their capacity.  This is
an important storage invariant: it makes the factor panels rectangular and
allows a batch to use a common GEMM shape even when the effective ranks differ.

The output left factors are maintained approximately orthonormal:

```text
Uc[i,j]' * Uc[i,j] ~= I.
```

This lets the final pruning operation compute coordinate energies from the
right factors, without reconstructing a dense `bm x bn` tile.

For one output row, the algorithm builds a shared orthonormal basis `Q[i]` and
coefficient blocks `P[i,k]` such that

```text
U[i,k] ~= Q[i] * P[i,k],          Q[i]' * Q[i] ~= I,
P[i,k]: t[i] x rA.
```

The product contribution is consequently

```text
DeltaC[i,j] ~= Q[i] * M[i,j]',
M[i,j] = sum_k Z[k,j] * R[i,k,j],             # bn x t[i]
R[i,k,j] = W[k,j]' * V[i,k] * P[i,k]'.        # rB x t[i]
```

Only `M[i,j]` is accumulated; the row-basis path never forms a dense product
tile.

---

## 2. Numerical and precision policy

* Ordinary factor products use `precision_gemm!` or
  `precision_gemm_batched!` with the caller's GEMM compute mode.
* Gram matrices, Cholesky QR, numerical-rank detection, and squared norms use
  the promoted TLR orthogonalisation type.
* The output rank cap is authoritative.  If it prevents the requested error
  tolerance from being met, the achieved residual reports that fact; the code
  must not silently claim convergence.
* `residuals(C)` includes the row-basis truncation diagnostic in addition to
  the tile-level final-prune error.
* All factor tails past a detected rank are explicitly zeroed.  The batched
  beta merge relies on this, not merely on a convention in the caller.

The current end-to-end driver uses uniform global capacities.  A future
panel-capacity layout retains the same invariant within a physical panel; see
section 7.

---

## 3. Stage 1 -- build a shared row basis

For logical row-major `A`, the stored row panel is already

```text
Ubar = [ U[i,1] ... U[i,K] ]                    # bm x (K*rA), zero-copy.
```

The current driver uses unit block weights (`gamma[k] = 1`), although the
standalone basis routine supports general nonnegative per-`k` weights.  It
performs the following calculation.

```text
FUNCTION build_row_basis!(Ubar, Omega, gamma, eps_basis, S, tmax, tguard)
    # Omega has shape (K*rA) x S.  Copy and scale it; Ubar is never modified.
    Omegagamma <- Omega
    FOR k = 1:K
        Omegagamma[(k-1)*rA+1 : k*rA, :] *= gamma[k]
    END

    Y  <- Ubar * Omegagamma                    # bm x S
    Q0 <- mixed_cholqr2_basis!(Y)              # approximately orthonormal
    Pfull <- Q0' * Ubar                        # S x (K*rA)

    # Weight selection, but retain an unweighted representation coefficient.
    Pweighted <- scale_each_k_block(Pfull, gamma)
    Ksmall <- Pweighted * Pweighted'           # promoted S x S covariance
    lambda, R <- eigh(Ksmall)
    t <- choose_rank(lambda, eps_basis, tmax)

    IF t >= tguard
        RETURN saturated(t)                    # caller will route this row to M4
    ELSE IF t == 0
        RETURN Q0[:, 1:0], Pfull[1:0, :], 0
    END

    Q <- Q0 * R[:, 1:t]
    P <- R[:, 1:t]' * Pfull                    # t x (K*rA), unweighted
    Ebar <- Ubar - Q * P                       # explicit basis-error diagnostic
    RETURN Q, reshape(P, t, rA, K), t, ||Ebar||_F^2
END
```

The rotations and explicit residual are intentionally retained.  The former
returns coefficients in the unweighted factor representation; the latter gives
a diagnostic for the approximation introduced by sharing a row basis.

### Saturation guard and M4 fallback

The row basis is useful only while `t` is appreciably smaller than `bm`.  For
`beta == 0`, a row that reaches `sat_threshold * bm` is routed to the existing
M4 dense-slab/recompression path when that path has a write-once row-family
lowering.  While this guard is armed, the sketch is capped at the threshold and
the basis build can return early.  After a short streak of saturated rows, the
remaining rows bypass the probe and use M4 directly.

This is a performance routing choice, not a change in the error contract.  It
does not currently apply to `beta != 0`, because M4 overwrites rather than
merges an existing TLR output.

---

## 4. Stage 2 -- batched coefficient accumulation

For an output row, Stage 2 computes all `M[i,j]` at once.  It chooses one of
two exact associations.

```text
IF t <= rA
    FOR k = 1:K
        T[k] <- V[i,k] * P[i,k]'               # bk x t; independent of j
    END
    R[k,j] <- W[k,j]' * T[k]                   # rB x t, batched over (k,j)
ELSE
    S[k,j] <- V[i,k]' * W[k,j]                 # rA x rB, batched over (k,j)
    R[k,j] <- S[k,j]' * P[i,k]'                # rB x t, batched over (k,j)
END

FOR j = 1:qn
    Zstack[j] <- [ Z[1,j] ... Z[K,j] ]         # bn x (K*rB)
    Rstack[j] <- [ R[1,j]; ...; R[K,j] ]       # (K*rB) x t
    M[i,j] <- alpha * Zstack[j] * Rstack[j]
END
```

The `t <= rA` association uses batches of sizes `K`, `K*qn`, and `qn`.
The other association uses `K*qn`, `K*qn`, and `qn`.  With column-major `B`,
`Zstack[j]` is a zero-copy reshape of the stored column panel.  With another
layout it is packed; the same algebra and batch structure are retained.

`alpha` is folded into the terminal `Zstack * Rstack` GEMM.  Stage 3 therefore
receives `Vm[:,:,j] = alpha * M[i,j]`.

---

## 5. Stage 3 -- output update and pruning

There are two materially different cases.

### 5.1 `beta == 0`: product only

Every output tile in the row has the same left factor `Q`.  Broadcast it into
one slab per output column, place `Vm` in the right factors, then prune the
whole row in one batch.

```text
Qm[:,:,j] <- Q                              # bm x t x qn
Vm[:,:,j] <- alpha * M[i,j]                 # bn x t x qn

prune_orthogonal_columns!(Qm, Vm,
                          active_columns = t,
                          maxrank = min(rC, t),
                          tolerance)
copy the zero-padded factor slabs and the rank/error vectors into C
```

The only device-to-host transfers on this path are the final rank and error
vectors needed by the container's host-resident diagnostics.

### 5.2 `beta != 0`: C2a row-batched orthogonal merge

The old tile basis generally differs from `Q`, so simply adding right factors
is incorrect.  The old factors are deliberately read at the complete padded
capacity `rcap = maxrank(C)`, even if the effective old rank is smaller.

For every `j` in the output row, C2a computes the following, but all arrays
carry a third slab dimension `j = 1:qn` and every GEMM/CholQR operation is
batched over that dimension.

```text
FUNCTION merge_row_block!(Q, Vm, Uold[:,:,:], Vold[:,:,:], beta)
    # Uold and Vold are bm/bn x rcap x qn and are zero-padded.
    D    <- Q' * Uold
    Ures <- Uold - Q * D

    # First residual factorisation: Ures ~= Q0 * V0'.
    Q0, V0 <- mixed_cholqr2_compress!(Ures)

    # Reorthogonalise Q0 against Q, then factor again:
    # Q0 ~= Q * D2 + Qres * V1'.
    D2   <- Q' * Q0
    Qres <- Q0 - Q * D2
    Qres, V1 <- mixed_cholqr2_compress!(Qres)

    # Preserve the represented old factor in the new coordinates.
    D    <- D + D2 * V0'
    Vtmp <- Vold * V0

    Qmerge <- [ Q | Qres ]
    Vmerge[:, 1:t, :]     <- Vm + beta * Vold * D'
    Vmerge[:, t+1:t+rcap, :] <- beta * Vtmp * V1

    # Qres/V1 tails are zero after CholQR pruning.  Consequently the full
    # width t+rcap is equivalent to each tile's t+rho[j] active width.
    prune_orthogonal_columns!(Qmerge, Vmerge,
                              active_columns = t + rcap,
                              maxrank = rcap,
                              tolerance)
    RETURN Qmerge, Vmerge, rank_vector, error_vector
END
```

This full-width formulation is intentional.  It avoids a host read of each
data-dependent residual rank `rho[j]`, avoids host-created `1:rho[j]` views,
and turns the old sequential per-tile merge into two batched CholQR2 calls,
batched small GEMMs, and one batched final prune.  The rank and error vectors
are copied to the host once per row after the merge completes.

The sequential `merge_row_basis_tile!` remains a useful one-tile reference and
test primitive.  The end-to-end driver uses the C2a row-batched merge for the
normal `beta != 0`, positive-capacity case.

### Zero product basis

If `t == 0`, the product is zero.  For a nonzero `beta` the existing right
factor is scaled in place; no basis merge is required.  If `maxrank(C) == 0`,
there is no stored old factor to fold, so the product-only batched path is used.

---

## 6. Current layout support and fallback behaviour

| Situation | Current behaviour |
| --- | --- |
| A row-major, B column-major | Preferred zero-copy A row panel and B Z stack. |
| A row-major, B row-major | A row panel is zero-copy; B column panels are packed. |
| A column-major | A row panels are packed before basis construction. |
| Non-transposed, regular-grid TLR output | Row-basis driver is selected. |
| Saturated row, `beta == 0`, row-family M4 available | Single-row M4 dense slab followed by compression. |
| Transposed TLR output | Existing M4 fallback rules apply; it does not support `beta != 0`. |
| Boundary/tail output tiles | Not supported by the current TLR-output row-basis interface. |

Packing is bounded but is not yet a performance-equivalent substitute for the
preferred layout.  A right-basis dual is explicitly out of scope; it should not
be smuggled in as a FoldLeft/FoldRight option.

---

## 7. Panel-specific rank capacities: target extension

The intended storage extension replaces one global capacity with a capacity per
physical panel:

```text
TileRowMajor:  rcap[i] for all factors in tile row i
TileColMajor:  rcap[j] for all factors in tile column j.
```

Effective tile ranks remain separate metadata.  Padding is retained *within*
a panel, so the key fusions remain valid:

```text
Stage 1, B row-major:  V[i,k]' * [W[k,j0] ... W[k,j1]]
Stage 3, FoldRight:    [U[i,1] ... U[i,K]] * Tstack[i]
Stage 3, FoldLeft:     Tstack[:,j] * [Z[1,j] ... Z[K,j]]'
```

Each expression holds a fixed physical panel and therefore a fixed capacity.
What changes is the surrounding batch: operations with different `(m,n,k,ld*)`
must be bucketed or issued through grouped GEMM.

For C2a specifically:

* a row-major output `C` has one capacity for an output row, so its beta merge
  remains one homogeneous row batch;
* a column-major output `C` can have different capacities across `j`, so the
  row is partitioned into equal-capacity buckets before the same C2a algorithm
  is applied.

The first ragged implementation should use an aligned arena plus panel offsets,
not one independent GPU allocation per panel.  Capacity values should be
quantised to a small aligned palette where possible.  This limits the number of
GEMM groups and protects tensor-core-friendly shapes.

An optional later primitive may accept per-slab active/max-rank vectors in the
prune kernel.  It is useful for more general ragged compression, but C2a does
not require it because its full-width residual tails are zero.

---

## 8. Workspace and performance contracts

The current driver allocates phase workspaces directly; `RowBasisWorkspacePlan`
is a sizing model rather than the runtime allocator.  Its liveness principle is
still the intended direction:

```text
single-stream peak = max(basis-build peak,
                         coefficient-accumulation peak,
                         merge peak).
```

The important live shapes are

```text
basis:        Ubar bm x (K*rA), Omega (K*rA) x S,
              Pfull S x (K*rA), promoted S x S covariance
coefficients: Rstack (K*rB) x t x qn, M bn x t x qn
C2a merge:    Q/V merge bm/bn x (t+rcap) x qn,
              residual/CholQR work bm x rcap x qn and rcap x rcap x qn.
```

`max_workspace` is a correctness budget for the direct/M4 paths.  It is not a
claim that a particular allocation saturates a GPU.  Any pipeline across rows
must be measured: concurrent large GEMMs often contend for the same device,
and the batched C2a merge has deliberately removed the host synchronisations
that would otherwise make such a pipeline tempting.

Performance acceptance should measure warmed execution, GPU kernel/API counts,
and accuracy.  In particular, the beta path must be compared with the same
coefficient workload at `beta == 0`; a row-batched beta merge should eliminate
per-tile device-to-host synchronisation, though it still performs more
numerical work than a product-only update.

---

## 9. Non-goals

* No generic contraction IR or output-sink abstraction.
* No dense `bm x bn` product tile on the row-basis path.
* No recompression inside the contraction-index loop.
* No scalar tile loop merely because a panel is not contiguous; use bounded
  packing and batches.
* No B-side/right-basis algorithm hidden behind an association switch.
* No claim that grouped GEMM alone implements panel-ragged TLR: storage
  accessors, scratch scheduling, compression, and output bucketing must agree
  on the same panel capacities.
