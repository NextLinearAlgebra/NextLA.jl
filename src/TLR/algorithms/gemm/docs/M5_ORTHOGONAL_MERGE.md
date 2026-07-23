# Milestone 5 — shared-panel orthogonal TLR merge

> **Algorithm selected; full GEMM integration remains deferred.** The reusable
> numerical foundation is implemented first. Panel scheduling and the complete
> TLR-output driver must not begin until the standalone merge primitive passes
> its numerical and performance gates.

## 1. Performance rule

Reuse is a source-organization concern, not a requirement to decompose hot
paths into separately launched primitives.

- Prefer fused kernels whenever splitting would add launches, global
  intermediates, synchronization, or workspace traffic.
- Duplicate a specialized kernel when compile-time specialization cannot
  preserve the optimized path.
- Shared APIs may expose standalone diagnostic operations, but production
  algorithms are not required to assemble themselves from those operations.
- Allocation and launch-count regressions are correctness failures for
  workspace-bounded paths.

The rank-pruning implementation follows this rule: coordinate norms, stable
selection, factor compaction, rank/error output, and zero padding are one fused
kernel. There is no materialized energy array.

## 2. Output invariant

Each output tile is stored as

```text
C_ij = U_ij V_ij',       U_ij' U_ij ≈ I.
```

The left-orthogonal invariant makes coordinate-energy pruning exact:

```text
‖Q V'‖²_F = ‖V‖²_F = Σ_l ‖V[:,l]‖².
```

Column-energy pruning is not SVD-optimal, but the error of the columns actually
dropped is known exactly while the left basis remains orthonormal.

## 3. Shared row-panel preprocessing

For a contraction panel `K` and output row `i`, concatenate the update's left
factors:

```text
W_i = [U^A_i,k₁ | ... | U^A_i,kq]       b × s.
```

Compress only its numerical nullspace:

```text
W_i ≈ Q_i P_i
Q_i' Q_i ≈ I
```

The numerical primitive stores the coefficient in TLR orientation:

```text
F_i = P_i'
W_i ≈ Q_i F_i'.
```

`Q_i` and `F_i` are shared by every output tile `(i,j)` in the row for this
panel. This is the principal reuse that makes the panel-first algorithm
preferable to independently orthogonalizing the update against each old tile.

The panel rank threshold is a numerical-rank threshold, not the user
approximation tolerance. If `W_i = Q_i P_i + E_i`, the tile update error is
`E_i H_ij'` and cannot be budgeted from `‖E_i‖` alone.

## 4. Tile update

Stage 1 and the update-factor construction produce

```text
ΔC_ij = W_i H_ij'.
```

For the existing tile

```text
C_ij = U V',
```

project its small left basis onto the shared panel basis:

```text
D    = Q_i' U
Ures = U - Q_i D.
```

Form the residual Gram explicitly in orthogonalization precision:

```text
Gres = Ures' Ures.
```

Do not use `U'U - D'D`; it has the same catastrophic-cancellation structure as
the old randQB error indicator.

Compress only the numerical nullspace of the residual:

```text
Ures ≈ Qres Rres
Eres = Rres'
Ures ≈ Qres Eres'.
```

The merged right coefficients are then

```text
Vshared = H_ij F_i + V D'     b × t
Vres    = V Eres              b × ρ

Qmerge  = [Q_i | Qres]
Vmerge  = [Vshared | Vres].
```

The identity is

```text
Qmerge Vmerge'
    = Q_i(P_i H_ij' + D V') + Qres Rres V'
    ≈ W_i H_ij' + U V'.
```

Scalars are folded into the right factors before the merge:

```text
H_ij ← alpha H_ij
V    ← beta V.
```

## 5. Orthogonality safeguards

The energy identity requires the active columns of `Qmerge` to be orthonormal.

1. Form `Ures` explicitly.
2. Recompute `Gres` from `Ures` in higher precision.
3. Apply shifted mixed-precision CholQR2.
4. Check `‖Q_i'Qres‖`; re-project and repeat orthogonalization when needed.
5. Check the retained basis after numerical-rank pruning.
6. Route a tile to the dense/randQB or small-core-SVD fallback if the invariant
   cannot be restored cheaply.

The factorization primitive returns the composite factor from both CholQR
passes. Returning only the second Cholesky factor is incorrect:

```text
R = R₂ R₁.
```

## 6. Final fused pruning

For every active coordinate:

```text
energy[l] = ‖Vmerge[:,l]‖².
```

The production implementation performs energy accumulation, deterministic
selection, in-place compaction, rank/error output, and tail clearing in one
kernel.

Drop the smallest coordinates while they fit the remaining tile budget. If
more than `rmax` coordinates remain, enforce the hard cap and report the
achieved discarded energy even when it exceeds the tolerance. Rank overflow
must never be hidden.

Only this final pruning step consumes the application tile tolerance.

## 7. Shared numerical layer

The implementation lives under `src/TLR/numerics/` and is included before
compression and GEMM:

- `precision.jl`
  - `tlr_orthogonalization_type`
  - batch/adjoint helpers
- `norms.jl`
  - fused batched Frobenius norm reduction
  - dense-source tile-norm kernels
- `cholqr2.jl`
  - minimal-workspace `mixed_cholqr2_basis!`
  - factor-producing `mixed_cholqr2_factor!`
  - `CholQR2FactorWorkspace`
  - numerical-rank floor policy
- `rank_pruning.jl`
  - fused randQB error-indicator pruning
  - fused exact-coordinate pruning
  - `mixed_cholqr2_compress!`

Compression keeps its existing minimal-workspace basis-only CholQR and fused
randQB pruning path. The factor-producing path has additional `s×s` scratch
because it must retain and combine both triangular factors.

## 8. Implementation sequence

### Step 0 — numerical foundation

- Extract the shared numerical layer without changing compression behavior.
- Prove `X ≈ QV'` for the factor-producing mixed CholQR2.
- Prove numerical-rank removal on zero, repeated, and nearly dependent columns.
- Preserve fused pruning, deterministic compaction, and zero padding.
- Pass CPU and CUDA compression regressions.

### Step 1 — standalone shared-panel merge

Implement a batched primitive with no GEMM scheduling:

```text
(U, V, W, H) → (Unew, Vnew, rank, error)
```

Gate:

- merge identity to storage-precision roundoff;
- retained left-basis orthogonality;
- discarded coordinate energy equals reconstruction error when no preliminary
  numerical-rank error is present;
- correlated and uncorrelated panels;
- zero ranks, `alpha`, and `beta`;
- rank inflation measured against a reference SVD;
- no dense `b×b` temporary;
- CPU and CUDA.

### Step 2 — row-family GEMM integration

- Reuse output-independent Stage 1.
- Build `H_ij` directly.
- Compress each `W_i` once per row and panel.
- Merge all `(i,j)` tiles with the shared basis.
- Keep the dense/recompression path as oracle and overflow fallback.

### Step 3 — remaining scheduling

- Column-family traversal.
- Multi-panel error accounting and drift guards.
- Workspace byte query.
- Overflow routing and fallback batching.
- Final benchmark and allocation gates.

Boundary and dense-diagonal output integration remain later work.
