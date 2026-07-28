# TLR compression algorithm

This page describes the compression implementation in
`src/TLR/algorithms/compression/`. The public entry point is `compress!`; the
numerical core is the reusable `compress_tiles!` pipeline.

## 1. Public interface and supported containers

```julia
compress!(A_tlr, A; tol=0.0, rel=false)
compress!(A_tlr, A, workspace; tol=0.0, rel=false)
```

Both methods overwrite the factor storage, ranks, and residual diagnostics of
`A_tlr`. The first allocates a workspace for the call. Repeated compression should
reuse one:

```julia
ws = alloc_workspace(A_tlr)
for A in matrices
    compress!(A_tlr, A, ws; tol=1e-3)
end
```

The workspace is tied to the container's backend, element type, tile layout,
`maxrank`, and factor arrays. Reuse it only with the container for which it was
created.

The implemented container behavior is:

| container | matrix shape | compressed tiles | dense tiles |
| --- | --- | --- | --- |
| `PaddedFTLRMatrix` | rectangular supported | every tile, including the corner | none |
| `TLRMatrix` | currently square only | interior off-diagonal, right, and bottom regions | regular diagonal and dense corner |

`A` must have the same dimensions and element type as `A_tlr`, and its storage must
be compatible with the container backend. `tol` must be nonnegative.

For `TLRMatrix`, diagonal tiles are copied exactly before low-rank
compression. Their recorded rank is `min(tile_m, tile_n)` and their residual is zero.

## 2. Tile categories and workspace

Compression is grouped by physical low-rank region because every tile in one region
has the same shape:

- interior;
- right boundary;
- bottom boundary;
- corner, for `PaddedFTLRMatrix` only.

`lowrank_regions(A_tlr)` selects the categories belonging to the container.
`alloc_workspace` creates one `CompressCategoryWorkspace` per category and one stream
per category on a non-CPU backend.

For a category of `tm × tn` tiles, define

```text
r        = maxrank(A_tlr)
S        = min(r, tm, tn)       sketch width
R_keep   = min(r, S)            maximum stored rank
n        = number of tiles in the category
```

The output capacity is also the randomized sketch capacity. There is no separate
oversampling parameter: any desired buffer must be included in `maxrank`.

The first `S` columns of the destination factor panels are used as work-precision
scratch:

| field | role |
| --- | --- |
| `Q_T` | aliases output `U`; holds `Y = AΩ`, then the orthogonalized `Q` |
| `V_T` | aliases output `V`; holds `Ω`, then `AᴴQ` |
| `R_work` | leading `S × S` part of the expired `Ω` storage |
| `Y_hi` | accumulation-precision copy used by Cholesky-QR |
| `G_hi` | accumulation-precision Gram matrices |
| `ranks_local` | category-local detected ranks |
| `norm_err_sq` | tile energy on entry to pruning, achieved squared error on exit |

Thus the allocated numerical scratch is the high-precision `Y_hi` and `G_hi`
storage. For one category its size is

```text
(tm*S + S*S) * n
```

elements of `_compress_accum_type(T)`. `compress_bytes` reports the corresponding
byte count. `carve_tile_workspace` and `alloc_tile_workspace` expose the same layout
for callers that compress standalone tile batches.

When `S == 0`, no sketch is formed. Every category tile receives rank zero and its
squared Frobenius norm becomes its squared residual.

## 3. Top-level orchestration

Conceptually, `compress!` performs:

```text
validate dimensions and tolerance

if A_tlr is TLRMatrix:
    copy dense diagonal and corner tiles
    set their rank/residual diagnostics

for each low-rank category:
    build DenseTiles views into A
    compress_tiles!(source, category_workspace)

scatter category-local ranks and sqrt(squared_errors)
into A_tlr.ranks and A_tlr.resid
```

On CPU, categories run sequentially. On a non-CPU backend, every category is submitted
to its own stream; all streams are synchronized before results are scattered into the
container diagnostics.

Tile order affects the category-to-global rank-index mapping, but not the numerical
pipeline.

## 4. Reusable tile-source pipeline

`compress_tiles!` operates on a `TileSource`, not directly on a TLR container.
Two sources currently exist:

- `DenseTiles`: views of category tiles inside a dense matrix, used by `compress!`;
- `PackedTiles`: a packed `[tm, tn, ntiles]` tile batch, used by intermediate-producing
  algorithms and tests.

Both provide batched tile views and a high-precision squared-norm operation. The
compression pipeline is otherwise input-independent.

For every tile `A_t` in a category:

```text
1. Ω_t ← randn(tn, S)
2. Q_t ← A_t * Ω_t
3. Q_t ← cholqr2(Q_t)
4. V_t ← A_tᴴ * Q_t
5. nA²_t ← ‖A_t‖²_F
6. choose rank, compact retained columns, and clear padding
```

The resulting approximation is

```text
A_t ≈ Q_k * V_kᴴ = Q_k * (A_tᴴ Q_k)ᴴ.
```

Real inputs use transpose GEMMs and complex inputs use adjoint GEMMs.

All tiles of a category use batched GEMM, batched Cholesky factorization, and batched
triangular solves. Randomness is drawn directly into the output `V` workspace and is
overwritten after the range sketch.

## 5. Shifted Cholesky-QR2

`cholqr2!` applies two shifted Cholesky-QR passes. For a current basis `Q` with
`m = size(Q,1)` and width `S`, one pass is:

```text
Y_hi  ← accumulation_precision(Q)
G_hi  ← Y_hiᴴ * Y_hi
shift ← δ * max(diag(G_hi))
G_hi  ← G_hi + shift*I
R     ← chol(G_hi).U
Q     ← Q * inv(R)
```

where

```text
δ = 11 * (m*S + S*(S+1)) * eps(real(T_hi))/2.
```

No unshifted factorization is attempted. If the Gram matrix is exactly zero, the
shift is replaced by `eps(real(T_hi))`; the zero basis remains zero after the solve.

On the first pass, failed Cholesky slabs have their shift multiplier doubled and are
retried, with a fresh Gram matrix, up to 40 times. A remaining failure throws
`PosDefException`. The second pass uses the prescribed shift without the escalation
loop.

Gram formation and Cholesky use the accumulation type. The triangular solve and
stored basis stay in the container element type.

Accumulation types are:

| factor type | Cholesky-QR accumulation type |
| --- | --- |
| `Float16` | `Float32` |
| `Float32` | `Float64` |
| `Float64` | `Float64` |
| `ComplexF32` | `ComplexF64` |
| `ComplexF64` | `ComplexF64` |

## 6. Error indicator and rank selection

After orthogonalization,

```text
V = AᴴQ
```

and, for an orthonormal `Q`, the squared error after retaining a subset `K` is

```text
‖A - Q_K V_Kᴴ‖²_F
    = range_residual + Σ(dropped j) ‖v_j‖²
```

with the randQB_EI range-capture indicator

```text
range_residual = max(‖A‖²_F - Σ(j=1:S) ‖v_j‖², 0).
```

Tile energy and factor-column energy are accumulated in `Float64`, including for
complex input by separately converting real and imaginary parts.

Because the indicator subtracts two nearly equal sums, the implementation removes
cancellation noise below

```text
size(Q,1) * eps(real(T)) * ‖A‖²_F.
```

The requested squared target is

```text
target = rel ? tol² * ‖A‖²_F : tol².
```

Rank selection uses the effective budget

```text
budget = max(target, 2δ * ‖A‖²_F) - range_residual.
```

The `2δ` term is the realized Cholesky-QR orthogonality/energy-accounting floor.
Consequently `tol == 0` does not request accuracy below the numerical floor.

The pruning kernel then:

1. computes every `V`-column energy;
2. stably sorts column indices by ascending energy, breaking ties by source index;
3. greedily drops the smallest columns while each fits in the remaining budget;
4. enforces the hard `R_keep` capacity by dropping additional smallest columns;
5. records `range_residual + dropped_energy`;
6. compacts the retained set into the first `k` columns with a deterministic
   minimum-move map;
7. zeroes every factor column after `k`.

Compaction preserves the selected column set, but it does not sort retained columns by
energy. Downstream kernels use the recorded rank and the compact leading columns.

The category stores squared errors during computation. `_store_category_results!`
takes a nonnegative square root and writes per-tile Frobenius residuals into
`residuals(A_tlr)`.

## 7. Saturation and residual semantics

There is no separate failure flag. Callers determine whether the requested tolerance
was met from the stored residual:

```text
threshold(tile) = rel ? tol * ‖A_tile‖_F : tol
success(tile)   = residual(tile) <= threshold(tile)
```

A tile whose numerical content does not fit inside the fixed sketch width normally
retains the full available rank `S` and reports a residual above the threshold. For a
thin or boundary tile, saturation means `rank == S`, not necessarily
`rank == maxrank`.

The residual includes both:

- range error left by the fixed-width randomized sketch;
- energy of columns removed by tolerance pruning or the hard capacity.

A saturated tile is not automatically a failure: it may still satisfy the requested
tolerance. Conversely, the numerical precision floor can permit a reported residual
slightly above a tighter requested threshold without a dedicated status bit. Callers
that require strict acceptance must compare the residual explicitly and can route
unsatisfied tiles to dense storage or a higher-capacity second pass.

## 8. Precision and storage invariants

- Sketch and co-sketch GEMMs use the factor/storage type `T`.
- `Q` and `V` are stored in `T`.
- Gram matrices and Cholesky use `_compress_accum_type(T)`.
- Tile norms, column energies, and reported squared-error accounting use `Float64`.
- Cholesky triangular factors are copied into expired work-precision `Ω` storage
  before the work-precision batched TRSM.
- Factor columns beyond the detected rank are always zeroed. This is required because
  downstream padded GEMMs may use the full allocated `maxrank`.
- `ranks(A_tlr)` stores the compact leading-column count; `residuals(A_tlr)` stores an
  estimated Frobenius norm, not its square.

## 9. Important limitations

- Compression is one-shot at a fixed sketch width; it does not adaptively extend a
  failed sketch.
- `maxrank` is both sketch width and output capacity.
- The tolerance is per tile, not a global matrix error allocation.
- Workspace is allocated per category; GPU categories may hold their scratch
  concurrently.
- `TLRMatrix` compression currently requires a square matrix.
- The implementation estimates error using the randQB_EI energy identity; it does not
  recompute the dense reconstruction error.
