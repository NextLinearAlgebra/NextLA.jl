# TLR `compress!` — algorithm

One-shot, fixed-budget randomized compression of a dense matrix into a
Tile Low-Rank (TLR) container. Every off-diagonal tile is compressed
independently with a randomized range finder (sketch → orthogonalize →
co-range → truncate); diagonal tiles are stored dense.

The design goal is *scheduling regularity*: unlike the Adaptive Randomized
Algorithm (ARA), the sketch width is fixed at `maxrank` for all tiles, so the
work is a batch of identically-shaped GEMM / Cholesky / TRSM calls with no
per-tile adaptive control flow. The per-tile rank is recovered *after* the
fixed-width sketch by truncation.

## Notation

| symbol | meaning |
| --- | --- |
| `A` | dense input, `m × n` |
| `b` | tile size (`tile_m = tile_n = b`) |
| `r = maxrank` | fixed sketch width |
| `r_eff` | `min(r, tile_m, tile_n)` — clamp for boundary / thin tiles |
| `T` | working precision (e.g. `Float32`) |
| `Thi` | accumulation precision for orthogonalization (`Float64` for `Float32`) |
| `Ω` | random Gaussian sketch matrix, `tile_n × r` |
| `U, V` | retained factors of a tile, `A_tile ≈ U · Vᴴ` |
| `tol`, `rel` | error budget and whether it is relative to `‖A_tile‖_F` |
| `eps(T)` | machine epsilon of `T` (`≈1.2e-7` for `Float32`) |

## Top level

```text
compress!(A_tlr, A; tol, rel):
    assert A is square with square tiles
    eps_sq ← tol²

    copy the dense diagonal tiles of A into A_tlr.D (+ corner)

    # three tile categories: interior (b×b), right-boundary (b×tail_n),
    # bottom-boundary (tail_m×b). On GPU each runs on its own stream.
    for cat in {interior, right, bottom}:
        compress_category!(A_tlr, A, cat, eps_sq, rel)
    for cat in {interior, right, bottom}:
        store cat.ranks, cat.residuals into A_tlr.ranks, A_tlr.resid
    return A_tlr
```

## Per-category pipeline

All tiles of a category are processed as one batch of width `r_eff`.

```text
compress_category!(A_tlr, A, cat, eps_sq, rel):
    if cat has no tiles: return
    r_eff ← cat.r_eff
    if r_eff == 0:                       # maxrank == 0 → every tile is rank 0
        cat.normA_sq ← ‖A_tile‖²_F for each tile
        cat.ranks    ← 0
        cat.resid_sq ← cat.normA_sq
        return

    # ── Step 0: reference term for the error indicator ──────────────
    cat.normA_sq[t] ← Σ |A_tile[t]|²   (accumulated in Float64)   for each tile t

    # ── Step 1: range sketch  Y = A · Ω ─────────────────────────────
    Ω ← randn!(...)                      # drawn into the V buffer
    U ← A_tile · Ω                       # batched GEMM, U is tile_m × r_eff

    # ── Step 2: orthogonalize the columns of U (in precision Thi) ────
    cholqr_pass!(U; rescue = true)       # shifted Cholesky-QR, pass 1
    cholqr_pass!(U; rescue = false)      # shifted Cholesky-QR, pass 2

    # ── Step 3: co-range  V = Aᴴ · U  (overwrites Ω) ─────────────────
    V ← A_tileᴴ · U                      # batched GEMM, V is tile_n × r_eff

    # ── Step 4: rank detection + truncation ─────────────────────────
    truncate!(U, V, cat.ranks, cat.resid_sq, cat.normA_sq, eps_sq, rel)
    zero the padding columns r_eff+1 … r of V
    return
```

`A_tile ≈ U · Vᴴ = U · (AᴴU)ᴴ = U Uᴴ A`, i.e. the orthogonal projection of the
tile onto `range(U)` — accurate exactly when `U` has orthonormal columns, which
is what Step 2 guarantees.

## Shifted Cholesky-QR

Cholesky-QR orthogonalizes via the Gram matrix `G = UᴴU = RᴴR`, then
`U ← U R⁻¹`. A diagonal shift keeps `G` positive-definite when the sketch is
rank-deficient (the common oversampled case, `r > numerical rank`).

```text
cholqr_pass!(U; rescue):
    G ← Uᴴ U                                    # batched, precision Thi
    for each tile (slab) b:
        tr  ← trace(G_b);   mx ← max diag(G_b)
        if rescue:  shift ← √eps(Thi) · tr / r  # survives rank deficiency
        else:       shift ← eps(Thi) · r · mx   # eps-level; restores the norms
                                                #   the rescue shift deflated
        if shift == 0 (zero tile): shift ← 1    # keep potrf PD, U stays 0
        G_b += shift · I
    R ← chol(G)   (upper, batched potrf)
    U ← U · R⁻¹   (batched trsm)
```

## Truncation (rank detection)

Run as one fused shared-memory kernel per tile: `R = r_eff` threads, one thread
per column. The reconstruction error of keeping the top-`k` columns decomposes
(with `U` orthonormal) as

```text
‖A_tile − U_k V_kᴴ‖²  =  resid  +  Σ_{dropped} ‖v_j‖²
       resid  =  ‖A_tile‖²  −  Σ_j ‖v_j‖²      (randQB_EI error indicator:
                                                the range-capture error the
                                                fixed-width sketch left behind)
```

so the greedy drop spends only the budget left after accounting for `resid`.

```text
truncate!(U, V, ranks, resid_sq, normA_sq, eps_sq, rel):
  for each tile (one workgroup):
    norms[j] ← ‖V[:, j]‖²                       (Float64 accumulation)
    sort columns by norms descending            (parallel rank sort, O(R))

    nA_sq ← normA_sq[tile]
    resid ← max(nA_sq − Σ norms, 0)

    # ── numerical-safety floor #1: resid cancellation ──────────────
    # resid = ‖A‖² − ‖V‖² subtracts two O(‖A‖²) sums; below the fp
    # rounding floor of V it is pure cancellation noise, not real
    # range-capture error → treat the range as captured.
    if resid < size(U,1) · eps(T) · nA_sq:  resid ← 0

    # ── budget, with safety floor #2: √eps(T) accuracy floor ───────
    # a precision-T sketch cannot resolve relative error below √eps(T),
    # so a tol below that floor would keep the sketch's noise columns
    # and inflate the rank; floor the budget at eps(T)·‖A‖².
    target ← rel ? eps_sq · nA_sq : eps_sq
    budget ← max(target, eps(T) · nA_sq) − resid

    # ── greedy drop from the smallest column upward ────────────────
    k ← R;  dropped ← 0
    if budget ≥ 0:
        for j from smallest to largest:
            if norms[j] > budget: break
            budget −= norms[j];  dropped += norms[j];  k −= 1

    ranks[tile]    ← k
    resid_sq[tile] ← resid + dropped            # reported Frobenius² error
    gather the k retained columns of U, V; zero-pad the rest
```

## FAIL semantics

A tile *fails* to compress within budget exactly when

```text
FAIL(tile)  ⟺  rank(tile) == maxrank  ∧  residual(tile) > tol
```

i.e. the sketch saturated its fixed width and still could not capture the range.
Because `resid` (the range-capture term) survives the cancellation floor only
when it is genuinely large, an under-captured tile keeps full rank and reports
an honest residual above `tol` instead of silently claiming convergence. Callers
inspect `residuals(A_tlr)` to route such tiles to dense storage or a
higher-rank second pass — nothing is recomputed on the tiles that succeeded.

## Precision notes (fp32)

- The characteristic noise scale of the fp32 sketch is `√eps(Float32) ≈ 3.4e-4`;
  columns below it are numerical noise, real singular directions sit above it.
- `tol ≥ √eps(T)`: exact rank recovery *and* `rel_err ≤ tol`, no drama.
- `tol < √eps(T)`: below the sketch noise floor. The budget floor recovers the
  **exact rank** at the best accuracy fp32 gives (`rel_err ≈ 2e-5`, the fp32
  ceiling), rather than retaining 1–2 noise columns to chase an unreachable
  target. A hard `rel_err ≤ tol` at such a `tol` requires fp64.
- The error *indicator* `√(‖A‖²−‖V‖²)` is cancellation-limited to `~√(b·eps)`;
  the *direct* reconstruction error `‖A−UVᴴ‖` is not, and reaches `~eps·√b`.
