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
| `b` | nominal tile size |
| `r = maxrank` | fixed sketch width |
| `r_eff` | `min(r, m_tile, n_tile)` — clamp for boundary / thin tiles |
| `T` | working precision (e.g. `Float32`) |
| `Thi` | accumulation precision for orthogonalization (`Float64` for `Float32`) |
| `Ω` | random Gaussian sketch matrix, `n_tile × r` |
| `U, V` | retained factors of a tile, `A_tile ≈ U · Vᴴ` |
| `tol`, `rel` | error budget and whether it is relative to `‖A_tile‖_F` |
| `eps(T)` | machine epsilon of `T` (`≈1.2e-7` for `Float32`) |

## Top level

```text
compress!(A_tlr, A; tol, rel):
    assert A is square
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
        cat.norm_err_sq ← ‖A_tile‖²_F for each tile
        cat.ranks    ← 0
        return

    # ── Step 1: range sketch  Y = A · Ω ─────────────────────────────
    Ω ← randn!(...)                      # drawn into the V buffer
    U ← A_tile · Ω                       # batched GEMM, U is m_tile × r_eff

    # ── Step 2: orthogonalize the columns of U (in precision Thi) ────
    cholqr_pass!(U; rescue = true)       # shifted Cholesky-QR, pass 1
    cholqr_pass!(U; rescue = false)      # shifted Cholesky-QR, pass 2

    # ── Step 3: co-range  V = Aᴴ · U  (overwrites Ω) ─────────────────
    V ← A_tileᴴ · U                      # batched GEMM, V is n_tile × r_eff

    # ── Step 4: reference term, reusing the now-dead Gram storage ────
    cat.norm_err_sq[t] ← Σ |A_tile[t]|²   (Float64, stored in G[1,1,t])

    # ── Step 5: rank detection + truncation ─────────────────────────
    truncate!(U, V, cat.ranks, cat.norm_err_sq, eps_sq, rel)
    zero the padding columns rank+1 … maxrank of U and V
    return
```

`A_tile ≈ U · Vᴴ = U · (AᴴU)ᴴ = U Uᴴ A`, i.e. the orthogonal projection of the
tile onto `range(U)` — accurate exactly when `U` has orthonormal columns, which
is what Step 2 guarantees.

## Shifted Cholesky-QR

Cholesky-QR orthogonalizes via the Gram matrix `G = UᴴU = RᴴR`, then
`U ← U R⁻¹`. A diagonal shift keeps `G` positive-definite when the sketch is
rank-deficient (the common buffered-sketch case, `r > numerical rank`).

```text
coeff ← 11 · (m·r + r·(r+1)) · eps(Thi)/2
for pass = 1:2:
    U_hi ← Thi.(U)
    G ← U_hiᴴ U_hi                              # batched, precision Thi
    shift ← coeff · max(diag(G))
    if shift == 0: shift ← eps(real(Thi))       # zero tile remains zero
    G += shift · I
    on pass 1 only: double failed slabs' shifts and retry (at most 40 times)
    R ← Twork.(chol(G).U)                       # stored in expired V/Ω slots
    U ← U · R⁻¹                                 # work-precision batched TRSM
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
truncate!(U, V, ranks, norm_err_sq, eps_sq, rel):
  for each tile (one workgroup):
    norms[j] ← ‖V[:, j]‖²                       (Float64 accumulation)
    sort columns by norms descending            (parallel rank sort, O(R))

    nA_sq ← norm_err_sq[tile]
    resid ← max(nA_sq − Σ norms, 0)

    # ── numerical-safety floor #1: resid cancellation ──────────────
    # resid = ‖A‖² − ‖V‖² subtracts two O(‖A‖²) sums; below the fp
    # rounding floor of V it is pure cancellation noise, not real
    # range-capture error → treat the range as captured.
    if resid < size(U,1) · eps(T) · nA_sq:  resid ← 0

    # ── budget, with the realized CholQR orthogonality floor ───────
    target ← rel ? eps_sq · nA_sq : eps_sq
    budget ← max(target, 2·coeff·nA_sq) − resid

    # ── greedy drop from the smallest column upward ────────────────
    k ← R;  dropped ← 0
    if budget ≥ 0:
        for j from smallest to largest:
            if norms[j] > budget: break
            budget −= norms[j];  dropped += norms[j];  k −= 1

    ranks[tile]       ← k
    norm_err_sq[tile] ← resid + dropped          # reported Frobenius² error
    compact retained columns in place; zero-pad the rest
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

- The sketch GEMMs and triangular solves stay in `Twork`; only the copied basis,
  Gram formation, and Cholesky factorization use `Thi` (`Float64` for fp32).
- Tile energies, residual accounting, and factor-column energies are accumulated
  in `Float64`.
- The cancellation guard treats `‖A‖²−‖V‖² < m·eps(Twork)·‖A‖²` as zero.
- The truncation target is floored at twice the realized shifted-CholQR
  orthogonality coefficient. Accuracy materially below the fp32 GEMM floor still
  requires a higher-precision throughput path.
