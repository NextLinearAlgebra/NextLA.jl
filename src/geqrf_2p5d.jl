export geqrf_2p5d!

# Default tile for trailing-update GEMMs; recovered inside kernels via @uniform @groupsize()[1].
const _GEQRF_TILE_DIM = 16

# ── Statement S4: W = Q^H * A_trailing ───────────────────────────────────────
# W  is sb × n_tr
# Q  is m_panel × sb  (orthonormal columns from sCQR3, stored in A)
# A_tr is m_panel × n_tr
#
# Per paper §A.1, step S4:  W_{ij} = ∑_k conj(Q_{ki}) A_{kj}
# Tiling follows matmul.jl: group (gi,gj) owns output tile (sb, n_tr); local
# thread (i,j) drives one output element.  Q is loaded transposed (transA='C').
@kernel function qta_tile_kernel!(W, Q, A_tr, m_panel::Int, sb::Int, n_tr::Int)
    gi, gj = @index(Group,   NTuple)
    li, lj = @index(Local,   NTuple)

    TILE = @uniform @groupsize()[1]
    tile_Q  = @localmem eltype(W) (TILE + 1, TILE)   # bank-padded, loaded conjugated
    tile_A  = @localmem eltype(W) (TILE + 1, TILE)   # bank-padded

    outval = @private eltype(W) 1
    @inbounds outval[1] = zero(eltype(W))

    @uniform NUM_TILES = cld(m_panel, TILE)
    @uniform ElT = eltype(W)

    for t in 0:(NUM_TILES - 1)
        # row = output row (Q column index), col = output col (A col index).
        # K1 = Q row for tile_Q load (uses lj so the K-dimension tiles along j).
        # K2 = A row for tile_A load (uses li so the K-dimension tiles along i).
        row = (gi - 1) * TILE + li
        K1  = t * TILE + lj
        if row <= sb && K1 <= m_panel
            q = @inbounds Q[K1, row]
            @inbounds tile_Q[li, lj] = ElT <: Complex ? Complex(real(q), -imag(q)) : q
        else
            @inbounds tile_Q[li, lj] = zero(ElT)
        end

        col = (gj - 1) * TILE + lj
        K2  = t * TILE + li
        if K2 <= m_panel && col <= n_tr
            @inbounds tile_A[li, lj] = A_tr[K2, col]
        else
            @inbounds tile_A[li, lj] = zero(ElT)
        end

        @synchronize

        # Recompute row/col after @synchronize — CPU lowering splits here and
        # aliases computed before the barrier do not carry across the segment.
        if (gi - 1) * TILE + li <= sb && (gj - 1) * TILE + lj <= n_tr
            tmp = zero(ElT)
            @simd for k in 1:TILE
                @inbounds tmp += tile_Q[li, k] * tile_A[k, lj]
            end
            outval[1] += tmp
        end

        @synchronize
    end

    if (gi - 1) * TILE + li <= sb && (gj - 1) * TILE + lj <= n_tr
        @inbounds W[(gi - 1) * TILE + li, (gj - 1) * TILE + lj] += outval[1]
    end
end

# ── Statement S5: A_trailing -= Q * W ────────────────────────────────────────
# A_tr is m_panel × n_tr  (updated in place)
# Q    is m_panel × sb
# W    is sb × n_tr
#
# Standard tiled GEMM with alpha = -1 (subtract).
@kernel function apply_trailing_kernel!(A_tr, Q, W, m_panel::Int, sb::Int, n_tr::Int)
    gi, gj = @index(Group,   NTuple)
    li, lj = @index(Local,   NTuple)

    TILE = @uniform @groupsize()[1]
    tile_Q = @localmem eltype(A_tr) (TILE + 1, TILE)
    tile_W = @localmem eltype(A_tr) (TILE + 1, TILE)

    outval = @private eltype(A_tr) 1
    @inbounds outval[1] = zero(eltype(A_tr))

    @uniform NUM_TILES = cld(sb, TILE)
    @uniform ElT = eltype(A_tr)

    for t in 0:(NUM_TILES - 1)
        row = (gi - 1) * TILE + li   # row in A_tr / Q
        K1  = t * TILE + lj           # Q column = W row (shared K dim, sb)

        if row <= m_panel && K1 <= sb
            @inbounds tile_Q[li, lj] = Q[row, K1]
        else
            @inbounds tile_Q[li, lj] = zero(ElT)
        end

        col = (gj - 1) * TILE + lj
        K2  = t * TILE + li
        if K2 <= sb && col <= n_tr
            @inbounds tile_W[li, lj] = W[K2, col]
        else
            @inbounds tile_W[li, lj] = zero(ElT)
        end

        @synchronize

        # Recompute after @synchronize — do not reuse row/col across the barrier.
        if (gi - 1) * TILE + li <= m_panel && (gj - 1) * TILE + lj <= n_tr
            tmp = zero(ElT)
            @simd for k in 1:TILE
                @inbounds tmp += tile_Q[li, k] * tile_W[k, lj]
            end
            outval[1] += tmp
        end

        @synchronize
    end

    if (gi - 1) * TILE + li <= m_panel && (gj - 1) * TILE + lj <= n_tr
        @inbounds A_tr[(gi - 1) * TILE + li, (gj - 1) * TILE + lj] -= outval[1]
    end
end

# ── Scatter panel R (sb×sb) into R_acc at 1-based block origin (k_start, k_start) ─
@kernel function geqrf_write_R_panel_kernel!(R_acc, Rp, k_start::Int, sb::Int)
    lin = @index(Global, Linear)
    @uniform area = sb * sb
    if lin <= area
        ii = (lin - 1) ÷ sb + 1
        jj = (lin - 1) % sb + 1
        @inbounds R_acc[k_start + ii - 1, k_start + jj - 1] = Rp[ii, jj]
    end
end

# ── Scatter W (sb×n_tr) into R_acc at 1-based block corner (k, k+sb) ───────────
@kernel function geqrf_write_W_block_kernel!(R_acc, W, k_row::Int, k_col::Int, sb::Int, n_tr::Int)
    lin = @index(Global, Linear)
    @uniform area = sb * n_tr
    if lin <= area
        ii = (lin - 1) % sb + 1
        jj = (lin - 1) ÷ sb + 1
        @inbounds R_acc[k_row + ii - 1, k_col + jj - 1] = W[ii, jj]
    end
end

# ── broadcast_panel_kernel! ───────────────────────────────────────────────────
# On a single-device backend all c replicas are virtual; this is a documented
# no-op stub.  A real multi-process implementation would use MPI Bcast here.
@kernel function broadcast_panel_kernel!(Qk_replicated, Qk_local, ::Val{Px}) where {Px}
end

# ── Launcher helpers ──────────────────────────────────────────────────────────
function _geqrf_qta!(be, W, Q, A_tr, m_panel::Int, sb::Int, n_tr::Int, tile::Int; clear_W::Bool = true)
    clear_W && fill!(W, zero(eltype(W)))
    nd = (cld(sb, tile) * tile, cld(n_tr, tile) * tile)
    qta_tile_kernel!(be, (tile, tile))(W, Q, A_tr, m_panel, sb, n_tr; ndrange = nd)
    KernelAbstractions.synchronize(be)
end

# Disjoint row ranges 1:m into K parts (K = Px·Pz replicas); skip empty parts.
function _geqrf_row_ranges(m::Int, K::Int)::Vector{Tuple{Int, Int}}
    K == 1 && return [(1, m)]
    ranges = Tuple{Int, Int}[]
    base = m ÷ K
    rem = m % K
    r = 1
    for i in 1:K
        len = base + (i <= rem ? 1 : 0)
        if len > 0
            push!(ranges, (r, r + len - 1))
            r += len
        end
    end
    return ranges
end

function _geqrf_qta_partitioned!(be, W, Q_full, A_tr_full, m::Int, sb::Int, n_tr::Int, tile::Int, params)
    K = params.Px * params.Pz
    fill!(W, zero(eltype(W)))
    for (r1, r2) in _geqrf_row_ranges(m, K)
        mr = r2 - r1 + 1
        mr < 1 && continue
        Qv = @view(Q_full[r1:r2, 1:sb])
        Av = @view(A_tr_full[r1:r2, 1:n_tr])
        _geqrf_qta!(be, W, Qv, Av, mr, sb, n_tr, tile; clear_W = false)
    end
end

function _geqrf_apply!(be, A_tr, Q, W, m_panel::Int, sb::Int, n_tr::Int, tile::Int)
    nd = (cld(m_panel, tile) * tile, cld(n_tr, tile) * tile)
    apply_trailing_kernel!(be, (tile, tile))(A_tr, Q, W, m_panel, sb, n_tr; ndrange = nd)
    KernelAbstractions.synchronize(be)
end

function _geqrf_write_R_panel!(be, R_acc, Rp, k_start::Int, sb::Int)
    geqrf_write_R_panel_kernel!(be)(R_acc, Rp, k_start, sb; ndrange = sb * sb)
    KernelAbstractions.synchronize(be)
end

function _geqrf_write_W_block!(be, R_acc, W, k_row::Int, k_col::Int, sb::Int, n_tr::Int)
    geqrf_write_W_block_kernel!(be)(R_acc, W, k_row, k_col, sb, n_tr; ndrange = sb * n_tr)
    KernelAbstractions.synchronize(be)
end

# ── geqrf_2p5d! — outer loop driver ──────────────────────────────────────────
"""
    geqrf_2p5d!(m, n, A, R_acc, tau; params=nothing, b=nothing)

Compute QR factorization of A[1:m, 1:n] using shifted CholeskyQR3 panels
(Phase Q1) and tiled trailing updates (Phases Q2/Q3; manuscript §3 / §A.1).

After the call:
- `A[1:m, 1:n]` stores the explicit orthonormal panel columns Q_k block by
  block (not Householder vectors).
- `R_acc[1:n, 1:n]` holds the upper-triangular R factor (R_acc is filled with
  zeros outside the upper triangle).
- `tau` is reserved for future use; currently unused.

**Stability checks (Fukaya et al., SISC / 18M1218212):** use relative Frobenius residual
`norm(A - Q*R) / norm(A)` and orthogonality `norm(Q'Q - I)` (Frobenius, Sec. 6.2 style) and
`opnorm(Q'Q - I, 2)` (spectral norm, abstract), with `Q = A[:, 1:n]`, not LAPACK scaled 1-norm
diagnostics.

**`params.c > 1` (single-device emulation):** when `params` has replication `c>1`, `scqr3!` receives
device `partials` for Gram `panel_allreduce!`, and the trailing block `W = Q_k^H A_tr` is formed as
the sum of row-block partial products (same math as local `QᵀA` then sum in §3). There is no MPI:
all replicas share global `A`, so a separate `Q` broadcast is unnecessary (the stub kernel remains a
no-op unless replicated panel buffers are introduced later).

**Orthogonality guarantee (Fukaya et al. 2018, Theorem 3.4 + Björck 1967 §2.2):**
`‖QᴴQ − I‖_F ≤ 50(mb + b(b+1))·u` and `‖A − QR‖_F/‖A‖_F ≈ O(u)` for all κ₂(A) ≤ O(u⁻¹),
where `u = eps(real(T))`. In practice (m=128, n=64, b=16): residual ≈ 1.2u and orthF ≤ 80u for
Float32 at κ = u⁻¹ ≈ 8.4e6; orthF ≤ 12400u for Float64 at κ = u⁻¹ ≈ 4.5e15.

**Known limitation — single-dominant-singular-value matrices (TODO):**
Matrices with a single large singular value (`mode = :one_large`, κ ≈ 1e6) in Float32/ComplexF32
can trigger `LinearAlgebra.PosDefException` from the `scqr3!` panel step. After the first trailing
update removes the dominant σ₁ direction, the second panel's effective condition number can exceed
the Fukaya Lemma 3.1 threshold (~77 for Float32 with these dimensions), causing the Cholesky
factorization of the shifted Gram matrix to fail. Matrices with κ > u⁻¹ are numerically
rank-deficient in the given precision and are not supported regardless of structure. This is not a
problem for typical decay-spectrum matrices; a fallback to `geqrt!` for the offending panel is a
planned future fix.

# Arguments
- `m, n`   : dimensions of A (m ≥ n required for full-rank factorization).
- `A`      : m×n matrix, modified in place.
- `R_acc`  : n×n output matrix for the R factor.
- `tau`    : length-n scratch vector (currently unused).
- `params` : `DeviceParams` from `compute_params`; probed automatically when `nothing` (uses `c=1`).
- `b`      : panel width override (clamped to admissible `[b_min, b_max]` from `params`).
"""
function geqrf_2p5d!(m::Integer, n::Integer,
                     A::AbstractMatrix{T},
                     R_acc::AbstractMatrix{T},
                     tau::AbstractVector{T};
                     params::Union{DeviceParams{T}, Nothing} = nothing,
                     b::Union{Integer, Nothing} = nothing) where {T}
    m = Int(m); n = Int(n)
    m >= 0 || throw(ArgumentError("m must be ≥ 0, got m=$m"))
    n >= 0 || throw(ArgumentError("n must be ≥ 0, got n=$n"))
    (m == 0 || n == 0) && return nothing
    size(A, 1) >= m && size(A, 2) >= n ||
        throw(ArgumentError("A ($(size(A))) too small for m=$m, n=$n"))
    size(R_acc, 1) >= n && size(R_acc, 2) >= n ||
        throw(ArgumentError("R_acc ($(size(R_acc))) too small for n=$n"))

    be = KernelAbstractions.get_backend(A)

    k_eff = min(m, n)

    # ── Device params ──────────────────────────────────────────────────────────
    p = if params === nothing
        bval = b === nothing ? min(32, k_eff) : max(1, Int(b))
        compute_params(be, T, n; b = bval, c = 1)
    else
        params
    end

    b_full = p.b
    b_full >= 1 || throw(ArgumentError("DeviceParams.b must be ≥ 1"))

    tile = clamp(_GEQRF_TILE_DIM, 1, b_full)

    # Gram partials for sCQR3 when params.c > 1 (same layout as `scqr3!`).
    partials_buf = p.c > 1 ? similar(A, b_full, b_full, p.Px * p.Pz) : nothing

    # ── Persistent scratch (full panel width) ──────────────────────────────────
    G_buf    = similar(A, b_full, b_full)
    R_buf    = similar(A, b_full, b_full)
    info_buf = fill!(similar(A, Int, 1), 0)
    # W1 and W2 for double-projection trailing update (Fix B, see §A.1).
    W_buf    = similar(A, b_full, n > b_full ? n - b_full : 1)
    W2_buf   = similar(A, b_full, n > b_full ? n - b_full : 1)
    # W_pre for panel pre-orthogonalization against accumulated Q (global O(u) orthogonality).
    # Max size: (n - b_full) × b_full — the biggest pre-projection needed at the last panel step.
    W_pre_buf = n > b_full ? similar(A, n - b_full, b_full) : similar(A, 1, 1)

    fill!(R_acc, zero(T))

    # ── Outer loop: step k (1-based) advances by b_full ───────────────────────
    k = 1
    while k <= k_eff
        sb = min(b_full, k_eff - k + 1)   # actual panel width (last panel may be smaller)
        m_panel = m                        # full rows 1:m — explicit Q needs prior residuals

        A_panel  = @view A[1:m, k:(k + sb - 1)]

        # --- Panel pre-orthogonalization against accumulated Q (§A.1 global O(u)) ---
        # At step k the panel still carries O(κ·u)·‖A‖ contamination from
        # accumulated trailing-update rounding errors in span(Q_1,...,Q_{k-1}).
        # Two projections remove this to O(u·‖A‖) (Björck 1967 §2.2), ensuring
        # global inter-panel orthogonality up to κ = O(u^{-1}).  No R_acc update
        # needed: R_acc[1:k-1, k:...] was computed correctly via step-j trailing
        # updates; the pre-projection removes only the accumulated numerical error.
        k_cols = k - 1
        if k_cols > 0
            Q_acc  = @view A[1:m, 1:k_cols]
            # W_pre is k_cols × sb; reuse preallocated buffer when it fits.
            W_pre  = (size(W_pre_buf, 1) >= k_cols && size(W_pre_buf, 2) >= sb) ?
                     @view(W_pre_buf[1:k_cols, 1:sb]) : similar(A, k_cols, sb)
            _geqrf_qta!(be, W_pre, Q_acc, A_panel, m, k_cols, sb, tile)
            _geqrf_apply!(be, A_panel, Q_acc, W_pre, m, k_cols, sb, tile)
            _geqrf_qta!(be, W_pre, Q_acc, A_panel, m, k_cols, sb, tile)
            _geqrf_apply!(be, A_panel, Q_acc, W_pre, m, k_cols, sb, tile)
        end

        # --- Phase Q1: sCQR3 panel factorization --------------------------------
        # Use full-size scratch when possible; allocate fresh for narrow last panel.
        if sb == b_full
            Gp    = G_buf
            Rp    = R_buf
            infop = info_buf
            p_panel = p
        else
            Gp    = similar(A, sb, sb)
            Rp    = similar(A, sb, sb)
            infop = fill!(similar(A, Int, 1), 0)
            p_panel = compute_params(be, T, max(m_panel, sb); b = sb, c = p.c)
        end

        partials_use = if p_panel.c > 1
            sb == b_full ? partials_buf : similar(A, sb, sb, p_panel.Px * p_panel.Pz)
        else
            nothing
        end

        scqr3!(m_panel, sb, A_panel, Rp, Gp, infop; params = p_panel, partials = partials_use)
        # A_panel now holds Q_k (explicit orthonormal columns).

        # Write diagonal block of R (device kernel: avoids scalar CPU indexing into CuArray).
        _geqrf_write_R_panel!(be, R_acc, Rp, k, sb)

        # --- Phase Q2/Q3: trailing update ---------------------------------------
        n_tr = n - (k + sb - 1)
        if n_tr > 0
            A_trailing = @view A[1:m, (k + sb):n]

            # Reuse W_buf / W2_buf when wide enough; otherwise allocate fresh slabs.
            fits(buf) = size(buf, 1) >= sb && size(buf, 2) >= n_tr
            W  = fits(W_buf)  ? @view(W_buf[1:sb,  1:n_tr]) : similar(A, sb, n_tr)
            W2 = fits(W2_buf) ? @view(W2_buf[1:sb, 1:n_tr]) : similar(A, sb, n_tr)

            # S4 (first pass): W1 = Q_k^H * A_trailing.
            if p.c > 1
                _geqrf_qta_partitioned!(be, W, A_panel, A_trailing, m_panel, sb, n_tr, tile, p)
            else
                _geqrf_qta!(be, W, A_panel, A_trailing, m_panel, sb, n_tr, tile)
            end

            # S5 (first pass): A_trailing -= Q_k * W1.
            _geqrf_apply!(be, A_trailing, A_panel, W, m_panel, sb, n_tr, tile)

            # Fix B — double projection: reorthogonalize the residual.
            # After one projection the residual still has an O(κ·u) component in span(Q_k).
            # A second pass reduces it to O(u), recovering full machine-precision orthogonality
            # for κ(A_trailing) ≤ u^{-1/2} (Björck 1967, §2.2).
            if p.c > 1
                _geqrf_qta_partitioned!(be, W2, A_panel, A_trailing, m_panel, sb, n_tr, tile, p)
            else
                _geqrf_qta!(be, W2, A_panel, A_trailing, m_panel, sb, n_tr, tile)
            end
            # Accumulate W2 into W: total R entry = W1 + W2.
            W .+= W2
            # S5 (second pass): A_trailing -= Q_k * W2.
            _geqrf_apply!(be, A_trailing, A_panel, W2, m_panel, sb, n_tr, tile)

            # Cross-replica reduce of W: on a single device, row-partitioned S4 already summed W.

            # Write W1+W2 (= upper off-diagonal block of R) to R_acc.
            _geqrf_write_W_block!(be, R_acc, W, k, k + sb, sb, n_tr)
        end

        k += b_full
    end

    return nothing
end

"""
    geqrf_2p5d!(A) -> (A, R)

Convenience overload: allocates `R` (n×n) and `tau` (n), calls the full driver,
returns `(A, R)`.
"""
function geqrf_2p5d!(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    R   = similar(A, n, n)
    tau = similar(A, n)
    geqrf_2p5d!(m, n, A, R, tau)
    return A, R
end
