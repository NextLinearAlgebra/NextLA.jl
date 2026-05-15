export geqrf_2p5d!

# Tile for trailing-update GEMMs. GPU uses 32 (1024 threads/block); CPU uses BLAS so this
# only affects the (unused) kernel fallback path — kept at 16 for safety.
_geqrf_tile(::KernelAbstractions.CPU, b_full::Int) = clamp(16, 1, b_full)
_geqrf_tile(::Any,                    b_full::Int) = clamp(256, 1, b_full)

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

# CPU path: route through LinearAlgebra.mul! (BLAS DGEMM/ZGEMM) instead of the tile kernel.
# W = Qᴴ * A_tr  (sb × n_tr); accumulates when clear_W=false.
function _geqrf_qta!(::KernelAbstractions.CPU, W, Q, A_tr, m_panel::Int, sb::Int, n_tr::Int, ::Int; clear_W::Bool = true)
    Qv = @view Q[1:m_panel, 1:sb]
    Av = @view A_tr[1:m_panel, 1:n_tr]
    if clear_W
        mul!(W, adjoint(Qv), Av)
    else
        mul!(W, adjoint(Qv), Av, one(eltype(W)), one(eltype(W)))
    end
end

# GPU path (CUDA/ROCM/etc.): use BLAS/cuBLAS mul! for BlasFloat; KA kernel otherwise.
function _geqrf_qta!(be, W, Q, A_tr, m_panel::Int, sb::Int, n_tr::Int, tile::Int; clear_W::Bool = true)
    T = eltype(W)
    if T <: LinearAlgebra.BlasFloat
        Wv = view(W,    1:sb,      1:n_tr)
        Qv = view(Q,    1:m_panel, 1:sb)
        Av = view(A_tr, 1:m_panel, 1:n_tr)
        clear_W ? mul!(Wv, Qv', Av) : mul!(Wv, Qv', Av, one(T), one(T))
        return
    end
    clear_W && fill!(W, zero(T))
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

# CPU path: A_tr[1:m_panel, 1:n_tr] -= Q[1:m_panel, 1:sb] * W  via BLAS DGEMM.
function _geqrf_apply!(::KernelAbstractions.CPU, A_tr, Q, W, m_panel::Int, sb::Int, n_tr::Int, ::Int)
    mul!(@view(A_tr[1:m_panel, 1:n_tr]), @view(Q[1:m_panel, 1:sb]), W,
         -one(eltype(A_tr)), one(eltype(A_tr)))
end

# GPU path: cuBLAS mul! for BlasFloat; KA kernel otherwise.
function _geqrf_apply!(be, A_tr, Q, W, m_panel::Int, sb::Int, n_tr::Int, tile::Int)
    T = eltype(A_tr)
    if T <: LinearAlgebra.BlasFloat
        mul!(view(A_tr, 1:m_panel, 1:n_tr),
             view(Q,    1:m_panel, 1:sb),
             view(W,    1:sb,      1:n_tr),
             -one(T), one(T))
        return
    end
    nd = (cld(m_panel, tile) * tile, cld(n_tr, tile) * tile)
    apply_trailing_kernel!(be, (tile, tile))(A_tr, Q, W, m_panel, sb, n_tr; ndrange = nd)
    KernelAbstractions.synchronize(be)
end

function _geqrf_write_R_panel!(be, R_acc, Rp, k_start::Int, sb::Int)
    geqrf_write_R_panel_kernel!(be)(R_acc, Rp, k_start, sb; ndrange = sb * sb)
end

function _geqrf_write_W_block!(be, R_acc, W, k_row::Int, k_col::Int, sb::Int, n_tr::Int)
    geqrf_write_W_block_kernel!(be)(R_acc, W, k_row, k_col, sb, n_tr; ndrange = sb * n_tr)
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
- `params` : `DeviceParams` from `compute_params`; probed automatically when `nothing` (uses `c` from the `P·M / N²` budget with `N = max(m, n)`, matching `verify_budget` / `scqr3!`).
- `b`      : panel width override (clamped to admissible `[b_min, b_max]` from `params`).
- `ortho`  : orthogonality strategy. `:fast` (default) — single trailing projection per panel,
  giving per-panel O(u) orthogonality (Fukaya §3.4) and global ortho ≈ O(κ·u); fastest path,
  matches the paper's §3.5 per-step structure. `:safe` — adds Fix B (Björck 1967 §2.2 double
  trailing projection) and Fix C (double Gram-Schmidt pre-projection of the panel against
  accumulated `Q`); enforces global O(u) orthogonality up to κ ≈ O(u⁻¹) at ≈ 5× the flop cost.
"""
function geqrf_2p5d!(m::Integer, n::Integer,
                     A::AbstractMatrix{T},
                     R_acc::AbstractMatrix{T},
                     tau::AbstractVector{T};
                     params::Union{DeviceParams{T}, Nothing} = nothing,
                     b::Union{Integer, Nothing} = nothing,
                     ortho::Symbol = :fast,
                     passes::Int = 3,
                     mixed_precision::Bool = false) where {T}
    ortho in (:fast, :safe) ||
        throw(ArgumentError("ortho must be :fast or :safe, got :$ortho"))
    1 <= passes <= 3 || throw(ArgumentError("passes must be 1..3, got $passes"))
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
    # `compute_params` budget uses c ≈ ⌊P·M / N²⌋ (see `verify_budget`); N must scale like the
    # problem order. Using only `n` inflates c by (max(m,n)/n)² for tall-skinny matrices — same as
    # `scqr3!`, use `max(m, n)` (not `min(m, n)` / `k_eff`).
    N_budget = max(m, n)

    # ── Device params ──────────────────────────────────────────────────────────
    p = if params === nothing
        bval = b === nothing ? min(32, k_eff) : max(1, Int(b))
        compute_params(be, T, N_budget; b = bval, c = nothing)
    else
        params
    end

    b_full = p.b
    b_full >= 1 || throw(ArgumentError("DeviceParams.b must be ≥ 1"))

    tile = _geqrf_tile(be, b_full)

    # `c_eff` honours `NEXTLA_FORCE_C1`: short-circuits the single-device fanout /
    # partitioned QTA to a no-op when set, while keeping `p.c` in `params` so the
    # X-partition cube sizing (b_min, TILE_DIM) stays paper-aligned.
    c_eff = effective_c(p)

    # Gram partials for sCQR3 when c_eff > 1 (same layout as `scqr3!`).
    partials_buf = c_eff > 1 ? similar(A, b_full, b_full, p.Px * p.Pz) : nothing

    # ── Persistent scratch (full panel width) ──────────────────────────────────
    G_buf    = similar(A, b_full, b_full)
    R_buf    = similar(A, b_full, b_full)
    info_buf = fill!(similar(A, Int, 1), 0)
    # Preallocated scqr3 scratch (avoids per-panel `similar(...)` calls and is a
    # prerequisite for CUDA graph capture, which forbids allocations inside the
    # captured region).
    racc_buf = similar(A, b_full, b_full)
    rwrk_buf = similar(A, b_full, b_full)
    RT       = real(T)
    Ntr_max  = nextpow(2, b_full)
    use_trace_scratch = Ntr_max <= 1024   # _WORKGROUP_REDUCE_MAX in xpartition.jl
    trace_src_buf = use_trace_scratch ? similar(A, RT, (Ntr_max,)) : similar(A, RT, (0,))
    trace_out_buf = use_trace_scratch ? similar(A, RT, (1,))       : similar(A, RT, (0,))
    # W1 for the trailing update (always needed); W2/W_pre only for :safe mode.
    # At b=256, n=16384 each :safe slab is ~30 MB FP64 — non-trivial HBM and L2
    # footprint we'd waste on the :fast path where they are never read.
    W_buf     = similar(A, b_full, n > b_full ? n - b_full : 1)
    W2_buf    = ortho === :safe ?
                similar(A, b_full, n > b_full ? n - b_full : 1) :
                similar(A, 0, 0)
    # W_pre for panel pre-orthogonalization against accumulated Q (global O(u)
    # orthogonality). Max size: (n - b_full) × b_full — the biggest pre-projection
    # needed at the last panel step. Only :safe path uses it.
    W_pre_buf = (ortho === :safe && n > b_full) ?
                similar(A, n - b_full, b_full) :
                similar(A, 0, 0)

    fill!(R_acc, zero(T))

    # CUDA graph capture (NEXTLA_USE_GRAPH=1) folds the ~26 per-panel kernel
    # launches into one replayable graph. When `update()` cannot patch the cached
    # executable (e.g. last panel has different sb), a fresh instantiation
    # happens; `capture_panel!` handles this transparently.
    use_graph = get(ENV, "NEXTLA_USE_GRAPH", "0") == "1" &&
                _graph_capture_supported(be)
    panel_exec_ref = Ref{Any}()   # cuGraphExec on CUDA; unused otherwise

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
        if ortho === :safe && k_cols > 0
            Q_acc  = @view A[1:m, 1:k_cols]
            # W_pre is k_cols × sb; reuse preallocated buffer when it fits.
            W_pre  = (size(W_pre_buf, 1) >= k_cols && size(W_pre_buf, 2) >= sb) ?
                     @view(W_pre_buf[1:k_cols, 1:sb]) : similar(A, k_cols, sb)
            # Double Gram-Schmidt (Björck 1967 §2.2): two unconditional projections
            # give global O(u) inter-panel orthogonality for all κ ≤ O(u^{-1}).
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

        partials_use = if effective_c(p_panel) > 1
            sb == b_full ? partials_buf : similar(A, sb, sb, p_panel.Px * p_panel.Pz)
        else
            nothing
        end

        # Choose capture-eligibility: only :fast / c_eff=1 / full-panel iterations are
        # captured. :safe has a device→host check (`nw2 > sqrt(u)·‖W1‖`) that breaks
        # capture, and the last partial panel has a different topology than the full
        # panels (graph `update` would fail and re-instantiate every time).
        capture_eligible = use_graph && ortho === :fast && c_eff == 1 && sb == b_full

        n_tr = n - (k + sb - 1)

        # Panel + trailing update. Inlined sequence rather than a closure to avoid
        # the per-iteration Julia anonymous-function dispatch overhead (each
        # iteration would otherwise instantiate a new closure with boxed captured
        # variables — measured 4× regression at N=8000 FP64).
        if capture_eligible
            # Capture-only path: the only place we pay the closure cost is when
            # graph capture is actually requested. capture_panel! falls back to
            # direct execution if capture fails (lifetime issues, etc.).
            capture_panel!(be, panel_exec_ref, () -> begin
                scqr3!(m_panel, sb, A_panel, Rp, Gp, infop; params = p_panel, partials = partials_use,
                       passes = passes,
                       racc = racc_buf, rwrk = rwrk_buf,
                       trace_src = use_trace_scratch ? trace_src_buf : nothing,
                       trace_out = use_trace_scratch ? trace_out_buf : nothing)
                _geqrf_write_R_panel!(be, R_acc, Rp, k, sb)
                if n_tr > 0
                    A_tr_ = @view A[1:m, (k + sb):n]
                    W_ = (size(W_buf, 1) >= sb && size(W_buf, 2) >= n_tr) ?
                         @view(W_buf[1:sb, 1:n_tr]) : similar(A, sb, n_tr)
                    _geqrf_qta!(be, W_, A_panel, A_tr_, m_panel, sb, n_tr, tile)
                    _geqrf_apply!(be, A_tr_, A_panel, W_, m_panel, sb, n_tr, tile)
                    _geqrf_write_W_block!(be, R_acc, W_, k, k + sb, sb, n_tr)
                end
                nothing
            end)
        else
            # Non-capture path: do NOT pass preallocated scqr3 scratch. Threading
            # SubArray views (view(racc_buf, 1:b, 1:b)) through scqr3 measured ~4×
            # slower at N=8000 FP64 — the inner `mul!(Rwrk, UpperTriangular(Gv),
            # Racc)` falls off the StridedCuMatrix fast path when all three args
            # are SubArrays. The capture path above explicitly passes the buffers
            # because graph capture forbids allocations inside the recorded body.
            scqr3!(m_panel, sb, A_panel, Rp, Gp, infop; params = p_panel, partials = partials_use, passes = passes)
            _geqrf_write_R_panel!(be, R_acc, Rp, k, sb)
            # :fast first-pass trailing update (the only one needed in :fast mode).
            if ortho === :fast && n_tr > 0
                A_trailing = @view A[1:m, (k + sb):n]
                fits_fast(buf) = size(buf, 1) >= sb && size(buf, 2) >= n_tr
                W = fits_fast(W_buf) ? @view(W_buf[1:sb, 1:n_tr]) : similar(A, sb, n_tr)
                if mixed_precision && T == Float64
                    # Mixed-precision trailing: cast panel + trailing to FP32, do
                    # both GEMMs in FP32 (TF32 Tensor Cores on Hopper if
                    # NEXTLA_TF32=1), accumulate back to FP64. The R-block (W) is
                    # cast back into the FP64 W view. Numerically this changes
                    # only the trailing-update precision; the FP64 sCQR3 panel
                    # still produces FP64 Q with the full Fukaya orthogonality
                    # guarantee on each panel.
                    A_panel32   = Float32.(A_panel)
                    A_tr32      = Float32.(A_trailing)
                    W32         = similar(A_panel32, sb, n_tr)
                    mul!(W32, A_panel32', A_tr32)
                    mul!(A_tr32, A_panel32, W32, -1.0f0, 1.0f0)
                    A_trailing .= Float64.(A_tr32)
                    W .= Float64.(W32)
                elseif c_eff > 1
                    _geqrf_qta_partitioned!(be, W, A_panel, A_trailing, m_panel, sb, n_tr, tile, p)
                    _geqrf_apply!(be, A_trailing, A_panel, W, m_panel, sb, n_tr, tile)
                else
                    _geqrf_qta!(be, W, A_panel, A_trailing, m_panel, sb, n_tr, tile)
                    _geqrf_apply!(be, A_trailing, A_panel, W, m_panel, sb, n_tr, tile)
                end
                _geqrf_write_W_block!(be, R_acc, W, k, k + sb, sb, n_tr)
            end
        end

        # --- :safe extra projection (Fix B) — outside the captured region ----
        if ortho === :safe && n_tr > 0
            A_trailing = @view A[1:m, (k + sb):n]
            fits(buf) = size(buf, 1) >= sb && size(buf, 2) >= n_tr
            W = fits(W_buf) ? @view(W_buf[1:sb, 1:n_tr]) : similar(A, sb, n_tr)

            # S4 (first pass): W1 = Q_k^H * A_trailing.
            if c_eff > 1
                _geqrf_qta_partitioned!(be, W, A_panel, A_trailing, m_panel, sb, n_tr, tile, p)
            else
                _geqrf_qta!(be, W, A_panel, A_trailing, m_panel, sb, n_tr, tile)
            end

            # S5 (first pass): A_trailing -= Q_k * W1.
            _geqrf_apply!(be, A_trailing, A_panel, W, m_panel, sb, n_tr, tile)

            # Fix B — second projection (Björck 1967 §2.2). Branch always taken here
            # because the enclosing block is gated on `ortho === :safe`.
            W2 = fits(W2_buf) ? @view(W2_buf[1:sb, 1:n_tr]) : similar(A, sb, n_tr)
            nw1 = fits(W_buf) ? norm(copy(W)) : real(T)(norm(W))
            # S4 (second pass): W2 = Q_k^H * A_trailing (residual after first correction).
            if c_eff > 1
                _geqrf_qta_partitioned!(be, W2, A_panel, A_trailing, m_panel, sb, n_tr, tile, p)
            else
                _geqrf_qta!(be, W2, A_panel, A_trailing, m_panel, sb, n_tr, tile)
            end
            # Accumulate W2 into W: R entry = W1 + W2 regardless of whether we apply S5.
            W .+= W2
            # S5 (second pass): skip when residual is below √u·‖W1‖ — one projection sufficed.
            nw2 = fits(W2_buf) ? norm(copy(W2)) : real(T)(norm(W2))
            if nw2 > sqrt(eps(real(T))) * nw1
                _geqrf_apply!(be, A_trailing, A_panel, W2, m_panel, sb, n_tr, tile)
            end

            # Cross-replica reduce of W: on a single device, row-partitioned S4 already summed W.

            # Write W1 (or W1+W2 in :safe) to the upper off-diagonal block of R.
            _geqrf_write_W_block!(be, R_acc, W, k, k + sb, sb, n_tr)
        end

        k += b_full
    end

    return nothing
end

"""
    geqrf_2p5d!(A; params=nothing, b=nothing) -> (A, R)

Convenience overload: allocates `R` (n×n) and `tau` (n), calls the full driver,
returns `(A, R)`.

`params` and `b` are forwarded to [`geqrf_2p5d!(m, n, A, R, tau; ...)`](@ref) — same meaning as there.
"""
function geqrf_2p5d!(A::AbstractMatrix{T};
                     params::Union{DeviceParams{T}, Nothing} = nothing,
                     b::Union{Integer, Nothing} = nothing,
                     ortho::Symbol = :fast) where {T}
    m, n = size(A)
    R   = similar(A, n, n)
    tau = similar(A, n)
    geqrf_2p5d!(m, n, A, R, tau; params = params, b = b, ortho = ortho)
    return A, R
end
