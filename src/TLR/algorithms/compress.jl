export compress!

# ----------- helpers -------------------

# accumulation precision used for cholqr and norms accumulation
@inline _compress_accum_type(::Type{Float16}) = Float32
@inline _compress_accum_type(::Type{Float32}) = Float64
@inline _compress_accum_type(::Type{Float64}) = Float64
@inline _compress_accum_type(::Type{ComplexF32}) = ComplexF64
@inline _compress_accum_type(::Type{ComplexF64}) = ComplexF64
@inline _compress_accum_type(::Type{T}) where {T} = T

@inline _adjoint_blas_char(::Type{<:Complex}) = 'C'
@inline _adjoint_blas_char(::Type) = 'T'

"""
    _batch_views(A, r=size(A, 2)) -> Vector{<:AbstractMatrix}

Per-batch-entry views into a `[m, maxrank, n]` factor array, trimmed to the
first `r` columns. Used to build `gemm_batched!` operand vectors; shared with
`uncompress.jl`.
"""
@inline _batch_views(A::AbstractArray{T,3}, r::Int=size(A, 2)) where {T} =
    [view(A, :, 1:r, k) for k in axes(A, 3)]

# ------- kernels --------

@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
    A::AbstractMatrix{T}, tile_m::Int, tile_n::Int) where {T}
    row, col, b = @index(Global, NTuple)
    p0 = (b - 1) * tile_m + 1
    q0 = (b - 1) * tile_n + 1
    @inbounds D[row, col, b] = A[p0+row-1, q0+col-1]
end

"""
    _copy_diagonal_from_dense!(A_tlr, A) -> A_tlr

Populate `A_tlr`'s dense diagonal storage (`A_tlr.D` and, if present,
`A_tlr.D_corner`) from the corresponding tiles of dense matrix `A`.
"""
function _copy_diagonal_from_dense!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T}) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    _copy_diag_kernel!(A_tlr.backend)(
        A_tlr.D, A, A_tlr.tile_m, A_tlr.tile_n;
        ndrange=(A_tlr.tile_m, A_tlr.tile_n, n_full_diag),
    )
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(view(A_tlr.D_corner, 1:tm, 1:tn, 1), _dense_tile_view(A, A_tlr, tile_k, tile_k))
    end
    return A_tlr
end

#= Squared magnitude accumulated in Float64: the error indicator is a difference
   of two nearly equal sums, so per-element squaring must not round in T. =#
@inline _abs2_f64(x::Real) = abs2(Float64(x))
@inline _abs2_f64(x::Complex) = abs2(Float64(real(x))) + abs2(Float64(imag(x)))

#= Per-slab Cholesky-QR shift from the Gram diagonal:
     rescue pass:  √eps · tr(G)/r      — rescues rank-deficient sketches
     refine pass:  eps · r · max(diag) — eps-level, so the second pass undoes the
                   column-norm deflation the rescue shift introduced
   A zero Gram (zero tile) gets shift 1: potrf stays PD and Y stays zero. =#
@kernel function _cholqr_shift_kernel!(
    G::AbstractArray{Thi,3}, trace_coeff::RT, maxdiag_coeff::RT) where {Thi,RT}
    b = @index(Group, Linear)
    j = @index(Local, Linear)
    shift = @localmem RT (1,)
    r = size(G, 1)
    if j == 1
        tr = zero(RT)
        mx = zero(RT)
        @inbounds for i in 1:r
            d = RT(real(G[i, i, b]))
            tr += d
            mx = max(mx, d)
        end
        reg = trace_coeff * (tr / r) + maxdiag_coeff * mx
        @inbounds shift[1] = ifelse(reg > zero(RT), reg, one(RT))
    end
    @synchronize
    @inbounds G[j, j, b] += Thi(shift[1])
end

# Per-tile squared Frobenius norm of the dense tiles, with hp accumulation
# TODO use warp intrinsics for the in-warp reduction
@kernel function _tile_norm_sq_kernel!(out::AbstractVector{Float64},
    A::AbstractMatrix{T}, p0s, q0s, tm::Int, tn::Int, ::Val{R}) where {T,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    partial = @localmem Float64 (R,)
    p0 = Int(@inbounds p0s[ob]) - 1
    q0 = Int(@inbounds q0s[ob]) - 1
    acc = 0.0
    col = j
    while col <= tn
        @inbounds for row in 1:tm
            acc += _abs2_f64(A[p0+row, q0+col])
        end
        col += R
    end
    @inbounds partial[j] = acc
    @synchronize
    if j == 1
        total = 0.0
        @inbounds for i in 1:R
            total += partial[i]
        end
        @inbounds out[ob] = total
    end
end

"""
    _fused_truncate_kernel!(U_out, V_out, Q, V, rk, err_sq, normA_sq,
                            eps_sq, rel, R_keep, ::Val{S}, ::Val{W})

Select the retained rank for each off-diagonal tile and gather the selected
columns from the `S`-wide sketch factors into the maxrank-wide output panels.
One workgroup handles one tile. Its local threads are split into subgroups of
width `W`; each subgroup cooperatively computes column energies for one sketch
column at a time, then lane 1 of each subgroup acts as that column's
representative for norm reduction.

The squared error of the truncated factorization decomposes, with orthonormal
`Q`, as

    ‖A - Q_k V_k'‖² = resid + Σ dropped ‖v_j‖²
    resid = ‖A‖² - Σ_j ‖v_j‖²

where `resid` is the randQB_EI range-capture error left by the sketch. The
kernel greedily drops the currently-smallest remaining `V`-column energy while
it fits in the remaining error budget, then drops extra smallest columns if
needed to satisfy `R_keep = min(maxrank, S)`. Surviving source columns are
compacted in original order.

The final gather runs threads along rows so consecutive threads touch
contiguous column-major memory.
"""
@kernel function _fused_truncate_kernel!(
    U_out::AbstractArray{T,3}, V_out::AbstractArray{T,3},
    Q::AbstractArray{T,3}, V::AbstractArray{T,3},
    rk::AbstractVector, err_sq::AbstractVector{Float64},
    normA_sq::AbstractVector{Float64},
    eps_sq::Float64, rel::Bool, R_keep::Int, ::Val{S}, ::Val{W}
) where {T,S,W}

    ob = @index(Group, Linear)
    tid = @index(Local, Linear)

    nthreads = @uniform @groupsize()[1]

    lane = ((tid - 1) % W) + 1
    subgroup = ((tid - 1) ÷ W) + 1
    nsubgroups = nthreads ÷ W

    norms = @localmem Float64 (S,)          # unsorted column norms
    partial = @localmem Float64 (S, W)      # per-lane partial sums

    dropped_flag = @localmem Int32 (S,)     # 1 if column is dropped, else 0
    kept_src = @localmem Int32 (S,)         # compacted surviving source columns
    k_buf = @localmem Int32 (1,)

    # Phase A — squared norm of each V column, one subgroup per column.
    for col in subgroup:nsubgroups:S
        acc = 0.0

        for row in lane:W:size(V, 1)
            @inbounds acc += _abs2_f64(V[row, col, ob])
        end

        @inbounds partial[col, lane] = acc
    end

    @synchronize

    # KA's CPU backend drops plain scalar locals across @synchronize (only @index
    # and @uniform values survive), so recompute the per-lane indices here.
    lane = ((tid - 1) % W) + 1
    subgroup = ((tid - 1) ÷ W) + 1
    nsubgroups = nthreads ÷ W

    for col in subgroup:nsubgroups:S
        if lane == 1
            total = 0.0

            @inbounds for i in 1:W
                total += partial[col, i]
            end

            @inbounds norms[col] = total
        end
    end

    @synchronize

    # Phase B greedy tail removal.
    if tid == 1
        total = 0.0

        @inbounds for i in 1:S
            total += norms[i]
            dropped_flag[i] = Int32(0)
        end

        nA_sq = @inbounds normA_sq[ob]

        resid = max(nA_sq - total, 0.0)

        # resid = ‖A‖² − ‖V‖² 
        epsT = Float64(eps(real(T)))
        resid_floor = Float64(size(Q, 1)) * epsT * nA_sq
        resid = ifelse(resid < resid_floor, 0.0, resid)

        # precision floor
        target = rel ? eps_sq * nA_sq : eps_sq
        budget = max(target, epsT * nA_sq) - resid

        k_val = S
        dropped = 0.0

        # find the smallest undropped column which fits the budget and drop it
        if budget >= 0.0
            while k_val > 0
                best_col = Int32(0)
                best_norm = Inf

                @inbounds for col in 1:S
                    if dropped_flag[col] == 0
                        nc = norms[col]

                        # tie-break by smaller column index
                        if nc < best_norm || (nc == best_norm && col < Int(best_col))
                            best_norm = nc
                            best_col = Int32(col)
                        end
                    end
                end

                if best_col != 0 && best_norm <= budget
                    @inbounds dropped_flag[Int(best_col)] = Int32(1)
                    budget -= best_norm
                    dropped += best_norm
                    k_val -= 1
                else
                    break
                end
            end
        end

        # pass twoL if tolerance keeps more than `maxrank` columns, 
        # drop the smallest columns until the stored rank fits.
        while k_val > R_keep
            best_col = Int32(0)
            best_norm = Inf

            @inbounds for col in 1:S
                if dropped_flag[col] == 0
                    nc = norms[col]

                    if nc < best_norm || (nc == best_norm && col < Int(best_col))
                        best_norm = nc
                        best_col = Int32(col)
                    end
                end
            end

            @inbounds dropped_flag[Int(best_col)] = Int32(1)
            dropped += best_norm
            k_val -= 1
        end

        # Phase C - compact kept columns
        pos = 1

        @inbounds for col in 1:S
            if dropped_flag[col] == 0
                kept_src[pos] = Int32(col)
                pos += 1
            end
        end

        @inbounds rk[ob] = eltype(rk)(k_val)
        @inbounds err_sq[ob] = resid + dropped
        @inbounds k_buf[1] = Int32(k_val)
    end

    @synchronize

    k = Int(@inbounds k_buf[1])
    Rmax = size(U_out, 2)
    # U_out has size(Q,1)=tm rows, V_out has size(V,1)=tn rows; these differ on
    # boundary tiles, so each gather uses its own row bound.
    bU = size(U_out, 1)
    bV = size(V_out, 1)

    @inbounds for jj in 1:Rmax
        src = jj <= k ? Int(kept_src[jj]) : 0
        for row in tid:nthreads:bU
            U_out[row, jj, ob] = jj <= k ? Q[row, src, ob] : zero(T)
        end
        for row in tid:nthreads:bV
            V_out[row, jj, ob] = jj <= k ? V[row, src, ob] : zero(T)
        end
    end
end

"""
    cholqr!(Y_hi, G_hi, backend; rescue::Bool) -> Y_hi

One shifted Cholesky-QR pass, orthogonalising the columns of each batch entry
of `Y_hi` in place (`Y_hi ← Y_hi · R⁻¹` via `G_hi = Y_hiᴴY_hi = RᴴR`).

`rescue=true` uses the `√eps·tr(G)/r` shift that survives rank-deficient
sketches; `rescue=false` uses an `eps`-level shift so the pass restores the
column norms the rescue shift deflated — the truncation step's error
indicator needs orthonormal columns to be trustworthy. See
[`_cholqr_shift_kernel!`](@ref) for the shift formulas.
"""
function cholqr!(Y_hi::AbstractArray{Thi,3}, G_hi, backend; rescue::Bool) where {Thi}
    count = size(Y_hi, 3)
    count == 0 && return Y_hi
    r = size(Y_hi, 2)
    RT = real(Thi)

    # G = Y'Y 
    gemm_batched!(_adjoint_blas_char(Thi), 'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
    
    trace_coeff   = rescue ? RT(sqrt(eps(RT))) : zero(RT)
    maxdiag_coeff = rescue ? zero(RT) : RT(r) * eps(RT)
    # G = G + shift·I
    _cholqr_shift_kernel!(backend)(G_hi, trace_coeff, maxdiag_coeff;
        ndrange=(r * count,), workgroupsize=r)
    
    # R = chol(G)
    potrf_batched!('U', G_hi)
    
    # Q = Y * inv(R)
    trsm_batched!('R', 'U', 'N', 'N', G_hi, Y_hi, one(Thi))
    
    return Y_hi
end

"""
    detect_rank!(U, V, Q_T, V_T, rk, err_sq, normA_sq, R_keep, eps_sq, rel, backend) -> U

Per-tile rank detection and truncation ([`_fused_truncate_kernel!`](@ref)):
gather the retained columns from the sketch scratch `Q_T`/`V_T` into the panels
`U`/`V`, writing the detected rank to `rk` and squared error to `err_sq`.
"""
function detect_rank!(U::AbstractArray{T,3}, V, Q_T, V_T, rk, err_sq, normA_sq,
    R_keep::Int, eps_sq::Float64, rel::Bool, backend) where {T}
    noff = size(U, 3)
    noff == 0 && return U
    S = size(Q_T, 2)
    W = unwrap(SUBGROUP_SIZE(typeof(backend)))
    nthreads = W * min(S, 8)
    kernel! = _fused_truncate_kernel!(backend, nthreads)
    kernel!(U, V, Q_T, V_T, rk, err_sq, normA_sq, eps_sq, rel, R_keep, Val{S}(), Val{W}();
        ndrange=(nthreads * noff,), workgroupsize=nthreads)
    U
end

# scratch for one tile category (interior / right / bottom) 
#`U`/`V` alias the A_tlr output panels (maxrank-wide); the sketch
struct CompressCategoryWorkspace{ObsT,PanelT,ScratchT,ScratchHiT,TileVT,RankVT,F64V,I32V}
    obs::ObsT              # global off-diagonal indices in this category
    S::Int                 # sketch width    = min(maxrank + oversample, tm, tn)
    R_keep::Int            # max stored rank  = min(maxrank, S)
    U::PanelT              # output left factors  (aliases A_tlr, maxrank-wide)
    V::PanelT              # output right factors (aliases A_tlr, maxrank-wide)
    Q_T::ScratchT          # orthonormal basis Q          (tm × S × n)
    V_T::ScratchT          # random Ω, then co-range Aᴴ·Q (tn × S × n)
    Q_tiles::TileVT        # per-tile GEMM operand views into Q_T / V_T
    V_tiles::TileVT
    Y_hi::ScratchHiT       # accumulation-precision Q copy for cholqr (tm × S × n)
    G_hi::ScratchHiT       # accumulation-precision Gram matrix        (S × S × n)
    ranks_local::RankVT    # per-tile detected rank
    err_sq_local::F64V     # per-tile squared error estimate
    normA_sq::F64V         # per-tile ‖A_tile‖²_F
    p0s::I32V              # per-tile dense-source row origin (1-based)
    q0s::I32V              # per-tile dense-source col origin (1-based)
end

# full scratch: one CompressCategoryWorkspace per tile category,
# reusable for matrix layout + oversampling
struct CompressWorkspace{IntWS, RightWS, BottomWS, StreamV}
    interior::IntWS
    right::RightWS
    bottom::BottomWS
    streams::StreamV # one execution stream for each category on gpu
end

# (obs, output U/V panels, tile dims) per off-diagonal category. `tail_m`/`tail_n`
# are the boundary tile dimensions (both == b in the interior).
@inline _category_specs(A_tlr, b, tail_m, tail_n) = (
    (; obs=A_tlr.obs_int, U=A_tlr.int_U, V=A_tlr.int_V, tm=b, tn=b),
    (; obs=A_tlr.obs_right, U=A_tlr.right_U, V=A_tlr.right_V, tm=b, tn=tail_n),
    (; obs=A_tlr.obs_bottom, U=A_tlr.bottom_U, V=A_tlr.bottom_V, tm=tail_m, tn=b),
)

# prepare category's scratch at sketch width S
function _alloc_category_workspace(A_tlr::TLRMatrix{<:Any,T}, spec, r::Int, p::Int, ::Type{Thi}) where {T,Thi}
    backend = get_backend(A_tlr)
    rank_type = eltype(A_tlr.ranks)
    n = length(spec.obs)
    S = max(min(r + p, spec.tm, spec.tn), 1)
    
    Q_T = zeros(backend, T, spec.tm, S, n)
    V_T = zeros(backend, T, spec.tn, S, n)
    Y_hi = zeros(backend, Thi, spec.tm, S, n)
    G_hi = zeros(backend, Thi, S, S, n)
    p0_host = Vector{Int32}(undef, n)
    q0_host = Vector{Int32}(undef, n)

    @inbounds for (k, ob) in enumerate(spec.obs)
        p0, q0 = tile_origin_coords(A_tlr, _offdiag_coords(A_tlr, ob)...)
        p0_host[k] = Int32(p0)
        q0_host[k] = Int32(q0)
    end

    p0s = copyto!(allocate(backend, Int32, n), p0_host)
    q0s = copyto!(allocate(backend, Int32, n), q0_host)
    
    return CompressCategoryWorkspace(
        spec.obs, S, min(r, S), spec.U, spec.V, Q_T, V_T,
        _batch_views(Q_T, S), _batch_views(V_T, S), Y_hi, G_hi,
        zeros(backend, rank_type, n),
        zeros(backend, Float64, n),
        zeros(backend, Float64, n),
        p0s,
        q0s,
    )
end

"""
    alloc_workspace(A_tlr; oversample=0) → CompressWorkspace

Pre-allocate `compress!` scratch for `A_tlr`, one bundle per off-diagonal tile
category at sketch width `S = min(maxrank + oversample, tile)`; `U`/`V` alias
`A_tlr` storage directly. Reuse across repeated calls on the same layout and
`oversample`:

    ws = alloc_workspace(A_tlr; oversample=8)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end
"""
function alloc_workspace(A_tlr::TLRMatrix{<:Any,T}; oversample::Int=0) where {T}
    oversample >= 0 || throw(ArgumentError("oversample must be >= 0"))
    Thi = _compress_accum_type(T)
    
    b = blocksize(A_tlr)[1]
    mt, nt = tilegrid_size(A_tlr)
    tail_m = max(A_tlr.m - (mt - 1) * b, 1)
    tail_n = max(A_tlr.n - (nt - 1) * b, 1)
    
    specs = _category_specs(A_tlr, b, tail_m, tail_n)
    cats = map(spec -> _alloc_category_workspace(A_tlr, spec, A_tlr.maxrank, oversample, Thi), specs)
    
    CompressWorkspace(cats..., create_streams(A_tlr.backend, 3))
end

# Randomized-sketch compression of one tile category (randQB_EI). Writes retained
# factors into cat.U/cat.V and per-tile rank/error into cat.ranks_local/err_sq_local.
# Degenerates to rank 0 for every tile when R_keep == 0 (maxrank == 0). The numbered
# steps below are the range → orthogonalise → co-range → truncate pipeline.
function _compress_category!(
    A_tlr::TLRMatrix,
    A::AbstractMatrix{T},
    cat::CompressCategoryWorkspace,
    eps_sq::Float64,
    rel::Bool,
) where {T}

    isempty(cat.obs) && return cat
    backend = get_backend(A_tlr)
    n = length(cat.obs)
    S = cat.S
    R_keep = cat.R_keep
    tm = size(cat.Q_T, 1)
    tn = size(cat.V_T, 1)

    # Step 0: per-tile ‖A‖²_F — reference term for the error indicator (step 4).
    Rnorm = max(S, 1)
    _tile_norm_sq_kernel!(backend, Rnorm)(
        cat.normA_sq, A, cat.p0s, cat.q0s, tm, tn, Val{Rnorm}();
        ndrange=(Rnorm * n,), workgroupsize=Rnorm)

    if R_keep == 0   # maxrank == 0: every tile degenerates to rank 0
        fill!(cat.ranks_local, zero(eltype(cat.ranks_local)))
        cat.err_sq_local .= cat.normA_sq
        return cat
    end

    A_tiles = map(ob -> _offdiag_tile_view(A, A_tlr, ob), cat.obs)

    # Step 1: range sampling  Q = A·Ω  (Ω drawn into V_T; step 3 overwrites it)
    Random.randn!(cat.V_T)
    gemm_batched!('N', 'N', one(T), A_tiles, cat.V_tiles, zero(T), cat.Q_tiles)

    # Step 2: orthogonalise Q (in higher precision)
    cat.Y_hi .= cat.Q_T
    cholqr!(cat.Y_hi, cat.G_hi, backend; rescue=true)
    cholqr!(cat.Y_hi, cat.G_hi, backend; rescue=false)
    cat.Q_T .= cat.Y_hi

    # Step 3: co-range  V = Aᴴ·Q  (overwrites the Ω we no longer need)
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), A_tiles, cat.Q_tiles, zero(T), cat.V_tiles)

    # Step 4: rank detection + truncation (fused SMEM kernel, EI-corrected budget)
    detect_rank!(cat.U, cat.V, cat.Q_T, cat.V_T, cat.ranks_local, cat.err_sq_local,
        cat.normA_sq, R_keep, eps_sq, rel, backend)
    cat
end

# ─── Storage helpers ──────────────────────────────────────────────────────────

# Scatter one category's local ranks / squared errors back into the global
# A_tlr.ranks / A_tlr.resid (converting squared error to a Frobenius residual).
function _store_category_results!(A_tlr::TLRMatrix, cat::CompressCategoryWorkspace)
    isempty(cat.obs) && return
    rk_host  = cat.ranks_local isa Vector ? cat.ranks_local : Array(cat.ranks_local)
    err_host = cat.err_sq_local isa Vector ? cat.err_sq_local : Array(cat.err_sq_local)
    @inbounds for (k, ob) in enumerate(cat.obs)
        A_tlr.ranks[ob] = rk_host[k]
        A_tlr.resid[ob] = sqrt(max(err_host[k], 0.0))
    end
end

# ─── Orchestration ────────────────────────────────────────────────────────────

# Compress all three tile categories and scatter their results into A_tlr. On GPU
# each category runs on its own stream (overlap) and is synced before storing; on
# CPU they run sequentially.
function _compress_all_categories!(
    A_tlr::TLRMatrix{<:Any,T},
    A::AbstractMatrix{T},
    ws::CompressWorkspace,
    eps_sq::Float64,
    rel::Bool,
) where {T}
    cats = (ws.interior, ws.right, ws.bottom)
    backend = get_backend(A_tlr)
    if backend isa KernelAbstractions.CPU
        for cat in cats
            _compress_category!(A_tlr, A, cat, eps_sq, rel)
        end
    else
        for (cat, stream) in zip(cats, ws.streams)
            with_stream(backend, stream) do
                _compress_category!(A_tlr, A, cat, eps_sq, rel)
            end
        end
        for stream in ws.streams
            sync_stream(backend, stream)
        end
    end
    for cat in cats
        _store_category_results!(A_tlr, cat)
    end
    A_tlr
end

"""
    compress!(A_tlr, A [, ws]; tol=0.0, rel=false)

Compress dense matrix `A` into the TLR container `A_tlr` in-place.

Per-tile effective ranks are detected via greedy V-column-norm thresholding
against an error-indicator-corrected budget and stored in `ranks(A_tlr)`; the
estimated per-tile Frobenius error lands in `residuals(A_tlr)`.  The indicator
(`‖A_tile‖²_F − ‖V‖²_F`, à la randQB_EI) accounts for the range-capture error
of the sketch, so a tile whose spectrum does not fit within `maxrank` keeps
full rank and reports a residual above `tol` instead of silently claiming
convergence — check `residuals` to route such tiles to dense storage or a
higher-rank second pass.

Factor arrays are updated in-place; call `alloc_workspace` once to amortise
device allocations across repeated calls:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end

## Keywords

`tol` — per-tile Frobenius error budget (default `0.0`). The budget is floored
at `√eps(T)·‖A_tile‖_F`: a precision-`T` sketch cannot resolve relative error
below `√eps(T)` (≈ `3.4e-4` for `Float32`), so a `tol` finer than that recovers
the true rank at the `√eps(T)` accuracy floor rather than retaining spurious
noise columns to chase an unreachable target.

`rel` — when `true`, the budget for each tile is `tol * ‖A_tile‖_F` instead of
the absolute `tol`.

`oversample` — extra sketch columns `p` beyond `maxrank` for better range
capture; the sketch width is `S = min(maxrank + p, tile)` and the stored rank is
capped at `maxrank`. Must match the `oversample` passed to `alloc_workspace`.

The sketch basis is orthogonalised with two shifted Cholesky-QR passes in
higher precision.
"""
compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T}; oversample::Int=0, kwargs...) where {T} =
    compress!(A_tlr, A, alloc_workspace(A_tlr; oversample); kwargs...)

function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::CompressWorkspace;
    tol::Real=0.0, rel::Bool=false) where {T}

    size(A) == (A_tlr.m, A_tlr.n) ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n && A_tlr.tile_m == A_tlr.tile_n ||
        throw(ArgumentError("compress! currently requires square matrices with square tiles"))
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))

    _copy_diagonal_from_dense!(A_tlr, A)

    eps_sq = Float64(tol)^2
    _compress_all_categories!(A_tlr, A, ws, eps_sq, rel)

    A_tlr
end
