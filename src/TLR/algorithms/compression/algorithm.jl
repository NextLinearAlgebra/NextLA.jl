"""
    cholqr2!(Q, Y_hi, G_hi, R_work, R_tiles, Q_tiles, multipliers) -> Q

Two FKNYY-shifted Cholesky-QR passes. Both use
`11(mS + S(S+1)) * (eps(Tgram)/2) * max(diag(YᴴY))`; no unshifted
factorisation is attempted.
"""
function cholqr2!(Q::AbstractArray{T,3}, Y_hi::AbstractArray{Thi,3}, G_hi,
    R_work, R_tiles, Q_tiles, multipliers) where {T,Thi}
    count = size(Q, 3)
    count == 0 && return Q

    backend = get_backend(Y_hi)
    m, r = size(Y_hi, 1), size(Y_hi, 2)
    RT = real(Thi)
    coeff = _cholqr_shift_coeff(Thi, m, r)
    nt_shift = _reduce_threads(r)
    fill!(multipliers, one(RT))

    for pass in 1:2
        Y_hi .= Q
        gemm_batched!(_adjoint_blas_char(Thi), 'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
        _cholqr_shift_kernel!(backend, nt_shift)(G_hi, coeff, multipliers, Val{nt_shift}();
            ndrange=(nt_shift * count,), workgroupsize=nt_shift)
        result = potrf_batched!('U', G_hi)

        # POTRF failure is extraordinarily rare with the prescribed shift.
        # Pass one nevertheless escalates all failed slabs as specified. A
        # fresh Gram restores any slabs overwritten by a failed factorisation.
        if pass == 1 && result isa Tuple
            status = Array(result[2])
            attempts = 0
            while any(!iszero, status) && attempts < 40
                host_mult = Array(multipliers)
                @inbounds for b in eachindex(status)
                    status[b] == 0 || (host_mult[b] *= RT(2))
                end
                copyto!(multipliers, host_mult)
                gemm_batched!(_adjoint_blas_char(Thi), 'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
                _cholqr_shift_kernel!(backend, nt_shift)(G_hi, coeff, multipliers, Val{nt_shift}();
                    ndrange=(nt_shift * count,), workgroupsize=nt_shift)
                result = potrf_batched!('U', G_hi)
                status = Array(result[2])
                attempts += 1
            end
            any(!iszero, status) && throw(LinearAlgebra.PosDefException(findfirst(!iszero, status)))
        end

        R_work .= G_hi
        trsm_batched!('R', 'U', 'N', 'N', R_tiles, Q_tiles, one(T))
        fill!(multipliers, one(RT))
    end

    return Q
end

"""
    prune_ranks!(U, V, rk, norm_err_sq, S, R_keep, eps_sq, rel) -> U

Per-tile rank detection and truncation ([`_prune_rank_kernel`](@ref)):
compact the retained columns in-place in the output panels `U`/`V`, writing the
detected rank to `rk`. `norm_err_sq` supplies the tile norm on entry and is
overwritten by the achieved squared error.
"""
function prune_ranks!(U::AbstractArray{T,3}, V, rk, norm_err_sq,
    S::Int, R_keep::Int, eps_sq::Float64, rel::Bool, delta_floor::Float64) where {T}
    noff = size(U, 3)
    noff == 0 && return U
    backend = get_backend(U)
    W = unwrap(SUBGROUP_SIZE(typeof(backend)))
    nthreads = W * min(S, 8)
    kernel! = _prune_rank_kernel(backend, nthreads)
    kernel!(U, V, rk, norm_err_sq, eps_sq, rel, R_keep, delta_floor,
        Val{S}(), Val{W}();
        ndrange=(nthreads * noff,), workgroupsize=nthreads)
    U
end

abstract type TileSource{T} end

@inline _ntiles(src::TileSource) = length(src.tiles)

# Off-diagonal tiles carved from a dense matrix (today's `compress!` path).
struct DenseTiles{T,AT<:AbstractMatrix{T},TV<:AbstractVector,CV} <: TileSource{T}
    A::AT           # dense source matrix
    tiles::TV       # per-tile views into A (gemm operands)
    p0s::CV         # per-tile row origin (device Int32) — for the norm kernel
    q0s::CV         # per-tile col origin (device Int32)
    tm::Int         # tile rows
    tn::Int         # tile cols
end

# out[k] = ‖A_tile_k‖²_F.
function _tile_norms_sq!(out, src::DenseTiles)
    n = _ntiles(src)
    n == 0 && return out
    backend = get_backend(src.A)
    W, _, NT = _norm_launch(backend, src.tn)
    _tile_norm_sq_kernel!(backend, NT)(out, src.A, src.p0s, src.q0s, src.tm, src.tn,
        Val{W}(), Val{NT}(); ndrange=(NT * n,), workgroupsize=NT)
    return out
end

# A packed [tm, tn, ntiles] batch of dense tiles (e.g. gemm intermediates).
struct PackedTiles{T,PT<:AbstractArray{T,3},TV<:AbstractVector} <: TileSource{T}
    data::PT        # [tm, tn, ntiles]
    tiles::TV       # per-slab views (gemm operands)
end
PackedTiles(data::AbstractArray{<:Any,3}) =
    PackedTiles(data, [view(data,:,:,k) for k in axes(data, 3)])

function _tile_norms_sq!(out, src::PackedTiles)
    n = _ntiles(src)
    n == 0 && return out
    backend = get_backend(src.data)
    W, _, NT = _norm_launch(backend, size(src.data, 2))
    _tile_norm_sq_kernel!(backend, NT)(out, src.data, Val{W}(), Val{NT}();
        ndrange=(NT * n,), workgroupsize=NT)
    return out
end

# Q = A·Ω and V = Aᴴ·Q, batched over tiles
@inline _sketch!(Q_tiles, src::TileSource{T}, Ω_tiles) where {T} =
    gemm_batched!('N', 'N', one(T), src.tiles, Ω_tiles, zero(T), Q_tiles)
@inline _cosketch!(V_tiles, src::TileSource{T}, Q_tiles) where {T} =
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), src.tiles, Q_tiles, zero(T), V_tiles)

"""
    compress_tiles!(src, cat; eps_sq, rel) -> cat

Randomized-sketch compression (randQB_EI) of the tile batch described by `src`
into the workspace `cat`: writes the retained factors into `cat.U`/`cat.V` and the
per-tile rank / squared error into `cat.ranks_local` / `cat.norm_err_sq`. Input-
agnostic — see [`TileSource`](@ref). Degenerates to rank 0 when `cat.R_keep == 0`.
"""
function compress_tiles!(src::TileSource{T}, cat::CompressCategoryWorkspace; eps_sq::Float64, rel::Bool) where {T}
    _ntiles(src) == 0 && return cat

    if cat.R_keep == 0   # maxrank == 0: every tile degenerates to rank 0
        _tile_norms_sq!(cat.norm_err_sq, src)
        fill!(cat.ranks_local, zero(eltype(cat.ranks_local)))
        return cat
    end

    # Step 1: range sampling  Q = A·Ω  (Ω drawn into V_T; step 3 overwrites it)
    Random.randn!(cat.V_T)
    _sketch!(cat.Q_tiles, src, cat.V_tiles)

    # Step 2: form/factor each Gram matrix in high precision, then apply its
    # triangular factor to Q with a work-precision TRSM. Ω is dead after the
    # sketch, so its leading S×S rows hold the triangular factors.
    R_work = view(cat.V_T, 1:cat.S, :, :)
    cholqr2!(cat.Q_T, cat.Y_hi, cat.G_hi, R_work, cat.R_tiles,
        cat.Q_tiles, cat.shift_mult)

    # Step 3: co-range  V = Aᴴ·Q  (overwrites the Ω we no longer need)
    _cosketch!(cat.V_tiles, src, cat.Q_tiles)

    # Step 4: the final Gram matrix is dead, so its first element in each slab
    # stores ‖A‖² and is then overwritten by the achieved squared error.
    _tile_norms_sq!(cat.norm_err_sq, src)

    # Step 5: rank detection + truncation (fused SMEM kernel, EI-corrected budget)
    delta_floor = Float64(_cholqr_shift_coeff(eltype(cat.G_hi), size(cat.Y_hi, 1), cat.S))
    prune_ranks!(cat.U, cat.V, cat.ranks_local, cat.norm_err_sq,
        cat.S, cat.R_keep, eps_sq, rel, delta_floor)
    return cat
end
