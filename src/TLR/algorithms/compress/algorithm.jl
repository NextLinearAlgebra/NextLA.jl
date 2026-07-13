"""
    cholqr2!(Y_hi, G_hi) -> Y_hi

Two FKNYY-shifted Cholesky-QR passes. Both use
`11(mS + S(S+1)) * (eps(Tgram)/2) * max(diag(YᴴY))`; no unshifted
factorisation is attempted.
"""
function cholqr2!(Y_hi::AbstractArray{Thi,3}, G_hi, multipliers) where {Thi}
    count = size(Y_hi, 3)
    count == 0 && return Y_hi
    backend = get_backend(Y_hi)
    m, r = size(Y_hi, 1), size(Y_hi, 2)
    RT = real(Thi)
    coeff = _cholqr_shift_coeff(Thi, m, r)
    fill!(multipliers, one(RT))

    for pass in 1:2
        gemm_batched!(_adjoint_blas_char(Thi), 'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
        _cholqr_shift_kernel!(backend)(G_hi, coeff, multipliers;
            ndrange=(r * count,), workgroupsize=r)
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
                _cholqr_shift_kernel!(backend)(G_hi, coeff, multipliers;
                    ndrange=(r * count,), workgroupsize=r)
                result = potrf_batched!('U', G_hi)
                status = Array(result[2])
                attempts += 1
            end
            any(!iszero, status) && throw(LinearAlgebra.PosDefException(findfirst(!iszero, status)))
        end

        trsm_batched!('R', 'U', 'N', 'N', G_hi, Y_hi, one(Thi))
        fill!(multipliers, one(RT))
    end

    return Y_hi
end

"""
    detect_rank!(U, V, Q_T, V_T, rk, err_sq, normA_sq, R_keep, eps_sq, rel) -> U

Per-tile rank detection and truncation ([`_fused_truncate_kernel!`](@ref)):
gather the retained columns from the sketch scratch `Q_T`/`V_T` into the panels
`U`/`V`, writing the detected rank to `rk` and squared error to `err_sq`.
"""
function detect_rank!(U::AbstractArray{T,3}, V, Q_T, V_T, rk, err_sq, normA_sq,
    R_keep::Int, eps_sq::Float64, rel::Bool, delta_floor::Float64) where {T}
    noff = size(U, 3)
    noff == 0 && return U
    backend = get_backend(U)
    S = size(Q_T, 2)
    W = unwrap(SUBGROUP_SIZE(typeof(backend)))
    nthreads = W * min(S, 8)
    kernel! = _fused_truncate_kernel!(backend, nthreads)
    kernel!(U, V, Q_T, V_T, rk, err_sq, normA_sq, eps_sq, rel, R_keep,
        delta_floor, Val{S}(), Val{W}();
        ndrange=(nthreads * noff,), workgroupsize=nthreads)
    U
end

# ─── Tile sources: decouple the sketch pipeline from where the tiles live ──────
#
# The compression core (`compress_tiles!`) touches the input only through three
# operations: `_tile_norms_sq!` (‖A_tile‖²_F, the EI reference term), `_sketch!`
# (Q = A·Ω) and `_cosketch!` (V = Aᴴ·Q). A `TileSource` supplies a `tiles` operand
# vector (fed straight to `gemm_batched!`) plus a norm method. This lets compress
# run on a dense matrix, a packed tile batch, etc. without changing the algorithm.
abstract type TileSource end

@inline _ntiles(src::TileSource) = length(src.tiles)

# Off-diagonal tiles carved from a dense matrix (today's `compress!` path).
struct DenseTiles{AT<:AbstractMatrix,TV<:AbstractVector,CV} <: TileSource
    A::AT           # dense source matrix
    tiles::TV       # per-tile views into A (gemm operands)
    p0s::CV         # per-tile row origin (device Int32) — for the norm kernel
    q0s::CV         # per-tile col origin (device Int32)
    tm::Int         # tile rows
    tn::Int         # tile cols
end

# out[k] = ‖A_tile_k‖²_F. `R` is the norm kernel's reduction width (any ≥ 1).
function _tile_norms_sq!(out, src::DenseTiles; R::Int)
    n = _ntiles(src)
    n == 0 && return out
    _tile_norm_sq_kernel!(get_backend(src.A), R)(out, src.A, src.p0s, src.q0s, src.tm, src.tn, Val{R}();
        ndrange=(R * n,), workgroupsize=R)
    return out
end

# A packed [tm, tn, ntiles] batch of dense tiles (e.g. gemm intermediates).
struct PackedTiles{PT<:AbstractArray,TV<:AbstractVector} <: TileSource
    data::PT        # [tm, tn, ntiles]
    tiles::TV       # per-slab views (gemm operands)
end
PackedTiles(data::AbstractArray{<:Any,3}) =
    PackedTiles(data, [view(data,:,:,k) for k in axes(data, 3)])

function _tile_norms_sq!(out, src::PackedTiles; R::Int)
    n = _ntiles(src)
    n == 0 && return out
    _tile_norm_sq_kernel!(get_backend(src.data), R)(out, src.data, Val{R}();
        ndrange=(R * n,), workgroupsize=R)
    return out
end

# Q = A·Ω and V = Aᴴ·Q, batched over tiles
@inline _sketch!(Q_tiles, src::TileSource, Ω_tiles, ::Type{T}) where {T} =
    gemm_batched!('N', 'N', one(T), src.tiles, Ω_tiles, zero(T), Q_tiles)
@inline _cosketch!(V_tiles, src::TileSource, Q_tiles, ::Type{T}) where {T} =
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), src.tiles, Q_tiles, zero(T), V_tiles)

"""
    compress_tiles!(src, cat; eps_sq, rel) -> cat

Randomized-sketch compression (randQB_EI) of the tile batch described by `src`
into the workspace `cat`: writes the retained factors into `cat.U`/`cat.V` and the
per-tile rank / squared error into `cat.ranks_local` / `cat.err_sq_local`. Input-
agnostic — see [`TileSource`](@ref). Degenerates to rank 0 when `cat.R_keep == 0`.
"""
function compress_tiles!(src::TileSource, cat::CompressCategoryWorkspace; eps_sq::Float64, rel::Bool)
    _ntiles(src) == 0 && return cat
    T = eltype(cat.Q_T)

    # Step 0: per-tile ‖A‖²_F — reference term for the error indicator (step 4).
    _tile_norms_sq!(cat.normA_sq, src; R=max(cat.S, 1))

    if cat.R_keep == 0   # maxrank == 0: every tile degenerates to rank 0
        fill!(cat.ranks_local, zero(eltype(cat.ranks_local)))
        cat.err_sq_local .= cat.normA_sq
        return cat
    end

    # Step 1: range sampling  Q = A·Ω  (Ω drawn into V_T; step 3 overwrites it)
    Random.randn!(cat.V_T)
    _sketch!(cat.Q_tiles, src, cat.V_tiles, T)

    # Step 2: orthogonalise Q (in higher precision)
    cat.Y_hi .= cat.Q_T
    cholqr2!(cat.Y_hi, cat.G_hi, cat.shift_mult)
    cat.Q_T .= cat.Y_hi

    # Step 3: co-range  V = Aᴴ·Q  (overwrites the Ω we no longer need)
    _cosketch!(cat.V_tiles, src, cat.Q_tiles, T)

    # Step 4: rank detection + truncation (fused SMEM kernel, EI-corrected budget)
    delta_floor = Float64(_cholqr_shift_coeff(eltype(cat.G_hi), size(cat.Y_hi, 1), cat.S))
    detect_rank!(cat.U, cat.V, cat.Q_T, cat.V_T, cat.ranks_local, cat.err_sq_local,
        cat.normA_sq, cat.R_keep, eps_sq, rel, delta_floor)
    return cat
end
