include("workspace.jl")

# Dense-diagonal copy ----------------------------------------------------------

@kernel function _copy_diag_from_dense_kernel!(D::AbstractArray{T,3},
                                                A::AbstractMatrix{T},
                                                tile_m::Int,
                                                tile_n::Int) where {T}
    row, col, batch = @index(Global, NTuple)
    p0 = (batch - 1) * tile_m + 1
    q0 = (batch - 1) * tile_n + 1
    @inbounds D[row, col, batch] = A[p0+row-1, q0+col-1]
end

function _copy_diagonal_from_dense!(A_tlr::TLRMatrix{<:Any,T},
                                    A::AbstractMatrix{T}) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    bm, bn = nominal_tile_size(A_tlr)
    if n_full_diag > 0
        _copy_diag_from_dense_kernel!(A_tlr.backend)(
            A_tlr.D, A, bm, bn; ndrange=(bm, bn, n_full_diag))
    end
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(view(A_tlr.D_corner, 1:tm, 1:tn, 1),
                _dense_tile_view(A, A_tlr, tile_k, tile_k))
    end
    return _set_dense_diagonal_diagnostics!(A_tlr)
end

# Tile-batch compression core --------------------------------------------------

abstract type TileSource{T} end

@inline _ntiles(src::TileSource) = length(src.tiles)

struct DenseTiles{T,AT<:AbstractMatrix{T},TV<:AbstractVector,CV} <: TileSource{T}
    A::AT
    tiles::TV
    p0s::CV
    q0s::CV
    tm::Int
    tn::Int
end

function _tile_norms_sq!(out, src::DenseTiles)
    n = _ntiles(src)
    n == 0 && return out
    backend = get_backend(src.A)
    W, _, NT = _norm_launch(backend, src.tn)
    _tile_norm_sq_kernel!(backend, NT)(
        out, src.A, src.p0s, src.q0s, src.tm, src.tn, Val{W}(), Val{NT}();
        ndrange=(NT * n,), workgroupsize=NT)
    return out
end

"""A homogeneous `[tile_rows, tile_cols, tile_count]` dense tile batch."""
struct PackedTiles{T,PT<:AbstractArray{T,3},TV<:AbstractVector} <: TileSource{T}
    data::PT
    tiles::TV
end
PackedTiles(data::AbstractArray{<:Any,3}) =
    PackedTiles(data, [view(data, :, :, k) for k in axes(data, 3)])

_tile_norms_sq!(out, src::PackedTiles) = batch_frobenius_norms_sq!(out, src.data)

@inline _cosketch!(V_tiles, src::TileSource{T}, Q_tiles) where {T} =
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T),
                  src.tiles, Q_tiles, zero(T), V_tiles)

struct ARATileSampler{S,C}
    src::S
    cat::C
end

function (sampler::ARATileSampler)(Y, width)
    src, cat = sampler.src, sampler.cat
    T = eltype(cat.omega)
    Random.randn!(cat.omega)
    gemm_batched!('N', 'N', one(T), src.tiles,
                  cat.omega_tiles, zero(T), cat.Y_tiles)
    return Y
end

@inline _ara_tile_sampler(src, cat) = ARATileSampler(src, cat)

# Consecutive negligible columns required before a tile is declared converged.
const _ARA_CONSECUTIVE = 10

"""
    compress_tiles!(src, workspace; eps_sq, rel)

Run blocked ARA on one homogeneous tile batch. Results are left in the
workspace's [`LowRankFactorBatch`](@ref); no matrix container is constructed.
"""
function compress_tiles!(src::TileSource{T}, cat::CompressCategoryWorkspace;
                         eps_sq::Float64, rel::Bool) where {T}
    _ntiles(src) == 0 && return cat
    f = cat.factors

    if cat.R_keep == 0
        _tile_norms_sq!(f.errors_sq, src)
        fill!(f.ranks, zero(eltype(f.ranks)))
        return cat
    end

    _tile_norms_sq!(f.errors_sq, src)
    eps_rel = max(sqrt(eps_sq), ara_stopping_floor(tlr_orthogonalization_type(T)))
    ara_build_basis!(cat.ara, _ara_tile_sampler(src, cat);
                     eps_rel, r_required=_ARA_CONSECUTIVE)
    _cosketch!(cat.V_tiles, src, cat.Q_tiles)
    ara_truncate!(view(f.U, :, 1:cat.R_keep, :),
                  view(f.V, :, 1:cat.R_keep, :),
                  f.ranks, f.errors_sq, cat.ara.Q, cat.Z;
                  tol=sqrt(eps_sq), relative=rel, maxrank=cat.R_keep,
                  energy=f.errors_sq)
    return cat
end

function _compress_category!(A::AbstractMatrix, cat::CompressCategoryWorkspace,
                             tile_size::NTuple{2,Int},
                             eps_sq::Float64, rel::Bool)
    f = cat.factors
    isempty(f.tile_ids) && return cat
    bm, bn = tile_size
    tm, tn = f.tile_shape
    tiles = [view(A,
                  (i-1)*bm+1:(i-1)*bm+tm,
                  (j-1)*bn+1:(j-1)*bn+tn) for (i, j) in f.tile_ids]
    return compress_tiles!(DenseTiles(A, tiles, cat.p0s, cat.q0s, tm, tn), cat;
                           eps_sq, rel)
end

function _compress_all_categories!(A::AbstractMatrix{T},
                                   ws::FTLRCompressionWorkspace,
                                   eps_sq::Float64, rel::Bool) where {T}
    cats = ws.cats
    backend = get_backend(A)
    tile_size = ws.key.tile_size
    if backend isa KernelAbstractions.CPU
        for cat in cats
            _compress_category!(A, cat, tile_size, eps_sq, rel)
        end
    else
        for (cat, stream) in zip(cats, ws.streams)
            with_stream(backend, stream) do
                _compress_category!(A, cat, tile_size, eps_sq, rel)
            end
        end
        for stream in ws.streams
            sync_stream(backend, stream)
        end
    end
    return ws
end

# Final packed-factor construction --------------------------------------------

@kernel function _scatter_lowrank_factor_kernel!(destination, source,
                                                 offsets, ranks, ld::Int)
    factor = @index(Group, Linear)
    lane = @index(Local, Linear)
    logical_rows = size(source, 1)
    active = logical_rows * Int(@inbounds ranks[factor])
    k = lane
    while k <= active
        row = (k - 1) % logical_rows + 1
        col = (k - 1) ÷ logical_rows + 1
        @inbounds destination[Int(offsets[factor]) + (col-1)*ld + row-1] =
            source[row, col, factor]
        k += @groupsize()[1]
    end
end

@inline _scatter_workgroupsize(backend) =
    backend isa KernelAbstractions.CPU ? 1 : 128

function _compression_diagnostics(ws::FTLRCompressionWorkspace)
    qm, qn = cld(ws.key.m, ws.key.tile_size[1]), cld(ws.key.n, ws.key.tile_size[2])
    rank_grid = Base.zeros(Int, qm, qn)
    residual_grid = Base.zeros(Float64, qm, qn)
    for cat in ws.cats
        f = cat.factors
        rk = f.ranks isa Vector ? f.ranks : Array(f.ranks)
        err = f.errors_sq isa Vector ? f.errors_sq : Array(f.errors_sq)
        @inbounds for (k, (i, j)) in enumerate(f.tile_ids)
            rank_grid[i, j] = Int(rk[k])
            residual_grid[i, j] = sqrt(max(Float64(real(err[k])), 0.0))
        end
    end
    return rank_grid, residual_grid
end

function _factor_offsets_for_batch(C::CompressedFTLRMatrix,
                                   f::LowRankFactorBatch, side::Symbol)
    factors = side === :outer ? C.outer : C.inner
    offsets = Vector{Int}(undef, length(f.tile_ids))
    @inbounds for (k, (i, j)) in enumerate(f.tile_ids)
        offsets[k] = factors.offsets[_compressed_ftlr_slot(factors, i, j)]
    end
    return offsets
end

function _scatter_factor_batch!(C::CompressedFTLRMatrix,
                                f::LowRankFactorBatch, side::Symbol)
    isempty(f.tile_ids) && return C
    backend = get_backend(C)
    factors = side === :outer ? C.outer : C.inner
    source = side === :outer ? f.U : f.V
    tile_axis = side === :outer ? first(f.tile_ids[1]) : last(f.tile_ids[1])
    ld = factors.leading_dimensions[tile_axis]
    offsets_host = _factor_offsets_for_batch(C, f, side)
    offsets = copyto!(allocate(backend, Int, length(offsets_host)), offsets_host)
    wg = _scatter_workgroupsize(backend)
    _scatter_lowrank_factor_kernel!(backend, wg)(
        factors.data, source, offsets, f.ranks, ld;
        ndrange=(wg * length(f.tile_ids),), workgroupsize=wg)
    return C
end

function _finalize_compressed_ftlr(ws::FTLRCompressionWorkspace;
                                   rank_multiple::Integer=0,
                                   outer_order=TileRowMajor,
                                   inner_order=TileColMajor)
    rank_grid, residual_grid = _compression_diagnostics(ws)
    key = ws.key
    backend = key.device
    C = CompressedFTLRMatrix(
        backend, key.T, key.m, key.n, key.tile_size, rank_grid;
        outer_order, inner_order, rank_multiple, rank_type=key.rank_type)
    @inbounds for j in axes(rank_grid, 2), i in axes(rank_grid, 1)
        C.resid[_rank_index(C, i, j)] = residual_grid[i, j]
    end
    for cat in ws.cats
        _scatter_factor_batch!(C, cat.factors, :outer)
        _scatter_factor_batch!(C, cat.factors, :inner)
    end
    KernelAbstractions.synchronize(backend)
    return C
end

# Public dense constructors ----------------------------------------------------

"""
    CompressedFTLRMatrix(A, tile_size; maxrank, tol=0, rel=false,
                         rank_multiple=0, workspace=nothing)

Compress dense `A` in one adaptive pass and return finalized packed factors.
`rank_multiple == 0` stores exact widths; a positive value rounds each nonzero
tile rank up to that multiple without changing its logical rank.
"""
function CompressedFTLRMatrix(A::AbstractMatrix{T},
                              tile_size::NTuple{2,Int};
                              maxrank::Int,
                              tol::Real=0.0,
                              rel::Bool=false,
                              rank_multiple::Integer=0,
                              workspace::Union{Nothing,FTLRCompressionWorkspace}=nothing,
                              outer_order=TileRowMajor,
                              inner_order=TileColMajor,
                              rank_type::Type{<:Integer}=Int32) where {T}
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    _validate_rank_multiple(rank_multiple)
    ws = workspace === nothing ? FTLRCompressionWorkspace(
        A, tile_size; maxrank, diagonal=:compressed, rank_type) : workspace
    _validate_compression_workspace(
        ws, A, tile_size, maxrank, :compressed, rank_type)
    _compress_all_categories!(A, ws, Float64(tol)^2, rel)
    return _finalize_compressed_ftlr(
        ws; rank_multiple, outer_order, inner_order)
end

CompressedFTLRMatrix(A::AbstractMatrix, b::Int; kwargs...) =
    CompressedFTLRMatrix(A, (b, b); kwargs...)

"""
    TLRMatrix(A, tile_size; maxrank, tol=0, rel=false,
              rank_multiple=0, workspace=nothing)

Compress only the off-diagonal tiles of dense `A`, keep the diagonal dense,
and return a finalized `TLRMatrix`.
"""
function TLRMatrix(A::AbstractMatrix{T}, tile_size::NTuple{2,Int};
                   maxrank::Int,
                   tol::Real=0.0,
                   rel::Bool=false,
                   rank_multiple::Integer=0,
                   workspace::Union{Nothing,FTLRCompressionWorkspace}=nothing,
                   rank_type::Type{<:Integer}=Int32) where {T}
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    _validate_rank_multiple(rank_multiple)
    ws = workspace === nothing ? FTLRCompressionWorkspace(
        A, tile_size; maxrank, diagonal=:dense, rank_type) : workspace
    _validate_compression_workspace(ws, A, tile_size, maxrank, :dense, rank_type)
    _compress_all_categories!(A, ws, Float64(tol)^2, rel)
    offdiag = _finalize_compressed_ftlr(ws; rank_multiple)
    C = TLRMatrix(offdiag)
    _copy_diagonal_from_dense!(C, A)
    KernelAbstractions.synchronize(get_backend(C))
    return C
end

TLRMatrix(A::AbstractMatrix, b::Int; kwargs...) = TLRMatrix(A, (b, b); kwargs...)
