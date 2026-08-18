"""
    TLRMatrix

Dense-diagonal tile-low-rank matrix. `offdiag` is a finalized full-grid
[`CompressedFTLRMatrix`](@ref); its diagonal ranks are zero. Dense diagonal
tiles are stored separately in `D`, with an optional ragged final tile in
`D_corner`.
"""
struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},
                 RankT<:Integer,OrderT<:TileOrderStyle,OffdiagT} <:
       AbstractTLRMatrix{BackendT,T,OrderT}
    backend::BackendT
    order::OrderT
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}
    offdiag::OffdiagT
    D::Arr3T
    D_corner::Arr3T
end

@inline offdiagonal(A::TLRMatrix) = A.offdiag
@inline ranks(A::TLRMatrix) = ranks(A.offdiag)
@inline residuals(A::TLRMatrix) = residuals(A.offdiag)
@inline maxrank(A::TLRMatrix) = maxrank(A.offdiag)
@inline rank_multiple(A::TLRMatrix) = rank_multiple(A.offdiag)
@inline maximum_storage_rank(A::TLRMatrix) = maximum_storage_rank(A.offdiag)

@inline ndiag_tiles(A::TLRMatrix) = min(grid_size(A)...)
@inline dense_diag(A::TLRMatrix) = A.D
@inline dense_diag_corner(A::TLRMatrix) = A.D_corner
@inline _nfull_diag_tiles(A::TLRMatrix) = size(A.D, 3)

@inline function _diag_tile_view(A::TLRMatrix, tile_k::Int)
    1 <= tile_k <= ndiag_tiles(A) || throw(BoundsError(1:ndiag_tiles(A), tile_k))
    if tile_k <= _nfull_diag_tiles(A)
        return view(A.D, :, :, tile_k)
    end
    size(A.D_corner, 3) != 0 || throw(BoundsError(1:_nfull_diag_tiles(A), tile_k))
    return view(A.D_corner, :, :, 1)
end

function _set_dense_diagonal_diagnostics!(A::TLRMatrix)
    @inbounds for k in 1:ndiag_tiles(A)
        idx = _rank_index(A, k, k)
        ranks(A)[idx] = 0
        residuals(A)[idx] = 0.0
    end
    return A
end

"""Return an exact-rank factor pair for an off-diagonal tile."""
@inline function get_factors(A::TLRMatrix, i::Int, j::Int)
    i == j && throw(ArgumentError("tile ($i, $j) is diagonal and stored densely"))
    return get_factors(A.offdiag, i, j)
end

function _validate_zero_compressed_diagonal(offdiag::CompressedFTLRMatrix)
    compressed_ftlr_outer_order(offdiag) isa TileRowMajor || throw(ArgumentError(
        "TLRMatrix requires row-major outer-factor packing"))
    compressed_ftlr_inner_order(offdiag) isa TileColMajor || throw(ArgumentError(
        "TLRMatrix requires column-major inner-factor packing"))
    @inbounds for k in 1:min(grid_size(offdiag)...)
        _compressed_ftlr_rank(offdiag, k, k) == 0 || throw(ArgumentError(
            "TLRMatrix off-diagonal storage requires rank zero at ($k, $k)"))
    end
    return offdiag
end

function _allocate_tlr_diagonal(backend, ::Type{T}, m::Int, n::Int,
                                tile_size::NTuple{2,Int}) where {T}
    bm, bn = tile_size
    mt, nt = cld(m, bm), cld(n, bn)
    n_diag = min(mt, nt)
    corner_tm = n_diag == mt ? _last_dim(m, bm) : bm
    corner_tn = n_diag == nt ? _last_dim(n, bn) : bn
    has_corner = n_diag > 0 && (corner_tm != bm || corner_tn != bn)
    D = zeros(backend, T, bm, bn, n_diag - Int(has_corner))
    D_corner = zeros(
        backend, T, max(corner_tm, 1), max(corner_tn, 1), has_corner ? 1 : 0)
    return D, D_corner
end

"""
    TLRMatrix(offdiag::CompressedFTLRMatrix)

Wrap finalized full-grid compressed off-diagonal storage and allocate a
separate dense diagonal. Diagonal ranks in `offdiag` must be zero.
"""
function TLRMatrix(offdiag::CompressedFTLRMatrix{BackendT,T}) where {BackendT,T}
    _validate_zero_compressed_diagonal(offdiag)
    D, D_corner = _allocate_tlr_diagonal(
        get_backend(offdiag), T, size(offdiag)..., nominal_tile_size(offdiag))
    RankT = eltype(ranks(offdiag))
    return TLRMatrix{BackendT,T,typeof(D),RankT,typeof(tile_order(offdiag)),typeof(offdiag)}(
        get_backend(offdiag), tile_order(offdiag), size(offdiag)...,
        nominal_tile_size(offdiag), tail_tile_size(offdiag), offdiag, D, D_corner)
end

"""
    TLRMatrix(backend, T, m, n, tile_size, ranks; rank_multiple=0)

Allocate a finalized dense-diagonal TLR matrix from known off-diagonal ranks.
The supplied full-grid rank matrix must contain zeros on its diagonal.
"""
function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int,
    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{<:Integer};
    outer_order=TileRowMajor, inner_order=TileColMajor,
    rank_multiple::Integer=0,
    rank_type::Type{<:Integer}=Int32,
) where {T}
    _order_instance(outer_order) isa TileRowMajor || throw(ArgumentError(
        "TLRMatrix requires outer_order=TileRowMajor"))
    _order_instance(inner_order) isa TileColMajor || throw(ArgumentError(
        "TLRMatrix requires inner_order=TileColMajor"))
    offdiag = CompressedFTLRMatrix(
        backend, T, m, n, tile_size, ranks_in;
        outer_order, inner_order, rank_multiple, rank_type)
    return TLRMatrix(offdiag)
end

function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
    ranks_in::AbstractMatrix{<:Integer}; kwargs...,
) where {T}
    return TLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end
