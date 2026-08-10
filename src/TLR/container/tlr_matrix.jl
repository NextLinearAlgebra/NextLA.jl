"""
    TLRMatrix

Dense-diagonal tile-low-rank matrix. `offdiag` is a full-grid
[`CompressedFTLRMatrix`](@ref): diagonal slots are present in its rank grid but
have logical and execution rank zero. Dense diagonal tiles are stored
separately in `D`, with an optional ragged final tile in `D_corner`.

The full compressed grid deliberately removes the old skip-diagonal and
interior/right/bottom storage arithmetic. Boundary dimensions and exact ranks
are represented by the packed-factor offsets in `offdiag`.
"""
mutable struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},
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
    # Compression capacity. The packed representation itself may use less.
    maxrank::Int
end

@inline offdiagonal(A::TLRMatrix) = A.offdiag
@inline ranks(A::TLRMatrix) = ranks(A.offdiag)
@inline residuals(A::TLRMatrix) = residuals(A.offdiag)
@inline execution_ranks(A::TLRMatrix) = execution_ranks(A.offdiag)
@inline execution_maxrank(A::TLRMatrix) = execution_maxrank(A.offdiag)
@inline execution_rank_policy(A::TLRMatrix) = execution_rank_policy(A.offdiag)

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
            "TLRMatrix off-diagonal storage requires logical rank zero at ($k, $k)"))
        _compressed_ftlr_execution_rank(offdiag, k, k) == 0 || throw(ArgumentError(
            "TLRMatrix off-diagonal storage requires execution rank zero at ($k, $k)"))
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
    TLRMatrix(offdiag::CompressedFTLRMatrix; maxrank=maxrank(offdiag))

Wrap full-grid compressed off-diagonal storage and allocate a separate dense
diagonal. Both logical and execution diagonal ranks must be zero.
"""
function TLRMatrix(offdiag::CompressedFTLRMatrix{BackendT,T};
                   maxrank::Int=maxrank(offdiag)) where {BackendT,T}
    maxrank >= getfield(offdiag, :maxrank) || throw(ArgumentError(
        "TLRMatrix maxrank capacity cannot be smaller than an existing logical rank"))
    _validate_zero_compressed_diagonal(offdiag)
    D, D_corner = _allocate_tlr_diagonal(
        get_backend(offdiag), T, size(offdiag)..., nominal_tile_size(offdiag))
    RankT = eltype(ranks(offdiag))
    return TLRMatrix{BackendT,T,typeof(D),RankT,typeof(tile_order(offdiag)),typeof(offdiag)}(
        get_backend(offdiag), tile_order(offdiag), size(offdiag)...,
        nominal_tile_size(offdiag), tail_tile_size(offdiag), offdiag,
        D, D_corner, maxrank)
end

"""
    TLRMatrix(backend, T, m, n, tile_size, ranks; kwargs...)

Allocate a dense-diagonal TLR matrix with exact off-diagonal logical ranks.
The supplied full-grid rank matrix must contain zeros on its diagonal.
"""
function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int,
    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{<:Integer};
    outer_order=TileRowMajor, inner_order=TileColMajor,
    execution_rank_policy::Symbol=:exact,
    rank_type::Type{<:Integer}=Int32,
) where {T}
    _order_instance(outer_order) isa TileRowMajor || throw(ArgumentError(
        "TLRMatrix requires outer_order=TileRowMajor"))
    _order_instance(inner_order) isa TileColMajor || throw(ArgumentError(
        "TLRMatrix requires inner_order=TileColMajor"))
    offdiag = CompressedFTLRMatrix(
        backend, T, m, n, tile_size, ranks_in;
        outer_order, inner_order, execution_rank_policy, rank_type)
    return TLRMatrix(offdiag)
end

function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
    ranks_in::AbstractMatrix{<:Integer}; kwargs...,
) where {T}
    return TLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end

"""
    TLRMatrix(backend, T, m, n, tile_size, maxrank; kwargs...)

Allocate an empty TLR matrix with zero logical off-diagonal ranks and reserved
`maxrank` execution capacity. This preserves the existing in-place compression
API. After `compress!`, the capacity storage is replaced by exact-rank packed
storage.
"""
function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int,
    tile_size::NTuple{2,Int}, maxrank::Int;
    rank_type::Type{<:Integer}=Int32,
    tile_order=TileRowMajor,
    outer_order=tile_order,
    inner_order=TileColMajor,
) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 && maxrank >= 0 || throw(ArgumentError(
        "m, n, and tile dimensions must be positive; maxrank must be non-negative"))
    mt, nt = cld(m, bm), cld(n, bn)
    _order_instance(outer_order) isa TileRowMajor || throw(ArgumentError(
        "TLRMatrix uses complementary packing and requires tile_order/outer_order=TileRowMajor"))
    _order_instance(inner_order) isa TileColMajor || throw(ArgumentError(
        "TLRMatrix uses complementary packing and requires inner_order=TileColMajor"))
    logical = Base.zeros(Int, mt, nt)
    capacity = fill(maxrank, mt, nt)
    @inbounds for k in 1:min(mt, nt)
        capacity[k, k] = 0
    end
    offdiag = CompressedFTLRMatrix(
        backend, T, m, n, tile_size, logical;
        outer_order, inner_order, execution_ranks=capacity, rank_type)
    return TLRMatrix(offdiag; maxrank)
end

function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int, b::Int, maxrank::Int;
    kwargs...,
) where {T}
    return TLRMatrix(backend, T, m, n, (b, b), maxrank; kwargs...)
end

function TLRMatrix(A::AbstractMatrix{T}, b::Int, ranks_or_maxrank; kwargs...) where {T}
    return TLRMatrix(
        get_backend(A), T, size(A, 1), size(A, 2), b, ranks_or_maxrank; kwargs...)
end

function TLRMatrix(A::AbstractMatrix{T}, tile_size::NTuple{2,Int}, ranks_or_maxrank;
                   kwargs...) where {T}
    return TLRMatrix(
        get_backend(A), T, size(A, 1), size(A, 2), tile_size,
        ranks_or_maxrank; kwargs...)
end
