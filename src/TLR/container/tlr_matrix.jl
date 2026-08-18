"""
    TLRMatrix

Dense-diagonal tile-low-rank matrix. `offdiag` is a finalized full-grid
[`CompressedFTLRMatrix`](@ref); its diagonal ranks are zero. Dense diagonal
tiles are stored separately in `D`, with an optional ragged final tile in
`D_corner`.
"""
struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},
                 OffdiagT<:CompressedFTLRMatrix{BackendT,T}} <: AbstractTLRMatrix{T}
    offdiag::OffdiagT
    D::Arr3T
    D_corner::Arr3T
end

@inline offdiagonal(A::TLRMatrix) = A.offdiag
@inline offdiagonal(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    transpose(offdiagonal(parent(A)))
Base.size(A::TLRMatrix) = size(A.offdiag)
@inline nominal_tile_size(A::TLRMatrix) = nominal_tile_size(A.offdiag)
@inline tail_tile_size(A::TLRMatrix) = tail_tile_size(A.offdiag)
@inline KernelAbstractions.get_backend(A::TLRMatrix) = get_backend(A.offdiag)
@inline tile_order(A::TLRMatrix) = tile_order(A.offdiag)
@inline ranks(A::TLRMatrix) = ranks(A.offdiag)
@inline residuals(A::TLRMatrix) = residuals(A.offdiag)
@inline maxrank(A::TLRMatrix) = maxrank(A.offdiag)
@inline rank_multiple(A::TLRMatrix) = rank_multiple(A.offdiag)
@inline maximum_storage_rank(A::TLRMatrix) = maximum_storage_rank(A.offdiag)

@inline ndiag_tiles(A::TLRMatrix) = min(grid_size(A)...)
@inline ndiag_tiles(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    ndiag_tiles(parent(A))
@inline dense_diag(A::TLRMatrix) = A.D
@inline dense_diag_corner(A::TLRMatrix) = A.D_corner
@inline dense_diag(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    PermutedDimsArray(dense_diag(parent(A)), (2, 1, 3))
@inline dense_diag_corner(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    PermutedDimsArray(dense_diag_corner(parent(A)), (2, 1, 3))

@inline function _diag_tile_view(A::TLRMatrix, tile_k::Int)
    1 <= tile_k <= ndiag_tiles(A) || throw(BoundsError(1:ndiag_tiles(A), tile_k))
    if tile_k <= size(A.D, 3)
        return view(A.D, :, :, tile_k)
    end
    size(A.D_corner, 3) != 0 || throw(BoundsError(1:size(A.D, 3), tile_k))
    return view(A.D_corner, :, :, 1)
end

@inline _diag_tile_view(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}, tile_k::Int) =
    transpose(_diag_tile_view(parent(A), tile_k))

"""Return an exact-rank factor pair for an off-diagonal tile."""
@inline function get_factors(A::TLRMatrix, i::Int, j::Int)
    i == j && throw(ArgumentError("tile ($i, $j) is diagonal and stored densely"))
    return get_factors(A.offdiag, i, j)
end
@inline get_factors(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}, i::Int, j::Int) =
    get_factors(offdiagonal(A), i, j)

"""
    TLRMatrix(offdiag::CompressedFTLRMatrix)

Wrap finalized full-grid compressed off-diagonal storage and allocate a
separate dense diagonal. Diagonal ranks in `offdiag` must be zero.
"""
function TLRMatrix(offdiag::CompressedFTLRMatrix{BackendT,T}) where {BackendT,T}
    compressed_ftlr_outer_order(offdiag) isa TileRowMajor || throw(ArgumentError(
        "TLRMatrix requires row-major outer-factor packing"))
    compressed_ftlr_inner_order(offdiag) isa TileColMajor || throw(ArgumentError(
        "TLRMatrix requires column-major inner-factor packing"))
    @inbounds for k in 1:min(grid_size(offdiag)...)
        _compressed_ftlr_rank(offdiag, k, k) == 0 || throw(ArgumentError(
            "TLRMatrix off-diagonal storage requires rank zero at ($k, $k)"))
    end
    bm, bn = nominal_tile_size(offdiag)
    mt, nt = grid_size(offdiag)
    n_diag = min(mt, nt)
    tail_m, tail_n = tail_tile_size(offdiag)
    corner_tm = n_diag == mt && !iszero(tail_m) ? tail_m : bm
    corner_tn = n_diag == nt && !iszero(tail_n) ? tail_n : bn
    has_corner = n_diag > 0 && (corner_tm != bm || corner_tn != bn)
    backend = get_backend(offdiag)
    D = zeros(backend, T, bm, bn, n_diag - Int(has_corner))
    D_corner = zeros(
        backend, T, max(corner_tm, 1), max(corner_tn, 1), has_corner ? 1 : 0)
    return TLRMatrix{BackendT,T,typeof(D),typeof(offdiag)}(offdiag, D, D_corner)
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
    (outer_order === TileRowMajor || outer_order isa TileRowMajor) || throw(ArgumentError(
        "TLRMatrix requires outer_order=TileRowMajor"))
    (inner_order === TileColMajor || inner_order isa TileColMajor) || throw(ArgumentError(
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
