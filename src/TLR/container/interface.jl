"""
    TileOrderStyle

Policy type selecting how a logical tile grid is linearized.
"""
abstract type TileOrderStyle end

"""Column-major traversal of the logical tile grid."""
struct TileColMajor <: TileOrderStyle end
"""Row-major traversal of the logical tile grid."""
struct TileRowMajor <: TileOrderStyle end

Base.transpose(::TileColMajor) = TileRowMajor()
Base.transpose(::TileRowMajor) = TileColMajor()

"""
    tile_linear_index(order, mt, nt, i, j)

Return the linear traversal index associated with logical tile `(i, j)`.
"""
@inline function tile_linear_index(order, mt::Int, nt::Int, i::Int, j::Int)
    1 <= i <= mt || throw(BoundsError((mt, nt), (i, :)))
    1 <= j <= nt || throw(BoundsError((mt, nt), (:, j)))
    (order === TileColMajor || order isa TileColMajor) &&
        return i + (j - 1) * mt
    (order === TileRowMajor || order isa TileRowMajor) &&
        return j + (i - 1) * nt
    throw(ArgumentError("unsupported tile order $(typeof(order))"))
end

"""
    inverse_tile_index(order, mt, nt, linear)

Return the logical tile coordinate corresponding to `linear`.
"""
@inline function inverse_tile_index(order, mt::Int, nt::Int, linear::Int)
    1 <= linear <= mt * nt || throw(BoundsError((mt, nt), linear))
    k = linear - 1
    (order === TileColMajor || order isa TileColMajor) &&
        return (k % mt + 1, k ÷ mt + 1)
    (order === TileRowMajor || order isa TileRowMajor) &&
        return (k ÷ nt + 1, k % nt + 1)
    throw(ArgumentError("unsupported tile order $(typeof(order))"))
end

"""
    AbstractTLRMatrix{T}

Shared interface for tile low-rank matrix containers. Concrete subtypes decide
how diagonal tiles are represented and how low-rank tile indices are mapped to
storage.
"""
abstract type AbstractTLRMatrix{T} end

"""Zero-copy container view representing the transpose of a TLR matrix."""
struct TransposeTLRMatrix{T,A<:AbstractTLRMatrix{T}} <: AbstractTLRMatrix{T}
    parent::A
end

TransposeTLRMatrix(A::AbstractTLRMatrix{T}) where {T} =
    TransposeTLRMatrix{T,typeof(A)}(A)

Base.parent(A::TransposeTLRMatrix) = getfield(A, :parent)
Base.transpose(A::AbstractTLRMatrix) = TransposeTLRMatrix(A)
Base.transpose(A::TransposeTLRMatrix) = parent(A)

Base.eltype(::Type{<:AbstractTLRMatrix{T}}) where {T} = T
Base.eltype(::AbstractTLRMatrix{T}) where {T} = T
Base.size(A::AbstractTLRMatrix) = (A.m, A.n)
Base.size(A::AbstractTLRMatrix, d::Int) = size(A)[d]
Base.size(A::TransposeTLRMatrix) = reverse(size(parent(A)))

@inline nominal_tile_size(A::AbstractTLRMatrix) = A.nominal_tile_size
@inline nominal_tile_size(A::TransposeTLRMatrix) = reverse(nominal_tile_size(parent(A)))
@inline nominal_tile_size(A::AbstractTLRMatrix, axis::Int) =
    nominal_tile_size(A)[axis]

@inline tail_tile_size(A::AbstractTLRMatrix) = A.tail_tile_size
@inline tail_tile_size(A::TransposeTLRMatrix) = reverse(tail_tile_size(parent(A)))
@inline tail_tile_size(A::AbstractTLRMatrix, axis::Int) =
    tail_tile_size(A)[axis]

"""Full tile grid including partial boundary tiles: `(n_tile_rows, n_tile_cols)`."""
@inline grid_size(A::AbstractTLRMatrix) =
    (cld(size(A, 1), nominal_tile_size(A, 1)),
     cld(size(A, 2), nominal_tile_size(A, 2)))

"""
    regular_grid_size(A) -> (q_m, q_n)

Sub-grid of full-size regular tiles, `(⌊m/bm⌋, ⌊n/bn⌋)` — `grid_size` minus
any partial boundary row/column. Equals `grid_size` when the matrix tiles
evenly. This is the interior grid the gemm hard term operates over.
"""
@inline regular_grid_size(A::AbstractTLRMatrix) =
    (fld(size(A, 1), nominal_tile_size(A, 1)),
     fld(size(A, 2), nominal_tile_size(A, 2)))

@inline function tile_size(A, tile_i::Int, tile_j::Int)
    mt, nt = grid_size(A)
    bm, bn = nominal_tile_size(A)
    tail_m, tail_n = tail_tile_size(A)
    row_size = tile_i == mt && !iszero(tail_m) ? tail_m : bm
    col_size = tile_j == nt && !iszero(tail_n) ? tail_n : bn
    return row_size, col_size
end

@inline maxrank(A::AbstractTLRMatrix) = A.maxrank
@inline maxrank(A::TransposeTLRMatrix) = maxrank(parent(A))
@inline ranks(A::AbstractTLRMatrix) = A.ranks
@inline ranks(A::TransposeTLRMatrix) = ranks(parent(A))
@inline residuals(A::AbstractTLRMatrix) = A.resid
@inline residuals(A::TransposeTLRMatrix) = residuals(parent(A))
@inline KernelAbstractions.get_backend(A::AbstractTLRMatrix) = A.backend
@inline KernelAbstractions.get_backend(A::TransposeTLRMatrix) = get_backend(parent(A))
@inline tile_order(A::AbstractTLRMatrix) = A.order
@inline tile_order(A::TransposeTLRMatrix) = transpose(tile_order(parent(A)))

@inline function tile_origin_coords(A, tile_i::Int, tile_j::Int)
    return ((tile_i - 1) * nominal_tile_size(A, 1) + 1,
        (tile_j - 1) * nominal_tile_size(A, 2) + 1)
end

@inline function _dense_tile_view(dense::AbstractMatrix, A, tile_i::Int, tile_j::Int)
    p0, q0 = tile_origin_coords(A, tile_i, tile_j)
    tm, tn = tile_size(A, tile_i, tile_j)
    return view(dense, p0:(p0+tm-1), q0:(q0+tn-1))
end

"""Logical row/column range of one TLR tile along `axis`."""
@inline function _tile_axis_range(A::AbstractTLRMatrix, tile::Int, axis::Int)
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    extent = tile_size(A, axis == 1 ? tile : 1, axis == 2 ? tile : 1)[axis]
    return first:(first + extent - 1)
end
