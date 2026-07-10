"""
    AbstractTLRMatrix{BackendT, T, OrderT}

Shared interface for tile low-rank matrix containers. Concrete subtypes decide
how diagonal tiles are represented and how low-rank tile indices are mapped to
storage.
"""
abstract type AbstractTLRMatrix{BackendT<:Backend,T,OrderT<:TileOrderStyle} end

const _TILE_INT = UInt8(1)      # regular interior tile category
const _TILE_RIGHT = UInt8(2)    # right boundary tile category
const _TILE_BOTTOM = UInt8(3)   # bottom boundary tile category
const _TILE_CORNER = UInt8(4)   # bottom-right corner tile category

Base.eltype(::Type{<:AbstractTLRMatrix{<:Any,T}}) where {T} = T
Base.eltype(::AbstractTLRMatrix{<:Any,T}) where {T} = T
Base.size(A::AbstractTLRMatrix) = (A.m, A.n)
Base.size(A::AbstractTLRMatrix, d::Int) = size(A)[d]

@inline function _axis_index(axis::Integer)
    i = Int(axis)
    1 <= i <= 2 || throw(BoundsError(1:2, i))
    return i
end

@inline nominal_tile_size(A::AbstractTLRMatrix) = A.nominal_tile_size
@inline nominal_tile_size(A::AbstractTLRMatrix, axis::Integer) = A.nominal_tile_size[_axis_index(axis)]

@inline tail_tile_size(A::AbstractTLRMatrix) = A.tail_tile_size
@inline tail_tile_size(A::AbstractTLRMatrix, axis::Integer) = A.tail_tile_size[_axis_index(axis)]

@inline function _last_tile_size(A::AbstractTLRMatrix, axis::Integer)
    tail = tail_tile_size(A, axis)
    return iszero(tail) ? nominal_tile_size(A, axis) : tail
end

@inline tilegrid_size(A::AbstractTLRMatrix) = (cld(A.m, nominal_tile_size(A, 1)), cld(A.n, nominal_tile_size(A, 2)))

"""
    _full_regular_grid(A) -> (q_m, q_n)

Sub-grid of full-size regular tiles, `(⌊m/bm⌋, ⌊n/bn⌋)` — `tilegrid_size` minus
any partial boundary row/column. Equals `tilegrid_size` when the matrix tiles
evenly. This is the interior grid the gemm hard term operates over.
"""
@inline _full_regular_grid(A::AbstractTLRMatrix) =
    (fld(A.m, nominal_tile_size(A, 1)), fld(A.n, nominal_tile_size(A, 2)))

@inline function tile_size(A::AbstractTLRMatrix, tile_i::Int, tile_j::Int)
    mt, nt = tilegrid_size(A)
    bm, bn = nominal_tile_size(A)

    row_size = tile_i == mt ?
        _last_tile_size(A, 1) :
        bm

    col_size = tile_j == nt ?
        _last_tile_size(A, 2) :
        bn

    return row_size, col_size
end

@inline maxrank(A::AbstractTLRMatrix) = A.maxrank
@inline ranks(A::AbstractTLRMatrix) = A.ranks
@inline residuals(A::AbstractTLRMatrix) = A.resid
@inline KernelAbstractions.get_backend(A::AbstractTLRMatrix) = A.backend
@inline tile_order(A::AbstractTLRMatrix) = A.order

"""
    _rank_index(A, i, j) -> Int

Index into `ranks(A)` / `residuals(A)` for tile `(i, j)`. All TLR containers
keep these diagnostic vectors tile-grid aligned, even when a concrete container
stores some tiles densely. Dense tiles therefore still occupy a rank/residual
slot; their values describe the represented tile, not necessarily low-rank
factor storage.
"""
@inline function _rank_index(A::AbstractTLRMatrix, i::Int, j::Int)
    mt, nt = tilegrid_size(A)
    return tile_linear_index(A.order, mt, nt, i, j)
end

"""
    _rank_index(A, category, k) -> Int

Map a category-local storage slot to the tile-grid-aligned diagnostic slot.
"""
@inline _rank_index(A::AbstractTLRMatrix, cat::UInt8, k::Int) =
    _rank_index(A, _category_coords(A, cat, k)...)

@inline function tile_origin_coords(A::AbstractTLRMatrix, tile_i::Int, tile_j::Int)
    return ((tile_i - 1) * nominal_tile_size(A, 1) + 1,
            (tile_j - 1) * nominal_tile_size(A, 2) + 1)
end

@inline function _dense_tile_view(dense::AbstractMatrix, A::AbstractTLRMatrix, tile_i::Integer, tile_j::Integer)
    p0, q0 = tile_origin_coords(A, Int(tile_i), Int(tile_j))
    tm, tn = tile_size(A, Int(tile_i), Int(tile_j))
    return view(dense, p0:(p0 + tm - 1), q0:(q0 + tn - 1))
end

@inline _last_dim(dim::Int, nominal::Int) = (tail = dim % nominal; iszero(tail) ? nominal : tail)
