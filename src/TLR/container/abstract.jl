"""
    AbstractTLRMatrix{BackendT, T, OrderT}

Shared interface for tile low-rank matrix containers. Concrete subtypes decide
how diagonal tiles are represented and how low-rank tile indices are mapped to
storage.
"""
abstract type AbstractTLRMatrix{BackendT<:Backend,T,OrderT<:TileOrderStyle} end

abstract type TLRRegion end
struct InteriorRegion <: TLRRegion end
struct RightRegion <: TLRRegion end
struct BottomRegion <: TLRRegion end
struct CornerRegion <: TLRRegion end

const _INTERIOR = InteriorRegion()
const _RIGHT = RightRegion()
const _BOTTOM = BottomRegion()
const _CORNER = CornerRegion()

@inline transpose_region(::InteriorRegion) = _INTERIOR
@inline transpose_region(::RightRegion) = _BOTTOM
@inline transpose_region(::BottomRegion) = _RIGHT
@inline transpose_region(::CornerRegion) = _CORNER

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

@inline function _last_tile_size(A, axis::Integer)
    tail = tail_tile_size(A, axis)
    return iszero(tail) ? nominal_tile_size(A, axis) : tail
end

@inline tilegrid_size(A::AbstractTLRMatrix) = (cld(A.m, nominal_tile_size(A, 1)), cld(A.n, nominal_tile_size(A, 2)))

"""
    regular_tilegrid_size(A) -> (q_m, q_n)

Sub-grid of full-size regular tiles, `(⌊m/bm⌋, ⌊n/bn⌋)` — `tilegrid_size` minus
any partial boundary row/column. Equals `tilegrid_size` when the matrix tiles
evenly. This is the interior grid the gemm hard term operates over.
"""
@inline regular_tilegrid_size(A::AbstractTLRMatrix) =
    (fld(A.m, nominal_tile_size(A, 1)), fld(A.n, nominal_tile_size(A, 2)))

"""Number and factor dimensions of the low-rank tiles stored in `region`."""
@inline region_tile_count(A, region::TLRRegion) = size(outer_factors(A, region), 3)
@inline region_tile_size(A, region::TLRRegion) =
    (size(outer_factors(A, region), 1), size(inner_factors(A, region), 1))

"""Global tile coordinates of region-local storage slot `k`."""
@inline region_tile_coords(A::AbstractTLRMatrix, ::RightRegion, k::Int) =
    (k, tilegrid_size(A)[2])
@inline region_tile_coords(A::AbstractTLRMatrix, ::BottomRegion, k::Int) =
    (tilegrid_size(A)[1], k)
@inline region_tile_coords(A::AbstractTLRMatrix, ::CornerRegion, ::Int) =
    tilegrid_size(A)

@inline function tile_size(A, tile_i::Int, tile_j::Int)
    mt, nt = tilegrid_size(A)
    bm, bn = nominal_tile_size(A)

    row_size = tile_i == mt ? _last_tile_size(A, 1) : bm

    col_size = tile_j == nt ? _last_tile_size(A, 2) : bn

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
    _rank_index(A, region, k) -> Int

Map a region-local storage slot to the tile-grid-aligned diagnostic slot.
"""
@inline _rank_index(A::AbstractTLRMatrix, region::TLRRegion, k::Int) =
    _rank_index(A, region_tile_coords(A, region, k)...)

@inline function tile_origin_coords(A, tile_i::Int, tile_j::Int)
    return ((tile_i - 1) * nominal_tile_size(A, 1) + 1,
        (tile_j - 1) * nominal_tile_size(A, 2) + 1)
end

@inline function _dense_tile_view(dense::AbstractMatrix, A, tile_i::Integer, tile_j::Integer)
    p0, q0 = tile_origin_coords(A, Int(tile_i), Int(tile_j))
    tm, tn = tile_size(A, Int(tile_i), Int(tile_j))
    return view(dense, p0:(p0+tm-1), q0:(q0+tn-1))
end

@inline _last_dim(dim::Int, nominal::Int) = (tail=dim % nominal; iszero(tail) ? nominal : tail)
