using LinearAlgebra

"""
    TileOrderStyle

Policy type selecting how a logical tile grid is linearized.
"""
abstract type TileOrderStyle end

"""Column-major traversal of the logical tile grid."""
struct ColMajor <: TileOrderStyle end
"""Row-major traversal of the logical tile grid."""
struct RowMajor <: TileOrderStyle end

"""
    TileOrder{S}(mt, nt)

Map logical tile coordinates `(i, j)` to a linear traversal order over an
`mt × nt` tile grid.
"""
struct TileOrder{S<:TileOrderStyle}
    mt::Int
    nt::Int

    function TileOrder{S}(mt::Integer, nt::Integer) where {S<:TileOrderStyle}
        mt > 0 || throw(ArgumentError("mt must be positive"))
        nt > 0 || throw(ArgumentError("nt must be positive"))
        new{S}(Int(mt), Int(nt))
    end
end

const TileColMajor = TileOrder{ColMajor}
const TileRowMajor = TileOrder{RowMajor}

"""Return the logical tile-grid dimensions `(mt, nt)`."""
Base.size(order::TileOrder) = (order.mt, order.nt)

"""Return the stride used by the linear tile traversal."""
@inline tile_stride(order::TileOrder{ColMajor}) = order.mt
@inline tile_stride(order::TileOrder{RowMajor}) = order.nt

"""Map logical tile coordinates into the order-specific flattened coordinates."""
@inline tile_coords(::TileOrder{ColMajor}, i::Int, j::Int) = (i, j)
@inline tile_coords(::TileOrder{RowMajor}, i::Int, j::Int) = (j, i)

"""Inverse of [`tile_coords`](@ref)."""
@inline inverse_tile_coords(::TileOrder{ColMajor}, a::Int, b::Int) = (a, b)
@inline inverse_tile_coords(::TileOrder{RowMajor}, a::Int, b::Int) = (b, a)

@inline function checkbounds_tile(order::TileOrder, i::Integer, j::Integer)
    1 <= i <= order.mt || throw(BoundsError(order, (i, :)))
    1 <= j <= order.nt || throw(BoundsError(order, (:, j)))
    return nothing
end

"""
    tile_linear_index(order, i, j)

Return the linear traversal index associated with logical tile `(i, j)`.
"""
@inline function tile_linear_index(order::TileOrder, i::Integer, j::Integer)
    checkbounds_tile(order, i, j)
    a, b = tile_coords(order, Int(i), Int(j))
    return a + (b - 1) * tile_stride(order)
end

@inline (order::TileOrder)(i::Integer, j::Integer) = tile_linear_index(order, i, j)

"""
    inverse_tile_index(order, linear)

Return the logical tile coordinate corresponding to `linear`.
"""
@inline function inverse_tile_index(order::TileOrder, linear::Integer)
    ntiles = order.mt * order.nt
    1 <= linear <= ntiles || throw(BoundsError(order, linear))

    stride = tile_stride(order)
    b = ((Int(linear) - 1) ÷ stride) + 1
    a = Int(linear) - (b - 1) * stride
    return inverse_tile_coords(order, a, b)
end

function Base.show(io::IO, order::TileOrder{ColMajor})
    print(io, "TileColMajor(", order.mt, ", ", order.nt, ")")
end

function Base.show(io::IO, order::TileOrder{RowMajor})
    print(io, "TileRowMajor(", order.mt, ", ", order.nt, ")")
end
