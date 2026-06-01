using LinearAlgebra

abstract type AbstractTileOrder end

struct TileColMajor <: AbstractTileOrder
    mt::Int
    nt::Int
end

struct TileRowMajor <: AbstractTileOrder
    mt::Int
    nt::Int
end

Base.size(order::AbstractTileOrder) = (order.mt, order.nt)

function (order::TileColMajor)(i::Integer, j::Integer)
    1 <= Int(i) <= order.mt || throw(BoundsError(order, (i, :)))
    1 <= Int(j) <= order.nt || throw(BoundsError(order, (:, j)))
    return Int(i) + (Int(j) - 1) * order.mt
end

function (order::TileRowMajor)(i::Integer, j::Integer)
    1 <= Int(i) <= order.mt || throw(BoundsError(order, (i, :)))
    1 <= Int(j) <= order.nt || throw(BoundsError(order, (:, j)))
    return Int(j) + (Int(i) - 1) * order.nt
end

tile_linear_index(order::AbstractTileOrder, i::Integer, j::Integer) = order(i, j)
