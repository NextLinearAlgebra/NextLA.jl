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
