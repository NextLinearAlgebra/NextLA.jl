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
@inline function tile_linear_index(order, mt::Integer, nt::Integer, i::Integer, j::Integer)
    mt_i, nt_i = Int(mt), Int(nt)
    i_i, j_i = Int(i), Int(j)
    1 <= i_i <= mt_i || throw(BoundsError((mt_i, nt_i), (i_i, :)))
    1 <= j_i <= nt_i || throw(BoundsError((mt_i, nt_i), (:, j_i)))
    (order === TileColMajor || order isa TileColMajor) &&
        return i_i + (j_i - 1) * mt_i
    (order === TileRowMajor || order isa TileRowMajor) &&
        return j_i + (i_i - 1) * nt_i
    throw(ArgumentError("unsupported tile order $(typeof(order))"))
end

"""
    inverse_tile_index(order, mt, nt, linear)

Return the logical tile coordinate corresponding to `linear`.
"""
@inline function inverse_tile_index(order, mt::Integer, nt::Integer, linear::Integer)
    mt_i, nt_i = Int(mt), Int(nt)
    linear_i = Int(linear)
    1 <= linear_i <= mt_i * nt_i || throw(BoundsError((mt_i, nt_i), linear_i))
    k = linear_i - 1
    (order === TileColMajor || order isa TileColMajor) &&
        return (k % mt_i + 1, k ÷ mt_i + 1)
    (order === TileRowMajor || order isa TileRowMajor) &&
        return (k ÷ nt_i + 1, k % nt_i + 1)
    throw(ArgumentError("unsupported tile order $(typeof(order))"))
end
