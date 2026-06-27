"""
    TileOrderStyle

Policy type selecting how a logical tile grid is linearized.
"""
abstract type TileOrderStyle end

"""Column-major traversal of the logical tile grid."""
struct TileColMajor <: TileOrderStyle end
"""Row-major traversal of the logical tile grid."""
struct TileRowMajor <: TileOrderStyle end

@inline _order_instance(::Type{O}) where {O<:TileOrderStyle} = O()
@inline _order_instance(order::O) where {O<:TileOrderStyle} = order

@inline _tile_stride(order, mt::Int, nt::Int) = _tile_stride(_order_instance(order), mt, nt)
@inline _tile_stride(::TileColMajor, mt::Int, nt::Int) = mt
@inline _tile_stride(::TileRowMajor, mt::Int, nt::Int) = nt

@inline _tile_coords(order, i::Int, j::Int) = _tile_coords(_order_instance(order), i, j)
@inline _tile_coords(::TileColMajor, i::Int, j::Int) = (i, j)
@inline _tile_coords(::TileRowMajor, i::Int, j::Int) = (j, i)

@inline _inverse_tile_coords(order, a::Int, b::Int) = _inverse_tile_coords(_order_instance(order), a, b)
@inline _inverse_tile_coords(::TileColMajor, a::Int, b::Int) = (a, b)
@inline _inverse_tile_coords(::TileRowMajor, a::Int, b::Int) = (b, a)

@inline function checkbounds_tile(mt::Int, nt::Int, i::Integer, j::Integer)
    1 <= i <= mt || throw(BoundsError((mt, nt), (i, :)))
    1 <= j <= nt || throw(BoundsError((mt, nt), (:, j)))
    return nothing
end

"""
    tile_linear_index(order, mt, nt, i, j)

Return the linear traversal index associated with logical tile `(i, j)`.
"""
@inline function tile_linear_index(order, mt::Integer, nt::Integer, i::Integer, j::Integer)
    checkbounds_tile(Int(mt), Int(nt), i, j)
    a, b = _tile_coords(order, Int(i), Int(j))
    return a + (b - 1) * _tile_stride(order, Int(mt), Int(nt))
end

"""
    inverse_tile_index(order, mt, nt, linear)

Return the logical tile coordinate corresponding to `linear`.
"""
@inline function inverse_tile_index(order, mt::Integer, nt::Integer, linear::Integer)
    mt_i, nt_i = Int(mt), Int(nt)
    1 <= linear <= mt_i * nt_i || throw(BoundsError((mt_i, nt_i), linear))
    stride = _tile_stride(order, mt_i, nt_i)
    b = ((Int(linear) - 1) ÷ stride) + 1
    a = Int(linear) - (b - 1) * stride
    return _inverse_tile_coords(order, a, b)
end

Base.show(io::IO, ::TileColMajor) = print(io, "TileColMajor()")
Base.show(io::IO, ::TileRowMajor) = print(io, "TileRowMajor()")
