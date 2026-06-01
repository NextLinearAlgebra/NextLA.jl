struct TileFactorBuffer{T,A<:AbstractArray{T,3},O<:AbstractTileOrder}
    data::A
    order::O
end

Base.size(buffer::TileFactorBuffer) = size(buffer.order)
Base.axes(buffer::TileFactorBuffer) = (Base.OneTo(buffer.order.mt), Base.OneTo(buffer.order.nt))

function Base.getindex(buffer::TileFactorBuffer, i::Integer, j::Integer)
    return @view buffer.data[:, :, buffer.order(i, j)]
end
