"""
    AbstractTLRMatrix{BackendT, T, OrderT}

Shared interface for tile low-rank matrix containers. Concrete subtypes decide
how diagonal tiles are represented and how low-rank tile indices are mapped to
storage.
"""
abstract type AbstractTLRMatrix{BackendT<:Backend,T,OrderT<:TileOrderStyle} end

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
