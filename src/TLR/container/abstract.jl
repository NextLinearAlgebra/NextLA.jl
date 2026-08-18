"""
    AbstractTLRMatrix{T}

Shared interface for tile low-rank matrix containers. Concrete subtypes decide
how diagonal tiles are represented and how low-rank tile indices are mapped to
storage.
"""
abstract type AbstractTLRMatrix{T} end

Base.eltype(::Type{<:AbstractTLRMatrix{T}}) where {T} = T
Base.eltype(::AbstractTLRMatrix{T}) where {T} = T
Base.size(A::AbstractTLRMatrix) = (A.m, A.n)
Base.size(A::AbstractTLRMatrix, d::Int) = size(A)[d]

@inline nominal_tile_size(A::AbstractTLRMatrix) = A.nominal_tile_size
@inline nominal_tile_size(A::AbstractTLRMatrix, axis::Integer) =
    nominal_tile_size(A)[Int(axis)]

@inline tail_tile_size(A::AbstractTLRMatrix) = A.tail_tile_size
@inline tail_tile_size(A::AbstractTLRMatrix, axis::Integer) =
    tail_tile_size(A)[Int(axis)]

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
@inline ranks(A::AbstractTLRMatrix) = A.ranks
@inline residuals(A::AbstractTLRMatrix) = A.resid
@inline KernelAbstractions.get_backend(A::AbstractTLRMatrix) = A.backend
@inline tile_order(A::AbstractTLRMatrix) = A.order

@inline function tile_origin_coords(A, tile_i::Int, tile_j::Int)
    return ((tile_i - 1) * nominal_tile_size(A, 1) + 1,
        (tile_j - 1) * nominal_tile_size(A, 2) + 1)
end

@inline function _dense_tile_view(dense::AbstractMatrix, A, tile_i::Integer, tile_j::Integer)
    p0, q0 = tile_origin_coords(A, Int(tile_i), Int(tile_j))
    tm, tn = tile_size(A, Int(tile_i), Int(tile_j))
    return view(dense, p0:(p0+tm-1), q0:(q0+tn-1))
end
