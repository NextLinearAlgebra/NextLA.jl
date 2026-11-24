"""
    TileOrderStyle

Policy type selecting how a logical tile grid is linearized.
"""
abstract type TileOrderStyle end

"""
    TileColMajor()

Column-major traversal of the logical tile grid.

Pass it (or the type itself) as `outer_order` or `inner_order` to a
`CompressedFTLRMatrix` constructor to choose the tile order in which the packed `U`
or `V` factors are stored. The defaults are `outer_order=TileRowMajor` and
`inner_order=TileColMajor`; `TLRMatrix` accepts only these defaults.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V';

julia> F = CompressedFTLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true,
                                outer_order=TileColMajor);

julia> B = similar(A); uncompress!(B, F);

julia> norm(B - A) / norm(A) < 1e-8
true
```
"""
struct TileColMajor <: TileOrderStyle end
"""
    TileRowMajor()

Row-major traversal of the logical tile grid.

This is the default `outer_order`, and the only one `TLRMatrix` accepts. The
per-tile vectors `ranks(A)` and `residuals(A)` follow the outer order, so
with `TileRowMajor` tile `(i, j)` is entry `(i - 1) * nt + j`, where `nt` is the
number of tile columns.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> using Random; Random.seed!(1);  # compression is randomized

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> A[1:128, 129:256] .= 0;  # tile (1, 2) becomes rank 0, tile (2, 1) stays rank 8

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> r = ranks(C);  # tile (i, j) is entry (i - 1) * 8 + j

julia> r[2] == 0 && r[9] == 8
true
```
"""
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

"""
    AbstractTLRMatrix{T}

Shared interface for tile low-rank matrix containers. Concrete subtypes decide
how diagonal tiles are represented and how low-rank tile indices are mapped to
storage.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8);

julia> C = TLRMatrix(U * V' + n * I, 128; maxrank=32, tol=1e-8, rel=true);

julia> F = CompressedFTLRMatrix(U * V', 128; maxrank=32, tol=1e-8, rel=true);

julia> C isa AbstractTLRMatrix{Float64} && F isa AbstractTLRMatrix{Float64}
true
```
"""
abstract type AbstractTLRMatrix{T} end

"""Zero-copy container view representing the transpose of a TLR matrix."""
struct TransposeTLRMatrix{T,A<:AbstractTLRMatrix{T}} <: AbstractTLRMatrix{T}
    parent::A
end

Base.parent(A::TransposeTLRMatrix) = getfield(A, :parent)
Base.transpose(A::AbstractTLRMatrix) = TransposeTLRMatrix(A)
Base.transpose(A::TransposeTLRMatrix) = parent(A)

Base.eltype(::Type{<:AbstractTLRMatrix{T}}) where {T} = T
Base.eltype(::AbstractTLRMatrix{T}) where {T} = T
Base.size(A::AbstractTLRMatrix) = (A.m, A.n)
Base.size(A::AbstractTLRMatrix, d::Int) = size(A)[d]
Base.size(A::TransposeTLRMatrix) = reverse(size(parent(A)))

"""
    nominal_tile_size(A) -> (bm, bn)
    nominal_tile_size(A, axis) -> Int

Return the full tile size `(bm, bn)` that `A` was built with, or one entry of it
(`axis = 1` for rows, `2` for columns). Tiles in the last tile row or column can be
smaller; see [`tail_tile_size`](@ref) and [`tile_size`](@ref).

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8);

julia> C = TLRMatrix(U * V' + n * I, (128, 256); maxrank=32, tol=1e-8, rel=true);

julia> nominal_tile_size(C) == (128, 256) && nominal_tile_size(C, 2) == 256
true
```
"""
@inline nominal_tile_size(A::AbstractTLRMatrix) = A.nominal_tile_size
@inline nominal_tile_size(A::TransposeTLRMatrix) = reverse(nominal_tile_size(parent(A)))
@inline nominal_tile_size(A::AbstractTLRMatrix, axis::Int) =
    nominal_tile_size(A)[axis]

"""
    tail_tile_size(A) -> (tail_m, tail_n)
    tail_tile_size(A, axis) -> Int

Return the size of the partial last tile row and column, `(m % bm, n % bn)` for an
`m × n` matrix with nominal tile size `(bm, bn)`, or one entry of it. A zero entry
means the tiles divide that dimension evenly and the last tile is full size.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1000; U, V = randn(n, 8), randn(n, 8);  # 1000 = 7 * 128 + 104

julia> C = TLRMatrix(U * V' + n * I, 128; maxrank=32, tol=1e-8, rel=true);

julia> tail_tile_size(C) == (104, 104) && tail_tile_size(C, 1) == 104
true
```
"""
@inline tail_tile_size(A::AbstractTLRMatrix) = A.tail_tile_size
@inline tail_tile_size(A::TransposeTLRMatrix) = reverse(tail_tile_size(parent(A)))
@inline tail_tile_size(A::AbstractTLRMatrix, axis::Int) =
    tail_tile_size(A)[axis]

"""
    grid_size(A) -> (n_tile_rows, n_tile_cols)

Full tile grid including partial boundary tiles: `(n_tile_rows, n_tile_cols)`.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8);

julia> C = TLRMatrix(U * V' + n * I, (128, 256); maxrank=32, tol=1e-8, rel=true);

julia> grid_size(C) == (8, 4)
true
```
"""
@inline grid_size(A::AbstractTLRMatrix) =
    (cld(size(A, 1), nominal_tile_size(A, 1)),
     cld(size(A, 2), nominal_tile_size(A, 2)))

"""
    regular_grid_size(A) -> (q_m, q_n)

Sub-grid of full-size tiles, `(⌊m/bm⌋, ⌊n/bn⌋)`. It excludes any partial
boundary row or column and equals `grid_size` when the matrix tiles evenly.
"""
@inline regular_grid_size(A::AbstractTLRMatrix) =
    (fld(size(A, 1), nominal_tile_size(A, 1)),
     fld(size(A, 2), nominal_tile_size(A, 2)))

"""
    tile_size(A, i, j) -> (rows, cols)

Return the size of tile `(i, j)` of `A`: the nominal tile size, except that tiles
in the last tile row or column take the partial size from `tail_tile_size(A)` when
it is nonzero. The tile indices are not bounds-checked.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1000; U, V = randn(n, 8), randn(n, 8);  # 1000 = 7 * 128 + 104

julia> C = TLRMatrix(U * V' + n * I, 128; maxrank=32, tol=1e-8, rel=true);

julia> tile_size(C, 1, 8) == (128, 104) && tile_size(C, 8, 8) == (104, 104)
true
```
"""
@inline function tile_size(A, tile_i::Int, tile_j::Int)
    mt, nt = grid_size(A)
    bm, bn = nominal_tile_size(A)
    tail_m, tail_n = tail_tile_size(A)
    row_size = tile_i == mt && !iszero(tail_m) ? tail_m : bm
    col_size = tile_j == nt && !iszero(tail_n) ? tail_n : bn
    return row_size, col_size
end

"""
    maxrank(A) -> Int

Return the largest logical rank of any tile stored in `A`. For a matrix built by
compression this is the rank actually reached, which can be smaller than the
`maxrank` cap passed to the constructor.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> using Random; Random.seed!(1);  # compression is randomized

julia> n = 1024; U, V = randn(n, 8), randn(n, 8);

julia> C = TLRMatrix(U * V' + n * I, 128; maxrank=32, tol=1e-8, rel=true);

julia> maxrank(C) == 8  # the cap was 32; the largest tile rank found is 8
true
```
"""
@inline maxrank(A::AbstractTLRMatrix) = A.maxrank
@inline maxrank(A::TransposeTLRMatrix) = maxrank(parent(A))
"""
    ranks(A) -> Vector{Int}

Return the logical rank of every tile of the full tile grid, one entry per tile,
ordered by the outer factor's tile order (row-major for a `TLRMatrix`, see
[`TileRowMajor`](@ref)). Diagonal tiles of a `TLRMatrix` are stored densely and
have rank 0. The vector is the matrix's own storage, not a copy.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> using Random; Random.seed!(1);  # compression is randomized

julia> n = 1024; U, V = randn(n, 8), randn(n, 8);

julia> C = TLRMatrix(U * V' + n * I, 128; maxrank=32, tol=1e-8, rel=true);

julia> r = ranks(C); length(r) == 64 && count(iszero, r) == 8 && count(==(8), r) == 56
true
```
"""
@inline ranks(A::AbstractTLRMatrix) = A.ranks
@inline ranks(A::TransposeTLRMatrix) = ranks(parent(A))
"""
    residuals(A) -> Vector{Float64}

Return the estimated compression error of every tile, in the Frobenius norm and
in absolute terms, in the same tile order as [`ranks`](@ref). Entries are zero for
tiles that were not compressed: diagonal tiles of a `TLRMatrix` and every tile of
a matrix allocated from known ranks. The vector is the matrix's own storage.
For the result of `NextLA.gemm`, a tile that stopped at `maxrank` also includes an
estimate of what the rank cap left out.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> length(residuals(C)) == 64 && maximum(residuals(C)) < 1e-8 * norm(A)
true
```
"""
@inline residuals(A::AbstractTLRMatrix) = A.resid
@inline residuals(A::TransposeTLRMatrix) = residuals(parent(A))
@inline KernelAbstractions.get_backend(A::AbstractTLRMatrix) = A.backend
@inline KernelAbstractions.get_backend(A::TransposeTLRMatrix) = get_backend(parent(A))
@inline tile_order(A::AbstractTLRMatrix) = A.order
@inline tile_order(A::TransposeTLRMatrix) = transpose(tile_order(parent(A)))

"""
    tile_origin_coords(A, i, j) -> (row, col)

Return the matrix row and column of the first (top-left) entry of tile `(i, j)`,
`((i - 1) * bm + 1, (j - 1) * bn + 1)` for nominal tile size `(bm, bn)`. The tile
indices are not bounds-checked.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8);

julia> C = TLRMatrix(U * V' + n * I, 128; maxrank=32, tol=1e-8, rel=true);

julia> tile_origin_coords(C, 2, 3) == (129, 257)
true
```
"""
@inline function tile_origin_coords(A, tile_i::Int, tile_j::Int)
    return ((tile_i - 1) * nominal_tile_size(A, 1) + 1,
        (tile_j - 1) * nominal_tile_size(A, 2) + 1)
end

@inline function dense_tile_view(dense::AbstractMatrix, A, tile_i::Int, tile_j::Int)
    p0, q0 = tile_origin_coords(A, tile_i, tile_j)
    tm, tn = tile_size(A, tile_i, tile_j)
    return view(dense, p0:(p0+tm-1), q0:(q0+tn-1))
end

"""Logical row/column range of one TLR tile along `axis`."""
@inline function tile_axis_range(A::AbstractTLRMatrix, tile::Int, axis::Int)
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    extent = tile_size(A, axis == 1 ? tile : 1, axis == 2 ? tile : 1)[axis]
    return first:(first + extent - 1)
end
