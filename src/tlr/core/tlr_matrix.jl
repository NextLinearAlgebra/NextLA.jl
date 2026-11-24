"""
    TLRMatrix

Dense-diagonal tile-low-rank matrix. `offdiag` is a finalized full-grid
[`CompressedFTLRMatrix`](@ref); its diagonal ranks are zero. Dense diagonal
tiles are stored separately in `D`, with an optional ragged final tile in
`D_corner`.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> B = similar(A); uncompress!(B, C);

julia> offdiagonal(C) isa CompressedFTLRMatrix && norm(B - A) / norm(A) < 1e-8
true
```
"""
struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},
                 OffdiagT<:CompressedFTLRMatrix{BackendT,T}} <: AbstractTLRMatrix{T}
    offdiag::OffdiagT
    D::Arr3T
    D_corner::Arr3T
end

"""
    offdiagonal(A::TLRMatrix) -> CompressedFTLRMatrix

Return the `CompressedFTLRMatrix` that holds the off-diagonal tiles of `A`. It
covers the full tile grid, with rank 0 on the diagonal tiles, and is `A`'s own
storage, not a copy.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> B = similar(A); uncompress!(B, offdiagonal(C));

julia> iszero(B[1:128, 1:128]) && norm(B[1:128, 129:n] - A[1:128, 129:n]) < 1e-8 * norm(A)
true
```
"""
@inline offdiagonal(A::TLRMatrix) = A.offdiag
@inline offdiagonal(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    transpose(offdiagonal(parent(A)))
Base.size(A::TLRMatrix) = size(A.offdiag)
@inline nominal_tile_size(A::TLRMatrix) = nominal_tile_size(A.offdiag)
@inline tail_tile_size(A::TLRMatrix) = tail_tile_size(A.offdiag)
@inline KernelAbstractions.get_backend(A::TLRMatrix) = get_backend(A.offdiag)
@inline tile_order(A::TLRMatrix) = tile_order(A.offdiag)
@inline ranks(A::TLRMatrix) = ranks(A.offdiag)
@inline residuals(A::TLRMatrix) = residuals(A.offdiag)
@inline maxrank(A::TLRMatrix) = maxrank(A.offdiag)
@inline rank_multiple(A::TLRMatrix) = rank_multiple(A.offdiag)
@inline maximum_storage_rank(A::TLRMatrix) = maximum_storage_rank(A.offdiag)

"""
    ndiag_tiles(A::TLRMatrix) -> Int

Return the number of diagonal tiles of `A`, `minimum(grid_size(A))`. These are
the tiles stored densely, including a ragged last one.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> m, n = 1024, 512; U, V = randn(m, 8), randn(n, 8);

julia> C = TLRMatrix(U * V', 128; maxrank=32, tol=1e-8, rel=true);

julia> grid_size(C) == (8, 4) && ndiag_tiles(C) == 4
true
```
"""
@inline ndiag_tiles(A::TLRMatrix) = min(grid_size(A)...)
@inline ndiag_tiles(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    ndiag_tiles(parent(A))
"""
    dense_diag(A::TLRMatrix) -> AbstractArray{T,3}

Return the dense diagonal tiles of `A` as a `bm × bn × k` array, where `(bm, bn)`
is the nominal tile size; slice `[:, :, t]` is diagonal tile `(t, t)`. Only
full-size diagonal tiles are included; a smaller last diagonal tile is in
[`dense_diag_corner`](@ref). The array is `A`'s own storage, not a copy.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> D = dense_diag(C); size(D) == (128, 128, 8) && D[:, :, 2] == A[129:256, 129:256]
true
```
"""
@inline dense_diag(A::TLRMatrix) = A.D
"""
    dense_diag_corner(A::TLRMatrix) -> AbstractArray{T,3}

Return the last diagonal tile of `A` when it is smaller than the nominal tile
size, as a `tm × tn × 1` array. When every diagonal tile is full size the array
has third dimension 0. The array is `A`'s own storage, not a copy.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1000; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> size(dense_diag(C), 3) == 7 && dense_diag_corner(C)[:, :, 1] == A[897:1000, 897:1000]
true
```
"""
@inline dense_diag_corner(A::TLRMatrix) = A.D_corner
@inline dense_diag(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    PermutedDimsArray(dense_diag(parent(A)), (2, 1, 3))
@inline dense_diag_corner(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}) =
    PermutedDimsArray(dense_diag_corner(parent(A)), (2, 1, 3))

@inline function diag_tile_view(A::TLRMatrix, tile_k::Int)
    1 <= tile_k <= ndiag_tiles(A) || throw(BoundsError(1:ndiag_tiles(A), tile_k))
    if tile_k <= size(A.D, 3)
        return view(A.D, :, :, tile_k)
    end
    size(A.D_corner, 3) != 0 || throw(BoundsError(1:size(A.D, 3), tile_k))
    return view(A.D_corner, :, :, 1)
end

@inline diag_tile_view(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}, tile_k::Int) =
    transpose(diag_tile_view(parent(A), tile_k))

"""
    get_factors(A::TLRMatrix, i, j) -> (U, V)
    get_factors(A::CompressedFTLRMatrix, i, j) -> (U, V)

Return an exact-rank factor pair for an off-diagonal tile.

Tile `(i, j)` is approximated by `U * V'`; `U` has the tile's row count and `V` its
column count, and both have exactly `r` columns, the tile's logical rank (padding
from `rank_multiple` is not included). They are views into `A`'s packed storage.
For a `TLRMatrix`, asking for a diagonal tile throws an `ArgumentError`; use
`dense_diag` instead.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> X, Y = get_factors(C, 1, 2); T12 = A[1:128, 129:256];

julia> size(X) == size(Y) == (128, 8) && norm(X * Y' - T12) < 1e-8 * norm(T12)
true
```
"""
@inline function get_factors(A::TLRMatrix, i::Int, j::Int)
    i == j && throw(ArgumentError("tile ($i, $j) is diagonal and stored densely"))
    return get_factors(A.offdiag, i, j)
end
@inline get_factors(A::TransposeTLRMatrix{<:Any,<:TLRMatrix}, i::Int, j::Int) =
    get_factors(offdiagonal(A), i, j)

"""
    TLRMatrix(offdiag::CompressedFTLRMatrix)

Wrap finalized full-grid compressed off-diagonal storage and allocate a
separate dense diagonal. Diagonal ranks in `offdiag` must be zero.

`offdiag` is used as is, not copied, and must have the default factor orders
(`outer_order=TileRowMajor`, `inner_order=TileColMajor`); otherwise an
`ArgumentError` is thrown. The new dense diagonal tiles are zero.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; U, V = randn(n, 8), randn(n, 8); A = U * V' + n * I;

julia> C = TLRMatrix(A, 128; maxrank=32, tol=1e-8, rel=true);

julia> C0 = TLRMatrix(offdiagonal(C));  # shares the off-diagonal factors, zero diagonal

julia> dense_diag(C0) .= dense_diag(C); B = similar(A); uncompress!(B, C0);

julia> norm(B - A) / norm(A) < 1e-8
true
```
"""
function TLRMatrix(offdiag::CompressedFTLRMatrix{BackendT,T}) where {BackendT,T}
    # layout and zero-diagonal-rank checks
    compressed_ftlr_outer_order(offdiag) isa TileRowMajor || throw(ArgumentError(
        "TLRMatrix requires row-major outer-factor packing"))
    compressed_ftlr_inner_order(offdiag) isa TileColMajor || throw(ArgumentError(
        "TLRMatrix requires column-major inner-factor packing"))
    @inbounds for k in 1:min(grid_size(offdiag)...)
        compressed_ftlr_rank(offdiag, k, k) == 0 || throw(ArgumentError(
            "TLRMatrix off-diagonal storage requires rank zero at ($k, $k)"))
    end

    # dense-diagonal sizing, with an optional ragged corner tile
    bm, bn = nominal_tile_size(offdiag)
    mt, nt = grid_size(offdiag)
    n_diag = min(mt, nt)
    tail_m, tail_n = tail_tile_size(offdiag)
    corner_tm = n_diag == mt && !iszero(tail_m) ? tail_m : bm
    corner_tn = n_diag == nt && !iszero(tail_n) ? tail_n : bn
    has_corner = n_diag > 0 && (corner_tm != bm || corner_tn != bn)
    backend = get_backend(offdiag)
    D = zeros(backend, T, bm, bn, n_diag - has_corner)
    D_corner = zeros(
        backend, T, max(corner_tm, 1), max(corner_tn, 1), has_corner ? 1 : 0)

    return TLRMatrix{BackendT,T,typeof(D),typeof(offdiag)}(offdiag, D, D_corner)
end

"""
    TLRMatrix(backend, T, m, n, tile_size, ranks; rank_multiple=0)

Allocate a finalized dense-diagonal TLR matrix from known off-diagonal ranks.
The supplied full-grid rank matrix must contain zeros on its diagonal.

`backend` is a KernelAbstractions backend such as `CPU()`, `T` the element type,
`m × n` the matrix size and `tile_size` a tuple `(bm, bn)` or one integer for square
tiles. `ranks` is a `cld(m, bm) × cld(n, bn)` integer matrix. All factors and
diagonal tiles start as zeros, ready to be filled in place.

# Examples
```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> backend = NextLA.KernelAbstractions.CPU();

julia> r = [i == j ? 0 : 4 for i in 1:8, j in 1:8];

julia> Z = TLRMatrix(backend, Float64, 1024, 1024, (128, 128), r);

julia> X, Y = get_factors(Z, 1, 2);

julia> size(X) == (128, 4) && iszero(X) && iszero(dense_diag(Z)) && maxrank(Z) == 4
true
```
"""
function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int,
    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{Int};
    outer_order=TileRowMajor, inner_order=TileColMajor,
    rank_multiple::Int=0,
) where {T}
    (outer_order === TileRowMajor || outer_order isa TileRowMajor) || throw(ArgumentError(
        "TLRMatrix requires outer_order=TileRowMajor"))
    (inner_order === TileColMajor || inner_order isa TileColMajor) || throw(ArgumentError(
        "TLRMatrix requires inner_order=TileColMajor"))
    offdiag = CompressedFTLRMatrix(
        backend, T, m, n, tile_size, ranks_in;
        outer_order, inner_order, rank_multiple)
    return TLRMatrix(offdiag)
end

function TLRMatrix(
    backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
    ranks_in::AbstractMatrix{Int}; kwargs...,
) where {T}
    return TLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end
