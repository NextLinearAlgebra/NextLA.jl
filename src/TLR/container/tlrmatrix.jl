"""
    TLRMatrix

Tile Low-Rank (TLR) matrix representation.

A `TLRMatrix` stores a dense matrix in a tile low-rank format.  Each
off-diagonal tile `A_{ij}` is approximated by a rank-`r` factorization

```math
A_{ij} \\approx U_{ij} V_{ij}^{\\mathsf T},
\\qquad i \\neq j,
```

where `U_{ij} \\in \\mathbb{R}^{b \\times r}` and
`V_{ij} \\in \\mathbb{R}^{b \\times r}`.

`TLRMatrix` is a **mutable** container: its `storage` field can be swapped
in-place (e.g. by [`compact!`](@ref)) to change the memory layout while
keeping the same geometry and backend.  The `backend`, `layout`, `m`, and `n`
fields are declared `const` and cannot be changed after construction.
"""
mutable struct TLRMatrix{BackendT<:Backend, T, L<:TileMap}
    const backend::BackendT
    const layout::L
    storage::AbstractTLRStorage{T}        # swappable: UniformTileStorage ↔ CompactTileStorage
    const m::Int
    const n::Int
end

Base.eltype(::Type{<:TLRMatrix{<:Any,T}}) where {T} = T
Base.eltype(A::TLRMatrix{<:Any,T}) where {T} = T
Base.size(A::TLRMatrix) = (A.m, A.n)
Base.size(A::TLRMatrix, d::Int) = size(A)[d]

"""Return the linear traversal index of logical tile `(i, j)`."""
@inline tile_linear_index(A::TLRMatrix, i::Integer, j::Integer) =
    tile_linear_index(A.layout.order, i, j)

"""Return the storage slot used by logical tile `(i, j)`."""
@inline tile_storage_index(A::TLRMatrix, i::Integer, j::Integer) =
    tile_storage_index(A.storage, A.layout, i, j)

"""
    TLRMatrix(backend, T, m, n, b, max_rank; kwargs...)

Allocate an empty TLR container with padded `UniformTileStorage`.
"""
function TLRMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, b::Int, max_rank::Int;
    compress_diag::Bool = false,
    rank_type::Type{<:Integer} = Int32,
    tile_order::Type{<:TileOrder} = TileColMajor,
) where {T}
    m > 0        || throw(ArgumentError("m must be positive"))
    n > 0        || throw(ArgumentError("n must be positive"))
    b > 0        || throw(ArgumentError("b must be positive"))
    max_rank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))

    layout  = TileMap(tile_order(cld(m, b), cld(n, b)), b, b, m, n)
    storage = allocate_storage(backend, T, layout, max_rank; compress_diag, rank_type)
    return TLRMatrix{typeof(backend), T, typeof(layout)}(backend, layout, storage, m, n)
end

"""
    TLRMatrix(A, blocksize, max_rank; kwargs...)

Allocate an empty TLR container on the same backend as dense matrix `A`.
"""
function TLRMatrix(
    A::AbstractMatrix{T}, blocksize::Int, max_rank::Int;
    compress_diag::Bool = false,
    rank_type::Type{<:Integer} = Int32,
    tile_order::Type{<:TileOrder} = TileColMajor,
) where {T}
    return TLRMatrix(get_backend(A), T, size(A, 1), size(A, 2), blocksize, max_rank;
                     compress_diag, rank_type, tile_order)
end
