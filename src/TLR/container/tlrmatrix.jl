"""
    TLRMatrix

Tile Low-Rank (TLR) matrix representation.

A `TLRMatrix` stores a dense matrix in a tile low-rank format. The matrix is
partitioned into square tiles of size `b × b` and each tile is represented
either as a low-rank factorization or as a dense block.

Given a tiled matrix

```math
A =
\\begin{bmatrix}
A_{11} & A_{12} & \\cdots & A_{1n_t} \\\\
A_{21} & A_{22} & \\cdots & A_{2n_t} \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
A_{m_t1} & A_{m_t2} & \\cdots & A_{m_tn_t}
\\end{bmatrix},
```

each off-diagonal tile is approximated by a rank-`r` factorization

```math
A_{ij} \\approx U_{ij} V_{ij}^{\\mathsf T},
\\qquad i \\neq j,
```

where `U_{ij} ∈ ℝ^{b×r}` and `V_{ij} ∈ ℝ^{b×r}`. Diagonal tiles may be stored
densely or compressed depending on the value of `compress_diag`.

###Notes

`TLRMatrix` owns backend and geometry metadata plus a concrete storage
object that holds the representation.
"""
struct TLRMatrix{BackendT<:Backend,T,L<:TileMap,S<:AbstractTLRStorage{T}}
    backend::BackendT
    layout::L
    storage::S
    m::Int
    n::Int
end

Base.eltype(::Type{<:TLRMatrix{<:Any,T}}) where {T} = T
Base.eltype(A::TLRMatrix{<:Any,T}) where {T} = T
Base.size(A::TLRMatrix) = (A.m, A.n)
Base.size(A::TLRMatrix, d::Int) = size(A)[d]

"""Return the linear traversal index of logical tile `(i, j)`."""
@inline tile_linear_index(A::TLRMatrix, i::Integer, j::Integer) = tile_linear_index(A.layout.order, i, j)
"""Return the storage slot used by logical tile `(i, j)`."""
@inline tile_storage_index(A::TLRMatrix, i::Integer, j::Integer) = tile_storage_index(A.storage, A.layout, i, j)

"""
    TLRMatrix(backend, T, m, n, b, max_rank; kwargs...)

Allocate an empty TLR container with a `UVTileStorage` destination format.
"""
function TLRMatrix(
    backend::Backend,
    ::Type{T},
    m::Int,
    n::Int,
    b::Int,
    max_rank::Int;
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:TileOrder}=TileColMajor,
) where {T}
    m > 0 || throw(ArgumentError("m must be positive"))
    n > 0 || throw(ArgumentError("n must be positive"))
    b > 0 || throw(ArgumentError("b must be positive"))
    max_rank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))

    layout = TileMap(tile_order(cld(m, b), cld(n, b)), b, b, m, n)
    storage = allocate_storage(
        backend,
        T,
        layout,
        max_rank;
        compress_diag,
        rank_type,
    )
    return TLRMatrix{typeof(backend),T,typeof(layout),typeof(storage)}(backend, layout, storage, m, n)
end

"""
    TLRMatrix(A, blocksize, max_rank; kwargs...)

Allocate an empty TLR container on the same backend as dense matrix `A`.
"""
function TLRMatrix(
    A::AbstractMatrix{T},
    blocksize::Int,
    max_rank::Int;
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:TileOrder}=TileColMajor,
) where {T}
    return TLRMatrix(
        get_backend(A),
        T,
        size(A, 1),
        size(A, 2),
        blocksize,
        max_rank;
        compress_diag,
        rank_type,
        tile_order,
    )
end
