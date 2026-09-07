"""
    TiledTriMixedPrec{T_Diag, T_OffDiag, T_Scale, M_Diag, M_OffDiag} <: AbstractMixedPrec{T_Diag}

A triangular matrix stored as a flat grid of tiles. The tile sizes come from the same
recursive splitting as `TriMixedPrec`, stopped at `threshold`; diagonal tiles are stored in
`T_Diag` and the off-diagonal tiles of the `uplo` triangle in `T_OffDiag`, one tile per grid
cell, each with its own scale factor of type `T_Scale`. The other triangle is not stored and
reads as zero.
"""
struct TiledTriMixedPrec{T_Diag<:AbstractFloat, T_OffDiag<:AbstractFloat, T_Scale<:AbstractFloat,
                         M_Diag<:AbstractMatrix{T_Diag}, M_OffDiag<:AbstractMatrix{T_OffDiag}} <:
       AbstractMixedPrec{T_Diag}
    diag::Vector{M_Diag}
    off::Matrix{Union{Nothing, M_OffDiag}}
    diag_scales::Vector{T_Scale}
    off_scales::Matrix{T_Scale}
    offsets::Vector{Int}
    uplo::Char
    n::Int
end

"""
    TiledTriMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType}, threshold, scale_type=Float32)

Constructs a `TiledTriMixedPrec` representation of the triangular matrix `A`.

`A` is split as `TriMixedPrec` splits it until a diagonal block has at most `threshold` rows;
those blocks set the tile sizes. As in the other containers, off-diagonal tiles are stored in
`precisions[1]` and diagonal tiles in `precisions[end]`, which is also `T_Diag`; `precisions`
holds one or two types. Only the `uplo` triangle of `A` is read. A tile whose largest
magnitude exceeds `floatmax` of its storage type is divided by a scale factor, clamped and
stored with that factor; a tile beyond `floatmax(scale_type) * floatmax` of its storage type
throws an `ArgumentError`. `precisions` may contain `Float8_E5M2` (q52), `BFloat16`,
`Float16`, `Float32` and `Float64`, and `scale_type` be `Float16`, `Float32` or `Float64`. `A` must
be square, non-empty and finite, `uplo` 'L' or 'U' and `threshold` at least 1.
"""
function TiledTriMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType},
    threshold::Integer,
    scale_type::Type{T_Scale}=Float32
) where {T_Scale<:AbstractFloat}
    _check_args(A, precisions, T_Scale)
    length(precisions) <= 2 ||
        throw(ArgumentError("precisions must hold one or two types: off-diagonal, then diagonal"))
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U'"))
    threshold >= 1 || throw(ArgumentError("threshold must be at least 1"))
    T_OffDiag, T_Diag = precisions[1], precisions[end]
    sizes = Int[]
    _tile_sizes!(sizes, size(A, 1), Int(threshold))
    offsets = cumsum([1; sizes])
    t = length(sizes)
    tile(k) = offsets[k]:offsets[k+1]-1
    M_Diag = typeof(similar(A, T_Diag, 0, 0))
    M_OffDiag = typeof(similar(A, T_OffDiag, 0, 0))
    diag = Vector{M_Diag}(undef, t)
    diag_scales = Vector{T_Scale}(undef, t)
    for k in 1:t
        block = view(A, tile(k), tile(k))
        diag[k], diag_scales[k] = _store(uplo == 'L' ? tril(block) : triu(block), T_Diag, T_Scale)
    end
    off = Matrix{Union{Nothing, M_OffDiag}}(nothing, t, t)
    off_scales = ones(T_Scale, t, t)
    for r in 1:t, c in 1:t
        r != c && _stored(r, c, uplo) || continue
        off[r, c], off_scales[r, c] = _store(view(A, tile(r), tile(c)), T_OffDiag, T_Scale)
    end
    return TiledTriMixedPrec{T_Diag, T_OffDiag, T_Scale, M_Diag, M_OffDiag}(
        diag, off, diag_scales, off_scales, offsets, uplo, size(A, 1)
    )
end

function _tile_sizes!(sizes::Vector{Int}, n::Int, threshold::Int)
    n <= threshold && return push!(sizes, n)
    mid = _rec_split(n)
    _tile_sizes!(sizes, mid, threshold)
    _tile_sizes!(sizes, n - mid, threshold)
    return sizes
end

Base.size(A::TiledTriMixedPrec) = (A.n, A.n)

Base.@propagate_inbounds function Base.getindex(A::TiledTriMixedPrec{T_Diag}, i::Int, j::Int) where {T_Diag}
    @boundscheck checkbounds(A, i, j)
    _stored(i, j, A.uplo) || return zero(T_Diag)
    r, c = searchsortedlast(A.offsets, i), searchsortedlast(A.offsets, j)
    li, lj = i - A.offsets[r] + 1, j - A.offsets[c] + 1
    r == c && return _unscaled(A.diag[r][li, lj], A.diag_scales[r], T_Diag)
    return _unscaled(A.off[r, c][li, lj], A.off_scales[r, c], T_Diag)
end

"""
    reconstruct_matrix(A::TiledTriMixedPrec{T_Diag})

Copies the tiled matrix back into one dense array with element type `T_Diag`, of the same
array type as the stored tiles, with each tile's scale applied and zeros in the other
triangle.
"""
function reconstruct_matrix(A::TiledTriMixedPrec{T_Diag}) where {T_Diag}
    C = fill!(similar(first(A.diag), T_Diag, A.n, A.n), zero(T_Diag))
    t = length(A.diag)
    for r in 1:t, c in 1:t
        rows, cols = A.offsets[r]:A.offsets[r+1]-1, A.offsets[c]:A.offsets[c+1]-1
        if r == c
            @views C[rows, cols] .= _unscaled.(A.diag[r], A.diag_scales[r], T_Diag)
        elseif A.off[r, c] !== nothing
            @views C[rows, cols] .= _unscaled.(A.off[r, c], A.off_scales[r, c], T_Diag)
        end
    end
    return C
end
