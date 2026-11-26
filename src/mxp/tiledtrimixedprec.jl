"""
    TiledTriMixedPrec{T_Diag, T_OffDiag, M_Diag, M_OffDiag} <: AbstractMixedPrec{T_Diag}

A triangular matrix stored as a flat grid of tiles. The tile sizes come from the same
recursive splitting as `TriMixedPrec`, stopped at `threshold`; diagonal tiles are stored in
`T_Diag` and the off-diagonal tiles of the `uplo` triangle in `T_OffDiag`, one tile per grid
cell, each with its own `Float32` scale factor. The other triangle is not stored and reads as
zero.
"""
struct TiledTriMixedPrec{T_Diag<:AbstractFloat, T_OffDiag<:AbstractFloat,
                         M_Diag<:AbstractMatrix{T_Diag}, M_OffDiag<:AbstractMatrix{T_OffDiag}} <:
       AbstractMixedPrec{T_Diag}
    diag::Vector{M_Diag}
    off::Matrix{Union{Nothing, M_OffDiag}}
    diag_scales::Vector{Float32}
    off_scales::Matrix{Float32}
    offsets::Vector{Int}
    uplo::Char
    n::Int
end

"""
    TiledTriMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType},
                      threshold) -> TiledTriMixedPrec

Constructs a `TiledTriMixedPrec` representation of the triangular matrix `A`.

`A` is split as `TriMixedPrec` splits it until a diagonal block has at most `threshold` rows;
those blocks set the tile sizes. As in the other containers, off-diagonal tiles are stored in
`precisions[1]` and diagonal tiles in `precisions[end]`, which is also `T_Diag`; `precisions`
holds one or two types. Only the `uplo` triangle of `A` is read. A tile whose largest
magnitude exceeds `floatmax` of its storage type is divided by a `Float32` scale factor and
stored with it; a tile beyond `floatmax(Float32) * floatmax` of its storage type throws an
`ArgumentError`. `precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`,
`Float32` and `Float64`; the diagonal tiles' type, `precisions[end]`, must be `Float32` or
`Float64`, which have the range of the scale factors. `A` must be square, non-empty and
finite, `uplo` 'L' or 'U' and `threshold` at least 1.
"""
function TiledTriMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType},
    threshold::Integer
)
    _check_mixedprec_args(A, precisions)
    length(precisions) <= 2 ||
        throw(ArgumentError("precisions must hold one or two types: off-diagonal, then diagonal"))
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U'"))
    threshold >= 1 || throw(ArgumentError("threshold must be at least 1"))
    T_OffDiag, T_Diag = precisions[1], precisions[end]
    offsets = _tile_offsets!([1], size(A, 1), Int(threshold))
    t = length(offsets) - 1
    M_Diag = typeof(similar(A, T_Diag, 0, 0))
    M_OffDiag = typeof(similar(A, T_OffDiag, 0, 0))
    diag = Vector{M_Diag}(undef, t)
    diag_scales = Vector{Float32}(undef, t)
    for k in 1:t
        block = view(A, offsets[k]:offsets[k+1]-1, offsets[k]:offsets[k+1]-1)
        diag[k], diag_scales[k] = _quantize_mixedprec_block(uplo == 'L' ? tril(block) : triu(block), T_Diag)
    end
    off = Matrix{Union{Nothing, M_OffDiag}}(nothing, t, t)
    off_scales = ones(Float32, t, t)
    for r in 1:t, c in 1:t
        r != c && (uplo == 'L' ? r >= c : r <= c) || continue
        off[r, c], off_scales[r, c] = _quantize_mixedprec_block(view(A, offsets[r]:offsets[r+1]-1, offsets[c]:offsets[c+1]-1), T_OffDiag)
    end
    return TiledTriMixedPrec{T_Diag, T_OffDiag, M_Diag, M_OffDiag}(
        diag, off, diag_scales, off_scales, offsets, uplo, size(A, 1)
    )
end

function _tile_offsets!(offsets::Vector{Int}, n::Int, threshold::Int)
    n <= threshold && return push!(offsets, offsets[end] + n)
    mid = _rec_split(n)
    _tile_offsets!(offsets, mid, threshold)
    _tile_offsets!(offsets, n - mid, threshold)
    return offsets
end

Base.size(A::TiledTriMixedPrec) = (A.n, A.n)

Base.@propagate_inbounds function Base.getindex(A::TiledTriMixedPrec{T_Diag}, i::Int, j::Int) where {T_Diag}
    @boundscheck checkbounds(A, i, j)
    (A.uplo == 'L' ? i >= j : i <= j) || return zero(T_Diag)
    r, c = searchsortedlast(A.offsets, i), searchsortedlast(A.offsets, j)
    li, lj = i - A.offsets[r] + 1, j - A.offsets[c] + 1
    r == c && return T_Diag(A.diag[r][li, lj] * T_Diag(A.diag_scales[r]))
    return T_Diag(A.off[r, c][li, lj] * T_Diag(A.off_scales[r, c]))
end

"""
    reconstruct_matrix(A::TiledTriMixedPrec{T_Diag}) -> AbstractMatrix{T_Diag}

Copies the tiled matrix back into one dense array with element type `T_Diag`, of the same
array type as the stored tiles, with each tile's scale applied and zeros in the other
triangle.
"""
function reconstruct_matrix(A::TiledTriMixedPrec{T_Diag}) where {T_Diag}
    C = fill!(similar(first(A.diag), T_Diag, A.n, A.n), zero(T_Diag))
    t = length(A.diag)
    for r in 1:t, c in 1:t
        if r == c
            @views C[A.offsets[r]:A.offsets[r+1]-1, A.offsets[c]:A.offsets[c+1]-1] .= T_Diag.(A.diag[r] .* T_Diag(A.diag_scales[r]))
        elseif A.off[r, c] !== nothing
            @views C[A.offsets[r]:A.offsets[r+1]-1, A.offsets[c]:A.offsets[c+1]-1] .= T_Diag.(A.off[r, c] .* T_Diag(A.off_scales[r, c]))
        end
    end
    return C
end
