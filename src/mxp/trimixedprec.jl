"""
    TriMixedPrec{T_Base<:AbstractFloat, T_Scale<:AbstractFloat} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to triangular matrices.
It partitions the matrix into two recursive diagonal blocks (`A11`, `A22`) and a single dense
off-diagonal block (`OffDiag`), structured according to its `uplo` ('U' for upper, 'L' for lower) character.
A node holds either `A11`, `A22` and `OffDiag`, or a leaf block `Base`, never both. Every
stored block carries a scale factor of type `T_Scale`, `one(T_Scale)` when it is unscaled.
The struct is mutable so that a block an update takes past `floatmax` of its type can be
stored back with a larger scale.
"""
mutable struct TriMixedPrec{T_Base<:AbstractFloat, T_Scale<:AbstractFloat} <:
       AbstractMixedPrec{T_Base}
    A11::Union{TriMixedPrec{T_Base, T_Scale}, Nothing}
    A22::Union{TriMixedPrec{T_Base, T_Scale}, Nothing}
    OffDiag::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    OffDiag_scale::T_Scale
    Base_scale::T_Scale
    Base::Union{AbstractMatrix{T_Base}, Nothing}
    uplo::Char
    n::Int
end

"""
    TriMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType}, scale_type=Float32)

Constructs a `TriMixedPrec` representation of the triangular matrix `A`.

The matrix is partitioned using a base-2 recursive splitting scheme, as for `FullMixedPrec`.
Each level stores the off-diagonal block of the `uplo` triangle in `precisions[1]` and
recurses on `A11` and `A22` with the rest. The recursion stops when one precision remains or
the block is 1×1; that block's `uplo` triangle is stored in `precisions[end]`, which is
also `T_Base`, with zeros in the other half.
Only the `uplo` triangle of `A` is read back; the other triangle reads as zero. Every block
is a copy, so later changes to `A` do not show through.

`precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`.
A block whose largest magnitude exceeds `floatmax` of its storage type is divided by a scale
factor, clamped and stored with that factor, of type `scale_type` (`Float16`, `Float32` or
`Float64`). A block whose largest magnitude exceeds `floatmax(scale_type) * floatmax` of its
storage type cannot be represented and throws an `ArgumentError`. `A` must be square,
non-empty and finite, and `uplo` must be 'L' or 'U'.
"""
function TriMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType},
    scale_type::Type{T_Scale}=Float32
) where {T_Scale<:AbstractFloat}
    _check_args(A, precisions, T_Scale)
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U'"))
    return _triangular_tree(TriMixedPrec, A, uplo, precisions, T_Scale)
end

Base.size(A::TriMixedPrec) = (A.n, A.n)

Base.@propagate_inbounds function Base.getindex(A::TriMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    @boundscheck checkbounds(A, i, j)
    _stored(i, j, A.uplo) || return zero(T_Base)
    A.Base !== nothing && return _unscaled(A.Base[i, j], A.Base_scale, T_Base)
    mid = size(A.A11, 1)
    i <= mid && j <= mid && return @inbounds A.A11[i, j]
    i > mid && j > mid && return @inbounds A.A22[i - mid, j - mid]
    x = A.uplo == 'L' ? A.OffDiag[i - mid, j] : A.OffDiag[i, j - mid]
    return _unscaled(x, A.OffDiag_scale, T_Base)
end

"""
    reconstruct_matrix(A::TriMixedPrec{T_Base})

Copies the triangular mixed-precision matrix back into one dense array with element type
`T_Base`, of the same array type as the stored blocks, with each block's scale applied and
zeros in the other triangle.
"""
function reconstruct_matrix(A::TriMixedPrec{T_Base}) where {T_Base}
    B = A.Base !== nothing ? A.Base : A.OffDiag
    return _reconstruct!(similar(B, T_Base, A.n, A.n), A)
end

function _reconstruct!(C::AbstractMatrix, A::TriMixedPrec{T_Base}) where {T_Base}
    if A.Base !== nothing
        r = 1:size(C, 1)
        C .= ifelse.(_stored.(r, r', A.uplo), _unscaled.(A.Base, A.Base_scale, T_Base), zero(T_Base))
        return C
    end
    mid, n = size(A.A11, 1), A.n
    @inbounds _reconstruct!(view(C, 1:mid, 1:mid), A.A11)
    @inbounds _reconstruct!(view(C, mid+1:n, mid+1:n), A.A22)
    if A.uplo == 'L'
        @inbounds @views C[mid+1:n, 1:mid] .= _unscaled.(A.OffDiag, A.OffDiag_scale, T_Base)
        @inbounds @views C[1:mid, mid+1:n] .= zero(T_Base)
    else
        @inbounds @views C[1:mid, mid+1:n] .= _unscaled.(A.OffDiag, A.OffDiag_scale, T_Base)
        @inbounds @views C[mid+1:n, 1:mid] .= zero(T_Base)
    end
    return C
end

"""
    TriMixedPrec(A::SymmMixedPrec{T_Base, T_Scale})

Views a `SymmMixedPrec` as a `TriMixedPrec` over the same stored blocks, without copying.
The result reads only the `uplo` triangle, so a `SymmMixedPrec` factored in place can be used
as its triangular factor.
"""
function TriMixedPrec(A::SymmMixedPrec{T_Base, T_Scale}) where {T_Base, T_Scale}
    A.Base !== nothing && return TriMixedPrec{T_Base, T_Scale}(
        nothing, nothing, nothing,
        one(T_Scale), A.Base_scale, A.Base, A.uplo, A.n
    )
    return TriMixedPrec{T_Base, T_Scale}(
        TriMixedPrec(A.A11), TriMixedPrec(A.A22), A.OffDiag,
        A.OffDiag_scale, one(T_Scale), nothing, A.uplo, A.n
    )
end
