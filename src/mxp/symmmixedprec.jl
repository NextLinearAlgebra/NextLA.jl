"""
    SymmMixedPrec{T_Base<:AbstractFloat, T_Scale<:AbstractFloat} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to symmetric matrices.
Only one triangle (specified by `uplo` as 'U' or 'L') is explicitly stored. It recursively
partitions the symmetric matrix into two symmetric diagonal sub-blocks (`A11`, `A22`) and
one corresponding off-diagonal block (`OffDiag`).
A node holds either `A11`, `A22` and `OffDiag`, or a leaf block `Base`, never both. Every
stored block carries a scale factor of type `T_Scale`, `one(T_Scale)` when it is unscaled.
The struct is mutable so that a block an update takes past `floatmax` of its type can be
stored back with a larger scale.
"""
mutable struct SymmMixedPrec{T_Base<:AbstractFloat, T_Scale<:AbstractFloat} <:
       AbstractMixedPrec{T_Base}
    A11::Union{SymmMixedPrec{T_Base, T_Scale}, Nothing}
    A22::Union{SymmMixedPrec{T_Base, T_Scale}, Nothing}
    OffDiag::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    OffDiag_scale::T_Scale
    Base_scale::T_Scale
    Base::Union{AbstractMatrix{T_Base}, Nothing}
    uplo::Char
    n::Int
end

"""
    SymmMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType}, scale_type=Float32)

Constructs a `SymmMixedPrec` representation of the symmetric matrix `A`.

The matrix is partitioned using a base-2 recursive splitting scheme, as for `FullMixedPrec`.
Each level stores the off-diagonal block of the `uplo` triangle in `precisions[1]` and
recurses on `A11` and `A22` with the rest. The recursion stops when one precision remains or
the block is 1×1; that block's `uplo` triangle is stored in `precisions[end]`, which is
also `T_Base`, with zeros in the other half.
`A` is assumed symmetric: only its `uplo` triangle is read back.

`precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`.
A block whose largest magnitude exceeds `floatmax` of its storage type is divided by a scale
factor, clamped and stored with that factor, of type `scale_type` (`Float16`, `Float32` or
`Float64`). A block whose largest magnitude exceeds `floatmax(scale_type) * floatmax` of its
storage type cannot be represented and throws an `ArgumentError`. `A` must be square,
non-empty and finite, and `uplo` must be 'L' or 'U'.
"""
function SymmMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType},
    scale_type::Type{T_Scale}=Float32
) where {T_Scale<:AbstractFloat}
    _check_args(A, precisions, T_Scale)
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U'"))
    return _triangular_tree(SymmMixedPrec, A, uplo, precisions, T_Scale)
end

Base.size(A::SymmMixedPrec) = (A.n, A.n)

Base.transpose(A::SymmMixedPrec) = A

Base.@propagate_inbounds function Base.getindex(A::SymmMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    @boundscheck checkbounds(A, i, j)
    if A.Base !== nothing
        _stored(i, j, A.uplo) || ((i, j) = (j, i))
        return _unscaled(A.Base[i, j], A.Base_scale, T_Base)
    end
    mid = size(A.A11, 1)
    i <= mid && j <= mid && return @inbounds A.A11[i, j]
    i > mid && j > mid && return @inbounds A.A22[i - mid, j - mid]
    r, c = i > mid ? (i - mid, j) : (j - mid, i)
    x = A.uplo == 'L' ? A.OffDiag[r, c] : A.OffDiag[c, r]
    return _unscaled(x, A.OffDiag_scale, T_Base)
end

"""
    reconstruct_matrix(A::SymmMixedPrec{T_Base})

Copies the symmetric mixed-precision matrix back into one dense array with element type
`T_Base`, of the same array type as the stored blocks, with both triangles filled from the
stored one and each block's scale applied.
"""
function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    B = A.Base !== nothing ? A.Base : A.OffDiag
    return _reconstruct!(similar(B, T_Base, A.n, A.n), A)
end

function _reconstruct!(C::AbstractMatrix, A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.Base !== nothing
        r = 1:size(C, 1)
        C .= _unscaled.(ifelse.(_stored.(r, r', A.uplo), A.Base, transpose(A.Base)), A.Base_scale, T_Base)
        return C
    end
    mid, n = size(A.A11, 1), A.n
    @inbounds _reconstruct!(view(C, 1:mid, 1:mid), A.A11)
    @inbounds _reconstruct!(view(C, mid+1:n, mid+1:n), A.A22)
    lo, up = A.uplo == 'L' ? (A.OffDiag, transpose(A.OffDiag)) : (transpose(A.OffDiag), A.OffDiag)
    @inbounds @views C[mid+1:n, 1:mid] .= _unscaled.(lo, A.OffDiag_scale, T_Base)
    @inbounds @views C[1:mid, mid+1:n] .= _unscaled.(up, A.OffDiag_scale, T_Base)
    return C
end
