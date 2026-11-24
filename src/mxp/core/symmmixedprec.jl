"""
    SymmMixedPrec{T_Base<:AbstractFloat} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to symmetric matrices. Only
one triangle (specified by `uplo` as 'U' or 'L') is explicitly stored. It recursively
partitions the symmetric matrix into two symmetric diagonal sub-blocks (`A11`, `A22`) and one
corresponding off-diagonal block (`OffDiag`). A node holds either `A11`, `A22` and `OffDiag`,
or a leaf block `Base`, never both. Every stored block carries a `Float32` scale factor, 1
when it is unscaled. The struct is mutable so that a block an update takes past `floatmax` of
its type can be stored back with a larger scale.
"""
mutable struct SymmMixedPrec{T_Base<:AbstractFloat} <: AbstractMixedPrec{T_Base}
    A11::Union{SymmMixedPrec{T_Base}, Nothing}
    A22::Union{SymmMixedPrec{T_Base}, Nothing}
    OffDiag::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    OffDiag_scale::Float32
    Base_scale::Float32
    Base::Union{AbstractMatrix{T_Base}, Nothing}
    uplo::Char
    n::Int
end

"""
    SymmMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType}) -> SymmMixedPrec

Constructs a `SymmMixedPrec` representation of the symmetric matrix `A`.

The matrix is partitioned using a base-2 recursive splitting scheme, as for `FullMixedPrec`.
Each level stores the off-diagonal block of the `uplo` triangle in `precisions[1]` and
recurses on `A11` and `A22` with the rest. The recursion stops when one precision remains or
the block is 1×1; that block's `uplo` triangle is stored in `precisions[end]`, which is
also `T_Base`, with zeros in the other half.
`A` is assumed symmetric: only its `uplo` triangle is read back.

`precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`.
The last, `precisions[end]`, must be `Float32` or `Float64`, which have the range of the scale
factors. A block whose largest magnitude exceeds `floatmax` of its storage type is divided by
a `Float32` scale factor and stored with it. A block whose largest magnitude exceeds
`floatmax(Float32) * floatmax` of its storage type cannot be represented and throws an
`ArgumentError`. `A` must be square, non-empty and finite, and `uplo` must be 'L' or 'U'.

# Examples
```jldoctest
julia> using NextLA.MixedPrecision, LinearAlgebra

julia> A = rand(1024, 1024); S = A + A';

julia> M = SymmMixedPrec(S, 'L'; precisions = DataType[Float32, Float64]);

julia> eltype(M.OffDiag) == Float32 && eltype(M.A11.Base) == Float64
true

julia> isapprox(reconstruct_matrix(M), S; rtol = 1e-5)
true
```
"""
function SymmMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType}
)
    _check_mixedprec_args(A, precisions)
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U'"))
    return _triangular_mixedprec(SymmMixedPrec, A, uplo, precisions)
end

Base.size(A::SymmMixedPrec) = (A.n, A.n)

Base.transpose(A::SymmMixedPrec) = A

Base.@propagate_inbounds function Base.getindex(A::SymmMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    @boundscheck checkbounds(A, i, j)
    if A.Base !== nothing
        (A.uplo == 'L' ? i >= j : i <= j) || ((i, j) = (j, i))
        return T_Base(A.Base[i, j] * T_Base(A.Base_scale))
    end
    mid = size(A.A11, 1)
    i <= mid && j <= mid && return @inbounds A.A11[i, j]
    i > mid && j > mid && return @inbounds A.A22[i - mid, j - mid]
    r, c = i > mid ? (i - mid, j) : (j - mid, i)
    x = A.uplo == 'L' ? A.OffDiag[r, c] : A.OffDiag[c, r]
    return T_Base(x * T_Base(A.OffDiag_scale))
end

"""
    reconstruct_matrix(A::SymmMixedPrec{T_Base}) -> AbstractMatrix{T_Base}

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
        # The triangle test is part of the one fused broadcast, so it runs where C
        # is (on the device for a GPU array) instead of building a CPU mask first.
        r, lower = 1:size(C, 1), A.uplo == 'L'
        C .= T_Base.(ifelse.(ifelse.(lower, r .>= r', r .<= r'), A.Base, transpose(A.Base)) .* T_Base(A.Base_scale))
        return C
    end
    mid, n = size(A.A11, 1), A.n
    @inbounds _reconstruct!(view(C, 1:mid, 1:mid), A.A11)
    @inbounds _reconstruct!(view(C, mid+1:n, mid+1:n), A.A22)
    lo, up = A.uplo == 'L' ? (A.OffDiag, transpose(A.OffDiag)) : (transpose(A.OffDiag), A.OffDiag)
    @inbounds @views C[mid+1:n, 1:mid] .= T_Base.(lo .* T_Base(A.OffDiag_scale))
    @inbounds @views C[1:mid, mid+1:n] .= T_Base.(up .* T_Base(A.OffDiag_scale))
    return C
end
