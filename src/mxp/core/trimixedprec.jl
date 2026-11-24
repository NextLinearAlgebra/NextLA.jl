"""
    TriMixedPrec{T_Base<:AbstractFloat} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to triangular matrices. It
partitions the matrix into two recursive diagonal blocks (`A11`, `A22`) and a single dense
off-diagonal block (`OffDiag`), placed by its `uplo` character ('U' for upper, 'L' for lower).
A node holds either `A11`, `A22` and `OffDiag`, or a leaf block `Base`, never both. Every
stored block carries a `Float32` scale factor, 1 when it is unscaled. The struct is mutable so
that a block an update takes past `floatmax` of its type can be stored back with a larger
scale.
"""
mutable struct TriMixedPrec{T_Base<:AbstractFloat} <: AbstractMixedPrec{T_Base}
    A11::Union{TriMixedPrec{T_Base}, Nothing}
    A22::Union{TriMixedPrec{T_Base}, Nothing}
    OffDiag::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    OffDiag_scale::Float32
    Base_scale::Float32
    Base::Union{AbstractMatrix{T_Base}, Nothing}
    uplo::Char
    n::Int
end

"""
    TriMixedPrec(A::AbstractMatrix, uplo::Char; precisions::Vector{DataType}) -> TriMixedPrec

Constructs a `TriMixedPrec` representation of the triangular matrix `A`.

The matrix is partitioned using a base-2 recursive splitting scheme, as for `FullMixedPrec`.
Each level stores the off-diagonal block of the `uplo` triangle in `precisions[1]` and
recurses on `A11` and `A22` with the rest. The recursion stops when one precision remains or
the block is 1×1; that block's `uplo` triangle is stored in `precisions[end]`, which is
also `T_Base`, with zeros in the other half.
Only the `uplo` triangle of `A` is read back; the other triangle reads as zero. Every block
is a copy, so later changes to `A` do not show through.

`precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`.
The last, `precisions[end]`, must be `Float32` or `Float64`, which have the range of the scale
factors. A block whose largest magnitude exceeds `floatmax` of its storage type is divided by
a `Float32` scale factor and stored with it. A block whose largest magnitude exceeds
`floatmax(Float32) * floatmax` of its storage type cannot be represented and throws an
`ArgumentError`. `A` must be square, non-empty and finite, and `uplo` must be 'L' or 'U'.

# Examples
```jldoctest
julia> using NextLA.MixedPrecision, LinearAlgebra

julia> A = rand(1024, 1024);

julia> M = TriMixedPrec(A, 'L'; precisions = DataType[Float32, Float64]);   # reads tril(A) only

julia> isapprox(reconstruct_matrix(M), tril(A); rtol = 1e-5)
true
```
"""
function TriMixedPrec(
    A::AbstractMatrix,
    uplo::Char;
    precisions::Vector{DataType}
)
    _check_mixedprec_args(A, precisions)
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U'"))
    return _triangular_mixedprec(TriMixedPrec, A, uplo, precisions)
end

Base.size(A::TriMixedPrec) = (A.n, A.n)

Base.@propagate_inbounds function Base.getindex(A::TriMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    @boundscheck checkbounds(A, i, j)
    (A.uplo == 'L' ? i >= j : i <= j) || return zero(T_Base)
    A.Base !== nothing && return T_Base(A.Base[i, j] * T_Base(A.Base_scale))
    mid = size(A.A11, 1)
    i <= mid && j <= mid && return @inbounds A.A11[i, j]
    i > mid && j > mid && return @inbounds A.A22[i - mid, j - mid]
    x = A.uplo == 'L' ? A.OffDiag[i - mid, j] : A.OffDiag[i, j - mid]
    return T_Base(x * T_Base(A.OffDiag_scale))
end

"""
    reconstruct_matrix(A::TriMixedPrec{T_Base}) -> AbstractMatrix{T_Base}

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
        # The triangle test is part of the one fused broadcast (see symmmixedprec.jl).
        r, lower = 1:size(C, 1), A.uplo == 'L'
        C .= ifelse.(ifelse.(lower, r .>= r', r .<= r'), T_Base.(A.Base .* T_Base(A.Base_scale)), zero(T_Base))
        return C
    end
    mid, n = size(A.A11, 1), A.n
    @inbounds _reconstruct!(view(C, 1:mid, 1:mid), A.A11)
    @inbounds _reconstruct!(view(C, mid+1:n, mid+1:n), A.A22)
    if A.uplo == 'L'
        @inbounds @views C[mid+1:n, 1:mid] .= T_Base.(A.OffDiag .* T_Base(A.OffDiag_scale))
        @inbounds @views C[1:mid, mid+1:n] .= zero(T_Base)
    else
        @inbounds @views C[1:mid, mid+1:n] .= T_Base.(A.OffDiag .* T_Base(A.OffDiag_scale))
        @inbounds @views C[mid+1:n, 1:mid] .= zero(T_Base)
    end
    return C
end

"""
    TriMixedPrec(A::SymmMixedPrec{T_Base}) -> TriMixedPrec{T_Base}

Views a `SymmMixedPrec` as a `TriMixedPrec` over the same stored blocks, without copying.
The result reads only the `uplo` triangle, so a `SymmMixedPrec` factored in place can be used
as its triangular factor.

# Examples
```jldoctest
julia> using NextLA.MixedPrecision, LinearAlgebra

julia> A = rand(1024, 1024); S = A + A';

julia> M = SymmMixedPrec(S, 'L'; precisions = DataType[Float32, Float64]);

julia> reconstruct_matrix(TriMixedPrec(M)) == tril(reconstruct_matrix(M))
true
```
"""
function TriMixedPrec(A::SymmMixedPrec{T_Base}) where {T_Base}
    A.Base !== nothing &&
        return TriMixedPrec{T_Base}(nothing, nothing, nothing, 1f0, A.Base_scale, A.Base, A.uplo, A.n)
    return TriMixedPrec{T_Base}(TriMixedPrec(A.A11), TriMixedPrec(A.A22), A.OffDiag,
                                A.OffDiag_scale, 1f0, nothing, A.uplo, A.n)
end
