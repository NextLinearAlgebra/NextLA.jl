"""
    FullMixedPrec{T_Base<:AbstractFloat} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to full dense square
matrices. It recursively partitions a matrix into four sub-blocks (`A11`, `A22`, `A21`,
`A12`), enabling varying precision levels to be stored dynamically at different depths of the
recursive hierarchy. A node holds either `A11`, `A22`, `A21` and `A12`, or a leaf block
`Base`, never both. Every stored block carries a `Float32` scale factor, 1 when it is
unscaled. The struct is mutable so that a block an update takes past `floatmax` of its type
can be stored back with a larger scale.
"""
mutable struct FullMixedPrec{T_Base<:AbstractFloat} <: AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base}, Nothing}
    A22::Union{FullMixedPrec{T_Base}, Nothing}
    A21::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    A12::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    A21_scale::Float32
    A12_scale::Float32
    Base_scale::Float32
    Base::Union{AbstractMatrix{T_Base}, Nothing}
    n::Int
end

"""
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType}) -> FullMixedPrec

Constructs a `FullMixedPrec` representation of the dense square matrix `A`.

The matrix is partitioned using a base-2 recursive splitting scheme. If the dimension `n` is a
power of 2, it splits evenly; otherwise, it splits at the largest power of 2 less than `n`.
Each level stores `A21` and `A12` in `precisions[1]` and recurses on `A11` and `A22` with the
rest. The recursion stops when one precision remains or the block is 1×1; that block
is stored in `precisions[end]`, which is also `T_Base`.

`precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`.
The last, `precisions[end]`, must be `Float32` or `Float64`, which have the range of the scale
factors. A block whose largest magnitude exceeds `floatmax` of its storage type is divided by
a `Float32` scale factor and stored with it. A block whose largest magnitude exceeds
`floatmax(Float32) * floatmax` of its storage type cannot be represented and throws an
`ArgumentError`. `A` must be square, non-empty and finite.
"""
function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    _check_mixedprec_args(A, precisions)
    return _fullmixedprec(A, precisions)
end

function _fullmixedprec(A::AbstractMatrix, precisions::AbstractVector{DataType})
    n = size(A, 1)
    T_Base = precisions[end]
    if length(precisions) == 1 || n == 1
        base, base_scale = _quantize_mixedprec_block(A, T_Base)
        return FullMixedPrec{T_Base}(nothing, nothing, nothing, nothing, 1f0, 1f0, base_scale, base, n)
    end
    mid = _rec_split(n)
    remaining_precisions = @view precisions[2:end]
    A11 = _fullmixedprec(view(A, 1:mid, 1:mid), remaining_precisions)
    A22 = _fullmixedprec(view(A, mid+1:n, mid+1:n), remaining_precisions)
    A21, A21_scale = _quantize_mixedprec_block(view(A, mid+1:n, 1:mid), precisions[1])
    A12, A12_scale = _quantize_mixedprec_block(view(A, 1:mid, mid+1:n), precisions[1])
    return FullMixedPrec{T_Base}(A11, A22, A21, A12, A21_scale, A12_scale, 1f0, nothing, n)
end

Base.size(A::FullMixedPrec) = (A.n, A.n)

Base.@propagate_inbounds function Base.getindex(A::FullMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    @boundscheck checkbounds(A, i, j)
    A.Base !== nothing && return T_Base(A.Base[i, j] * T_Base(A.Base_scale))
    mid = size(A.A11, 1)
    i <= mid && j <= mid && return @inbounds A.A11[i, j]
    i > mid && j > mid && return @inbounds A.A22[i - mid, j - mid]
    return i <= mid ? T_Base(A.A12[i, j - mid] * T_Base(A.A12_scale)) :
                      T_Base(A.A21[i - mid, j] * T_Base(A.A21_scale))
end

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base}) -> AbstractMatrix{T_Base}

Copies the hierarchical mixed-precision matrix back into one dense array with element type
`T_Base`, of the same array type as the stored blocks, with each block's scale applied.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    B = A.Base !== nothing ? A.Base : A.A21
    return _reconstruct!(similar(B, T_Base, A.n, A.n), A)
end

function _reconstruct!(C::AbstractMatrix, A::FullMixedPrec{T_Base}) where {T_Base}
    if A.Base !== nothing
        C .= T_Base.(A.Base .* T_Base(A.Base_scale))
        return C
    end
    mid, n = size(A.A11, 1), A.n
    @inbounds _reconstruct!(view(C, 1:mid, 1:mid), A.A11)
    @inbounds _reconstruct!(view(C, mid+1:n, mid+1:n), A.A22)
    @inbounds @views C[mid+1:n, 1:mid] .= T_Base.(A.A21 .* T_Base(A.A21_scale))
    @inbounds @views C[1:mid, mid+1:n] .= T_Base.(A.A12 .* T_Base(A.A12_scale))
    return C
end