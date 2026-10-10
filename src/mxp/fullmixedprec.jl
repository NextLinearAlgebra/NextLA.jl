"""
    FullMixedPrec{T_Base<:AbstractFloat, T_Scale<:AbstractFloat} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to full dense square matrices.
It recursively partitions a matrix into four sub-blocks (`A11`, `A22`, `A21`, `A12`), enabling
varying precision levels to be stored dynamically at different depths of the recursive hierarchy.
A node holds either `A11`, `A22`, `A21` and `A12`, or a leaf block `Base`, never both. Every
stored block carries a scale factor of type `T_Scale`, `one(T_Scale)` when it is unscaled.
The struct is mutable so that a block an update takes past `floatmax` of its type can be
stored back with a larger scale.
"""
mutable struct FullMixedPrec{T_Base<:AbstractFloat, T_Scale<:AbstractFloat} <:
       AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base, T_Scale}, Nothing}
    A22::Union{FullMixedPrec{T_Base, T_Scale}, Nothing}
    A21::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    A12::Union{AbstractMatrix{<:AbstractFloat}, Nothing}
    A21_scale::T_Scale
    A12_scale::T_Scale
    Base_scale::T_Scale
    Base::Union{AbstractMatrix{T_Base}, Nothing}
    n::Int
end

"""
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType}, scale_type=Float32)

Constructs a `FullMixedPrec` representation of the dense square matrix `A`.

The matrix is partitioned using a base-2 recursive splitting scheme. If the dimension `n` is a
power of 2, it splits evenly; otherwise, it splits at the largest power of 2 less than `n`.
Each level stores `A21` and `A12` in `precisions[1]` and recurses on `A11` and `A22` with the
rest. The recursion stops when one precision remains or the block is 1×1; that block
is stored in `precisions[end]`, which is also `T_Base`.

`precisions` may contain `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`.
A block whose largest magnitude exceeds `floatmax` of its storage type is divided by a scale
factor, clamped and stored with that factor, of type `scale_type` (`Float16`, `Float32` or
`Float64`). A block whose largest magnitude exceeds `floatmax(scale_type) * floatmax` of its
storage type cannot be represented and throws an `ArgumentError`. `A` must be square,
non-empty and finite.
"""
function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType},
    scale_type::Type{T_Scale}=Float32
) where {T_Scale<:AbstractFloat}
    _check_args(A, precisions, T_Scale)
    return _fullmixedprec(A, precisions, T_Scale)
end

function _fullmixedprec(A::AbstractMatrix, precisions::AbstractVector{DataType}, ::Type{S}) where {S<:AbstractFloat}
    n = size(A, 1)
    T_Base = precisions[end]
    unit_scale = one(S)
    if length(precisions) == 1 || n == 1
        base, base_scale = _store(A, T_Base, S)
        return FullMixedPrec{T_Base, S}(
            nothing, nothing, nothing, nothing,
            unit_scale, unit_scale, base_scale, base, n
        )
    end
    mid = _rec_split(n)
    T_OffDiag = precisions[1]
    remaining_precisions = @view precisions[2:end]
    A11 = _fullmixedprec(view(A, 1:mid, 1:mid), remaining_precisions, S)
    A22 = _fullmixedprec(view(A, mid+1:n, mid+1:n), remaining_precisions, S)
    A21, A21_scale = _store(view(A, mid+1:n, 1:mid), T_OffDiag, S)
    A12, A12_scale = _store(view(A, 1:mid, mid+1:n), T_OffDiag, S)
    return FullMixedPrec{T_Base, S}(
        A11, A22, A21, A12,
        A21_scale, A12_scale, unit_scale, nothing, n
    )
end

Base.size(A::FullMixedPrec) = (A.n, A.n)

Base.@propagate_inbounds function Base.getindex(A::FullMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    @boundscheck checkbounds(A, i, j)
    A.Base !== nothing && return _unscaled(A.Base[i, j], A.Base_scale, T_Base)
    mid = size(A.A11, 1)
    if i <= mid
        return j <= mid ? (@inbounds A.A11[i, j]) : _unscaled(A.A12[i, j - mid], A.A12_scale, T_Base)
    else
        return j <= mid ? _unscaled(A.A21[i - mid, j], A.A21_scale, T_Base) : (@inbounds A.A22[i - mid, j - mid])
    end
end

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base})

Copies the hierarchical mixed-precision matrix back into one dense array with element type
`T_Base`, of the same array type as the stored blocks, with each block's scale applied.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    A.Base !== nothing && return _unscaled.(A.Base, A.Base_scale, T_Base)
    return _reconstruct!(similar(A.A21, T_Base, A.n, A.n), A)
end

function _reconstruct!(C::AbstractMatrix, A::FullMixedPrec{T_Base}) where {T_Base}
    if A.Base !== nothing
        C .= _unscaled.(A.Base, A.Base_scale, T_Base)
        return C
    end
    mid, n = size(A.A11, 1), A.n
    @inbounds _reconstruct!(view(C, 1:mid, 1:mid), A.A11)
    @inbounds _reconstruct!(view(C, mid+1:n, mid+1:n), A.A22)
    @inbounds @views C[mid+1:n, 1:mid] .= _unscaled.(A.A21, A.A21_scale, T_Base)
    @inbounds @views C[1:mid, mid+1:n] .= _unscaled.(A.A12, A.A12_scale, T_Base)
    return C
end