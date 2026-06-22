"""
    AbstractTLROperator{T}

Marker type for matrix-free operators that can be compressed into a TLR
representation.
"""
abstract type AbstractTLROperator{T} end

Base.eltype(::Type{<:AbstractTLROperator{T}}) where {T} = T
Base.eltype(::AbstractTLROperator{T}) where {T} = T

"""
    TLRLinearOperator(T, m, n, apply!, apply_adjoint!)

Lightweight wrapper turning a pair of `mul!` callbacks into an explicit
matrix-free operator accepted by `compress!`.
"""
struct TLRLinearOperator{T,F,G} <: AbstractTLROperator{T}
    m::Int
    n::Int
    apply!::F
    apply_adjoint!::G

    function TLRLinearOperator{T}(m::Integer, n::Integer, apply!::F, apply_adjoint!::G) where {T,F,G}
        m > 0 || throw(ArgumentError("m must be positive"))
        n > 0 || throw(ArgumentError("n must be positive"))
        return new{T,F,G}(Int(m), Int(n), apply!, apply_adjoint!)
    end
end

function TLRLinearOperator(::Type{T}, m::Integer, n::Integer, apply!::F, apply_adjoint!::G) where {T,F,G}
    return TLRLinearOperator{T}(m, n, apply!, apply_adjoint!)
end

Base.size(op::TLRLinearOperator) = (op.m, op.n)
Base.size(op::TLRLinearOperator, d::Int) = size(op)[d]

"""Apply the forward operator in place."""
function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::TLRLinearOperator{T}, X::AbstractVecOrMat) where {T}
    op.apply!(Y, X)
    return Y
end

"""Internal adjoint view of an [`AbstractTLROperator`](@ref)."""
struct TLRAdjointOperator{T,Op<:AbstractTLROperator{T}} <: AbstractTLROperator{T}
    parent::Op
end

Base.size(op::TLRAdjointOperator) = reverse(size(op.parent))
Base.size(op::TLRAdjointOperator, d::Int) = size(op)[d]
Base.adjoint(op::AbstractTLROperator{T}) where {T} = TLRAdjointOperator{T,typeof(op)}(op)

"""Apply the adjoint operator in place."""
function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::TLRAdjointOperator{T}, X::AbstractVecOrMat) where {T}
    op.parent.apply_adjoint!(Y, X)
    return Y
end
