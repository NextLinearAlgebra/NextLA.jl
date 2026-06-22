@inline function require_compressible_tlr(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T}) where {T}
    get_backend(A) == A_tlr.backend || throw(ArgumentError("dense matrix and TLRMatrix must have the same backend"))
    size(A) == size(A_tlr) || throw(DimensionMismatch("dense matrix size must match the TLRMatrix"))
    return A
end

@inline function require_compressible_tlr(A_tlr::TLRMatrix{<:Any,T}, op) where {T}
    applicable(size, op) || throw(ArgumentError("operator must define size(op)"))
    applicable(eltype, op) || throw(ArgumentError("operator must define eltype(op)"))
    applicable(adjoint, op) || throw(ArgumentError("operator must define adjoint(op)"))
    applicable(LinearAlgebra.mul!, allocate(A_tlr.backend, T, size(A_tlr, 1), 1), op, allocate(A_tlr.backend, T, size(A_tlr, 2), 1)) ||
        throw(ArgumentError("operator must support mul!(Y, op, X)"))
    applicable(LinearAlgebra.mul!, allocate(A_tlr.backend, T, size(A_tlr, 2), 1), adjoint(op), allocate(A_tlr.backend, T, size(A_tlr, 1), 1)) ||
        throw(ArgumentError("operator must support mul!(Y, adjoint(op), X)"))

    size(op) == size(A_tlr) || throw(DimensionMismatch("operator size must match the TLRMatrix"))
    eltype(op) == T || throw(ArgumentError("operator element type must match the TLRMatrix element type"))
    return op
end
