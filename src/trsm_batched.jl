"""
    trsm_batched!(side, uplo, transa, diag, A, B, alpha=one(eltype(A)))

Solve a batch of triangular systems in place.

`A` and `B` must be three-dimensional arrays with equal batch sizes, or matching
vectors of matrices. Each `B[:, :, b]` is overwritten by the solution of the
corresponding triangular system defined by `A[:, :, b]`.

This entry point is reserved for backend batched-library dispatch. CPU arrays
fall back to a BLAS loop, CUDA/AMDGPU/oneAPI extensions override this with
native batched kernels, and Metal currently throws an error.
"""
function trsm_batched!(side,
                       uplo,
                       transa,
                       diag,
                       A::AbstractArray{<:Any,3},
                       B::AbstractArray{<:Any,3},
                       alpha=one(eltype(A)))
    backend = KernelAbstractions.get_backend(B)
    backend isa KernelAbstractions.CPU || throw(ArgumentError("NextLA.trsm_batched! has no generic non-CPU implementation; use a backend wrapper"))
    size(A, 3) == size(B, 3) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    for bid in axes(A, 3)
        BLAS.trsm!(side, uplo, transa, diag, convert(eltype(B), alpha), @view(A[:, :, bid]), @view(B[:, :, bid]))
    end
    return B
end

function trsm_batched!(side,
                       uplo,
                       transa,
                       diag,
                       A::AbstractVector{<:AbstractMatrix},
                       B::AbstractVector{<:AbstractMatrix},
                       alpha=one(eltype(first(A))))
    backend = KernelAbstractions.get_backend(first(B))
    backend isa KernelAbstractions.CPU || throw(ArgumentError("NextLA.trsm_batched! has no generic non-CPU implementation; use a backend wrapper"))
    length(A) == length(B) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    for bid in eachindex(A, B)
        BLAS.trsm!(side, uplo, transa, diag, convert(eltype(B[bid]), alpha), A[bid], B[bid])
    end
    return B
end
