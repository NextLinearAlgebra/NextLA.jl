export potrf_batched!

"""
    potrf_batched!(uplo, A)

Compute an in-place batched Cholesky factorization of `A`.

`A` must be a three-dimensional array whose slices `A[:, :, b]` are factored
independently.

This entry point is reserved for backend batched-library dispatch. CPU arrays
fall back to a LAPACK loop, CUDA/AMDGPU/oneAPI extensions override this with
native batched kernels, and Metal currently throws an error.

The return value follows the underlying backend wrapper as closely as practical:
CPU, CUDA, and AMDGPU return `(A, status)`, oneAPI returns `A`, and Metal
throws.
"""
function potrf_batched!(uplo::Char, A::AbstractArray{T,3}) where {T}
    backend = KernelAbstractions.get_backend(A)
    backend isa KernelAbstractions.CPU || throw(ArgumentError("NextLA.potrf_batched! has no generic non-CPU implementation; use a backend wrapper"))
    size(A, 1) == size(A, 2) ||
        throw(DimensionMismatch("potrf_batched!: A must be square in its first two dimensions"))
    status = similar(A, Int32, size(A, 3))
    @inbounds for b in axes(A, 3)
        _, info = LAPACK.potrf!(uplo, view(A, :, :, b))
        status[b] = Int32(info)
    end
    return A, status
end
