using LinearAlgebra
using KernelAbstractions
using CUDA

# Function documentation:
"""
This function implements a GPU-accelerated algorithm to solve a system of linear equations of the form:
    A * x = b, where A is a lower triangular matrix.

The solution involves two steps:
1. **Normalization**: For each row `i`, divide both `b[i]` and the lower triangular elements of `A` by the diagonal element `A[i, i]`.
2. **Substitution**: Update each element of `b` by subtracting contributions from previously solved variables using the elements of `A`.

Key assumptions:
- The matrix `A` is lower triangular.
- The algorithm is optimized for GPU acceleration using CUDA and KernelAbstractions.
- The input system is not transposed, and `A` is not upper triangular.

The result is computed in-place, modifying `B` to contain the solution `x`.
"""

@kernel function both_steps(A, B, n)
    # Allocate shared memory for diagonal and vector B
    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024

    # Determine the row index of the current thread
    row = @index(Global)

    if row <= n
        # Load the diagonal element and normalize B[row]
        diag[row] = A[row, row]
        B_c[row] = B[row] / diag[row]
    end

    # Substitution step
    for col in 1:n
        @synchronize
        if row > col
            B_c[row] -= (A[col, row] / diag[row]) * B_c[col]
        end
    end

    # Write back the result to B
    B[row, 1] = B_c[row]
end

function performant_trsm_2_2!(
    side::Char, uplo::Char, transposed::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where T
    if side == 'L' && uplo == 'L' && transposed == 'N'
        n = size(A, 1)

        # Get the backend (GPU)
        backend = get_backend(A)

        # Launch the kernel with GPU backend
        both_steps(backend, n)(transpose(A), B, n, ndrange=n)

        return B
    else
        error("Only the case where side='L', uplo='L', and transposed='N' is supported.")
    end
end
