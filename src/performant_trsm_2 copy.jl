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

@kernel function both_steps_parallel(A, B, n)
    col = @index(Group)  # Block ID corresponds to column
    row = @index(Local)  # Thread ID corresponds to row within the column

    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024

    if row <= n
        diag[row] = A[row, row]
        B_c[row] = B[row, col] / diag[row]
    end

    for i in 1:n
        @synchronize
        if row > i
            B_c[row] -= (A[i, row] / diag[row]) * B_c[i]
        end
    end

    if row <= n
        B[row, col] = B_c[row]
    end
end

function performant_trsm_2_2!(
    side::Char, uplo::Char, transposed::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where T
    if side == 'L' && uplo == 'L' && transposed == 'N'
        n, m = size(B)
        @assert size(A, 1) == size(A, 2) == n "Matrix A must be square and match the number of rows in B"

        backend = get_backend(A)
        
        # Launch kernel with m blocks (columns) and n threads per block (rows)
        both_steps_parallel(backend, (n,))(transpose(A), B, n, ndrange=(n, m))

        return B
    else
        error("Only the case where side='L', uplo='L', and transposed='N' is supported.")
    end
end
