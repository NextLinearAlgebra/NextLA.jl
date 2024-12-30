using LinearAlgebra
using KernelAbstractions
using CUDA


@kernel function both_steps(A, B, n)
    # Get the column index handled by this thread
    row = @index(Global)

    B[row, 1] = B[row, 1] / A[row, row]

    for i in 2:n
        if row < i
            A[i, row] = A[i, row] / A[i, i]
        end
    end

    @synchronize

    for col in 1:n
        @synchronize
        if row > col
            B[row, 1] = B[row, 1] - A[row, col] * B[col, 1]
        end
    end
end

function performant_trsm_2!(side::Char, uplo::Char, transpose::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    if side == 'L' && uplo == 'L' && transpose == 'N'
        # Get matrix dimensions
        n = size(A, 1)
        # m = size(B, 2)

        # @assert n == size(A, 2) "Matrix A must be square"
        # @assert n == size(B, 1) "Incompatible dimensions between A and B"

        # Get the backend (GPU)
        backend = get_backend(A)

        both_steps(backend, n)(A, B, n, ndrange=n)

        return B
    else
        error("Only 'L', 'L', 'N' case is supported.")
    end
end
