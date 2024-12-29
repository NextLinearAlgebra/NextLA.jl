using LinearAlgebra
using KernelAbstractions
using CUDA

@kernel function step_one(A, B, n)
    # Get the column index handled by this thread
    col = @index(Global)

    B[col, 1] = B[col, 1]/A[col, col]

    for row in 2:n
        if col < row
            A[row, col] = A[row, col]/A[row, row]
        end
    end
end

@kernel function step_two(A, B, n)
    row = @index(Global)
    for col in 1:n
        # @synchronize
        if row > col
            B[row, 1] = B[row, 1] - A[row, col]*B[col, 1]
        end
    end
end


function performant_trsm_2!(side::Char, uplo::Char, transpose::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    if side == 'L' && uplo == 'L' && transpose == 'N'
        # Get matrix dimensions
        n = size(A, 1)
        m = size(B, 2)

        @assert n == size(A, 2) "Matrix A must be square"
        @assert n == size(B, 1) "Incompatible dimensions between A and B"

        # Get the backend (GPU)
        backend = get_backend(A)

        # Step 1: Solve the diagonal and scale appropriately
        step_one(backend, n)(A, B, n, ndrange=n)

        # Synchronize before moving to the next step
        KernelAbstractions.synchronize(backend)

        # Step 2: Update the remaining rows based on the scaled solution
        step_two(backend, n)(A, B, n, ndrange=n)

        # Synchronize to ensure all computations are complete
        KernelAbstractions.synchronize(backend)        

        return B
    else
        error("Only 'L', 'L', 'N' case is supported.")
    end
end
