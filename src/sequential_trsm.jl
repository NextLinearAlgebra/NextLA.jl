using LinearAlgebra

function sequential_trsm!(side::Char, uplo::Char, transpose::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    if side == 'L' && uplo == 'L' && transpose == 'N'
        # Get matrix dimensions
        n = size(A, 1)

        # Step 1: Solve the diagonal and scale appropriately
        B[1, 1] = B[1, 1] / A[1, 1]
        for row in 2:n
            B[row, 1] = B[row, 1] / A[row, row]
            for col in 1:row-1
                A[row, col] = A[row, col] / A[row, row]
            end
        end

        # Step 2: Update the remaining rows based on the scaled solution
        for col in 1:n
            for row in col+1:n
                B[row, 1] -= A[row, col] * B[col, 1]
            end
        end

        return B
    else
        error("Only 'L', 'L', 'N' case is supported.")
    end
end
