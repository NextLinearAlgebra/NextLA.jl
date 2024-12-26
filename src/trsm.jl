using LinearAlgebra

"""
    trsm!(side, uplo, transpose, A, B)

Sequential implementation of the triangular solve matrix function (TRSM) for solving 
the matrix equation `op(A) * X = B`, where `A` is a triangular matrix.

# Parameters
- `side` (Character): Specifies left or right side multiplication (`'L'` or `'R'`).
- `uplo` (Character): Specifies if `A` is `'U'` (upper) or `'L'` (lower triangular).
- `transpose` (Character): Specifies if `A` should be transposed (`'T'` for transpose, `'N'` for no transpose).
- `A` (AbstractMatrix{T}): Triangular matrix.
- `B` (AbstractMatrix{T}): Right-hand side matrix.

# Notes
Only the case `(side = 'L', uplo = 'L', transpose = 'N')` is implemented for now.
"""
function trsm!(side::Char, uplo::Char, transpose::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    # Implement only the Lower, Left, No-transpose case for now
    if side == 'L' && uplo == 'L' && transpose == 'N'
        # Get dimensions of A and B to ensure compatibility
        n = size(A, 1)
        k = size(B, 2)
        
        # Check if A is square and dimensions match with B
        @assert n == size(A, 2) "Matrix A must be square"
        @assert n == size(B, 1) "Incompatible dimensions between A and B"

        # Perform the triangular solve manually
        for j in 1:k  # Loop over columns of B
            for i in 1:n  # Loop over rows of A and corresponding rows of B
                B[i, j] /= A[i, i]  # Divide by the diagonal element of A
                for l in i+1:n  # Update subsequent rows of B[j] for each i
                    B[l, j] -= A[l, i] * B[i, j]
                end
            end
        end

    # Placeholder for other cases:
    elseif side == 'L' && uplo == 'U'
        # CASE left side, upper triangular NOT IMPLEMENTED YET
    elseif side == 'R' && uplo == 'L'
        # CASE right side, lower triangular NOT IMPLEMENTED YET
    else
        # CASE right side, upper triangular NOT IMPLEMENTED YET
    end
end


