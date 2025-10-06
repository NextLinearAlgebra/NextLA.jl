export lauu2!

"""
    lauu2!(uplo, n, A)

Compute the product U * U^H or L^H * L, where the triangular factor U or L
is stored in the upper or lower triangular part of the array A.

This is an unblocked algorithm for computing the product of a triangular
matrix with its conjugate transpose. The result overwrites the original
triangular matrix.

# Arguments
- `uplo`: Character specifying which triangle is stored
  - 'U' or 'u': Upper triangular, computes U * U^H
  - 'L' or 'l': Lower triangular, computes L^H * L
- `n`: Order of the triangular matrix (≥ 0)
- `A`: Triangular matrix to be transformed (modified in-place)

# Algorithm
For upper triangular (uplo='U'):
- Computes A := U * U^H where U is upper triangular
- Result is Hermitian, only upper triangle is computed and stored

For lower triangular (uplo='L'):  
- Computes A := L^H * L where L is lower triangular
- Result is Hermitian, only lower triangle is computed and stored

The algorithm processes one column (or row) at a time using dot products
and matrix-vector operations. This is the unblocked version, suitable
for small matrices or as a building block for blocked algorithms.

# Input Validation
- uplo must be 'U', 'u', 'L', or 'l'
- n must be non-negative

# Notes
This routine is typically used in Cholesky factorization algorithms
and for computing covariance matrices from triangular factors.

# Example
```julia
n = 4
A = triu(randn(ComplexF64, n, n))  # Upper triangular matrix
lauu2!('U', n, A, n)  # A := U * U^H
```
"""
function lauu2!(uplo::Char, n::Int, A::AbstractMatrix{T}) where T

    # Input validation with descriptive error messages
    if !(uplo in ['U', 'u', 'L', 'l'])
        throw(ArgumentError("uplo must be 'U', 'u', 'L', or 'l', got '$uplo'"))
    end

    if n < 0
        throw(ArgumentError("n must be non-negative, got $n"))
    end

    # Quick return for degenerate case
    if n == 0
        return
    end

    if uplo in ['U', 'u']
        # Upper triangular case: Compute U * U^H
        for i in 1:n
            aii = A[i, i]  # Diagonal element of U

            if i < n
                # Update diagonal: A[i,i] = |U[i,i]|² + sum(|U[i,j]|² for j > i)
                A[i, i] = real(aii * conj(aii)) + real(dot(A[i, i+1:n], A[i, i+1:n]))

                # Update off-diagonal elements in column i
                for k in 1:i-1
                    A[k, i] = A[k, i] * aii + dot(A[k, i+1:n], conj(A[i, i+1:n]))
                end
            else
                # Final column: scale by diagonal element
                for k in 1:i
                    A[k, i] = A[k, i] * aii
                end
            end
        end
    else
        # Lower triangular case: Compute L^H * L
        for i in 1:n
            aii = A[i, i]  # Diagonal element of L

            if i < n
                # Update diagonal: A[i,i] = |L[i,i]|² + sum(|L[j,i]|² for j > i)
                A[i, i] = real(conj(aii) * aii) + real(dot(A[i+1:n, i], A[i+1:n, i]))

                # Update off-diagonal elements in row i
                for k in 1:i-1
                    A[i, k] = conj(aii) * A[i, k] + dot(A[i+1:n, k], conj(A[i+1:n, i]))
                end
            else
                # Final row: scale by conjugate of diagonal element
                for k in 1:i
                    A[i, k] = conj(aii) * A[i, k]
                end
            end
        end
    end
end

lauu2!(uplo::Char, A::AbstractMatrix{T}) where {T} = lauu2!(uplo, size(A, 1), A)
