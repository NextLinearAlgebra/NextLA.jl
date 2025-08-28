"""
    pemv!(trans, storev, m, n, l, alpha, A, X, beta, Y, work)

Perform panel matrix-vector multiplication with optimized block algorithms.

This function implements efficient matrix-vector multiplication for structured panels,
commonly used in block QR factorization algorithms. It performs operations of the form:
Y := alpha * op(A) * X + beta * Y

where op(A) can be A, A^T, or A^H depending on the trans parameter.

# Arguments
- `trans::Char`: Transpose operation ('N' for none, 'T' for transpose, 'C' for conjugate transpose)
- `storev::Char`: Storage format for vectors ('C' for columnwise, 'R' for rowwise)
- `m::Int`: Number of rows in matrix A
- `n::Int`: Number of columns in matrix A
- `l::Int`: Panel size (must be ≤ min(m,n))
- `alpha`: Scalar multiplier for the matrix-vector product
- `A::Matrix`: Input matrix of size m×n
- `X::Vector`: Input vector (modified in-place)
- `beta`: Scalar multiplier for the output vector Y
- `Y::Vector`: Output vector (modified in-place)
- `work::Vector`: Workspace array for intermediate computations

# Returns
- `Int`: Status code (0 for success, negative for invalid arguments)

# Algorithm
The function uses block-structured algorithms that partition the matrix and vectors
to take advantage of cache locality and vectorization, particularly effective for
panel-based factorizations.

# Implementation Notes
- Optimized for different storage formats (columnwise vs rowwise)
- Uses BLAS-3 operations where possible for performance
- Handles edge cases with l=1 efficiently
- Validates input parameters with descriptive error messages
"""
function pemv!(trans::Char, storev::Char, m::Integer, n::Integer, l::Integer, alpha::T, A::AbstractMatrix{T}, x::AbstractVector{T}, beta::T, y::AbstractVector{T}, work::AbstractVector{T}) where {T}
    # Input validation
    if trans != 'N' && trans != 'T' && trans != 'C'
        throw(ArgumentError("illegal value of trans"))
    end

    if storev != 'C' && storev != 'R'
        throw(ArgumentError("illegal value of storev"))
    end

    if !((storev == 'C' && trans != 'N') || (storev == 'R' && trans == 'N'))
        throw(ArgumentError("illegal values of trans/storev"))
    end

    if m < 0
        throw(ArgumentError("illegal value of m"))
    end

    if n < 0
        throw(ArgumentError("illegal value of n"))
    end

    if l > min(m, n)
        throw(ArgumentError("illegal value of l"))
    end

    # Quick return for trivial cases
    if m == 0 || n == 0
        return
    end

    if alpha == 0 && beta == 0
        return
    end

    # Handle special case where l=1 (convert to l=0 for efficiency)
    if l == 1
        l = 0
    end

    # Set up vector views based on storage format
    if storev == 'C'
        # Column-wise storage: partition X and Y based on m and l
        x1 = (@view x[1:m-l])
        x2 = (@view x[m-l+1:m])
        xf = (@view x[1:m])
    else 
        # Row-wise storage: partition X and Y based on n and l
        x1 = (@view x[1:n-l])
        x2 = (@view x[n-l+1:n])
        xf = (@view x[1:n])
    end

    # Determine Y partitioning based on storage format
    if storev != 'C'
        y1 = (@view y[1:l])
        y2 = (@view y[l+1:m])
    else 
        y1 = (@view y[1:l])
        y2 = (@view y[l+1:n])
    end


    # Apply the matrix-vector multiplication based on storage format and transpose
    if storev == 'C'
        if trans == 'N'
            throw(ErrorException("not implemented"))
        else
            # Column-wise storage with transpose/adjoint operation
            if l > 0
                # Copy relevant portion to workspace for triangular operations
                (@view work[1:l]) .= (@view x[m-l+1:m])

                # Apply triangular matrix multiplication
                if trans == 'C'
                    LinearAlgebra.generic_trimatmul!((@view work[1:l]), 'U', 'N', adjoint,
                        (@view A[m-l+1:m, 1:l]), (@view work[1:l]))
                else
                    LinearAlgebra.generic_trimatmul!((@view work[1:l]), 'U', 'N', transpose,
                        (@view A[m-l+1:m, 1:l]), (@view work[1:l]))
                end

                # Handle remaining matrix-vector operations
                if m > l
                    LinearAlgebra.generic_matvecmul!((@view y[1:l]), trans, (@view A[1:m-l, 1:l]),
                        (@view x[1:m-l]), LinearAlgebra.MulAddMul(alpha, beta))
                    LinearAlgebra.axpy!(alpha, (@view work[1:l]), (@view y[1:l]))
                else
                    # Handle case where m <= l
                    if beta == 0
                        (@view work[1:l]) .*= alpha
                        (@view y[1:l]) .= (@view work[1:l])
                    else
                        (@view y[1:l]) .*= beta
                        LinearAlgebra.axpy!(alpha, (@view work[1:l]), (@view y[1:l]))
                    end
                end
            end

            # Handle remaining columns if n > l
            if n > l
                k = n - l
                LinearAlgebra.generic_matvecmul!((@view y[l+1:n]), trans, (@view A[1:m, l+1:n]),
                    (@view x[1:m]), LinearAlgebra.MulAddMul(alpha, beta))
            end
        end
    else
        # Row-wise storage
        if trans == 'N'
            # Row-wise storage with no transpose
            if l > 0
                # Copy and apply triangular operations
                work[1:l] .= x2
                LinearAlgebra.generic_trimatmul!((@view work[1:l]), 'L', 'N', identity,
                    (@view A[1:l, n-l+1:n]), (@view work[1:l]))

                # Handle rectangular part if n > l
                if n > l
                    LinearAlgebra.generic_matvecmul!(y1, 'N', (@view A[1:l, 1:n-l]),
                        x1, LinearAlgebra.MulAddMul(alpha, beta))
                    LinearAlgebra.axpy!(alpha, (@view work[1:l]), y1)
                else
                    # Handle case where n <= l
                    if beta == 0
                        y1 .= alpha * (@view work[1:l])
                    else
                        y1 .*= beta
                        LinearAlgebra.axpy!(alpha, (@view work[1:l]), y1)
                    end
                end
            end

            # Handle remaining rows if m > l
            if m > l
                LinearAlgebra.generic_matvecmul!(y2, 'N', (@view A[l+1:m, 1:n]),
                    xf, LinearAlgebra.MulAddMul(alpha, beta))
            end
        else
            # Row-wise storage with transpose - not implemented
            throw(ErrorException("not implemented"))
        end
    end
end

"""
    pemv(trans, storev, A, X, Y, alpha=1.0, beta=0.0) -> Y

Performs panel matrix-vector multiplication with automatic workspace allocation.
This is a simplified interface that automatically computes required parameters.

# Arguments
- `trans::Char`: Transpose operation ('N' for none, 'T' for transpose, 'C' for conjugate transpose)
- `storev::Char`: Storage format for vectors ('C' for columnwise, 'R' for rowwise)
- `A::Matrix`: Matrix for multiplication
- `X::Vector`: Input vector
- `Y::Vector`: Output vector (modified in-place)
- `alpha`: Scalar multiplier for A*X (default: 1.0)
- `beta`: Scalar multiplier for Y (default: 0.0)

# Returns
- Updated vector Y

# Example
```julia
m, n, l = 6, 4, 3
A = complex.(randn(m, n), randn(m, n))
X = complex.(randn(n), randn(n))
Y = complex.(randn(m), randn(m))
Y_new = pemv('N', 'C', A, X, Y, 2.0, 1.0)
```
"""

function pemv(trans::Char, storev::Char, alpha::T, A::AbstractMatrix{T}, x::AbstractVector{T}, beta::T, y::AbstractVector{T}) where {T}
    # Determine dimensions
    m, n = size(A)
    l = min(m, n)  # Default panel size
    
    # Leading dimension
    
    # Allocate workspace
    work = similar(x, max(m, n))
    
    # Call the underlying kernel
    pemv!(trans, storev, m, n, l, alpha, A, x, beta, y, work)
end

export pemv!