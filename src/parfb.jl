"""
    parfb!(side, trans, direct, storev, m1, n1, m2, n2, k, l, A1, A2, V, T, work)

Apply a block reflector or its transpose/adjoint to a general matrix using parallel algorithms.

This function applies the block reflector H or its transpose/adjoint to two matrix blocks simultaneously,
making it efficient for parallel QR factorization algorithms. It performs operations of the form:
- C₁ := H^op · C₁ (left multiplication)
- C₁ := C₁ · H^op (right multiplication)

where H is represented in compact form by matrices V and T, and op can be N (no operation), T (transpose), 
or C (conjugate transpose).

# Arguments
- `side::Char`: Determines the side of multiplication ('L' for left, 'R' for right)
- `trans::Char`: Operation to apply ('N' for none, 'T' for transpose, 'C' for conjugate transpose)
- `direct::Char`: Direction of reflector storage ('F' for forward, 'B' for backward)
- `storev::Char`: Storage format of reflectors ('C' for columnwise, 'R' for rowwise)
- `m1::Int`: Number of rows in first matrix block A1
- `n1::Int`: Number of columns in first matrix block A1
- `m2::Int`: Number of rows in second matrix block A2
- `n2::Int`: Number of columns in second matrix block A2
- `k::Int`: Number of elementary reflectors
- `l::Int`: Order of the triangular factor in T
- `A1::Matrix`: First m1×n1 matrix block to be transformed (modified in-place)
- `A2::Matrix`: Second m2×n2 matrix block to be transformed (modified in-place)
- `V::Matrix`: Matrix containing elementary reflectors in compact form
- `T::Matrix`: Upper triangular factor matrix
- `work::Vector`: Workspace array

# Returns
- `Int`: Status code (0 for success, negative for invalid arguments)

# Algorithm
The function uses the compact WY representation where H = I - V·T·V^H, performing efficient
block operations to apply the transformation to both matrix blocks simultaneously.

# Implementation Notes
- Modifies A1 and A2 in-place for efficiency
- Uses optimized BLAS-3 operations for performance
- Handles different storage formats and operation types
- Validates all input parameters with descriptive error messages
"""
function parfb!(side::Char, trans::Char, direct::Char, storev::Char, m1::Integer, n1::Integer, m2::Integer, n2::Integer, k::Integer, l::Integer, 
                A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, T_mat::AbstractMatrix{T}, work::AbstractMatrix{T}) where {T}

    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("illegal value of trans"))
    end

    if direct != 'F' && direct != 'B'
        throw(ArgumentError("illegal value of direct"))
    end

    if storev != 'C' && storev != 'R'
        throw(ArgumentError("illegal value of storev"))
    end

    if m1 < 0
        throw(ArgumentError("illegal value of m1"))
    end

    if n1 < 0
        throw(ArgumentError("illegal value of n1"))
    end

    if m2 < 0 || (side == 'R' && m1 != m2)
        throw(ArgumentError("illegal value of m2"))
    end

    if n2 < 0 || (side == 'L' && n1 != n2)
        throw(ArgumentError("illegal value of n2"))
    end

    if k < 0
        throw(ArgumentError("illegal value of k"))
    end

    if l < 0 || l > k
        throw(ArgumentError("illegal value of l"))
    end

    # Quick return if any dimension is zero
    if m1 == 0 || n1 == 0 || n2 == 0 || k == 0
        return 
    end

    # Define scalar constants
    one = oneunit(eltype(A1))

    # Determine operation transformations based on flags
    if trans == 'N'
        tfun = identity
    else
        tfun = adjoint
    end

    if direct == 'F'
        forward = true
    else
        forward = false
    end

    if side == 'L'
        left = true
    else
        left = false
    end

    if storev == 'C'
        colmajor = true
    else
        colmajor = false
    end

    # Apply workspace computation using pamm kernel
    pamm!('W', side, storev, direct, m2, n2, k, l, A1, A2, V, work)

    # Apply block reflector transformation based on storage format and direction
    if colmajor && forward && left # colmajor, forward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'U', 'N', tfun, (@view T_mat[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if colmajor && forward && !left # colmajor, forward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'U', 'N', tfun, (@view work[1:m2, 1:k]), (@view T_mat[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    if colmajor && !forward && left # colmajor, backward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'L', 'N', tfun, (@view T_mat[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if colmajor && !forward && !left # colmajor, backward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'L', 'N', tfun, (@view work[1:m2, 1:k]), (@view T_mat[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    if !colmajor && forward && left # rowmajor, forward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'U', 'N', tfun, (@view T_mat[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if !colmajor && forward && !left # rowmajor, forward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'U', 'N', tfun, (@view work[1:m2, 1:k]), (@view T_mat[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    if !colmajor && !forward && left # rowmajor, backward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'L', 'N', tfun, (@view T_mat[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if !colmajor && !forward && !left # rowmajor, backward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'L', 'N', tfun, (@view work[1:m2, 1:k]), (@view T_mat[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    # Apply final transformation using pamm kernel
    pamm!('A', side, storev, direct, m2, n2, k, l, A1, A2, V, work)

end

"""
    parfb(side, trans, direct, storev, A1, A2, V, T) -> (A1, A2)

Applies a block reflector or its transpose to a pair of matrices A1 and A2.
This is a simplified interface that automatically computes required parameters.

# Arguments
- `side::Char`: Determines the side of multiplication ('L' for left, 'R' for right)
- `trans::Char`: Operation to apply ('N' for none, 'T' for transpose, 'C' for conjugate transpose)
- `direct::Char`: Direction of reflector storage ('F' for forward, 'B' for backward)
- `storev::Char`: Storage format of reflectors ('C' for columnwise, 'R' for rowwise)
- `A1::Matrix`: First matrix to be updated (modified in-place)
- `A2::Matrix`: Second matrix to be updated (modified in-place)
- `V::Matrix`: Matrix containing the elementary reflectors
- `T::Matrix`: Upper triangular matrix of the block reflector

# Returns
- Updated A1 and A2 matrices

# Example
```julia
m1, n1, m2, n2, k = 4, 6, 4, 6, 3
A1 = complex.(randn(m1, n1), randn(m1, n1))
A2 = complex.(randn(m2, n2), randn(m2, n2))
V = complex.(randn(m1+m2, k), randn(m1+m2, k))
T = complex.(randn(k, k), randn(k, k))
A1_new, A2_new = parfb('L', 'N', 'F', 'C', A1, A2, V, T)
```
"""
function parfb!(side::Char, trans::Char, direct::Char, storev::Char, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, T_mat::AbstractMatrix{T}) where {T}
    # Determine dimensions
    m1, n1 = size(A1)
    m2, n2 = size(A2)
    k = size(T_mat, 1)
    l = size(V, 2)

    # Allocate workspace
    if side == 'L'
        work = similar(A1, max(m1, m2), max(n1, n2))
    else
        work = similar(A1, max(m1, m2), max(n1, n2))
    end
    
    # Call the underlying kernel
    parfb!(side, trans, direct, storev, m1, n1, m2, n2, k, l,
          A1, A2, V, T_mat, work)
end

export parfb!
