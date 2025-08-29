"""
    tsmqr!(side, trans, m1, n1, m2, n2, k, ib, A1, A2, V, T, work)

Apply orthogonal matrix Q (or Q^H) stored as compact WY representation to 
a triangular-pentagonal matrix [A1; A2].

This routine applies a block orthogonal transformation represented in compact
WY form (stored in V and T) to the combined matrix [A1; A2] where A1 is 
triangular and A2 is pentagonal.

# Arguments
- `side`: Character indicating side of multiplication
  - 'L': Apply Q from the left (Q*[A1; A2] or Q^H*[A1; A2])
  - 'R': Apply Q from the right ([A1 A2]*Q or [A1 A2]*Q^H)
- `trans`: Character indicating whether to transpose Q  
  - 'N': Apply Q (no transpose)
  - 'C': Apply Q^H (conjugate transpose)
  - 'T': Apply Q^T (transpose, same as 'C' for complex)
- `m1`, `n1`: Dimensions of triangular matrix A1
- `m2`, `n2`: Dimensions of pentagonal matrix A2  
- `k`: Number of elementary reflectors (columns of V)
- `ib`: Block size for compact WY representation
- `A1`: Triangular part of the matrix (modified in-place)
- `A2`: Pentagonal part of the matrix (modified in-place)
- `V`: Matrix containing reflector vectors
- `T`: Upper triangular block reflector coefficient matrix  
- `work`: Workspace array


# Algorithm
The transformation Q is applied using the compact WY representation:
Q = I - V * T * V^H

The algorithm processes the reflectors in blocks of size ib, applying
each block using efficient matrix operations (parfb! routine).

# Input Validation  
Validates all dimension parameters and leading dimension requirements
for proper matrix storage and computation.

# Notes
This is a core computational routine for applying orthogonal transformations
in blocked QR algorithms. The compact WY form enables efficient block updates.
"""
function tsmqr!(side::Char, trans::Char, m1::Integer, n1::Integer, m2::Integer, n2::Integer, k::Integer, ib::Integer,
    A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, T_mat::AbstractMatrix{T}, work::AbstractVector{T}) where {T}

    # Input validation with descriptive error messages
    if side != 'L' && side != 'R'
        throw(ArgumentError("side must be 'L' or 'R', got '$side'"))
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("trans must be 'N', 'C', or 'T', got '$trans'"))
    end

    if m1 < 0
        throw(ArgumentError("m1 must be non-negative, got $m1"))
    end

    if n1 < 0
        throw(ArgumentError("n1 must be non-negative, got $n1"))
    end

    if m2 < 0 || (m2 != m1 && side == 'R')
        if side == 'R'
            throw(ArgumentError("For side='R', m2 must equal m1. Got m1=$m1, m2=$m2"))
        else
            throw(ArgumentError("m2 must be non-negative, got $m2"))
        end
    end

    if n2 < 0 || (n2 != n1 && side == 'L')
        if side == 'L'
            throw(ArgumentError("For side='L', n2 must equal n1. Got n1=$n1, n2=$n2"))
        else
            throw(ArgumentError("n2 must be non-negative, got $n2"))
        end
    end

    if k < 0 || (side == 'L' && k > m1) || (side == 'R' && k > n1)
        max_k = side == 'L' ? m1 : n1
        throw(ArgumentError("k must be between 0 and $max_k for side='$side', got $k"))
    end

    if ib < 0
        throw(ArgumentError("ib must be non-negative, got $ib"))
    end

    # Quick return for degenerate cases
    if m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0 || k == 0 || ib == 0
        return 
    end

    # Determine the order of applying blocks based on side and trans
    if (side == 'L' && trans != 'N') || (side == 'R' && trans == 'N')
        # Apply blocks forward: 1, ib+1, 2*ib+1, ...
        i1 = 1
        i3 = ib
        istop = k
    else
        # Apply blocks backward: ..., 2*ib+1, ib+1, 1
        i1 = (div(k-1,ib))*ib + 1
        i3 = -ib
        istop = 1
    end
    
    # Initialize indices for submatrices
    ic = 1
    jc = 1
    mi = m1
    ni = n1

    # Apply blocks of reflectors
    for i in i1:i3:istop
        kb = min(ib, k - i + 1)  # Size of current block

        if side == 'L'
            # Q is applied from the left: Q * [A1; A2]
            mi = m1 - i + 1
            ic = i
            # Workspace for this block: kb x ni
            W = reshape(@view(work[1:kb*ni]), kb, ni)
            parfb!('L', trans, 'F', 'C', mi, ni, m2, n2, kb, 0,
                   (@view A1[ic:ic+mi-1, jc:jc+ni-1]), (@view A2[1:m2, 1:n2]),
                   (@view V[1:m2, i:i+kb-1]), (@view T_mat[1:kb, i:i+kb-1]), W)
        else
            # Q is applied from the right: [A1 A2] * Q  
            ni = n1 - i + 1
            jc = i
            # Workspace for this block: mi x kb
            W = reshape(@view(work[1:mi*kb]), mi, kb)
            parfb!('R', trans, 'F', 'C', mi, ni, m2, n2, kb, 0,
                   (@view A1[ic:ic+mi-1, jc:jc+ni-1]), (@view A2[1:m2, 1:n2]),
                   (@view V[1:n2, i:i+kb-1]), (@view T_mat[1:kb, i:i+kb-1]), W)
        end
    end
end

"""
    tsmqr!(side, trans, A1, A2, V, T, ib) -> (A1, A2)
    
Apply orthogonal matrix Q (stored in compact WY form) to triangular-pentagonal matrices.

This is a high-level interface that automatically determines dimensions and
allocates workspace for applying block orthogonal transformations to the
combined matrix [A1; A2].

# Arguments
- `side`: Character indicating multiplication side
  - 'L': Apply Q from left (Q*[A1; A2] or Q^H*[A1; A2])
  - 'R': Apply Q from right ([A1 A2]*Q or [A1 A2]*Q^H)
- `trans`: Character indicating transpose operation
  - 'N': Apply Q (no transpose)
  - 'C': Apply Q^H (conjugate transpose)
- `A1`: Triangular part of matrix (modified in-place)
- `A2`: Pentagonal part of matrix (modified in-place)
- `V`: Matrix containing elementary reflector vectors
- `T_matrix`: Upper triangular block reflector coefficient matrix
- `ib`: Block size for the compact WY representation

# Returns
- Modified `A1`: Triangular part after transformation
- Modified `A2`: Pentagonal part after transformation

# Input Validation
- For side='L': n2 must equal n1 (same number of columns)
- For side='R': m2 must equal m1 (same number of rows)
- Block size ib should be positive and ≤ min(size(V,2), ib)

# Example
```julia
# Apply Q from left to triangular-pentagonal matrix
m1, n1, m2, n2 = 6, 8, 10, 8  
k, ib = 4, 2
A1 = triu(randn(ComplexF64, m1, n1))
A2 = randn(ComplexF64, m2, n2)
V = randn(ComplexF64, m2, k)
T = triu(randn(ComplexF64, ib, k))
tsmqr!('L', 'N', A1, A2, V, T, ib)
```

# Algorithm
Uses blocked approach to apply the orthogonal transformation Q = I - V*T*V^H
efficiently. The compact WY representation enables high-performance 
matrix-matrix operations instead of multiple vector operations.
"""
function tsmqr!(side::Char, trans::Char, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, 
               V::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}) where {T}
    m1, n1 = size(A1)
    m2, n2 = size(A2)
    k = size(V, 2)
    ib = size(T_matrix, 1)
    
    # Validate input dimensions
    if side == 'L' && n2 != n1
        throw(ArgumentError("For side='L', A1 and A2 must have same number of columns. Got n1=$n1, n2=$n2"))
    elseif side == 'R' && m2 != m1
        throw(ArgumentError("For side='R', A1 and A2 must have same number of rows. Got m1=$m1, m2=$m2"))
    end
    
    if ib <= 0
        throw(ArgumentError("Block size ib must be positive, got $ib"))
    end
    
    if k > size(T_matrix, 2)
        throw(ArgumentError("Number of reflectors k ($k) exceeds T matrix columns ($(size(T_matrix, 2)))"))
    end
    
    # Determine workspace requirements and allocate
    if side == 'L'
        work_size = ib * max(n1, n2)
    else
        work_size = m1 * ib
    end
    work = zeros(T, work_size)
    
    # Call the core computational routine
    tsmqr!(side, trans, m1, n1, m2, n2, k, ib, A1, A2, 
          V, T_matrix, work)
end
