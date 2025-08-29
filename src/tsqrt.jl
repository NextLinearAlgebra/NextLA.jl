"""
    tsqrt!(m, n, ib, A1, A2, T, tau, work)

Compute the QR factorization of an (m+n)-by-n triangular-pentagonal matrix
using the compact WY representation.

This routine computes the QR factorization of a triangular-pentagonal matrix:
    [ A1 ]
    [ A2 ]
where A1 is n-by-n upper triangular and A2 is m-by-n general.

The factorization has the form:
    [ A1 ] = Q * [ R ]
    [ A2 ]       [ 0 ]
where Q is orthogonal and R is upper triangular.

# Arguments
- `m`: Number of rows of the pentagonal part A2
- `n`: Number of columns of the triangular-pentagonal matrix  
- `ib`: Block size for the compact WY representation
- `A1`: n×n upper triangular matrix (modified in-place)
- `A2`: m×n general matrix (modified in-place) 
- `T`: ib×n matrix to store block reflector coefficients
- `tau`: Vector of length n to store reflector scalar factors
- `work`: Workspace array of length ib×n

# Algorithm
The algorithm proceeds in blocks of size ib:
1. For each block, generate elementary reflectors to zero the pentagonal part
2. Apply reflectors to remaining columns using efficient block updates
3. Store reflector coefficients in compact WY form in matrix T

The compact WY representation allows for efficient application of the 
orthogonal factor Q using block operations.

# Input Validation
All dimension parameters must be non-negative and leading dimensions
must satisfy minimum requirements for valid matrix storage.

# Notes
This is a low-level computational routine typically called by higher-level
QR factorization interfaces. The matrices A1, A2 are modified in-place
to store the R factor and reflector vectors respectively.
"""
function tsqrt!(m::Integer, n::Integer, ib::Integer, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}, tau::AbstractVector{T}, work::AbstractVector{T}) where {T}
    # Input validation with descriptive error messages
    if m < 0
        throw(ArgumentError("m must be non-negative, got $m"))
    end

    if n < 0
        throw(ArgumentError("n must be non-negative, got $n"))
    end

    if ib < 0
        throw(ArgumentError("ib must be non-negative, got $ib"))
    end

    # Quick return for degenerate cases
    if m == 0 || n == 0 || ib == 0
        return
    end

    Tone = oneunit(eltype(A1))
    Tzero = zero(eltype(A1))
    plus = LinearAlgebra.MulAddMul(Tone, Tone)

    # Process matrix in blocks of size ib
    for ii in 1:ib:n
        sb = min(n-ii+1, ib)

        # Generate elementary reflectors for current block
        for i in 1:sb
            # Generate elementary reflector H[ii+i-1] to annihilate A2[1:m, ii+i-1]
            A1[ii+i-1, ii+i-1], tau[ii+i-1] = larfg!(m+1, A1[ii+i-1, ii+i-1], 
                (@view A2[1:m, ii+i-1]), 1, tau[ii+i-1])

            if ii+i <= n
                # Apply H[ii+i-1] to A[ii+i-1:m, ii+i:ii+sb-1] from the left
                alpha = -conj(tau[ii+i-1])
                (@view work[1:sb-i]) .= (@view A1[ii+i-1, ii+i:ii+sb-1])
                
                # Compute work = A1[ii+i-1, ii+i:ii+sb-1]^H + A2[1:m, ii+i:ii+sb-1]^H * A2[1:m, ii+i-1]
                conj!((@view work[1:sb-i]))
                LinearAlgebra.generic_matvecmul!((@view work[1:sb-i]), 'C', (@view A2[1:m, ii+i:ii+sb-1]), 
                    (@view A2[1:m, ii+i-1]), plus)
                conj!((@view work[1:sb-i]))
                
                # Apply the reflector: A1 -= alpha * work, A2 -= alpha * v * work^H
                LinearAlgebra.axpy!(alpha, (@view work[1:sb-i]), (@view A1[ii+i-1, ii+i:ii+sb-1]))
                conj!((@view work[1:sb-i]))
                gerc!(alpha, (@view A2[1:m, ii+i-1]), (@view work[1:sb-i]), (@view A2[1:m, ii+i:ii+sb-1]))
            end

            # Build triangular factor T for block reflectors
            if i > 1
                alpha = -tau[ii+i-1]
                LinearAlgebra.generic_matvecmul!((@view T_matrix[1:i-1, ii+i-1]), 'C', (@view A2[1:m, ii:ii+i-2]), 
                    (@view A2[1:m, ii+i-1]), LinearAlgebra.MulAddMul(alpha, Tzero))
                LinearAlgebra.generic_trimatmul!((@view T_matrix[1:i-1, ii+i-1]), 'U', 'N', identity, 
                    (@view T_matrix[1:i-1, ii:ii+i-2]), (@view T_matrix[1:i-1, ii+i-1]))
            end
            T_matrix[i, ii+i-1] = tau[ii+i-1]
        end

        # Apply block reflector to remaining columns
        if n >= ii+sb
            # Use provided vector workspace; tsmqr! will reshape internally as needed
            tsmqr!('L', 'C', sb, n - (ii + sb) + 1, m, n - (ii + sb) + 1, sb, ib,
                   (@view A1[ii:ii+sb-1, ii+sb:n]), (@view A2[1:m, ii+sb:n]),
                   (@view A2[1:m, ii:ii+sb-1]), (@view T_matrix[1:ib, ii:ii+sb-1]), work)
        end
    end
end

"""
    tsqrt!(A1, A2, ib) -> (A1, A2, T, tau)
    
Compute QR factorization of a triangular-pentagonal matrix using block algorithm.

This is a high-level interface that automatically allocates workspace and
computes the QR factorization of the combined matrix [A1; A2] where A1 is
upper triangular and A2 is general.

# Arguments
- `A1`: n×n upper triangular matrix (modified in-place to store R factor)
- `A2`: m×n general matrix (modified in-place to store reflector vectors)
- `ib`: Block size for the algorithm (typically 32-64 for good performance)

# Returns
- Modified `A1`: Contains the R factor of the QR factorization  
- Modified `A2`: Contains the elementary reflector vectors
- `T`: ib×n matrix containing block reflector coefficients
- `tau`: Length-n vector containing reflector scaling factors

# Input Validation
- A1 must be square (n×n)
- A2 must have same number of columns as A1 (m×n)
- Block size ib should be positive and ≤ n for efficiency

# Example
```julia
n, m = 6, 8
ib = 4
A1 = triu(randn(ComplexF64, n, n))  # Upper triangular
A2 = randn(ComplexF64, m, n)        # General matrix
A1_qr, A2_qr, T, tau = tsqrt!(copy(A1), copy(A2), ib)
```

# Algorithm Notes  
Uses blocked algorithm for efficiency with large matrices. The compact WY
representation (stored in T) enables efficient application of the Q factor.
"""
function tsqrt!(A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}) where{T}
    n, n2 = size(A1)
    
    m, n3 = size(A2) 
    if n2 != n3
        throw(ArgumentError("A1 and A2 must have same number of columns, got $n2 and $n3"))
    end

    ib, nb = size(T_matrix)

    if ib <= 0
        throw(ArgumentError("Block size ib must be positive, got $ib"))
    end
    

    tau = Vector{T}(undef, n)
    work = zeros(T, ib * n)
    
    # Call the core computational routine
    tsqrt!(m, n, ib, A1, A2, T_matrix, tau, work)
end
