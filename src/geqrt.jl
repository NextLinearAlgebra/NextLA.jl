"""
    geqrt!(m, n, ib, A, T_matrix, tau, work)

Compute blocked QR factorization of an m-by-n matrix A using block size ib.
The matrix A is overwritten with the Q and R factors, and T contains the 
triangular factor of the block reflector.

# Arguments
- `m`: Number of rows in matrix A
- `n`: Number of columns in matrix A
- `ib`: Block size for the factorization (must be > 0 if m,n > 0)
- `A`: Input matrix (m × n), modified in place to contain Q and R factors
- `T`: Output triangular block reflector matrix (ib × n)
- `tau`: Output vector of scalar factors (length n)
- `work`: Workspace vector (length ib × n)

# Algorithm
Uses a block algorithm that processes ib columns at a time.
For each block, performs unblocked QR and then applies the 
block reflector to the remaining columns.
"""
function geqrt!(m::Integer, n::Integer, ib::Integer, A::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}, tau::AbstractVector{T}, work::AbstractVector{T}) where {T}
    # Input validation
    if m < 0
        throw(ArgumentError("illegal value of m: $m"))
    end

    if n < 0
        throw(ArgumentError("illegal value of n: $n"))
    end

    if (ib < 0) || ((ib == 0) && (m > 0) && (n > 0))
        throw(ArgumentError("illegal value of ib: $ib"))
    end

    # Quick return for empty matrices or zero block size
    if m == 0 || n == 0 || ib == 0
        return 
    end

    k = min(m, n)  # Number of reflectors to generate

    # Process matrix in blocks of size ib
    for i in 1:ib:k
        sb = min(ib, k-i+1)  # Current block size

        # Extract current block and corresponding parts of T and tau
    av = @view A[i:m, i:i+sb-1]           # Current block columns
    tv = @view T_matrix[1:sb, i:i+sb-1]          # Corresponding T block
        tauv = @view tau[i:i+sb-1]            # Corresponding tau values

    # Perform unblocked QR factorization on current block
    geqr2!(m-i+1, sb, av, tauv, work)
        
    # Form the triangular factor T for the block reflector
    larft!('F', 'C', m-i+1, sb, av, tauv, tv)

        # Apply block reflector to remaining columns if any exist
        if n >= i + sb
            # Reshape work array for block reflector application
            ww = reshape((@view work[1:(n-i-sb+1)*sb]), n-i-sb+1, sb)

            # Apply H^H to A[i:m, i+sb:n] from the left
            larfb!('L', 'C', 'F', 'C', m-i+1, n-i-sb+1, sb, av, 
                m-i+1, tv, (@view A[i:m, i+sb:n]), ww)
        end
    end
end

"""
    geqrt!(A, ib) -> (A, T, tau)
    
Helper function for blocked QR factorization. Computes A = Q*R where Q is orthogonal and R is upper triangular.

# Arguments
- `A`: Input matrix (m × n), modified in place to contain R in upper triangle and Q factors below
- `ib`: Block size for the factorization

# Returns
- Modified `A` matrix containing Q and R factors  
- `T`: Upper triangular block reflector matrix (ib × n)
- `tau`: Vector of scalar factors for elementary reflectors (length n)
"""
function geqrt!(A::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A)
    ib = size(T_matrix, 1)
    work = zeros(T, ib * n)

    geqrt!(m, n, ib, A, T_matrix, tau, work)
end
