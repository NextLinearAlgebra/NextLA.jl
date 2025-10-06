"""
    geqr2!(m, n, A, lda, tau, work)

Compute unblocked QR factorization of an m-by-n matrix A using Householder reflectors.
The matrix A is overwritten with the Q and R factors.

# Arguments
- `m`: Number of rows in matrix A
- `n`: Number of columns in matrix A  
- `A`: Input matrix (m × n), modified in place to contain Q and R factors
- `tau`: Output vector of scalar factors (length min(m,n))
- `work`: Workspace vector (length n)

# Algorithm
Uses Householder reflectors H(i) to zero out elements below the diagonal.
For each column i, generates H(i) and applies it to remaining columns.
"""
function geqr2!(m::Integer, n::Integer, A::AbstractMatrix{T}, tau::AbstractVector{T}, work::AbstractVector{T}) where {T}
    # Input validation
    if m < 0
        throw(ArgumentError("illegal value of m: $m"))
    end
    
    if n < 0
        throw(ArgumentError("illegal value of n: $n"))
    end

    # Quick return for empty matrices
    if m == 0 || n == 0
        return
    end

    k = min(m, n)  # Number of reflectors to generate
    one = oneunit(eltype(A))

    # Main QR factorization loop
    for i in 1:k
        # Generate elementary reflector H(i) to annihilate A(i+1:m, i)
        A[i, i], tau[i] = larfg!(m-i+1, A[i, i], (@view A[min(i+1,m):m, i]), 1, tau[i])
        
        if i < n
            # Apply H(i)^H to A(i:m, i+1:n) from the left
            alpha = A[i, i]
            A[i, i] = one  # Set diagonal element to 1 for reflector application

            # Apply the reflector to remaining columns
            larf!('L', m-i+1, n-i, (@view A[i:m, i]), 1, conj(tau[i]), (@view A[i:m, i+1:n]), work)

            A[i, i] = alpha  # Restore original diagonal element
        end
    end
end

"""
    geqr2!(A) -> (A, tau)
    
Helper function for unblocked QR factorization using Householder reflectors.

# Arguments  
- `A`: Input matrix (m × n), modified in place
- `tau`: Output vector of scalar factors (length min(m,n))

# Returns
- Modified `A` containing Q and R factors
- `tau`: Vector of scalar factors (length min(m,n))
"""
function geqr2!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A)
    work = zeros(T, n)
    
    geqr2!(m, n, A, tau, work)
end
