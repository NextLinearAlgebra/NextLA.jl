"""
    pamm!(op, side, storev, direct, m, n, k, l, A1, A2, V, W)

Parallel matrix multiplication kernel for block reflector applications.

This routine performs specialized matrix operations needed in blocked orthogonal
factorizations. It computes either:
- W = A1 + op(V) * A2 (when op='W')
- A2 = A2 + op(V) * W (when op='A')

where op(V) is V, V^H, V^T depending on the storage and direction parameters.

# Arguments
- `op`: Operation type
  - 'W': Compute W = A1 + op(V) * A2 or W = A1 + A2 * op(V)
  - 'A': Update A2 = A2 + op(V) * W or A2 = A2 + W * op(V)
- `side`: Which side V is applied
  - 'L': Left multiplication (op(V) * A2)
  - 'R': Right multiplication (A2 * op(V))
- `storev`: How reflector vectors are stored in V
  - 'C': Columnwise storage
  - 'R': Rowwise storage
- `direct`: Direction of reflector product
  - 'F': Forward (H = H₁H₂...Hₖ)
  - 'B': Backward (H = HₖHₖ₋₁...H₁)
- `m`, `n`: Dimensions of matrices A1, A2, W
- `k`: Number of elementary reflectors
- `l`: Number of columns/rows in triangular part of V
- `A1`: First input matrix
- `A2`: Second input/output matrix
- `V`: Matrix containing reflector vectors
- `W`: Workspace/output matrix

# Algorithm
The routine handles all combinations of storage formats and application sides
efficiently by dispatching to specialized kernels. Each kernel exploits the
structure of the reflector matrix V (triangular + rectangular parts) to
minimize computational cost.

# Input Validation
All parameters are validated for correctness. Dimensions must be non-negative
and leading dimensions must meet minimum requirements.

# Notes
This is a low-level computational kernel used internally by blocked QR
and LQ factorization routines. It is optimized for performance with
specific memory access patterns.
"""
function pamm!(op::Char, side::Char, storev::Char, direct::Char, m::Integer, n::Integer, k::Integer, l::Integer, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T}
    # Input validation with descriptive error messages
    if op ∉ ('W', 'A')
        throw(ArgumentError("op must be 'W' or 'A', got '$op'"))
    end
    if side ∉ ('L', 'R')
        throw(ArgumentError("side must be 'L' or 'R', got '$side'"))
    end
    if storev ∉ ('C', 'R')
        throw(ArgumentError("storev must be 'C' or 'R', got '$storev'"))
    end
    if direct ∉ ('F', 'B')
        throw(ArgumentError("direct must be 'F' or 'B', got '$direct'"))
    end
    
    # Dimension validation
    if m < 0
        throw(ArgumentError("m must be non-negative, got $m"))
    end
    if n < 0
        throw(ArgumentError("n must be non-negative, got $n"))
    end
    if k < 0
        throw(ArgumentError("k must be non-negative, got $k"))
    end
    if l < 0
        throw(ArgumentError("l must be non-negative, got $l"))
    end
    
    
    # Quick return for degenerate cases
    if m == 0 || n == 0 || k == 0
        return
    end

    # Convert parameters to boolean flags for efficiency
    forward = (direct == 'F')
    colmajor = (storev == 'C')
    left = (side == 'L')
    
    # Dispatch to appropriate kernel
    if op == 'W'
        pamm_w!(left, colmajor, forward, m, n, k, l, A1, A2, V, W)
    else
        pamm_a!(left, colmajor, forward, m, n, k, l, A2, V, W)
    end
end

function pamm_w!(left::Bool, colmajor::Bool, forward::Bool, m::Integer, n::Integer, k::Integer, l::Integer, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T}
    # W = A1 + op(V) * A2 or W = A1 + A2 * op(V)
    one = oneunit(eltype(A1))
    Tzero = zero(eltype(A1))
    plus = LinearAlgebra.MulAddMul(one, one)
    eqa = LinearAlgebra.MulAddMul(one, Tzero)

    if colmajor && forward && left # colmajor, forward, left
        mp = min(m-l+1, m)
        kp = min(l+1, k)

        for i in 1:l
            copyto!((@view W[i, 1:n]), (@view A2[m-l + i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'U', 'N', adjoint, (@view V[mp:mp+l-1, 1:l]), (@view W[1:l, 1:n]))
    
        LinearAlgebra.generic_matmatmul!((@view W[1:l, 1:n]), 'C', 'N', (@view V[1:m-l, 1:l]), (@view A2[1:m-l, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+k-l-1, 1:n]), 'C', 'N', (@view V[1:m, kp:kp+k-l-1]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end

    end
    if !colmajor && forward && left # rowmajor, forward, left
        mp = min(m-l+1, m)
        kp = min(l+1, k)

        for i in 1:l
            copyto!((@view W[i, 1:n]), (@view A2[m-l+i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'L', 'N', identity, (@view V[1:l, mp:mp+l-1]), (@view W[1:l, 1:n]))

        LinearAlgebra.generic_matmatmul!((@view W[1:l, 1:n]), 'N', 'N', (@view V[1:l, 1:m-l]), (@view A2[1:m-l, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+k-l-1, 1:n]), 'N', 'N', (@view V[kp:kp+k-l-1, 1:m]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end
    end
    if colmajor && forward && !left # colmajor, forward, right
        np = min(n-l+1, n)
        kp = min(l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, j]), (@view A2[1:m, n-l+j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'U', 'N', identity, (@view W[1:m, 1:l]), (@view V[np:np+l-1, 1:l]))

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:l]), 'N', 'N', (@view A2[1:m, 1:n-l]), (@view V[1:n-l, 1:l]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+k-l-1]), 'N', 'N', (@view A2[1:m, 1:n]), (@view V[1:n, kp:kp+k-l-1]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
    if !colmajor && forward && !left # rowmajor, forward, right
        np = min(n-l+1, n)
        kp = min(l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, j]), (@view A2[1:m, n-l+j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'L', 'N', adjoint, (@view W[1:m, 1:l]), (@view V[1:l,  np:np+l-1]))

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:l]), 'N', 'C', (@view A2[1:m, 1:n-l]), (@view V[1:l, 1:n-l]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+k-l-1]), 'N', 'C', (@view A2[1:m, 1:n]), (@view V[kp:kp+k-l-1, 1:n]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
    if colmajor && !forward && left # colmajor, backward, left
        mp = min(l+1, m)
        kp = min(k-l+1, k)

        for i in 1:l
            copyto!((@view W[k-l + i, 1:n]), (@view A2[i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'L', 'N', adjoint, (@view V[1:l, kp:kp+l-1]), (@view W[kp:kp+l-1, 1:n]))

        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+l-1, 1:n]), 'C', 'N', (@view V[mp:mp+m-l-1, kp:kp+l-1]), (@view A2[mp:mp+m-l-1, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:k-l, 1:n]), 'C', 'N', (@view V[1:m, 1:k-l]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end
    end

    if !colmajor && !forward && left # rowmajor, backward, left       
        mp = min(l+1, m)
        kp = min(k-l+1, k)

        for i in 1:l 
            copyto!((@view W[k-l + i, 1:n]), (@view A2[i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'U', 'N', identity, (@view V[kp:kp+l-1, 1:l]), (@view W[kp:kp+l-1, 1:n]))
        
        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+l-1, 1:n]), 'N', 'N', (@view V[kp:kp+l-1, mp:mp+m-l-1]), (@view A2[mp:mp+m-l-1, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:k-l, 1:n]), 'N', 'N', (@view V[1:k-l, 1:m]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end      
    end
    if !colmajor && !forward && !left # rowmajor, backward, right
        np = min(l+1, n)
        kp = min(k-l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, k-l+j]), (@view A2[1:m, j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'U', 'N', adjoint, (@view W[1:m, kp:kp+l-1]), (@view V[kp:kp+l-1, 1:l]))
        
        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+l-1]), 'N', 'C', (@view A2[1:m, np:np+n-l-1]), (@view V[kp:kp+l-1, np:np+n-l-1]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:k-l]), 'N', 'C', (@view A2[1:m, 1:n]), (@view V[1:k-l, 1:n]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
    if colmajor && !forward && !left # colmajor, backward, right
        np = min(l+1, n) 
        kp = min(k-l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, k-l+j]), (@view A2[1:m, j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'L', 'N', identity, (@view W[1:m, kp:kp+l-1]), (@view V[1:l, kp:kp+l-1]))
        
        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+l-1]), 'N', 'N', (@view A2[1:m, np:np+n-l-1]), (@view V[np:np+n-l-1, kp:kp+l-1]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:k-l]), 'N', 'N', (@view A2[1:m, 1:n]), (@view V[1:n, 1:k-l]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
end


function pamm_a!(left::Bool, colmajor::Bool, forward::Bool, m::Integer, n::Integer, k::Integer, l::Integer, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T}
        # A2 = A2 + op(V) * W or A2 = A2 + W * op(V)
        one = oneunit(eltype(A2))
        minus = LinearAlgebra.MulAddMul(one*(-1), one)

        if colmajor && forward && left # colmajor, forward, left
            mp = min( m-l+1, m )
            kp = min( l+1, k )

            LinearAlgebra.generic_matmatmul!((@view A2[1:m-l, 1:n]), 'N', 'N', (@view V[1:m-l, 1:k]), (@view W[1:k, 1:n]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+l-1, 1:n]), 'N', 'N', (@view V[mp:mp+l-1, kp:kp+k-l-1]), (@view W[kp:kp+k-l-1, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'U', 'N', identity, (@view V[mp:mp+l-1, 1:l]), (@view W[1:l, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[i, 1:n]), (@view A2[m-l+i, 1:n]))
            end
        end

        if !colmajor && forward && left # rowmajor, forward, left
            mp = min(m-l+1, m)
            kp = min(l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m-l, 1:n]), 'C', 'N', (@view V[1:k, 1:m-l]), (@view W[1:k, 1:n]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+l-1, 1:n]), 'C', 'N', (@view V[kp:kp+k-l-1, mp:mp+l-1]), (@view W[kp:kp+k-l-1, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'L', 'N', adjoint, (@view V[1:l, mp:mp+l-1]), (@view W[1:l, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!((-one), (@view W[i, 1:n]), (@view A2[m-l+i, 1:n])) 
            end
        end


        if colmajor && forward && !left # colmajor, forward, right
            np = min(n-l+1, n)
            kp = min(l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n-l]), 'N', 'C', (@view W[1:m, 1:k]), (@view V[1:n-l, 1:k]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+l-1]), 'N', 'C', (@view W[1:m, kp:kp+k-l-1]), (@view V[np:np+l-1, kp:kp+k-l-1]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'U', 'N', adjoint, (@view W[1:m, 1:l]), (@view V[np:np+l-1, 1:l]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[1:m, j]), (@view A2[1:m, n-l+j]))
            end
        end

        if !colmajor && forward && !left # rowmajor, forward, right
            np = min(n-l+1, n)
            kp = min(l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n-l]), 'N', 'N', (@view W[1:m, 1:k]), (@view V[1:k, 1:n-l]), minus)
 
            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+l-1]), 'N', 'N', (@view W[1:m, kp:kp+k-l-1]), (@view V[kp:kp+k-l-1, np:np+l-1]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'L', 'N', identity, (@view W[1:m, 1:l]), (@view V[1:l, np:np+l-1]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[1:m, j]), (@view A2[1:m, n-l+j]))
            end
        end

        if colmajor && !forward && left # colmajor, backward, left
            mp = min(l+1, m)
            kp = min(k-l+1, k)
            
            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+m-l-1, 1:n]), 'N', 'N', (@view V[mp:mp+m-l-1, 1:k]), (@view W[1:k, 1:n]), minus)
            
            LinearAlgebra.generic_matmatmul!((@view A2[1:l, 1:n]), 'N', 'N', (@view V[1:l, 1:k-l]), (@view W[1:k-l, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'L', 'N', identity, (@view V[1:l, kp:kp+l-1]), (@view W[kp:kp+l-1, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[k-l+i, 1:n]), (@view A2[i, 1:n]))
            end
        end
        
        if !colmajor && !forward && left # rowmajor, backward, left
            mp = min(l+1, m)
            kp = min(k-l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+m-l-1, 1:n]), 'C', 'N', (@view V[1:k, mp:mp+m-l-1]), (@view W[1:k, 1:n]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:l, 1:n]), 'C', 'N', (@view V[1:k-l, 1:l]), (@view W[1:k-l, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'U', 'N', adjoint, (@view V[kp:kp+l-1, 1:l]), (@view W[kp:kp+l-1, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[k-l+i, 1:n]), (@view A2[i, 1:n]))
            end
        end
        
        if !colmajor && !forward && !left # rowmajor, backward, right
            np = min(l+1, n)
            kp = min(k-l+1, k)
            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+n-l-1]), 'N', 'N', (@view W[1:m, 1:k]), (@view V[1:k, np:np+n-l-1]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:l]), 'N', 'N', (@view W[1:m, 1:k-l]), (@view V[1:k-l, 1:l]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'U', 'N', identity, (@view W[1:m, kp:kp+l-1]), (@view V[kp:kp+l-1, 1:l]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[1:m, k-l+j]), (@view A2[1:m, j]))
            end
        end
        
        if colmajor && !forward && !left # colmajor, backward, right
            np = min(l+1, n)
            kp = min(k-l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+n-l-1]), 'N', 'C', (@view W[1:m, 1:k]), (@view V[np:np+n-l-1, 1:k]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:l]), 'N', 'C', (@view W[1:m, 1:k-l]), (@view V[1:l, 1:k-l]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'L', 'N', adjoint, (@view W[1:m, kp:kp+l-1]), (@view V[1:l, kp:kp+l-1]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one, (@view W[1:m, k-l+j]), (@view A2[1:m, j]))
            end
        end
end

"""
    pamm(op, side, storev, direct, A1, A2, V) -> (A1, A2)

Performs panel matrix multiplication with automatic workspace allocation.
This is a simplified interface that automatically computes required parameters.

# Arguments
- 'op': operation type
    - 'W': compute workspace
    - 'A': apply operation
- 'side': 
    - 'L' : apply from the left
    - 'R' : apply from the right
- 'storev': indicates how the vectors are stored
    - 'C' : columnwise
    - 'R' : rowwise
- 'direct': indicates direction
    - 'F' : forward
    - 'B' : backward
- 'A1': first matrix to be updated
- 'A2': second matrix to be updated
- 'V': matrix containing the vectors

# Returns
- Updated A1 and A2 matrices

# Example
```julia
m, n, k, l = 6, 4, 3, 2
A1 = complex.(randn(m, k), randn(m, k))
A2 = complex.(randn(m, l), randn(m, l))
V = complex.(randn(m, k), randn(m, k))
A1_new, A2_new = pamm('A', 'L', 'C', 'F', A1, A2, V)
```
"""
function pamm(op::Char, side::Char, storev::Char, direct::Char, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T}
    # Determine dimensions
    m, k = size(A1)
    n = size(A2, 2)
    l = n

    W = similar(A1, m, k)
    
    # Call the underlying kernel
    pamm(op, side, storev, direct, m, n, k, l, A1, A2, V, W)
end

