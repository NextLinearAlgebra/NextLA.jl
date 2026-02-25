"""
    larfb!(side, trans, direct, storev, m, n, k, v, ldv, t, c, work)

Applies complex block reflector H or its transpose H^H to m-by-n matrix C from either the left or the right
Implemented with Julia internal functions for matrix multiplication

# Arguments
- 'side': 
    - 'L' : apply H or H^H from the left
    - 'R' : apply H or H^H from the right
- 'trans': 
    - 'N' : apply H
    - 'C' : apply H^H
- 'direct':  indicates how H is formed from product of elementary reflectors
    - 'F' : H = H(1) H(2) ... H(k) (Forward)
    - 'B' : H = H(k) ... H(2) H(1) (Backward)
- 'storev': indcicates how the vectors which define the elementary reflectors are stored
    - 'C' : columnwise
    - 'R' : rowwise
- 'm': the number of rows of matrix c
- 'n': the number of columns of matrix c
- 'k': the order of marix t (= the number of elementary reflectors whose roduct defines the block reflector)
- 'v': dimension 
        - (ldv, k) if storev = 'C'
        - (ldv, m) if storev = 'R' and side = 'L'
        - (ldv, n) if storev = 'R' and side = 'R'
- 'ldv': the leading dimension of array v
    - if storev = 'C' and side = 'L', ldv >= max(1,m)
    - if storev = 'C' and side = 'R', ldv >= max(1,n)
    - if storev = 'R', ldv >= k
- 't': dimension (ldv, k), the triangular k-by-k matrix t in representation of the block reflector
- 'c': 
    - on entry m-by-n matrix
    - on exit, overwritten by H*C or H^H*C or C*H or C*H^H
- 'work': dimension (ldwork, k)
"""
function larfb!(side::Char, trans::Char, direct::Char, storev::Char, m::Integer, n::Integer, k::Integer, V::AbstractMatrix{T}, ldv::Integer, T_mat::AbstractMatrix{T}, C::AbstractMatrix{T}, work::AbstractMatrix{T}) where {T}
    if m <= 0 || n <= 0 || k <= 0
        return
    end

    if storev != 'C' || direct != 'F'
        throw(ArgumentError("larfb! currently supports only forward, columnwise reflectors"))
    end

    if side != 'L' && side != 'R'
        throw(ArgumentError("side must be 'L' or 'R', got '$side'"))
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("trans must be 'N', 'C', or 'T', got '$trans'"))
    end

    Tzero = zero(T)
    one = oneunit(T)

    if side == 'L'
        W = @view work[1:n, 1:k]
        fill!(W, Tzero)

        C1 = @view C[1:k, :]

        V1 = UnitLowerTriangular(@view V[1:k, 1:k])
        Tblock = UpperTriangular(@view T_mat[1:k, 1:k])

        # W := C1^H  (conjugate transpose)
        W .= conj.(transpose(C1))

        # W := W * V1
        LinearAlgebra.mul!(W, W, V1)

        if m > k
            C2 = @view C[k+1:m, :]
            V2 = @view V[k+1:m, 1:k]
            # W += C2' * V2
            LinearAlgebra.mul!(W, adjoint(C2), V2, one, one)
        end

        if trans == 'N'
            # W := W * T'
            LinearAlgebra.mul!(W, W, adjoint(Tblock))
        else
            # W := W * T
            LinearAlgebra.mul!(W, W, Tblock)
        end

        if m > k
            C2 = @view C[k+1:m, :]
            V2 = @view V[k+1:m, 1:k]
            # C2 -= V2 * W'
            LinearAlgebra.mul!(C2, V2, adjoint(W), -one, one)
        end

        # W := W * V1'
        LinearAlgebra.mul!(W, W, adjoint(V1))

        # C1 -= W^H  (conjugate transpose)
        C1 .-= conj.(transpose(W))
    else
        W = @view work[1:m, 1:k]
        fill!(W, Tzero)

        C1 = @view C[:, 1:k]

        V1 = UnitLowerTriangular(@view V[1:k, 1:k])
        Tblock = UpperTriangular(@view T_mat[1:k, 1:k])

        # W := C1
        W .= C1

        # W := W * V1
        LinearAlgebra.mul!(W, W, V1)

        if n > k
            C2 = @view C[:, k+1:n]
            V2 = @view V[k+1:n, 1:k]
            # W += C2 * V2
            LinearAlgebra.mul!(W, C2, V2, one, one)
        end

        if trans == 'N'
            # W := W * T
            LinearAlgebra.mul!(W, W, Tblock)
        else
            # W := W * T'
            LinearAlgebra.mul!(W, W, adjoint(Tblock))
        end

        if n > k
            C2 = @view C[:, k+1:n]
            V2 = @view V[k+1:n, 1:k]
            # C2 -= W * V2'
            LinearAlgebra.mul!(C2, W, adjoint(V2), -one, one)
        end

        # W := W * V1'
        LinearAlgebra.mul!(W, W, adjoint(V1))

        # C1 -= W
        C1 .-= W
    end
end 

"""
    larfb!(side, trans, direct, storev, V, T, C)

Apply a complex block reflector H or its conjugate transpose H^H to a matrix C.

This is a high-level interface that automatically computes required dimensions
and allocates workspace for the block reflector application.

The block reflector H has the form:
H = I - V * T * V^H

where V contains k elementary reflector vectors and T is an upper triangular
block reflector coefficient matrix.

# Arguments
- `side`: Character specifying which side to apply the reflector
  - 'L': Apply H from the left (H*C or H^H*C)  
  - 'R': Apply H from the right (C*H or C*H^H)
- `trans`: Character specifying which form to apply
  - 'N': Apply H (no conjugate transpose)
  - 'C': Apply H^H (conjugate transpose)
- `direct`: Character indicating how H is formed from elementary reflectors
  - 'F': H = H(1) H(2) ... H(k) (Forward - first k reflectors)
  - 'B': H = H(k) ... H(2) H(1) (Backward - last k reflectors)  
- `storev`: Character indicating how reflector vectors are stored in V
  - 'C': Reflector vectors stored columnwise in V
  - 'R': Reflector vectors stored rowwise in V
- `V`: Matrix containing the elementary reflector vectors
- `T`: Upper triangular k×k matrix with block reflector coefficients
- `C`: m×n matrix to be transformed in-place

# Algorithm
Applies the block reflector efficiently by:
1. Computing W = C^H * V (or W = C * V for right multiplication)  
2. Multiplying by the triangular matrix T: W := W * T (or W * T^H)
3. Applying rank-k update: C := C - V * W^H (or C - W * V^H)

The algorithm exploits the triangular structure of the reflector matrix
to minimize computational cost.

# Example
```julia
m, n, k = 8, 6, 4
C = complex.(randn(m, n), randn(m, n))
V = complex.(randn(m, k), randn(m, k))  # k reflector vectors  
T = triu(complex.(randn(k, k), randn(k, k)))  # Upper triangular
larfb!('L', 'N', 'F', 'C', V, T, C)  # Apply H*C
```
"""
function larfb!(side::Char, trans::Char, direct::Char, storev::Char, V::AbstractMatrix{T}, T_mat::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    # Determine dimensions
    m, n = size(C)
    k = size(T_mat, 1)
    
    # Set leading dimensions
    ldv = size(V, 1) 
    
    # Allocate workspace: LAPACK convention is (ldwork, k) where ldwork=n (L) or m (R)
    if side == 'L'
        work = similar(C, n, k)
    else
        work = similar(C, m, k)
    end
    
    # Call the underlying kernel
    larfb!(side, trans, direct, storev, m, n, k, V, ldv, T_mat, C, work)
end