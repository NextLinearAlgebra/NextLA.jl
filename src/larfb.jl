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

    if m <= 0 || n <= 0
        return
    end

    one = oneunit(eltype(C))
    plus = LinearAlgebra.MulAddMul(one, one)
    minus = LinearAlgebra.MulAddMul(one*(-1),one)

    if storev == 'C'
        if direct == 'F'
            """
            V = (V1) (first k rows)
                (V2)
            where V1 is unit lower triangular
            """
            if side == 'L'
                """
                Form H*C or H^H * C where C = (C1)
                                              (C2)
                """

                c1 = @view C[1:k,:] 
                c2 = @view C[k+1:m,:]
                v1 = @view V[1:k,:]
                v2 = @view V[k+1:m,:]
            
                work .= c1'

                # W = W*V1           
                LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v1)

                if m > k
                    # W = W + C2^H * V2
                    LinearAlgebra.generic_matmatmul!(work, 'C', 'N', c2, v2, plus)
                end
                
                # W = W * T^H or W*T

                if trans == 'N' # W = W*T^H
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, T_mat)
                else
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, T_mat)
                end

                if m > k 
                    #C2 = C2 - V2*W^H
                    LinearAlgebra.generic_matmatmul!(c2, 'N', 'C', v2, work, minus)
                end

                # w = w*v1^H
                LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v1)

                c1 .-= (work)'

            else 
                if side == 'R'
                    """
                    Form C*H or C*H^H where C = (c1 c2)
                    """
                    c1 = @view C[:, 1:k]
                    c2 = @view C[:, k+1:n]
                    v1 = @view V[1:k,:]
                    v2 = @view V[k+1:n,:]

                    work .= c1

                    # w = w*v1
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v1)


                    if n > k
                        # w = w + c2*V2
                        LinearAlgebra.generic_matmatmul!(work, 'N', 'N', c2, v2, plus)
                    end
                    
                    #w = w*t or w*t^H

                    if trans == 'C' # W = W*T^H
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, T_mat)
                    else
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, T_mat)
                    end

                    if n > k
                        # c2 = c2 - w*v2^h
                        LinearAlgebra.generic_matmatmul!(c2, 'N', 'C', work, v2, minus)
                    end

                    #work = work*(v1')
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v1)

                    c1 .-= work
                end
            end
        else
            """
            V = (v1)
                (v2) (last k rows)
            where v2 is unit upper triangular
            """
            if side == 'L'
                """
                Form H*C or H^H*C where C = (c1)
                                            (c2)
                """
                c1 = @view C[1:m-k,:]
                c2 = @view C[m-k+1:m,:]
                v1 = @view V[1:ldv-k,:]
                v2 = @view V[ldv-k+1:ldv,:]
                
                work .= c2'

                #work = work*v2
                LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v2)

                if m > k
                    #work = work + (c1')*V1
                    LinearAlgebra.generic_matmatmul!(work, 'C', 'N', c1, v1, plus)
                end

                if trans == 'N'
                    #work = work*(t')
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, T_mat)
                else
                    #work = work*t
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, T_mat)
                end

                #c1 = c1 - v1*w^H
                if m > k
                    LinearAlgebra.generic_matmatmul!(c1, 'N', 'C', v1, work, minus)                    
                end

                #work = work*(v2')
                LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v2)

                #c2 = c2 - w^H
                for j in 1:k
                    for i in 1:n
                        C[m-k+j,i] = C[m-k+j,i] - conj(work[i,j])
                    end
                end
            else 
                if side == 'R'
                    """
                    Form C*H or C*H^H where C = (c1 c2)
                    """
                    c1 = @view C[:,1:n-k]
                    c2 = @view C[:,n-k+1:n]
                    v1 = @view V[1:ldv-k,:]
                    v2 = @view V[ldv-k+1:ldv,:]

                    work .= c2

                    #work = work*v2
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v2)

                    if n > k
                        #work = work + c1*V1
                        LinearAlgebra.generic_matmatmul!(work, 'N', 'N', c1, v1, plus)
                    end

                    if trans == 'C'
                        #work = work*(t')
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, T_mat)
                    else
                        #work = work*t
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, T_mat)
                    end
                    
                    #c1 = c1 - w*v1^H
                    if n > k
                        LinearAlgebra.generic_matmatmul!(c1, 'N', 'C', work, v1, minus)
                    end

                    #work = work*(v2')
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v2)

                    c2 .-= work
                end
            end
        end
    else 
        if storev == 'R'
            if direct == 'F'
                """
                Let V = (V1 V2) (v1: first k columns)
                where v1 is unit upper triangular
                """

                if side == 'L'
                    """
                    Form H*C or H^H*C where C = (c1)
                                                (c2)
                    """

                    v1 = @view V[:, 1:k]
                    v2 = @view V[:, k+1:m]
                    c1 = @view C[1:k, :]
                    c2 = @view C[k+1:m, :]

                    work .= c1'

                    #work = work*(v1')
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v1)

                    if m > k
                        #work = work + (c2')*(v2')
                        LinearAlgebra.generic_matmatmul!(work, 'C', 'C', c2, v2, plus)
                    end

                    if trans == 'N'
                        #work = work*(t')
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, T_mat)
                    else
                        #work = work*t
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, T_mat)
                    end

                    #c2 = c2 - v2^h*w^h
                    if m > k
                        LinearAlgebra.generic_matmatmul!(c2, 'C', 'C', v2, work, minus)
                    end

                    #work = work*v1
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v1)

                    c1 .-= work'

                else 
                    if side == 'R' || side == 'r'
                        """
                        Form C*H or C*H^H where C = (c1 c2)
                        """
                        
                        v1 = @view V[:, 1:k]
                        v2 = @view V[:, k+1:n]
                        c1 = @view C[:, 1:k]
                        c2 = @view C[:, k+1:n]

                        work .= c1

                        #work = work*(v1')
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v1)

                        if n > k
                            #work = work + c2*(v2')
                            LinearAlgebra.generic_matmatmul!(work, 'N', 'C', c2, v2, plus)
                        end

                        if trans == 'C'
                            #work = work*(t')
                            LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, T_mat)
                        else
                            #work = work*t
                            LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, T_mat)
                        end

                        #c2 = c2 - w*v2
                        if n > k
                            LinearAlgebra.generic_matmatmul!(c2, 'N', 'N', work, v2, minus)
                        end

                        #work = work*v1
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v1)
                        
                        c1 .-= work
                    end
                end
            else # direct = B
                """
                Let V = (v1  v2) (v2: last k columns)
                where v2 is unit lower triangular
                """
                if side == 'L' || side == 'l'
                    """
                    Form H*C or H^H*C where C = (c1)
                                                (c2)
                    """
                    v1 = @view V[:, 1:m-k]
                    v2 = @view V[:, m-k+1:m]
                    c1 = @view C[1:m-k,:]
                    c2 = @view C[m-k+1:m,:]

                    work .= c2'

                    #work = work * (v2')
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v2)
                    
                    if m > k
                        #work = work + (c1')*(v1')
                        LinearAlgebra.generic_matmatmul!(work, 'C', 'C', c1, v1, plus)
                    end

                    if trans == 'N'
                        #work = work*(t')
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, T_mat)
                    else
                        #work = work*t
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, T_mat)
                    end

                    #c1 = c1 - v1^h * w^h
                    if m > k
                        LinearAlgebra.generic_matmatmul!(c1, 'C', 'C', v1, work, minus)
                    end

                    #work = work*v2
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v2)
                
                    c2 .-= work'

                else 
                    if side == 'R'
                        """
                        Form C*H or C*H^H where C = (c1 c2)
                        """
                        v1 = @view V[:, 1:n-k]
                        v2 = @view V[:, n-k+1:n]
                        c1 = @view C[:, 1:n-k]
                        c2 = @view C[:,n-k+1:n]

                        work .= c2
                        
                        #work = work * (v2')
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v2)

                        if n > k
                            #work = work + c1*(v1')
                            LinearAlgebra.generic_matmatmul!(work, 'N', 'C', c1, v1, plus)
                        end

                        if trans == 'C'
                            #work = work*(t')
                            LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, T_mat)
                        else
                            #work = work*t
                            LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, T_mat)
                        end

                        #c1 = c1 - w*v1
                        if n > k
                            LinearAlgebra.generic_matmatmul!(c1, 'N', 'N', work, v1, minus)
                        end

                        #work = work*v2
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v2)

                        c2 .-= work
                    end
                end
            end
        end
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
    k = size(T, 1)
    
    # Set leading dimensions
    ldv = size(V, 1) 
    
    # Allocate workspace
    if side == 'L'
        ldwork = n
        work = similar(C, k, n)
    else
        ldwork = m
        work = similar(C, m, k)
    end
    
    # Call the underlying kernel
    larfb!(side, trans, direct, storev, m, n, k, V, ldv, T_mat, C, work)
end