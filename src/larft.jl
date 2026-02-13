"""
    larft!(direct, storev, n, k, v, tau, T_mat)

Form the triangular factor T of a complex block reflector H of order n,
where H is defined as a product of k elementary reflectors.

The block reflector H has the form:
H = I - V * T * V^H

where V is n-by-k and contains the elementary reflector vectors, and T is
the k-by-k upper triangular factor computed by this routine.

# Arguments
- `direct`: Character indicating the order of the elementary reflectors
  - 'F': H = H(1) H(2) ... H(k) (Forward)  
  - 'B': H = H(k) ... H(2) H(1) (Backward)
- `storev`: Character indicating how the reflector vectors are stored in V
  - 'C': Columnwise storage (V is n-by-k)
  - 'R': Rowwise storage (V is k-by-n)  
- `n`: Order of the reflector H
- `k`: Number of elementary reflectors (order of T)
- `v`: Matrix containing the elementary reflector vectors
- `tau`: Array containing the scalar factors of the elementary reflectors
- `T_mat`: k-by-k matrix where the triangular factor T will be stored

# Algorithm
The algorithm computes T such that H = I - V * T * V^H where each column
(or row) of V represents an elementary reflector. The triangular structure
ensures efficient application of the block reflector.

For forward direction (direct='F'):
- T[i,i] = tau[i] (diagonal elements)
- T[j,i] = -tau[i] * V[i,j] * T[j,j:i-1] for j < i (upper triangular part)

For backward direction (direct='B'):
- T[i,i] = tau[i] (diagonal elements)  
- T[j,i] = -tau[i] * V[j,i] * T[i+1:j,i] for j > i (lower triangular part)

# Notes
This is the core computational routine for forming block reflector coefficients.
The matrix T enables efficient application of multiple reflectors simultaneously.
"""
function larft!(direct::Char, storev::Char, n::Integer, k::Integer, V::AbstractMatrix{T}, tau::AbstractVector{T}, T_mat::AbstractMatrix{T}) where {T}
    if n == 0
        return
    end

    zero0 = zero(eltype(V))
    one0 = oneunit(eltype(V))

    if direct == 'F'
        prevlastv = n

        for i in 1:k
            prevlastv = max(prevlastv, i)

            if tau[i] == zero0
                # H(i) = I (no reflection)
                for j in 1:i
                    T_mat[j,i] = zero0
                end
            else
                # General case: compute T column
                if storev == 'C'
                    # Find the last non-zero element in v[:,i]
                    lastv = n
                    while lastv >= i+1
                        if V[lastv, i] != zero0
                            break
                        end
                        lastv -= 1
                    end

                    # Initialize T[1:i-1,i] with diagonal contribution
                    for j in 1:i-1
                        T_mat[j,i] = -tau[i] * conj(V[i,j])
                    end

                    # Add contribution from off-diagonal part
                    j = min(lastv, prevlastv)
                    LinearAlgebra.generic_matvecmul!((@view T_mat[1:i-1, i]), 'C', (@view V[i+1:j, 1:i-1]), 
                        (@view V[i+1:j,i]), LinearAlgebra.MulAddMul(-tau[i], one0))

                else  # storev == 'R'
                    # Find the last non-zero element in v[i,:]
                    lastv = n
                    while lastv >= i+1
                        if V[i, lastv] != zero0
                            break
                        end
                        lastv -= 1
                    end

                    # Initialize T[1:i-1,i] with diagonal contribution
                    for j in 1:i-1
                        T_mat[j,i] = -tau[i] * V[j,i]
                    end

                    # Add contribution from off-diagonal part
                    j = min(lastv, prevlastv)
                    if i-1 > 0 && i+1 <= j
                        LinearAlgebra.generic_matmatmul!((@view T_mat[1:i-1, i]), 'N', 'C', (@view V[1:i-1, i+1:j]), 
                            (@view V[i:i, i+1:j]), LinearAlgebra.MulAddMul(-tau[i], one0))
                    end
                end

                # Apply triangular solve: T[1:i-1,i] = T[1:i-1,1:i-1] * T[1:i-1,i]
                LinearAlgebra.generic_trimatmul!((@view T_mat[1:i-1,i]), 'U', 'N', identity, 
                    (@view T_mat[1:i-1, 1:i-1]), (@view T_mat[1:i-1, i]))

                # Set diagonal element
                T_mat[i,i] = tau[i]

                # Update tracking variable
                if i > 1
                    prevlastv = max(prevlastv, lastv)
                else
                    prevlastv = lastv
                end
            end
        end
    else  # direct == 'B'
        prevlastv = 1
        for i in k:-1:1
            if tau[i] == zero0
                # H(i) = I (no reflection)
                for j in i:k
                    T_mat[j,i] = zero0
                end
            else
                if i < k
                    if storev == 'C'
                        # Find the first non-zero element in v[:,i]
                        lastv = 1
                        while lastv <= i-1
                            if V[lastv,i] != zero0
                                break
                            end
                            lastv += 1
                        end

                        # Initialize T[i+1:k,i] with diagonal contribution
                        for j in i+1:k
                            T_mat[j,i] = -tau[i] * conj(V[n-k+i, j])
                        end
                        
                        # Add contribution from off-diagonal part
                        j = max(lastv, prevlastv)
                        LinearAlgebra.generic_matvecmul!((@view T_mat[i+1:k, i]), 'C', (@view V[j:n-k+i, i+1:k]), 
                            (@view V[j:n-k+i, k]), LinearAlgebra.MulAddMul(-tau[i], one0))
                            
                    else  # storev == 'R'
                        # Find the first non-zero element in v[i,:]
                        lastv = 1
                        while lastv <= i-1
                            if V[lastv,i] != zero0
                                break
                            end
                            lastv += 1
                        end

                        # Initialize T[i+1:k,i] with diagonal contribution
                        for j in i+1:k
                            T_mat[j,i] = -tau[i] * V[j, n-k+i]
                        end
                        
                        # Add contribution from off-diagonal part
                        j = max(lastv, prevlastv)
                        LinearAlgebra.generic_matmatmul!((@view T_mat[i+1:k, i]), 'N', 'C', (@view V[i+1:k, j:n-k+i-1]), 
                            (@view V[i:i, j:n-k+i-1]), LinearAlgebra.MulAddMul(-tau[i], one0))
                    end

                    # Apply triangular solve: T[i+1:k,i] = T[i+1:k,i+1:k] * T[i+1:k,i]
                    LinearAlgebra.generic_trimatmul!((@view T_mat[i+1:k, i]), 'L', 'N', identity, 
                        (@view T_mat[i+1:k, i+1:k]), (@view T_mat[i+1:k, i]))

                    # Update tracking variable
                    if i > 1
                        prevlastv = min(prevlastv, lastv)
                    else
                        prevlastv = lastv
                    end
                end

                # Set diagonal element
                T_mat[i,i] = tau[i]
            end
        end
    end
end

"""
    larft(direct, storev, V, tau) -> T

Form the triangular factor T of a complex block reflector H from elementary 
reflectors and their scalar factors.

This is a high-level interface that automatically determines dimensions and
allocates the output matrix. The block reflector H has the form:
H = I - V * T * V^H

# Arguments
- `direct`: Character indicating the order of elementary reflector products
  - 'F': H = H(1) H(2) ... H(k) (Forward)
  - 'B': H = H(k) ... H(2) H(1) (Backward)
- `storev`: Character indicating how reflector vectors are stored in V
  - 'C': Columnwise storage (V is n-by-k)
  - 'R': Rowwise storage (V is k-by-n)  
- `V`: Matrix containing the elementary reflector vectors
- `tau`: Vector containing scalar factors of the elementary reflectors

# Returns
- `T`: k-by-k upper triangular matrix (triangular factor of block reflector)

# Input Validation  
- Matrix V and vector tau must have compatible dimensions
- For 'C' storage: size(V,2) must equal length(tau)
- For 'R' storage: size(V,1) must equal length(tau)

# Example
```julia
m, k = 8, 4
V = complex.(randn(m, k), randn(m, k))  # Elementary reflector vectors
tau = complex.(randn(k), randn(k))      # Reflector scaling factors
T = larft('F', 'C', V, tau)             # Compute triangular factor
```

# Mathematical Background
The triangular factor T enables efficient block operations. Instead of applying
k individual reflectors H(1), H(2), ..., H(k), the block reflector 
H = I - V*T*V^H can be applied in O(n²k) operations rather than O(nk²).
"""
function larft!(direct::Char, storev::Char, V::AbstractMatrix{T}, tau::AbstractVector{T}, T_mat::AbstractMatrix{T}) where {T}
    # Determine dimensions based on storage format
    if storev == 'C'
        n, k = size(V)
        if length(tau) != k
            throw(ArgumentError("For columnwise storage, length(tau) must equal size(V,2)"))
        end
    else # storev == 'R'
        k, n = size(V)
        if length(tau) != k
            throw(ArgumentError("For rowwise storage, length(tau) must equal size(V,1)"))
        end
    end

    # Call the core computational routine
    larft!(direct, storev, n, k, V, tau, T_mat)
end