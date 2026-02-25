"""
    larft!(direct, storev, n, k, v, tau, T_mat)

Form the triangular factor T of a complex block reflector H of order n,
where H is defined as a product of k elementary reflectors.

The block reflector H has the form:
H = I - V * T * V^H

where V is n-by-k and contains the elementary reflector vectors, and T is
the k-by-k upper triangular factor computed by this routine.

Implemented as a KernelAbstractions kernel (single work item), runs on any KA backend
(CPU, CUDA, ROCm, oneAPI, Metal) without CPU copies.

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
# Kernel: single work item runs full sequential larft loop (no scalar indexing from host)
# unsafe_indices=true: no @index(Global) needed for single work-item
@kernel unsafe_indices=true function larft_kernel!(direct::Char, storev::Char, V, tau, T_mat, work, n::Int, k::Int)
    @uniform T = eltype(V)
    @uniform zero0 = zero(T)
    @uniform one0 = oneunit(T)

    if direct == 'F'
        prevlastv = n
        for i in 1:k
            prevlastv = max(prevlastv, i)
            tau_i = @inbounds tau[i]
            if tau_i == zero0
                for j in 1:i
                    @inbounds T_mat[j, i] = zero0
                end
            else
                if storev == 'C'
                    lastv = n
                    while lastv >= i + 1
                        if @inbounds V[lastv, i] != zero0
                            break
                        end
                        lastv -= 1
                    end
                    j = min(lastv, prevlastv)
                    # T[1:i-1,i] := -tau[i] * conj(V[i,1:i-1])
                    for p in 1:i-1
                        @inbounds T_mat[p, i] = -tau_i * conj(@inbounds V[i, p])
                    end
                    # T[1:i-1,i] += -tau[i] * V[i+1:j,1:i-1]^H * V[i+1:j,i]
                    if i + 1 <= j
                        for p in 1:i-1
                            acc = zero0
                            for r in (i+1):j
                                acc += conj(@inbounds V[r, p]) * @inbounds V[r, i]
                            end
                            @inbounds T_mat[p, i] += (-tau_i) * acc
                        end
                    end
                    prevlastv = (i > 1) ? max(prevlastv, lastv) : lastv
                else  # storev == 'R'
                    lastv = n
                    while lastv >= i + 1
                        if @inbounds V[i, lastv] != zero0
                            break
                        end
                        lastv -= 1
                    end
                    j = min(lastv, prevlastv)
                    # T[1:i-1,i] := -tau[i] * V[1:i-1,i]
                    for p in 1:i-1
                        @inbounds T_mat[p, i] = -tau_i * @inbounds V[p, i]
                    end
                    # T[1:i-1,i] += -tau[i] * V[1:i-1,i+1:j] * V[i,i+1:j]^H
                    if i - 1 > 0 && i + 1 <= j
                        for p in 1:i-1
                            acc = zero0
                            for r in (i+1):j
                                acc += @inbounds V[p, r] * conj(@inbounds V[i, r])
                            end
                            @inbounds T_mat[p, i] += (-tau_i) * acc
                        end
                    end
                    prevlastv = (i > 1) ? max(prevlastv, lastv) : lastv
                end
                # work[1:i-1] := T[1:i-1,i] (copy for triangular multiply)
                for p in 1:i-1
                    @inbounds work[p] = @inbounds T_mat[p, i]
                end
                # T[1:i-1,i] := T[1:i-1,1:i-1] * work[1:i-1] (upper tri mat-vec)
                for p in 1:i-1
                    acc = zero0
                    for q in p:i-1
                        acc += @inbounds T_mat[p, q] * @inbounds work[q]
                    end
                    @inbounds T_mat[p, i] = acc
                end
                @inbounds T_mat[i, i] = tau_i
            end
        end
    else  # direct == 'B'
        prevlastv = 1
        for i in k:-1:1
            tau_i = @inbounds tau[i]
            if tau_i == zero0
                for j in i:k
                    @inbounds T_mat[j, i] = zero0
                end
            else
                if i < k
                    if storev == 'C'
                        lastv = 1
                        while lastv <= i - 1
                            if @inbounds V[lastv, i] != zero0
                                break
                            end
                            lastv += 1
                        end
                        j = max(lastv, prevlastv)
                        # T[i+1:k,i] := -tau[i] * conj(V[n-k+i, i+1:k])
                        for p in (i+1):k
                            @inbounds T_mat[p, i] = -tau_i * conj(@inbounds V[n - k + i, p])
                        end
                        # T[i+1:k,i] += -tau[i] * V[j:n-k+i, i+1:k]^H * V[j:n-k+i, k]
                        if j <= n - k + i
                            for p in (i+1):k
                                acc = zero0
                                for r in j:(n - k + i)
                                    acc += conj(@inbounds V[r, p]) * @inbounds V[r, k]
                                end
                                @inbounds T_mat[p, i] += (-tau_i) * acc
                            end
                        end
                        prevlastv = (i > 1) ? min(prevlastv, lastv) : lastv
                    else  # storev == 'R'
                        lastv = 1
                        while lastv <= i - 1
                            if @inbounds V[lastv, i] != zero0
                                break
                            end
                            lastv += 1
                        end
                        j = max(lastv, prevlastv)
                        # T[i+1:k,i] := -tau[i] * V[i+1:k, n-k+i]
                        for p in (i+1):k
                            @inbounds T_mat[p, i] = -tau_i * @inbounds V[p, n - k + i]
                        end
                        # T[i+1:k,i] += -tau[i] * V[i+1:k, j:n-k+i-1] * V[i, j:n-k+i-1]^H
                        if i + 1 <= k && j <= n - k + i - 1
                            for p in (i+1):k
                                acc = zero0
                                for r in j:(n - k + i - 1)
                                    acc += @inbounds V[p, r] * conj(@inbounds V[i, r])
                                end
                                @inbounds T_mat[p, i] += (-tau_i) * acc
                            end
                        end
                        prevlastv = (i > 1) ? min(prevlastv, lastv) : lastv
                    end
                    # work[i+1:k] := T[i+1:k,i]
                    for p in (i+1):k
                        @inbounds work[p] = @inbounds T_mat[p, i]
                    end
                    # T[i+1:k,i] := T[i+1:k,i+1:k] * work[i+1:k] (lower tri mat-vec)
                    for p in (i+1):k
                        acc = zero0
                        for q in (i+1):p
                            acc += @inbounds T_mat[p, q] * @inbounds work[q]
                        end
                        @inbounds T_mat[p, i] = acc
                    end
                end
                @inbounds T_mat[i, i] = tau_i
            end
        end
    end
end

function larft!(direct::Char, storev::Char, n::Integer, k::Integer, V::AbstractMatrix{T}, tau::AbstractVector{T}, T_mat::AbstractMatrix{T}) where {T}
    if n == 0
        return
    end
    work = similar(tau, k)
    backend = KernelAbstractions.get_backend(V)
    larft_kernel!(backend, 1)(direct, storev, V, tau, T_mat, work, n, k, ndrange=1)
    KernelAbstractions.synchronize(backend)
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