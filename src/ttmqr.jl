function ttmqr!(side::Char, trans::Char, m1::Integer, n1::Integer, m2::Integer, n2::Integer, k::Integer, ib::Integer, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, V::AbstractMatrix{T}, T_mat::AbstractMatrix{T}, work::AbstractVector{T}) where {T}
    # check input arguments
    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
    end

    if trans != 'N' && trans != 'C' 
        throw(ArgumentError("illegal value of trans"))
    end

    if m1 < 0 
        throw(ArgumentError("illegal value of m1"))
    end

    if n1 < 0
        throw(ArgumentError("illegal value of n1"))
    end

    if (m2 < 0) || (m2 != m1 && side == 'R')
        throw(ArgumentError("illegal value of m2"))
    end

    if (n2 < 0) || (n2 != n1 && side == 'L')
        throw(ArgumentError("illegal value of n2"))
    end

    if (k < 0) || (side == 'L' && k > m1) || (side == 'R' && k > n1)
        throw(ArgumentError("illegal value of k"))
    end

    if ib < 0
        throw(ArgumentError("illegal value of ib"))
    end

    # quick return
    if m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0 || k == 0 || ib == 0
        return
    end

    if (side == 'L' && trans != 'N') || (side == 'R' && trans == 'N')
        i1 = 1  # Starting from 1 in Julia (0 in C)
        i3 = ib
    else
        i1 = div(k-1, ib) * ib + 1  # Convert from 0-based to 1-based indexing
        i3 = -ib
    end  

    # Main loop - replicate PLASMA's condition: i > -1 && i < k
    i = i1
    while i >= 1 && i <= k
        kb = min(ib, k-i+1)
        ic = 1
        jc = 1
        mi = m1
        ni = n1
        mi2 = m2
        ni2 = n2
        l = 0

        if side == 'L'
            # Apply from left on the current block rows
            mi = kb
            mi2 = min(i + kb - 1, m2)
            ic = i
            l = min(kb, max(0, m2 - i))
            # Workspace as kb x ni
            W = reshape(@view(work[1:kb*ni]), kb, ni)
            parfb!('L', trans, 'F', 'C', mi, ni, mi2, ni2, kb, l,
            (@view A1[ic:ic+mi-1, jc:jc+ni-1]),
            (@view A2[1:mi2, 1:ni2]),
            (@view V[1:m2, i:i+kb-1]),
            (@view T_mat[1:kb, i:i+kb-1]),
            W)
        else
            # Apply from right on the current block columns
            ni = kb
            ni2 = min(i + kb - 1, n2)
            jc = i
            l = min(kb, max(0, n2 - i))
            # Workspace as mi x kb
            W = reshape(@view(work[1:mi*kb]), mi, kb)
            parfb!('R', trans, 'F', 'C', mi, ni, mi2, ni2, kb, l,
                   (@view A1[ic:ic+mi-1, jc:jc+ni-1]),
                   (@view A2[1:mi2, 1:ni2]),
                   (@view V[1:n2, i:i+kb-1]),
                   (@view T_mat[1:kb, i:i+kb-1]),
                   W)
        end
        
        i += i3
    end
end

"""
    ttmqr!(side, trans, A1, A2, V, T, ib) -> (A1, A2)
    
Helper function for triangular-trapezoidal matrix transformation.

# Arguments
- `side`: 'L' (left) or 'R' (right)
- `trans`: 'N' (no transpose) or 'C' (conjugate transpose)  
- `A1`: Upper triangular matrix to be updated
- `A2`: Trapezoidal matrix to be updated
- `V`: Reflector vectors matrix
- `T`: Block reflector matrix
- `ib`: Block size

# Returns  
- Modified `A1` and `A2`
"""
function ttmqr!(side::Char, trans::Char, A1::AbstractMatrix{T}, A2::AbstractMatrix{T},
         V::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}, ib::Integer) where T
    m1, n1 = size(A1)
    m2, n2 = size(A2)
    # Use the common number of reflectors available in V and T
    k = size(T_matrix, 2)

    # Workspace size follows parfb!/TPMQRT requirements
    # - Left: W is (ib x n1) at most
    # - Right: W is (m1 x ib) at most
    work_size = side == 'L' ? ib * n1 : m1 * ib
    work = zeros(T, work_size)

    ttmqr!(side, trans, m1, n1, m2, n2, k, ib, A1, A2,
        V, T_matrix, work)
end
