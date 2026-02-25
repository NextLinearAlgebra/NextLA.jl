"""
    larf!(side, m, n, v, incv, tau, c, work)

Apply an elementary reflector H to a m-by-n matrix C from either 
the left or the right.

H = I - tau * v * v^H

where tau is a scalar and v is a vector.

# Arguments
- `side`: Character specifying the side of application
  - 'L': apply H from the left (H * C)
  - 'R': apply H from the right (C * H)
- `m`: Number of rows in matrix C
- `n`: Number of columns in matrix C  
- `v`: Array containing the elementary reflector vector
- `incv`: Increment for the elements of v (typically 1)
- `tau`: Scalar factor for the elementary reflector
- `c`: m-by-n matrix to be modified in-place
- `work`: Workspace array

# Algorithm
The elementary reflector H is applied optimally by exploiting the structure
of the reflector. The algorithm scans for the effective length of the reflector
vector and the effective dimensions of the matrix to minimize operations.

For side = 'L': Computes C := H * C = (I - tau * v * v^H) * C
For side = 'R': Computes C := C * H = C * (I - tau * v * v^H)

# Notes
This is a low-level computational routine used internally by higher-level
QR factorization algorithms. The workspace array must be properly allocated.
"""
function larf!(side::Char, m::Integer, n::Integer, v::AbstractVector{T}, incv::Integer, tau::T, C::AbstractMatrix{T}, work::AbstractVector{T}) where {T}
    one0 = oneunit(eltype(C))
    zero0 = zero(eltype(C))

    if tau == zero0
        return
    end

    # Use full dimensions (GPU-agnostic: avoids scalar indexing in ilazlc/ilazlr and v-scan)
    lastv = side == 'L' ? m : n
    lastc = side == 'L' ? n : m

    # Strided view of v (handles incv; for incv≠1 uses 1:incv:1+(lastv-1)*incv)
    vv = incv == 1 ? (@view v[1:lastv]) : (@view v[1:incv:1+(lastv-1)*incv])

    if side == 'L'
        # Form H*C = (I - tau * v * v^H) * C
        cv = @view C[1:lastv, 1:lastc]
        wv = @view work[1:lastc]

        # Step 1: w = C^H * v
        LinearAlgebra.generic_matvecmul!(wv, 'C', cv, vv, LinearAlgebra.MulAddMul(one0, zero0))

        # Step 2: C := C - tau * v * w^H (rank-1 update)
        gerc!(-tau, vv, wv, cv)
    else
        # Form C*H = C * (I - tau * v * v^H)
        cv = @view C[1:lastc, 1:lastv]
        wv = @view work[1:lastc]

        # Step 1: w = C * v
        LinearAlgebra.generic_matvecmul!(wv, 'N', cv, vv, LinearAlgebra.MulAddMul(one0, zero0))

        # Step 2: C := C - tau * w * v^H (rank-1 update)
        gerc!(-tau, wv, vv, cv)
    end
end

"""
    ilazlc(m, n, a) -> Int

Find the index of the last non-zero column in an m-by-n matrix.
Scans from column n backwards to column 1, checking all rows
in each column for non-zero elements.

# Arguments
- `m`: Number of rows in matrix a
- `n`: Number of columns in matrix a
- `a`: Matrix to scan

# Returns
- Index of last column containing at least one non-zero element,
  or 0 if all elements are zero
"""
function ilazlc(m, n, a)
    if n == 0
        return n
    end

    # Quick check of the last column boundaries
    if a[1,n] != 0 || a[m,n] != 0 
        return n
    end

    # Scan columns from right to left
    for j in n:-1:1
        for i in 1:m
            if a[i, j] != 0
                return j
            end
        end
    end
    
    return 0  # All elements are zero
end

"""
    ilazlr(m, n, a) -> Int

Find the index of the last non-zero row in an m-by-n matrix.
Scans all columns to determine the effective row dimension.

# Arguments
- `m`: Number of rows in matrix a
- `n`: Number of columns in matrix a  
- `a`: Matrix to scan

# Returns
- Index of last row containing at least one non-zero element,
  or 0 if all elements are zero
"""
function ilazlr(m, n, a)
    if m == 0
        return m
    end

    # Quick check of the last row boundaries
    if a[m,1] != 0 || a[m,n] != 0 
        return m
    end

    ila = 0

    # For each column, find the last non-zero row
    for j in 1:n
        i = m
        while (a[max(i,1), j] == 0) && (i > 1)
            i -= 1
        end
        ila = max(ila, i)
    end

    return ila
end

"""
    larf!(side, A, tau, C) -> C

Apply an elementary reflector H to a matrix C, where H = I - tau * A * A^H.

This is a high-level interface to the elementary reflector application routine.
The reflector can be applied from either the left (H*C) or right (C*H) side.

# Arguments
- `side`: Character specifying application side ('L' for left, 'R' for right)
- `A`: Vector defining the elementary reflector
- `tau`: Scalar factor for the reflector  
- `C`: Matrix to be transformed in-place

# Returns
- The modified matrix `C`

# Input Validation
- For side='L': length(A) must equal number of rows in C
- For side='R': length(A) must equal number of columns in C

# Example
```julia
# Apply reflector from left: C := H * C
larf!('L', v, tau, C)

# Apply reflector from right: C := C * H  
larf!('R', v, tau, C)
```
"""
function larf!(side::Char,  v::AbstractVector{T}, incv::Integer, tau::T, C::AbstractMatrix{T}) where {T}
    m, n = size(C)
    # Input validation with descriptive error messages
    if side == 'L'
        if length(v) != m
            throw(ArgumentError("For side='L', reflector length ($(length(v))) must equal matrix row dimension ($m)"))
        end
        work = similar(C, n)
    elseif side == 'R'
        if length(v) != n
            throw(ArgumentError("For side='R', reflector length ($(length(v))) must equal matrix column dimension ($n)"))
        end
        work = similar(C, m)
    else
        throw(ArgumentError("Invalid side parameter: '$side'. Must be 'L' or 'R'"))
    end
    # Call the core computational routine
    larf!(side, m, n, v, incv, tau, C, work)
end
