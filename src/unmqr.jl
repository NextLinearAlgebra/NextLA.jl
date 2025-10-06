"""
    unmqr!(side, trans, m, n, k, ib, A, lda, T_matrix, C, work)

Apply orthogonal matrix Q (or Q^H) from a QR factorization to a general matrix C.

Overwrites the general m-by-n matrix C with:
                    side = 'L'        side = 'R'
    trans = 'N'       Q * C           C * Q
    trans = 'C'       Q^H * C         C * Q^H

where Q is a unitary matrix defined as the product of k elementary reflectors:
Q = H(1) H(2) ... H(k)

as returned by geqrt!. Q is of order m if side = 'L' and of order n if side = 'R'.

# Arguments
- `side`: Character specifying which side to apply Q
  - 'L': Apply Q or Q^H from the left
  - 'R': Apply Q or Q^H from the right
- `trans`: Character specifying transpose operation
  - 'N': No transpose, apply Q
  - 'C': Conjugate transpose, apply Q^H
- `m`: Number of rows of matrix C (≥ 0)
- `n`: Number of columns of matrix C (≥ 0)
- `k`: Number of elementary reflectors defining Q
  - If side = 'L': m ≥ k ≥ 0
  - If side = 'R': n ≥ k ≥ 0
- `ib`: Inner block size (≥ 0)
- `A`: Matrix of dimension (lda, k) containing reflector vectors
  The i-th column contains the vector defining elementary reflector H(i),
  as returned by geqrt! in the first k columns
- `lda`: Leading dimension of array A
  - If side = 'L': lda ≥ max(1,m)
  - If side = 'R': lda ≥ max(1,n)
- `T`: ib×k triangular factor of the block reflector
  T is upper triangular by blocks (economic storage)
- `C`: m×n matrix to be transformed (modified in-place)
- `work`: Workspace array

# Algorithm
The routine applies Q using the compact WY representation stored in A and T.
It processes the elementary reflectors in blocks of size ib, using efficient
block operations (larfb!) for high performance.

The order of applying blocks depends on side and trans parameters to ensure
numerical stability and efficiency.

# Notes
This is a core computational routine for applying orthogonal transformations
from QR factorizations. It is typically called by higher-level interfaces.
"""
function unmqr!(side::Char, trans::Char, m::Integer, n::Integer, k::Integer, ib::Integer, A::AbstractMatrix{T}, lda::Integer, T_matrix::AbstractMatrix{T}, C::AbstractMatrix{T}, work::AbstractMatrix{T}) where {T}
    # Input validation with descriptive error messages
    if side != 'L' && side != 'R'
        throw(ArgumentError("side must be 'L' or 'R', got '$side'"))
    end

    if side == 'L'
        nq = m  # Order of Q when applied from left
        nw = n  # Width for workspace
    else
        nq = n  # Order of Q when applied from right  
        nw = m  # Width for workspace
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("trans must be 'N', 'C', or 'T', got '$trans'"))
    end

    if m < 0
        throw(ArgumentError("m must be non-negative, got $m"))
    end

    if n < 0
        throw(ArgumentError("n must be non-negative, got $n"))
    end

    if k < 0 || k > nq
        throw(ArgumentError("k must satisfy 0 ≤ k ≤ $nq, got $k"))
    end

    if ib < 0 
        throw(ArgumentError("ib must be non-negative, got $ib"))
    end

    if lda < max(1, nq) && nq > 0
        throw(ArgumentError("lda must be ≥ max(1,$nq), got $lda"))
    end

    # Quick return for degenerate cases
    if m == 0 || n == 0 || k == 0
        return
    end

    # Determine order of applying reflector blocks
    if ((side == 'L' && trans != 'N') || (side == 'R' && trans == 'N'))
        # Apply blocks forward: 1, ib+1, 2*ib+1, ...
        i1 = 1
        i3 = ib
        ibstop = k
    else
        # Apply blocks backward: ..., 2*ib+1, ib+1, 1
        i1 = div((k-1),ib)*ib + 1
        i3 = -ib
        ibstop = 1
    end
    
    # Initialize submatrix indices
    ic = 1
    jc = 1
    ni = n
    mi = m

    # Allocate workspace for block operations
    if side == 'L'
        wwork = ones(eltype(A), n, ib)
        ldw = n
    else
        wwork = ones(eltype(A), m, ib)
        ldw = m
    end

    # Apply blocks of elementary reflectors
    for i in i1 : i3 : ibstop
        kb = min(ib, k-i+1)  # Size of current block

        if side == 'L'
            # Apply to C[i:m, 1:n]
            mi = m - i + 1
            ic = i
        else
            # Apply to C[1:m, i:n]
            ni = n - i + 1
            jc = i
        end

        # Get view of submatrix to transform
        cv = @view C[ic:m, jc:n]

        # Apply current block of reflectors
        larfb!(side, trans, 'F', 'C', mi, ni, kb,
            (@view A[i:lda, i:i+kb-1]), lda-i+1,
            (@view T_matrix[1:kb, i:i+kb-1]),
            cv, (@view wwork[:, 1:kb]))
    end
end

"""
    unmqr!(side, trans, A_qr, T, C, ib) -> C
    
Apply orthogonal matrix Q from QR factorization to matrix C.

This is a high-level interface that automatically determines dimensions and
allocates workspace to apply the orthogonal factor Q from a QR factorization
to a general matrix C.

# Arguments
- `side`: Character specifying application side
  - 'L': Apply Q from left (Q*C or Q^H*C)
  - 'R': Apply Q from right (C*Q or C*Q^H)
- `trans`: Character specifying transpose operation  
  - 'N': Apply Q (no transpose)
  - 'C': Apply Q^H (conjugate transpose)
- `A_qr`: QR factorization result from geqrt! (contains reflector vectors)
- `T_matrix`: Block reflector coefficient matrix from geqrt!
- `C`: Matrix to transform (modified in-place)
- `ib`: Block size used in QR factorization

# Returns
- Modified matrix `C` after applying the orthogonal transformation

# Input Validation
- Matrix dimensions must be compatible with the QR factorization
- Block size ib must be positive and consistent with T matrix dimensions
- For side='L': number of rows of C must match Q dimension
- For side='R': number of columns of C must match Q dimension

# Example
```julia
# Apply Q from QR factorization to matrix C
m, n, k = 10, 8, 6
ib = 4
A = randn(ComplexF64, m, k)
A_qr, T, tau = geqrt!(copy(A), ib)
C = randn(ComplexF64, m, n)
unmqr!('L', 'N', A_qr, T, C, ib)  # C := Q * C
```

# Algorithm
Uses the blocked compact WY representation to apply Q efficiently through
matrix-matrix operations rather than individual elementary reflectors.
"""
function unmqr!(side::Char, trans::Char, A::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}, 
               C::AbstractMatrix{T}) where {T}
    m, n = size(C)
    ib, k = size(T_matrix)

    
    # Validate input dimensions
    if ib <= 0
        throw(ArgumentError("Block size ib must be positive, got $ib"))
    end
    
    if side == 'L'
        if size(A, 1) != m
            throw(ArgumentError("For side='L', A_qr rows ($(size(A, 1))) must match C rows ($m)"))
        end
        if size(A, 2) < k
            throw(ArgumentError("A_qr columns ($(size(A, 2))) must be ≥ k ($k)"))
        end
    else  # side == 'R'
        if size(A, 1) != n
            throw(ArgumentError("For side='R', A_qr rows ($(size(A, 1))) must match C columns ($n)"))
        end
        if size(A, 2) < k
            throw(ArgumentError("A_qr columns ($(size(A, 2))) must be ≥ k ($k)"))
        end
    end
    
    # Set leading dimensions
    lda = max(1, stride(A, 2))
    
    # Allocate workspace based on side (matrix workspace expected by low-level)
    if side == 'L'
        work = zeros(T, n, ib)
    else
        work = zeros(T, m, ib)
    end
    
    # Call the core computational routine
    unmqr!(side, trans, m, n, k, ib, A, lda, T_matrix, C, work)
end
