Perfect — here is a drop-in recursive nonpivoting in-place LU implementation that matches your constraints and style.

using LinearAlgebra
include("wrappers.jl")      # should expose CUSOLVER LU base-case wrapper
include("matmul.jl")        # recgemm!
include("rectrxm.jl")       # unified_rectrxm!
include("fullmixedprec.jl") # FullMixedPrec

export lu_nopiv_recursive!

# ------------------------------------------------------------------------------
# Base-case LU (dense): must be CUSOLVER non-pivoting in your wrappers
# ------------------------------------------------------------------------------
"""
    lu_basecase_nopiv!(A)

Base-case dense LU factorization in-place.
This must map to CUSOLVER non-pivoting LU in `wrappers.jl`.
"""
function lu_basecase_nopiv!(A::AbstractMatrix)
    # IMPORTANT:
    # Ensure your wrapper `getrf!(A)` is configured as NON-PIVOTING CUSOLVER LU.
    # If your wrapper name is different (e.g., getrf_nopiv!), replace this call.
    getrf!(A)
    return A
end

# ------------------------------------------------------------------------------
# Dense recursive LU
# ------------------------------------------------------------------------------
"""
    lu_nopiv_recursive!(A::AbstractMatrix, block_size::Int=256)

In-place nested recursive block LU (nonpivoting) on a dense matrix.

Block equations:
1) A11 = L11*U11
2) U12 = L11^{-1} * A12
3) L21 = A21 * U11^{-1}
4) A22 <- A22 - L21*U12
5) A22 = L22*U22
"""
function lu_nopiv_recursive!(A::AbstractMatrix, block_size::Int=256)
    n, m = size(A)
    @assert n == m "LU requires a square matrix"

    if n <= block_size
        lu_basecase_nopiv!(A)
        return A
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    A11 = @view A[1:mid,     1:mid]
    A12 = @view A[1:mid,     mid+1:n]
    A21 = @view A[mid+1:n,   1:mid]
    A22 = @view A[mid+1:n,   mid+1:n]

    # 1) A11 -> L11 U11
    lu_nopiv_recursive!(A11, block_size)

    # 2) A12 <- L11^{-1} A12   (L11 is unit lower)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A11, A12)

    # 3) A21 <- A21 U11^{-1}   (U11 is upper non-unit)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A11, A21)

    # 4) A22 <- A22 - A21*A12
    recgemm!(-1.0f0, A21, A12, 1.0f0, A22)

    # 5) A22 -> L22 U22
    lu_nopiv_recursive!(A22, block_size)

    return A
end

# ------------------------------------------------------------------------------
# FullMixedPrec recursive LU
# ------------------------------------------------------------------------------
"""
    lu_nopiv_recursive!(A::FullMixedPrec, block_size::Int=4096)

In-place nested recursive block LU (nonpivoting) on `FullMixedPrec`.

Uses:
- `unified_rectrxm!` for triangular solves on A12/A21 updates
- `recgemm!` for Schur complement
- Dense CUSOLVER base-case at `A.BaseCase`
"""
function lu_nopiv_recursive!(A::FullMixedPrec, block_size::Int=4096)
    if A.BaseCase !== nothing
        lu_nopiv_recursive!(A.BaseCase, block_size)
        return A
    end

    # 1) A11 -> L11 U11
    lu_nopiv_recursive!(A.A11, block_size)

    # 2) A12 <- L11^{-1} A12
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)

    # 3) A21 <- A21 U11^{-1}
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    # 4) A22 <- A22 - A21*A12
    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22)

    # 5) A22 -> L22 U22
    lu_nopiv_recursive!(A.A22, block_size)

    return A
end
Notes
This is fully in-place and follows your exact operation order.
No mul!, no /, no \ for block updates.
For correctness with your “nonpivoting + CUSOLVER base case” requirement, the only thing you must ensure is that getrf!(A) in wrappers.jl is truly no-pivot.
If your wrapper is named differently (e.g. getrf_nopiv!), just swap that one line in lu_basecase_nopiv!.
If you want, I can also give you a tiny extract_LU(A) utility for FullMixedPrec that reconstructs explicit L and U (with unit diagonal in L) for residual checks.

