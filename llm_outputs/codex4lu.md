Perfect — given your constraints, this is the recursive nonpivoting in-place LU you want.
It uses:
FullMixedPrec as the recursive structure
unified_rectrxm! for triangular solves
recgemm! for Schur updates
CUSOLVER nonpivoting LU at the dense base case (through a wrapper)
export getrf_recursive!

using CUDA
include("wrappers.jl")   # expected to provide getrf_nopiv! base-case wrapper
include("rectrxm.jl")    # unified_rectrxm!
include("matmul.jl")     # recgemm!

const LU_BASECASE = 4096

"""
    lu_basecase_nopiv!(A)

Base-case nonpivoting LU factorization (in place), routed to CUSOLVER.
Expected wrapper in `wrappers.jl`: `getrf_nopiv!(A)`.
"""
function lu_basecase_nopiv!(A::AbstractMatrix)
    getrf_nopiv!(A)   # <-- CUSOLVER no-pivot call
    return A
end

"""
    getrf_recursive!(A::AbstractMatrix, block_size::Int=LU_BASECASE)

In-place nested recursive nonpivoting LU on dense matrix `A`.

Block form:
    A = [A11 A12; A21 A22]
    A11 = L11*U11
    A12 <- L11 \\ A12
    A21 <- A21 / U11
    A22 <- A22 - A21*A12
Then recurse on `A22`.

Uses:
- `unified_rectrxm!` for triangular solves
- `recgemm!` for Schur update
- CUSOLVER nonpivoting LU at base case
"""
function getrf_recursive!(A::AbstractMatrix, block_size::Int=LU_BASECASE)
    n = size(A, 1)
    @assert n == size(A, 2) "LU requires a square matrix"

    if n <= block_size
        lu_basecase_nopiv!(A)
        return A
    end

    # Match the structural style of the provided recursive Cholesky
    n1 = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1,       1:n1]
    A12 = @view A[1:n1,       n1+1:end]
    A21 = @view A[n1+1:end,   1:n1]
    A22 = @view A[n1+1:end,   n1+1:end]

    # 1) Factor A11
    getrf_recursive!(A11, block_size)

    # 2) Triangular solves (in place)
    # U12 = L11^{-1} * A12  (L11 unit-lower)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)

    # L21 = A21 * U11^{-1}  (U11 non-unit upper)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)

    # 3) Schur complement update: A22 -= A21*A12
    recgemm!(-1.0, A21, A12, 1.0, A22)

    # 4) Recurse
    getrf_recursive!(A22, block_size)

    return A
end

"""
    getrf_recursive!(A::FullMixedPrec)

In-place nested recursive nonpivoting LU on `FullMixedPrec`.

Recurses over the hierarchy and uses:
- `unified_rectrxm!` for mixed-precision triangular solves
- `recgemm!` for mixed-precision Schur updates
- dense CUSOLVER base case on `A.BaseCase`
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, LU_BASECASE)
        return A
    end

    # Hierarchical partition already encoded in A.A11, A.A12, A.A21, A.A22
    getrf_recursive!(A.A11)

    # U12 = L11^{-1} * A12
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A.A11, A.A12)

    # L21 = A21 * U11^{-1}
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A.A11, A.A21)

    # A22 -= A21*A12
    recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)

    getrf_recursive!(A.A22)

    return A
end
Notes
This is strictly nonpivoting LU (A = LU, no permutations).
It is fully in-place in the standard packed LU layout.
If your wrapper name differs (e.g., getrf! with a no-pivot flag), just replace getrf_nopiv!(A) accordingly.

