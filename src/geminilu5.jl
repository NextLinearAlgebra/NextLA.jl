using LinearAlgebra
using StochasticRounding
include("fullmixedprec.jl")
include("rectrxm.jl")
include("wrappers.jl")
# Assumes a nonpivoting CUSOLVER wrapper getrf!(A) is available via wrappers.jl

"""
    getrf_recursive!(A, block_size)

Performs an in-place, nonpivoting, nested recursive block LU factorization on the dense matrix `A`.
The recursion dynamically splits the matrix until the sub-block size is less than or 
equal to `block_size`, at which point it falls back to standard hardware CUSOLVER routines (`getrf!`).
f"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        getrf!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2  
    
    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Step 1: Factorize upper-left block: A11 -> L11 U11
    getrf_recursive!(A11, block_size)

    # Step 2: Update upper-right block: Solve L11 * U12 = A12 (Unit lower triangular, Left side)
    if eltype(A11) == Float16
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
    end

    # Step 3: Update lower-left block: Solve L21 * U11 = A21 (Non-unit upper triangular, Right side)
    if eltype(A11) == Float16
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    # Step 4: Schur complement update: A22 <- A22 - (A21 * A12)
    if eltype(A21) == Float16
        recgemm!(-1.0, A21, A12, 1.0, A22)
    else
        gemm!('N', 'N', -1.0, A21, A12, 1.0, A22)
    end
    
    # Step 5: Factorize updated lower-right block: A22 -> L22 U22
    getrf_recursive!(A22, block_size)
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs an in-place, nonpivoting, nested recursive block LU factorization on a full 
mixed-precision matrix structure `A`. The recursion handles off-diagonal block updates 
through precision hierarchies and falls back to standard hardware CUSOLVER routines at the base case.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Step 1: Factorize upper-left block: A11 -> L11 U11
    getrf_recursive!(A.A11) 

    # Step 2: Update upper-right block: Solve L11 * U12 = A12 (Unit lower triangular, Left side)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A.A11, A.A12)

    # Step 3: Update lower-left block: Solve L21 * U11 = A21 (Non-unit upper triangular, Right side)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A.A11, A.A21)

    # Step 4: Schur complement update: A22 <- A22 - (A21 * A12)
    recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)

    # Step 5: Factorize updated lower-right block: A22 -> L22 U22
    getrf_recursive!(A.A22)
end

if hasproperty(A, :A21)
    OffDiag_block = (uplo == 'L') ? A.A21 : A.A12
else
    OffDiag_block = A.OffDiag
end
