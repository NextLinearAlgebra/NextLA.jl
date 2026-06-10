export lu_recursive_mixed!, lu_recursive!
using CUDA

"""
    lu_recursive!(A::AbstractMatrix, block_size::Int)

Flat recursive unpivoted LU factorization driver.
Bypasses the mixed-precision struct to directly chunk flat arrays, 
keeping the bulk of operations in their native precision.
"""
function lu_recursive!(A::AbstractMatrix, block_size::Int)
    n = size(A, 1)

    # 1. Base Case: We've chopped it down small enough for CUSOLVER
    if n <= block_size
        if eltype(A) == Float16
            # The memory-taxing spoof happens ONLY on tiny chunks now!
            A_f32 = Float32.(A)
            CUSOLVER.getrf!(A_f32)
            A .= Float16.(A_f32)
        else
            CUSOLVER.getrf!(A)
        end
        return
    end

    # 2. Recursive Split (Using the largest power of 2 less than N)
    n1 = 2^floor(Int, log2(n)) ÷ 2

    # View subblocks
    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # 3. Factor Top-Left: A11 <- LU(A11)
    lu_recursive!(A11, block_size)

    # 4. Solve Top-Right: A12 <- L11^{-1} A12 
    # (Side='L', Uplo='L', Trans='N', Diag='U')
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A11, A12)

    # 5. Solve Bottom-Left: A21 <- A21 U11^{-1} 
    # (Side='R', Uplo='U', Trans='N', Diag='N')
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A11, A21)

    # 6. Update Bottom-Right: A22 <- A22 - A21 * A12
    recgemm!(-1.0f0, A21, A12, 1.0f0, A22)

    # 7. Factor Bottom-Right
    lu_recursive!(A22, block_size)
end

"""
    lu_recursive_mixed!(A::FullMixedPrec{T}, threshold::Int) where T

Performs a recursive LU factorization (without pivoting) on a `FullMixedPrec` matrix.
Now uses the flat `lu_recursive!` driver at the leaf nodes for maximum performance.
"""
function lu_recursive_mixed!(A::FullMixedPrec{T_Base}, block_size::Int=2048) where {T_Base}
    # 1. Base Case: If we have reached a leaf node with data
    if A.BaseCase !== nothing
        # Hand off to the flat recursive driver to chunk it further if needed!
        lu_recursive!(A.BaseCase, block_size)
        return
    end

    # 2. Recursive Step: Strictly follow the tree structure
    lu_recursive_mixed!(A.A11, block_size)

    # 3. TRSM updates
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    # 4. GEMM update
    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22)

    # 5. Recursive Step
    lu_recursive_mixed!(A.A22, block_size)
end