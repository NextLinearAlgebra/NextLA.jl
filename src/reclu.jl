export lu_recursive_mixed!
using CUDA

"""
    lu_recursive_mixed!(A::FullMixedPrec{T}, threshold::Int) where T

Performs a recursive LU factorization (without pivoting) on a `FullMixedPrec` matrix.
Following the framework:
1. Factor the Top-Left: A11 <- LU(A11)
2. Solve the Top-Right: A12 <- L11^{-1} A12
3. Solve the Bottom-Left: A21 <- A21 U11^{-1}
4. Update the Bottom-Right: A22 <- A22 - A21 * A12
5. Factor the Bottom-Right: A22 <- LU(A22)
"""
function lu_recursive_mixed!(A::FullMixedPrec{T_Base}) where {T_Base}
    # 1. Base Case: If we have reached a leaf node with data
    if A.BaseCase !== nothing
        # Perform LU on the base case
        if eltype(A.BaseCase) == Float16
            A_f32 = Float32.(A.BaseCase)
            CUSOLVER.getrf!(A_f32)
            A.BaseCase .= Float16.(A_f32)
        else
            CUSOLVER.getrf!(A.BaseCase)
        end
        return
    end

    # 2. Recursive Step: Strictly follow the tree structure
    # Since we are not at a leaf, A.A11 and A.A22 must exist
    lu_recursive_mixed!(A.A11)

    # 3. TRSM updates
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    # 4. GEMM update
    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22)

    # 5. Recursive Step
    lu_recursive_mixed!(A.A22)
end