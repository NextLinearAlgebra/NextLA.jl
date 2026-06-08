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
function lu_recursive_mixed!(A::FullMixedPrec{T_Base}, threshold::Int) where {T_Base}
    n = size(A, 1)
    
    # 1 & 5. Base Case
    if n <= threshold
        if A.BaseCase !== nothing
            if eltype(A.BaseCase) == Float16
                A_f32 = Float32.(A.BaseCase)
                CUSOLVER.getrf!(A_f32)
                A.BaseCase .= Float16.(A_f32)
            else
                CUSOLVER.getrf!(A.BaseCase)
            end
        end
        return
    end

    # 1. Factor the Top-Left
    lu_recursive_mixed!(A.A11, threshold)

    # 2. Solve the Top-Right: A12 <- L11^{-1} * A12
    # L11 is Unit Lower Triangular, so diag='U', uplo='L', side='L', trans='N'
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)

    # 3. Solve the Bottom-Left: A21 <- A21 * U11^{-1}
    # U11 is Non-Unit Upper Triangular, so diag='N', uplo='U', side='R', trans='N'
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    # 4. Update the Bottom-Right: A22 <- A22 - A21 * A12
    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22)

    # 5. Factor the Bottom-Right
    lu_recursive_mixed!(A.A22, threshold)
end
