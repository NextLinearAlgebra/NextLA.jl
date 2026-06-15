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

    if n <= block_size
        if eltype(A) == Float16
            A_f32 = Float32.(A)
            CUSOLVER.getrf!(A_f32)
            A .= Float16.(A_f32)
        else
            CUSOLVER.getrf!(A)
        end
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    lu_recursive!(A11, block_size)

    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A11, A12)

    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A11, A21)

    recgemm!(-1.0f0, A21, A12, 1.0f0, A22)

    lu_recursive!(A22, block_size)
end

"""
    lu_recursive_mixed!(A::FullMixedPrec{T}, threshold::Int) where T

Performs a recursive LU factorization (without pivoting) on a `FullMixedPrec` matrix.
Now uses the flat `lu_recursive!` driver at the leaf nodes for maximum performance.
"""
function lu_recursive_mixed!(A::FullMixedPrec{T_Base}, block_size::Int=2048) where {T_Base}
    if A.BaseCase !== nothing
        lu_recursive!(A.BaseCase, block_size)
        return
    end

    lu_recursive_mixed!(A.A11, block_size)

    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22)

    lu_recursive_mixed!(A.A22, block_size)
end