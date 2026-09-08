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
