"""
    _recsyrk_impl!(alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec; parallel::Bool)

Internal implementation for nested recursive symmetric rank-k updates specifically for the `SymmMixedPrec` block structure.
Recursively divides matrices into sub-blocks and applies updates in-place, falling back to standard hardware routines using the dispatch helper at the base case.
"""
function _recsyrk_impl!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec;
    parallel::Bool
)
    if C.BaseCase !== nothing
        recsyrk!(alpha, A, beta, C.BaseCase, 4096)
        return
    end

    n1 = size(C.A11, 1)
    A1 = @view A[1:n1, :]; A2 = @view A[n1+1:end, :]

    _syrk_dispatch!(:GEMM, alpha, A2, A1, beta, C.OffDiag)

    if parallel
        @sync begin
            @async _recsyrk_impl!(alpha, A1, beta, C.A11, parallel=false)
            @async _recsyrk_impl!(alpha, A2, beta, C.A22, parallel=false)
        end
    else
        _recsyrk_impl!(alpha, A1, beta, C.A11, parallel=false)
        _recsyrk_impl!(alpha, A2, beta, C.A22, parallel=false)
    end
end

"""
    recsyrk!(alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec)

Performs an in-place, nested recursive block symmetric rank-k update on a symmetric mixed-precision matrix structure.
Falls back to standard hardware routines using the dispatch helper at the base case.
"""
function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec
)
    if C.BaseCase !== nothing
        recsyrk!(alpha, A, beta, C.BaseCase)
        return
    end
    n_subproblem = size(C.A11, 1)
    should_parallelize = n_subproblem > PARALLEL_THRESHOLD
    _recsyrk_impl!(alpha, A, beta, C, parallel=should_parallelize)
end
