include("wrappers.jl")
using CUDA
using StochasticRounding

"""
    _syrk_dispatch!(op::Symbol, alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, beta::Number, C::CUDA.StridedCuArray)

Handles type-conversion and hardware routing for symmetric rank-k updates and general matrix multiplications.
"""
function _syrk_dispatch!(
    op::Symbol,
    alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, beta::Number, C::CUDA.StridedCuArray
)
    TC = eltype(C)
    TA = eltype(A)

    if op === :SYRK
        if TA == TC && TC in (Float32, Float64)
            syrk!('L', 'N', TC(alpha), A, TC(beta), C)
        elseif TA == Float16 && TC in (Float16, Float32)
            gemmEx!('N', 'T', alpha, A, A, beta, C)
        else
            compute_type = Float32
            
            C_temp = (TC == compute_type) ? C : compute_type.(C)

            if TA == Float32
                syrk!('L', 'N', compute_type(alpha), A, compute_type(beta), C_temp)
            elseif TA == Float16
                gemmEx!('N', 'T', alpha, A, A, beta, C_temp)
            else
                A_temp = compute_type.(A)
                syrk!('L', 'N', compute_type(alpha), A_temp, compute_type(beta), C_temp)
            end
            
            if C !== C_temp
                copy!(C, C_temp)
            end
        end

    elseif op === :GEMM
        TB = eltype(B)
        if TA == TB == TC && TC in (Float32, Float64)
            gemm!('N', 'T', TC(alpha), A, B, TC(beta), C)
        elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
            gemmEx!('N', 'T', alpha, A, B, beta, C)
        else
            A_final = (TA == TC) ? A : TC.(A)
            B_final = (TB == TC) ? B : TC.(B)
            if TC in (Float32, Float64)
                gemm!('N', 'T', TC(alpha), A_final, B_final, TC(beta), C)
            else 
                gemmEx!('N', 'T', alpha, A_final, B_final, beta, C)
            end
        end
    end
end

"""
    _recsyrk_impl!(alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int; parallel::Bool)

Internal implementation for nested recursive symmetric rank-k updates.
Recursively divides matrices into sub-blocks and applies updates in-place, falling back to standard hardware routines using the dispatch helper at the base case.
"""
function _recsyrk_impl!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int; parallel::Bool
)
    n = size(C, 1)
    if n <= threshold
        _syrk_dispatch!(:SYRK, alpha, A, A, beta, C)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2
    m = size(A, 2)

    A1 = @view A[1:n1, 1:m]; A2 = @view A[n1+1:end, 1:m]
    C11 = @view C[1:n1, 1:n1]; C21 = @view C[n1+1:end, 1:n1]; C22 = @view C[n1+1:end, n1+1:end]

    _syrk_dispatch!(:GEMM, alpha, A2, A1, beta, C21)

    if parallel
        @sync begin
            @async _recsyrk_impl!(alpha, A1, beta, C11, threshold, parallel=false)
            @async _recsyrk_impl!(alpha, A2, beta, C22, threshold, parallel=false)
        end
    else
        _recsyrk_impl!(alpha, A1, beta, C11, threshold, parallel=false)
        _recsyrk_impl!(alpha, A2, beta, C22, threshold, parallel=false)
    end
end

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

const PARALLEL_THRESHOLD = 4096

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

"""
    recsyrk!(alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256)

Performs an in-place, nested recursive block symmetric rank-k update (`C = alpha*A*Aᵀ + beta*C`).
Falls back to standard hardware routines using the dispatch helper at the specified base case threshold.
"""
function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256
)
    should_parallelize = size(C, 1) > PARALLEL_THRESHOLD
    _recsyrk_impl!(alpha, A, beta, C, threshold, parallel=should_parallelize)
end