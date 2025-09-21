using AMDGPU
using CUDA
using oneAPI
using StochasticRounding



function _syrk_dispatch!(
    op::Symbol,
    alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, beta::Number, C::CUDA.StridedCuArray
)
    TC = eltype(C)
    TA = eltype(A)

    if op === :SYRK
        if TA == TC && TC in (Float32, Float64)
            CUBLAS.syrk!('L', 'N', TC(alpha), A, TC(beta), C)
        elseif TA == Float16 && TC in (Float16, Float32)
            CUBLAS.gemmEx!('N', 'T', alpha, A, A, beta, C)
        else
            compute_type = Float32
            
            C_temp = (TC == compute_type) ? C : compute_type.(C)

            if TA == Float32
                CUBLAS.syrk!('L', 'N', compute_type(alpha), A, compute_type(beta), C_temp)
            elseif TA == Float16
                CUBLAS.gemmEx!('N', 'T', alpha, A, A, beta, C_temp)
            else
                A_temp = compute_type.(A)
                CUBLAS.syrk!('L', 'N', compute_type(alpha), A_temp, compute_type(beta), C_temp)
            end
            
            if C !== C_temp
                copy!(C, C_temp)
            end
        end

    elseif op === :GEMM
        TB = eltype(B)
        if TA == TB == TC && TC in (Float32, Float64)
            CUBLAS.gemm!('N', 'T', TC(alpha), A, B, TC(beta), C)
        elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
            CUBLAS.gemmEx!('N', 'T', alpha, A, B, beta, C)
        else
            # print("type a:", eltype(A), "type b:", eltype(B), "type c:", eltype(C))
            A_final = (TA == TC) ? A : TC.(A)
            B_final = (TB == TC) ? B : TC.(B)
            if TC in (Float32, Float64)
                CUBLAS.gemm!('N', 'T', TC(alpha), A_final, B_final, TC(beta), C)
            else 
                CUBLAS.gemmEx!('N', 'T', alpha, A_final, B_final, beta, C)
            end
        end
    end
end

function _syrk_dispatch!(
    op::Symbol,
    alpha::Number, A::AMDGPU.StridedROCArray, B::AMDGPU.StridedROCArray, beta::Number, C::AMDGPU.StridedROCArray
)
    TC = eltype(C)
    TA = eltype(A)

    if op === :SYRK
        if TA == TC && TC in (Float32, Float64)
            syrk!('L', 'N', TC(alpha), A, TC(beta), C)
        elseif TA == Float16 && TC in (Float16, Float32)
            gemmEx!('N', 'T', alpha, A, A, beta, C)
        else
            A_converted = TC.(A)
            syrk!('L', 'N', TC(alpha), A_converted, TC(beta), C)
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
            gemm!('N', 'T', TC(alpha), A_final, B_final, TC(beta), C)
        end
    end
end

function _syrk_dispatch!(
    op::Symbol,
    alpha::Number, A::oneAPI.oneDeviceArray, B::oneAPI.oneDeviceArray, beta::Number, C::oneAPI.oneDeviceArray
)
    TC = eltype(C)
    TA = eltype(A)

    if op === :SYRK
        if TA == TC && TC in (Float32, Float64)
            oneMKL.syrk!('L', 'N', TC(alpha), A, TC(beta), C)
        elseif TA == Float16 && TC in (Float16, Float32)
            oneMKL.gemm!('N', 'T', alpha, A, A, beta, C)
        else
            A_converted = TC.(A)
            oneMKL.syrk!('L', 'N', TC(alpha), A_converted, TC(beta), C)
        end
    elseif op === :GEMM
        TB = eltype(B)
        if TA == TB == TC && TC in (Float32, Float64)
            oneMKL.gemm!('N', 'T', TC(alpha), A, B, TC(beta), C)
        elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
            oneMKL.gemm!('N', 'T', alpha, A, B, beta, C)
        else
            A_final = (TA == TC) ? A : TC.(A)
            B_final = (TB == TC) ? B : TC.(B)
            oneMKL.gemm!('N', 'T', TC(alpha), A_final, B_final, TC(beta), C)
        end
    end
end


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


function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix, threshold::Int=256
)
    should_parallelize = size(C, 1) > PARALLEL_THRESHOLD
    _recsyrk_impl!(alpha, A, beta, C, threshold, parallel=should_parallelize)
end
