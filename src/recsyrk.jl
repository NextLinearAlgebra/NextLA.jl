function _syrk_dispatch!(
    op::Symbol,
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix
)
    TC = eltype(C)
    TA = eltype(A)

    if op === :SYRK
        if TA == TC && TC in (Float32, Float64)
            CUBLAS.syrk!('L', 'N', TC(alpha), A, TC(beta), C)

        elseif TA == Float16 && TC in (Float16, Float32)
            CUBLAS.gemmEx!('N', 'T', alpha, A, A, beta, C)
        else
            A_converted = TC.(A)
            if TC in (Float32, Float64)
                CUBLAS.syrk!('L', 'N', TC(alpha), A_converted, TC(beta), C)
            else 
                CUBLAS.gemmEx!('N', 'T', alpha, A_converted, A_converted, beta, C)
            end
        end

    elseif op === :GEMM
        TB = eltype(B)

        if TA == TB == TC && TC in (Float32, Float64)
            CUBLAS.gemm!('N', 'T', TC(alpha), A, B, TC(beta), C)

        elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
            CUBLAS.gemmEx!('N', 'T', alpha, A, B, beta, C)
        else
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


function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int; parallel::Bool=true # Add the parallel flag
)
    n = size(C, 1)

    if n <= threshold
        _syrk_dispatch!(:SYRK, alpha, A, A, beta, C)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2
    m = size(A, 2)

    A1 = @view A[1:n1, 1:m]
    A2 = @view A[n1+1:end, 1:m]

    C11 = @view C[1:n1, 1:n1]
    C21 = @view C[n1+1:end, 1:n1]
    C22 = @view C[n1+1:end, n1+1:end]

    _syrk_dispatch!(:GEMM, alpha, A2, A1, beta, C21)

    if parallel
        # If parallel, create tasks but then call recursively with parallel=false
        @sync begin
            @async recsyrk!(alpha, A1, beta, C11, threshold, parallel=false)
            @async recsyrk!(alpha, A2, beta, C22, threshold, parallel=false)
        end
    else
        # If not parallel, just execute sequentially
        recsyrk!(alpha, A1, beta, C11, threshold, parallel=false)
        recsyrk!(alpha, A2, beta, C22, threshold, parallel=false)
    end
end


function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec;
    parallel::Bool=true # Add the parallel flag
)
    if C.BaseCase !== nothing
        # Propagate the parallel flag down to the base implementation
        recsyrk!(alpha, A, beta, C.BaseCase, 256, parallel=parallel)
        return
    end

    n1 = size(C.A11, 1)
    A1 = @view A[1:n1, :]
    A2 = @view A[n1+1:end, :]

    _syrk_dispatch!(:GEMM, alpha, A2, A1, beta, C.OffDiag)

    if parallel
        # If parallel, create tasks but then call recursively with parallel=false
        @sync begin
            @async recsyrk!(alpha, A1, beta, C.A11, parallel=false)
            @async recsyrk!(alpha, A2, beta, C.A22, parallel=false)
        end
    else
        # If not parallel, just execute sequentially
        recsyrk!(alpha, A1, beta, C.A11, parallel=false)
        recsyrk!(alpha, A2, beta, C.A22, parallel=false)
    end
end