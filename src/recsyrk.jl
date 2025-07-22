function GEMM_cublas!(C, A, B, alpha, beta)
    TC = eltype(C)

    if TC == Float16
        CUBLAS.gemmEx!('N', 'T', alpha, Float16.(A), Float16.(B), beta, C)
    else
        A_casted = (eltype(A) == TC) ? A : TC.(A)
        B_casted = (eltype(B) == TC) ? B : TC.(B)
        CUBLAS.gemm!('N', 'T', TC(alpha), A_casted, B_casted, TC(beta), C)
    end
end

# recurisve syrk with C = alpha * A * A^T + beta * C
function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec;
    gemm_order::Symbol=:first # Default to GEMM First
)
    # Base case remains unchanged
    if C.BaseCase !== nothing
        T_C = eltype(C.BaseCase)
        if T_C == Float16
            CUBLAS.gemmEx!('N', 'T', alpha, Float16.(A), Float16.(A), beta, C.BaseCase)
        else
            CUBLAS.syrk!('L', 'N', T_C(alpha), T_C.(A), T_C(beta), C.BaseCase)
        end
        return
    end

    n1 = size(C.A11, 1)
    A1 = @view A[1:n1, :]
    A2 = @view A[n1+1:end, :]

    # Use the parameter to determine the order of operations
    if gemm_order == :first
        GEMM_cublas!(C.OffDiag, A2, A1, alpha, beta)
        recsyrk!(alpha, A1, beta, C.A11; gemm_order=gemm_order)
        recsyrk!(alpha, A2, beta, C.A22; gemm_order=gemm_order)
    elseif gemm_order == :middle
        recsyrk!(alpha, A1, beta, C.A11; gemm_order=gemm_order)
        GEMM_cublas!(C.OffDiag, A2, A1, alpha, beta)
        recsyrk!(alpha, A2, beta, C.A22; gemm_order=gemm_order)
    elseif gemm_order == :end
        recsyrk!(alpha, A1, beta, C.A11; gemm_order=gemm_order)
        recsyrk!(alpha, A2, beta, C.A22; gemm_order=gemm_order)
        GEMM_cublas!(C.OffDiag, A2, A1, alpha, beta)
    else
        error("Invalid gemm_order: Must be :first, :middle, or :end")
    end
end
