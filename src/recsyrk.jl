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
    alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec
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

    GEMM_cublas!(C.OffDiag, A2, A1, alpha, beta)
    recsyrk!(alpha, A1, beta, C.A11)
    recsyrk!(alpha, A2, beta, C.A22)

end
