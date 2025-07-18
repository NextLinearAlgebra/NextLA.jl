function GEMM_cublas!(C, A, B, alpha, beta)
    if eltype(A) == Float16 && eltype(B) == Float16
        if eltype(C) == Float16
            C_op = Float32.(C)
            CUBLAS.gemmEx!('N', 'T', alpha, A, B, beta, C_op)
            clamp!(C_op, floatmin(Float16), floatmax(Float16))
            copy!(C, C_op)
        else
            CUBLAS.gemmEx!('N', 'T', alpha, A, B, beta, C)
        end
    else
        T_C = eltype(C)
        CUBLAS.gemm!('N', 'T', T_C(alpha), A, B, T_C(beta), C)
    end
end

# recurisve syrk with C = alpha * A * A^T + beta * C
function recsyrk!(
    alpha::Number, A::AbstractMatrix, beta::Number, C::SymmMixedPrec
)
    if C.BaseCase !== nothing
        T_C = eltype(C.BaseCase)
        CUBLAS.syrk!('L', 'N', T_C(alpha), T_C.(A), T_C(beta), C.BaseCase)
        return
    end
    n1 = size(C.A11, 1)
    A1 = @view A[1:n1, :]
    A2 = @view A[n1+1:end, :]
    GEMM_cublas!(C.OffDiag, eltype(C.OffDiag).(A2), eltype(C.OffDiag).(A1), alpha, beta)
    recsyrk!(alpha, A1, beta, C.A11)
    recsyrk!(alpha, A2, beta, C.A22)
    
end



"""
Tests that recsyrk! matches the output of the standard CUBLAS.syrk!.
"""
function test_recsyrk()
    n_values = [256, 512, 1024] # Sizes for the symmetric matrix C
    m_values = [64, 128, 256]   # Column sizes for the update matrix A

    # --- Loop through all combinations of n and m ---
    for n in n_values
        for m in m_values
            println("-"^50)
            println("Testing recsyrk! for C(n x n)=$n, A(n x m)=$m")

            # --- 1. Setup ---
            precisions = [Float32, Float64]
            T_out = precisions[end]
            alpha, beta = -1.0, 1.0

            # --- 2. Create Input Data on GPU ---
            d_A = CuArray(randn(T_out, n, m))
            h_C = randn(T_out, n, n)
            h_C = h_C * h_C' # Make it symmetric
            d_C_orig = CuArray(h_C)

            # --- 3. Run Custom Recursive Version ---
            C_for_custom = copy(d_C_orig)
            C_mixed = SymmMixedPrec(C_for_custom, 'L'; precisions=precisions)
            recsyrk!(alpha, d_A, beta, C_mixed)

            # --- 4. Run Standard BLAS Version (for Ground Truth) ---
            C_for_blas = copy(d_C_orig)
            CUBLAS.syrk!('L', 'N', alpha, d_A, beta, C_for_blas)

            # --- 5. Compare Results ---
            C_custom_result = reconstruct_matrix(C_mixed)
            
            error_norm = norm(tril(C_custom_result) - tril(C_for_blas))
            truth_norm = norm(tril(C_for_blas))
            relative_error = error_norm / truth_norm

            println("Relative Error vs. CUBLAS: ", relative_error)
            @assert relative_error < 1e-5 "Error is too high!"
            println("Test PASSED ✅")
        end
    end
    println("-"^50)
    println("All recsyrk! tests passed successfully!")
end

# recunstructs recursive matrix
function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end

    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.OffDiag
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = CuArray{T_Base}(undef, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end



test_recsyrk()



