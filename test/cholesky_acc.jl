using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions


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

function check_cholesky_accuracy()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
    )
    mixed_scenarios = Dict(
        "[F32, F64, F64, F64]" => [Float32, Float64, Float64, Float64],
        "[F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "[F32, F32, F64]" => [Float32, Float32, Float64],
        "[F32, F64, F64]" => [Float32, Float64, Float64],
        "[F16, F32, F32]" => [Float16, Float32, Float32],
        "[F16, F16, F32]" => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F64]" => [Float16, Float32, Float64],
        "[F32, F64]"      => [Float32, Float64],
        "[F16, F64]"      => [Float16, Float64],
        "[F16, F32]"      => [Float16, Float32],
    )

    println("🔬 Starting Cholesky Accuracy Check...")

    for n in n_values
        println("\n" * "="^80)
        println("Checking Accuracy for Matrix Size (n x n) = $n x $n")
        
        A_cpu = randn(Float64, n, n)
        A_spd_fp64 = CuArray(A_cpu * A_cpu' + (n*100) * I)
        
        A_ground_truth = copy(A_spd_fp64)
        CUSOLVER.potrf!('L', A_ground_truth)
        L_truth = tril(A_ground_truth)
        
        # Handle pure scenarios
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[1] 
            A_pure_input = T_prec.(A_spd_fp64)
            potrf_recursive!(A_pure_input, 4096) 
            L_result = tril(A_pure_input)
            
            error_norm = norm(L_result - L_truth)
            relative_error = max(error_norm / norm(L_truth), 1e-20)
            
            @printf("    %-25s | Rel. Error: %9.2e\n", name, relative_error)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            A_mixed_input = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
            potrf_recursive!(A_mixed_input)
            L_result = tril(reconstruct_matrix(A_mixed_input))
            
            error_norm = norm(L_result - L_truth)
            relative_error = max(error_norm / norm(L_truth), 1e-20)
            
            @printf("    %-25s | Rel. Error: %9.2e\n", name, relative_error)
        end
    end
    
    println("\n" * "="^80)
    println("✅ Accuracy check complete.")
    println("="^80)
end

check_cholesky_accuracy()