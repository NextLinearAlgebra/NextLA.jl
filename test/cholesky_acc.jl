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

    T_Recon = promote_type(eltype(C11), eltype(C22), eltype(C21))
    C_full = CuArray{T_Recon}(undef, n, n)
    
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end


function get_accuracy_pure(A_spd_fp64::CuMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_spd_fp64)
    
    potrf_recursive!(A_to_factor, 4096)
    # L_result = tril(A_to_factor)
    
    A_reconstructed = Float64.(tril(A_to_factor) * tril(A_to_factor)')
    
    error_norm = norm(A_reconstructed - A_spd_fp64)
    orig_norm = norm(A_spd_fp64)
    
    return max(error_norm / orig_norm, 1e-20)
end


function get_accuracy_mixed(A_spd_fp64::CuMatrix, precisions::Vector)
    A_mixed_input = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)

    potrf_recursive!(A_mixed_input)

    # L_result = tril(reconstruct_matrix(A_mixed_input))
    
    A_reconstructed = Float64.(tril(reconstruct_matrix(A_mixed_input)) * tril(reconstruct_matrix(A_mixed_input))')
    
    error_norm = norm(A_reconstructed - A_spd_fp64)
    orig_norm = norm(A_spd_fp64)
    
    return max(error_norm / orig_norm, 1e-20)
end

function get_accuracy_cusolver(A_spd_fp64::CuMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_spd_fp64)
    
    CUSOLVER.potrf!('L', A_to_factor)
    # L_result = tril(A_to_factor)
    
    A_reconstructed = Float64.(tril(A_to_factor) * tril(A_to_factor)')
    
    error_norm = norm(A_reconstructed - A_spd_fp64)
    orig_norm = norm(A_spd_fp64)
    
    return max(error_norm / orig_norm, 1e-20)
end


function check_cholesky_accuracy()
    n_values = [4096, 8192, 16384, 32768, 65536] #256, 512, 1024, 2048, 

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
    )
    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64,
    )
    mixed_scenarios = Dict(
        "[F32, F64, F64, F64]"      => [Float32, Float64, Float64, Float64],
        "[F32, F32, F32, F64]"      => [Float32, Float32, Float32, Float64],
        "[F32, F32, F64]"           => [Float32, Float32, Float64],
        "[F32, F64, F64]"           => [Float32, Float64, Float64],
        "[F16, F32, F32]"           => [Float16, Float32, Float32],
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "[F16, F32, F64]"           => [Float16, Float32, Float64],
        "[F32, F64]"                => [Float32, Float64],
        "[F16, F64]"                => [Float16, Float64],
        "[F16, F32]"                => [Float16, Float32],
    )

    println("Starting Cholesky Accuracy Check...")

    for n in n_values
        println("\n" * "="^80)
        println("Checking Accuracy for Matrix Size (n x n) = $n x $n")
        
        A_cpu_rand = randn(Float64, n, n)
        A_gpu_rand = CuArray(A_cpu_rand)
        A_cpu_rand = nothing
        
        A_spd_fp64 = A_gpu_rand * A_gpu_rand' + (n*100) * I
        A_gpu_rand = nothing
        
        println("\n--- CUSOLVER Library Scenarios ---")
        for (name, T_prec) in cusolver_scenarios
            relative_error = get_accuracy_cusolver(A_spd_fp64, T_prec)
            @printf("      %-25s | Rel. Error: %9.2e\n", name, relative_error)
        end
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[1]
            relative_error = get_accuracy_pure(A_spd_fp64, T_prec)
            @printf("      %-25s | Rel. Error: %9.2e\n", name, relative_error)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            relative_error = get_accuracy_mixed(A_spd_fp64, precisions)
            @printf("      %-25s | Rel. Error: %9.2e\n", name, relative_error)
        end
    end
    
    println("\n" * "="^80)
    println("✅ Accuracy check complete.")
    println("="^80)
end

check_cholesky_accuracy()