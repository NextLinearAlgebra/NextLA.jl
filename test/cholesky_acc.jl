using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions

function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        if A.base_scale !== nothing
            print("this is the base scale:", A.base_scale)
            return copy(A.BaseCase) .* A.base_scale
        else
            return copy(A.BaseCase)
        end
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    
    local C21
    if A.offDiag_scale !== nothing
        print("this is the off diag scale:", A.offDiag_scale)
        C21 = A.OffDiag .* A.offDiag_scale
    else
        C21 = A.OffDiag
    end

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
    local A_to_factor, scale_factor
    
    if T_prec == Float16
        scale_factor = maximum(abs, A_spd_fp64)
        A_to_factor = Float16.(A_spd_fp64 ./ scale_factor) #+ 100*I
    else
        scale_factor = 1.0
        A_to_factor = T_prec.(A_spd_fp64)
    end
    
    potrf_recursive!(A_to_factor, 4096)
    A_tri = tril(A_to_factor)
    A_reconstructed = Float64.(A_tri * Transpose(A_tri) * scale_factor)
    A_to_factor = nothing
    A_tri = nothing
    GC.gc(true); CUDA.reclaim()
    
    if T_prec == Float16
        error_norm = norm(A_reconstructed - (A_spd_fp64 + scale_factor*100*I))
        orig_norm = norm(A_spd_fp64 + scale_factor*100*I)
    else
        orig_norm = norm(A_spd_fp64)
        A_reconstructed .-= A_spd_fp64
        error_norm = norm(A_reconstructed)
    end
    
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
        "Pure F16" => [Float16]
    )
    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64,
    )
    # mixed_scenarios = Dict(
    #     "[F32, F64, F64, F64]"      => [Float32, Float64, Float64, Float64],
    #     "[F32, F32, F32, F64]"      => [Float32, Float32, Float32, Float64],
    #     "[F32, F32, F64]"           => [Float32, Float32, Float64],
    #     "[F32, F64, F64]"           => [Float32, Float64, Float64],
    #     "[F16, F32, F32]"           => [Float16, Float32, Float32],
    #     "[F16, F16, F32]"           => [Float16, Float16, Float32],
    #     "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F32, F32, F32, F32, F32, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
    #     "[F16, F32, F64]"           => [Float16, Float32, Float64],
    #     "[F32, F64]"                => [Float32, Float64],
    #     "[F16, F64]"                => [Float16, Float64],
    #     "[F16, F32]"                => [Float16, Float32],
    # )
    mixed_scenarios = Dict(
        "[F16, F32, F64]" => [Float16, Float32, Float64],
        "[F16, F16, F32, F64]" => [Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64]
    )

    println("Starting Cholesky Accuracy Check...")

    for n in n_values
        println("\n" * "="^80)
        println("Checking Accuracy for Matrix Size (n x n) = $n x $n")
        
        # A_cpu_rand = randn(Float64, n, n)* 0.01
        # A_gpu_rand = CuArray(A_cpu_rand)
        # A_cpu_rand = nothing
        
        # A_spd_fp64 = A_gpu_rand * A_gpu_rand' + (n*10) * I
        # A_gpu_rand = nothing
        
        # GC.gc(true); CUDA.reclaim()
        A_raw = CUDA.rand(Float64, n, n)
        
        # 2. Symmetrize it! (A + A')
        # This is CRITICAL. If input isn't symmetric, error calculation 
        # compares LL' (symmetric) vs A (non-symmetric), causing constant high error.
        
        scale_factor = 1.0 / sqrt(n)
        A_raw .*= scale_factor

        A_spd_fp64 = A_raw + A_raw'
        
        # 3. Make it SPD (Add n to diagonal)
        view(A_spd_fp64, diagind(A_spd_fp64)) .+= 1000
        
        # Free temp memory
        A_raw = nothing
        CUDA.synchronize()
        
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
        A_spd_fp64 = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
    
    println("\n" * "="^80)
    println("✅ Accuracy check complete.")
    println("="^80)
end

check_cholesky_accuracy()