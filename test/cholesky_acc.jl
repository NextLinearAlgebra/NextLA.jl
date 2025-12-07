using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions, GPUArrays



function reconstruct_matrix!(A::SymmMixedPrec{T_Base},A_orig::AbstractGPUMatrix )where {T_Base}
    if A.BaseCase !== nothing
	A_orig_view = view(A_orig,A.loc[1].+(1:size(A.BaseCase,1)),A.loc[1].+(1:size(A.BaseCase,2)))
	A_orig_view .=A.BaseCase
	tril!(A_orig_view)
        if A.base_scale !== nothing
            #print("this is the base scale:", A.base_scale)
            A_orig_view.*=A.base_scale
        end
	return A_orig
    end
    
    reconstruct_matrix!(A.A11,A_orig)
    reconstruct_matrix!(A.A22,A_orig)
    
    n1,n2 = size(A.OffDiag)
    n_start=A.loc[1]
    n_mid=A.loc[1]+A.loc[2]
    C21_orig= view(A_orig,n_mid.+(1:n1),n_start.+(1:n2))
    C12_orig= view(A_orig,n_start.+(1:n2),n_mid.+(1:n1))
    C21_orig .= A.OffDiag
    if A.offDiag_scale !== nothing
        #print("this is the off diag scale:", A.offDiag_scale)
        C21_orig .*= A.offDiag_scale
    end
    C12_orig.=0#C21_orig'
    return A_orig
end


function get_accuracy_pure(A_spd_fp64::AbstractGPUMatrix, T_prec::DataType)
    local A_to_factor, scale_factor
    n=size(A_spd_fp64,1)
    A_T =  KernelAbstractions.allocate(backend,T_prec,n,n)
    A_out =  KernelAbstractions.allocate(backend,Float64,n,n)
    if T_prec == Float16
	    scale_factor = n*10#maximum(abs.( A_spd_fp64))
	    A_spd_fp64./=scale_factor
	    copyto!(A_T,A_spd_fp64)
	    A_spd_fp64.*=scale_factor
	    A_T=A_T+100*I
    else
        scale_factor = 1.0
        copyto!(A_T,A_spd_fp64)
    end
    
    potrf_recursive!(A_T, 4096)
    tril!(A_T)
    copyto!(A_out,A_T)
    A_out= A_out*A_out'*scale_factor
    if T_prec == Float16
        A_spd_fp64+=scale_factor*100*I
    end
    A_T = nothing
    A_out.-=A_spd_fp64
        orig_norm = norm(A_spd_fp64)
        error_norm = norm(A_out)
    
    return max(error_norm / orig_norm, 1e-20)
end


function get_accuracy_mixed(A_spd_fp64::AbstractGPUMatrix, precisions::Vector)
    memspace = SymmMixedPrec_prealloc(A_spd_fp64,precisions)
    A_mixed_input = SymmMixedPrec(A_spd_fp64, 'L',memspace; precisions=precisions)
    CPU_A = Array(A_spd_fp64)
    potrf_recursive!(A_mixed_input)
    # L_result = tril(reconstruct_matrix(A_mixed_input))
    L_reconstructed=reconstruct_matrix!(A_mixed_input,A_spd_fp64)
    for i in 1:length(memspace)
	    memspace[i]=nothing
    end
    #tril!(L_reconstructed)
    A_reconstructed = L_reconstructed * L_reconstructed'
    copyto!(A_spd_fp64, CPU_A)
    myAminB!(A_reconstructed,A_spd_fp64)
    error_norm = norm(Array(A_reconstructed))
    orig_norm = norm(Array(A_spd_fp64))
    
    return max(error_norm / orig_norm, 1e-20)
end

function get_accuracy_cusolver(A_spd_fp64::AbstractGPUMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_spd_fp64)
    
    potrf!('L', A_to_factor)
    # L_result = tril(A_to_factor)
    tril!(A_to_factor)
    A_reconstructed = Float64.((A_to_factor) *(A_to_factor)')
    A_to_factor=nothing
    error_norm = norm(A_reconstructed - A_spd_fp64)
    orig_norm = norm(A_spd_fp64)
    
    return max(error_norm / orig_norm, 1e-20)
end


function check_cholesky_accuracy()
	n_values = [1000, 4096, 5000, 8192, 9999, 16384, 32768, 65536] #256, 512, 1024, 2048, 

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
       "Pure F64" => [Float64],
       "Pure F16" => [Float16]
    )
    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64,
    )
    mixed_scenarios = Dict(
			   "[F32,F64]"=>[Float32,Float64],		   
        "[F32, F64, F64, F64]"      => [Float32, Float64, Float64, Float64],
        "[F32, F32, F32, F64]"      => [Float32, Float32, Float32, Float64],
        "[F32, F32, F64]"           => [Float32, Float32, Float64],
        "[F32, F64, F64]"           => [Float32, Float64, Float64],
        "[F16, F32, F32]"           => [Float16, Float32, Float32],
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F32, F32, F32, F32, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32],
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
        
        A_cpu_rand = randn(Float64, n, n)* 0.01
        A_gpu_rand =  KernelAbstractions.allocate(backend,Float64,n,n)
	copyto!(A_gpu_rand,A_cpu_rand)
        A_cpu_rand = nothing
        
        A_spd_fp64 = A_gpu_rand * A_gpu_rand' + (n*10) * I
        A_gpu_rand = nothing
        
        if n<32768
        	println("\n--- CUSOLVER Library Scenarios ---")
		for (name, T_prec) in cusolver_scenarios
            		relative_error = get_accuracy_cusolver(A_spd_fp64, T_prec)
            		@printf("      %-25s | Rel. Error: %9.2e\n", name, relative_error)
        	end
	end
        
        if n<65536
		println("\n--- Pure Precision Scenarios ---")
        	for (name, precisions) in pure_scenarios
            		T_prec = precisions[1]
            		relative_error = get_accuracy_pure(A_spd_fp64, T_prec)
            		@printf("      %-25s | Rel. Error: %9.2e\n", name, relative_error)
        	end
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
