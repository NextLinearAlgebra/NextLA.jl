using CUDA
using LinearAlgebra
using Plots
using NextLA  # Assuming NextLA or DLA exports the necessary types

include("benchmark.jl")

function test_fullmixedprec_rectrxm()
    sizes = [256, 512, 1024, 2048] 
    uplo = 'U'
    side = 'L'
    alpha = 1.0f0 
    trans = 'N'

    test_scenarios = Dict(
        "FullMixed: [F16, F32]" => [Float16, Float32], 
        "FullMixed: [F32, F64]" => [Float32, Float64],
        "FullMixed: [F16, F16, F32]" => [Float16, Float16, Float32]
    )
    
    for func in ['S', 'M']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^70)
        println("🚀 Starting Benchmark for $op_name (uplo='$uplo') with FullMixedPrec...")
        println("="^70)

        for n in sizes
            println("\n--- Testing Matrix Size: $n x $n ---")

            # Create a full matrix but make it strictly diagonally dominant to avoid singular issues
            A_cpu = rand(Float64, n, n)
            A_cpu .+= Diagonal(fill(n * 2.0, n))
            B_cpu = rand(Float64, n, n)
            
            # --- Calculate Ground Truth Solution (FP64) ---
            A_sol_gpu = CuArray(A_cpu)
            B_sol_gpu = CuArray(B_cpu)
            if func == 'S' # TRSM: B <- alpha * inv(A) * B
                CUBLAS.trsm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu)
            else # TRMM: B <- alpha * A * B
                B_sol_gpu_copy = copy(B_sol_gpu)
                CUBLAS.trmm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu_copy, B_sol_gpu)
            end

            # --- Benchmark Recursive and Mixed-Precision Implementations ---
            for (name, prec_list) in test_scenarios
                T_Base = prec_list[1]
                A_test_gpu = CuArray(A_cpu) 
                B_test_gpu = CuArray{T_Base}(B_cpu)

                # Initialize FullMixedPrec
                A_mixed = FullMixedPrec(A_test_gpu; precisions=prec_list)
                
                unified_rectrxm!(side, uplo, trans, 'N', alpha, func, A_mixed, B_test_gpu)

                error_norm = norm(CuArray{Float64}(B_test_gpu) .- B_sol_gpu)
                solution_norm = norm(B_sol_gpu)
                relative_error = iszero(solution_norm) ? 0.0 : error_norm / solution_norm
                
                println("  '$name' | Rel. Error: $(round(relative_error, sigdigits=3))")
            end
        end
    end
end

# Run the test
test_fullmixedprec_rectrxm()
