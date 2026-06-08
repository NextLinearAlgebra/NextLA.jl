using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions
using NextLA  # Assuming NextLA or DLA exports the necessary types

include("benchmark.jl")

# --- Timing Helper Function ---
function benchmark_op(op, reset_op, backend)
    reset_op()
    op()
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:5
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

function test_fullmixedprec_rectrxm()
    # Capped at 32768 to avoid 64k OOM error on H200
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768] 
    uplo = 'U'
    side = 'L'
    alpha = 1.0f0 
    trans = 'N'

    test_scenarios = Dict(
        # Pure Recursive Baselines
        "Recursive Float64" => [Float64],
        "Recursive Float32" => [Float32],
        "Recursive Float16" => [Float16],
        
        # Basic comparisons
        "FullMixed: [F16, F32]" => [Float16, Float32], 
        "FullMixed: [F32, F64]" => [Float32, Float64],
        
        # Deep recursion (more F16 leaves)
        "FullMixed: [F16, F16, F32]" => [Float16, Float16, Float32],
        "FullMixed: [F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32],
        
        # The "Smooth Gradient" (Deep F16 -> F32 -> F64)
        "FullMixed: [F16, F16, F32, F64]" => [Float16, Float16, Float32, Float64],
        "FullMixed: [F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        
        # The Extreme Jump (testing precision loss on F16 -> F64)
        "FullMixed: [F16, F64]" => [Float16, Float64],
        "FullMixed: [F16, F16, F64]" => [Float16, Float16, Float64]
    )
    
    for func in ['S']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^85)
        println("🚀 Starting Accuracy & Timing Benchmark for $op_name (uplo='$uplo')...")
        println("="^85)

        for n in sizes
            println("\n--- Testing Matrix Size: $n x $n ---")

            # Create a full matrix but make it strictly diagonally dominant
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
            solution_norm = norm(B_sol_gpu)

            # --- Benchmark Recursive and Mixed-Precision Implementations ---
            for (name, prec_list) in test_scenarios
                local runtime_ms, relative_error
                
                if startswith(name, "Recursive")
                    T_Base = prec_list[1]
                    A_test_gpu = CuArray{T_Base}(A_cpu)
                    B_test_gpu = CuArray{T_Base}(B_cpu)
                    B_clean = copy(B_test_gpu)
                    backend = KernelAbstractions.get_backend(A_test_gpu)

                    # Accuracy Pass
                    unified_rectrxm!(side, uplo, trans, 'N', alpha, func, A_test_gpu, B_test_gpu)
                    error_norm = norm(CuArray{Float64}(B_test_gpu) .- B_sol_gpu)
                    relative_error = iszero(solution_norm) ? 0.0 : error_norm / solution_norm

                    # Timing Pass
                    op = () -> unified_rectrxm!(side, uplo, trans, 'N', alpha, func, A_test_gpu, B_test_gpu)
                    reset_op = () -> copyto!(B_test_gpu, B_clean)
                    min_time_ns = benchmark_op(op, reset_op, backend)
                    runtime_ms = min_time_ns / 1_000_000

                else
                    T_Base = prec_list[end] # Prevent accumulation error
                    A_test_gpu = CuArray(A_cpu) 
                    B_test_gpu = CuArray{T_Base}(B_cpu)
                    B_clean = copy(B_test_gpu)
                    backend = KernelAbstractions.get_backend(B_test_gpu)

                    # Initialize FullMixedPrec
                    A_mixed = FullMixedPrec(A_test_gpu; precisions=prec_list)
                    
                    # Accuracy Pass
                    unified_rectrxm!(side, uplo, trans, 'N', alpha, func, A_mixed, B_test_gpu)
                    error_norm = norm(CuArray{Float64}(B_test_gpu) .- B_sol_gpu)
                    relative_error = iszero(solution_norm) ? 0.0 : error_norm / solution_norm
                    
                    # Timing Pass
                    op = () -> unified_rectrxm!(side, uplo, trans, 'N', alpha, func, A_mixed, B_test_gpu)
                    reset_op = () -> copyto!(B_test_gpu, B_clean)
                    min_time_ns = benchmark_op(op, reset_op, backend)
                    runtime_ms = min_time_ns / 1_000_000
                end
                
                @printf("  %-40s | Rel. Error: %.3e | Runtime: %8.3f ms\n", "'$name'", relative_error, runtime_ms)
            end

            # --- Benchmark CUBLAS Baselines ---
            println("  " * "-"^80)
            for T_prec in [Float64, Float32]
                A_blas = CuArray{T_prec}(A_cpu)
                B_blas_clean = CuArray{T_prec}(B_cpu)
                B_blas = copy(B_blas_clean)
                backend = KernelAbstractions.get_backend(A_blas)

                op = () -> begin
                    if func == 'S'
                        CUBLAS.trsm!(side, uplo, trans, 'N', T_prec(alpha), A_blas, B_blas)
                    else
                        C_blas = similar(B_blas)
                        CUBLAS.trmm!(side, uplo, trans, 'N', T_prec(alpha), A_blas, B_blas, C_blas)
                    end
                end
                reset_op = () -> copyto!(B_blas, B_blas_clean)

                min_time_ns = benchmark_op(op, reset_op, backend)
                runtime_ms = min_time_ns / 1_000_000

                cublas_name = "'CUBLAS F$(sizeof(T_prec)*8)'"
                @printf("  %-40s | %-23s | Runtime: %8.3f ms\n", cublas_name, "Rel. Error: Base", runtime_ms)
            end

            # Cleanup memory for next iteration
            A_cpu, B_cpu, A_sol_gpu, B_sol_gpu = (nothing, nothing, nothing, nothing)
            GC.gc(true); CUDA.reclaim()
        end
    end
end

# Run the test
test_fullmixedprec_rectrxm()