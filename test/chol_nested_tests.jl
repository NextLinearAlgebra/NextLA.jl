using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

# Assuming benchmark.jl and flops.jl are in the same directory
include("benchmark.jl")
include("flops.jl")

flops_potrf(T_prec, n) = (1/3 * n^3 + 1/2 * n^2)

calculate_gflops(flops, time_ns) = (flops / time_ns)

function benchmark_op(op, reset_op, backend)
    # 1. Warm-up
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

function run_cholesky_test(A_spd_fp64::CuMatrix, n::Int, T_prec::DataType, factorization_func!)
    # Create a clean copy in the target precision for repeated tests
    A_clean = T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op = () -> factorization_func!(A_perf)
    reset_op = () -> copyto!(A_perf, A_clean)
    
    min_time_ns = benchmark_op(op, reset_op, backend)
    
    runtime_ms = min_time_ns / 1_000_000
    flops = flops_potrf(T_prec, n)
    gflops = calculate_gflops(flops, min_time_ns)
    
    # Clean up GPU memory for this specific test
    A_clean = nothing
    A_perf = nothing
    # Note: GC calls are not strictly necessary here but can help manage memory in tight situations
    
    return runtime_ms, gflops
end


function run_all_benchmarks()
    n_values = [4096, 8192, 16384, 32768, 65536]
    precisions = [Float32, Float64] 
    block_size = 256 

    println("="^80)
    println("🚀 Starting Cholesky Factorization Benchmark...")
    println("="^80)

    for n in n_values
        println("\n" * "-"^80)
        @printf(" Matrix Size = %d x %d\n", n, n)
        println("-"^80)
        
        # --- EFFICIENT GPU-BASED MATRIX GENERATION ---
        local A_spd_fp64 # Ensure A_spd_fp64 is available in the whole loop scope
        try
            @printf("   Generating %d x %d SPD matrix directly on GPU...\n", n, n)
            t_gen = @elapsed begin
                # 1. Create the initial random matrix directly on the GPU
                A_gpu_rand = CUDA.randn(Float64, n, n) * 0.01

                # 2. Pre-allocate the final matrix, initializing it with the diagonal component.
                #    This avoids creating a separate large temporary matrix for A*A'.
                A_spd_fp64 = CuMatrix(CUDA.I(n) * (n * 10.0))

                # 3. Use syrk! (Symmetric Rank-k Update) for an efficient, in-place update.
                #    This computes C = α*A*A' + β*C. Here, C=A_spd_fp64, A=A_gpu_rand, α=1.0, β=1.0.
                LinearAlgebra.syrk!('L', 'N', 1.0, A_gpu_rand, 1.0, A_spd_fp64)
                
                # 4. Since syrk! only fills the lower triangle, copy it to the upper part.
                copytri!(A_spd_fp64, 'L')

                # 5. Immediately free the now-unnecessary random matrix to conserve VRAM.
                A_gpu_rand = nothing
                GC.gc(true); CUDA.reclaim()
            end
            @printf("   Matrix generation took %.2f seconds.\n", t_gen)

        catch e
            if isa(e, CUDA.OutOfGPUMemoryError)
                println("   ERROR: GPU out of memory while trying to create the matrix. Skipping this size.")
                continue # Skip to the next n_value
            else
                rethrow(e)
            end
        end
        # --- END OF MATRIX GENERATION ---

        @printf("\n%-28s | %-12s | %12s | %12s\n", "Implementation", "Precision", "Runtime (ms)", "GFLOPS")
        println("-"^73)

        for T_prec in precisions
            # These two implementations are run for all precisions
            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> potrf_recursive_nested!(A, block_size))
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "Recursive Nested", string(T_prec), runtime, gflops)

            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> potrf_recursive_nonnested!(A, block_size))
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "Recursive Non-Nested", string(T_prec), runtime, gflops)
            
            # --- Conditional CUSOLVER Benchmark ---
            # Only run the CUSOLVER benchmark if the precision is NOT Float16
            if T_prec != Float16
                runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> CUSOLVER.potrf!('L', A))
                @printf("%-28s | %-12s | %12.3f | %12.2f\n", "CUSOLVER Standard", string(T_prec), runtime, gflops)
            end
        end
        
        # Free the large SPD matrix before the next iteration
        A_spd_fp64 = nothing
        GC.gc(true); CUDA.reclaim()
    end
    
    println("\n" * "="^80)
    println("✅ Benchmark complete.")
    println("="^80)
end

# Run the main benchmark function
run_all_benchmarks()