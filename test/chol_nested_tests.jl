using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

include("benchmark.jl")
include("flops.jl")

flops_potrf(T_prec, n) = (1/3 * n^3 + 1/2 * n^2)

calculate_gflops(flops, time_ns) = (flops / time_ns)

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

function run_cholesky_test(A_spd_fp64::CuMatrix, n::Int, T_prec::DataType, factorization_func!)
    A_clean = T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op = () -> factorization_func!(A_perf)
    reset_op = () -> copyto!(A_perf, A_clean)
    
    min_time_ns = benchmark_op(op, reset_op, backend)
    
    runtime_ms = min_time_ns / 1_000_000
    flops = flops_potrf(T_prec, n)
    gflops = calculate_gflops(flops, min_time_ns)
    
    A_clean = nothing
    A_perf = nothing

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
        
        local A_spd_fp64
        try
            @printf("   Generating %d x %d SPD matrix directly on GPU...\n", n, n)
            t_gen = @elapsed begin
                # 1. Create a random matrix directly on the GPU.
                A_gpu_rand = CUDA.randn(Float64, n, n) * 0.01
                
                # 2. Perform the multiplication and addition on the GPU.
                #    This is the correct, high-level way to do it.
                #    CUDA.jl overloads these operators to call optimized CUBLAS routines.
                A_spd_fp64 = A_gpu_rand * A_gpu_rand' + (CuMatrix{Float64}(I, n, n) * (n * 10.0))

                # 3. Immediately free the intermediate matrix to save VRAM.
                A_gpu_rand = nothing
                GC.gc(true); CUDA.reclaim()
            end
            @printf("   Matrix generation took %.2f seconds.\n", t_gen)
        
        catch e
            if isa(e, CUDA.OutOfGPUMemoryError)
                println("   ERROR: GPU out of memory. Skipping this size.")
                continue
            else
                rethrow(e)
            end
        end

        @printf("\n%-28s | %-12s | %12s | %12s\n", "Implementation", "Precision", "Runtime (ms)", "GFLOPS")
        println("-"^73)

        for T_prec in precisions
            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> potrf_recursive_nested!(A, block_size))
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "Recursive Nested", string(T_prec), runtime, gflops)

            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> potrf_recursive_nonnested!(A, block_size))
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "Recursive Non-Nested", string(T_prec), runtime, gflops)
            
            if T_prec != Float16
                runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> CUSOLVER.potrf!('L', A))
                @printf("%-28s | %-12s | %12.3f | %12.2f\n", "CUSOLVER Standard", string(T_prec), runtime, gflops)
            end
        end
        
        A_spd_fp64 = nothing
        GC.gc(true); CUDA.reclaim()
    end
    
    println("\n" * "="^80)
    println("✅ Benchmark complete.")
    println("="^80)
end

run_all_benchmarks()