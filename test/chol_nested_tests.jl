using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

include("benchmark.jl")
include("flops.jl")

flops_potrf(T_prec, n) = (1/3 * n^3 + 1/2 * n^2)

calculate_gflops(flops, time_ns) = (flops / time_ns)



function run_cholesky_test(A_spd_fp64::CuMatrix, n::Int, T_prec::DataType, factorization_func!)
    A_clean = T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op_with_reset = () -> begin
        copyto!(A_perf, A_clean)
        factorization_func!(A_perf)
    end
    
    min_time_ns = run_manual_benchmark(op_with_reset, backend; min_time_s=1.0, min_iters=5)
    
    runtime_ms = min_time_ns / 1_000_000
    flops = flops_potrf(T_prec, n)
    gflops = calculate_gflops(flops, min_time_ns)
    
    A_clean = nothing
    A_perf = nothing
    GC.gc(true); CUDA.reclaim()

    return runtime_ms, gflops
end


"""
    run_all_benchmarks()

Main function to orchestrate the Cholesky factorization benchmarks across
different matrix sizes, precisions, and implementations.
"""
function run_all_benchmarks()
    n_values = [4096, 8192, 16384, 32768, 65536]
    precisions = [Float32, Float64]
    block_size = 256 

    println("="^80)
    println("🚀 Starting Cholesky Factorization Benchmark...")
    println("   (Accuracy tests have been removed to conserve memory)")
    println("="^80)

    for n in n_values
        println("\n" * "-"^80)
        @printf(" Matrix Size = %d x %d\n", n, n)
        println("-"^80)
        
        A_cpu_rand = randn(Float64, n, n) * 0.01
        A_spd_fp64 = CuArray(A_cpu_rand * A_cpu_rand' + (n * 10) * I)
        A_cpu_rand = nothing
        GC.gc(true); CUDA.reclaim()

        # Updated table header (Rel. Error column removed)
        @printf("%-28s | %-12s | %12s | %12s\n", "Implementation", "Precision", "Runtime (ms)", "GFLOPS")
        println("-"^73) # Adjusted separator length

        for T_prec in precisions
            # Updated function call to unpack only two values
            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> potrf_recursive_nested!(A, block_size))
            # Updated print statement
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "Recursive Nested", string(T_prec), runtime, gflops)

            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> potrf_recursive_nonnested!(A, block_size))
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "Recursive Non-Nested", string(T_prec), runtime, gflops)
            
            runtime, gflops = run_cholesky_test(A_spd_fp64, n, T_prec, A -> CUSOLVER.potrf!('L', A))
            @printf("%-28s | %-12s | %12.3f | %12.2f\n", "CUSOLVER Standard", string(T_prec), runtime, gflops)
        end
        
        A_spd_fp64 = nothing
        GC.gc(true); CUDA.reclaim()
    end
    
    println("\n" * "="^80)
    println("✅ Benchmark complete.")
    println("="^80)
end


run_all_benchmarks()