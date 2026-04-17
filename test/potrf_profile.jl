using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

function benchmark_op(op, reset_op, backend)
    reset_op()
    op() # Warmup
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:50
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

function run_profile_sweep()
    # Sweeping from small to large to show how the bottleneck shifts
    n_sizes = [256, 512, 1024, 2048, 4096, 8192]
    
    println("="^60)
    @printf("%-6s | %-12s | %-12s | %-12s\n", "N", "Chol Base %", "TRSM Panel %", "GEMM Update %")
    println("="^60)

    for N in n_sizes
        # 1. Setup PD Matrix
        A_rand = rand(Float64, N, N)
        A_host = A_rand * A_rand' + N * I 
        d_A = CuArray(A_host)
        
        # 2. Warmup (crucial for accurate timing on the first run)
        d_A_warmup = copy(d_A)
        cholesky_lower_left_profiled!(d_A_warmup)
        
        # 3. Profile the actual run
        t_chol, t_trsm, t_gemm = cholesky_lower_left_profiled!(d_A)
        
        total_time = t_chol + t_trsm + t_gemm
        
        p_chol = (t_chol / total_time) * 100
        p_trsm = (t_trsm / total_time) * 100
        p_gemm = (t_gemm / total_time) * 100
        
        @printf("%-6d | %11.2f%% | %11.2f%% | %11.2f%%\n", N, p_chol, p_trsm, p_gemm)
        
        # Free up memory before the next huge matrix allocation
        CUDA.reclaim()
    end
    println("="^60)
end

run_profile_sweep()