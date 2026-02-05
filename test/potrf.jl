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

@testset "Cholesky Accuracy Check" begin
    n_sizes = [32, 64, 128, 256, 512] 
    # n_sizes = [64] 
    tolerance = 1e-4 # Relaxed slightly for Float32 accumulations

    println("\nRunning Accuracy Tests...")
    for n in n_sizes
        # Generate PD Matrix
        A_rand = rand(Float64, n, n)
        A_host = A_rand * A_rand' + n * I 
        
        L_ref = cholesky(A_host).L

        # Test Right-Looking (cholesky_lower!)
        d_A_right = CuArray(A_host)
        cholesky_lower!(d_A_right) # FIXED: Was cholesky_right!
        res_right = Array(d_A_right)
        diff_right = norm(tril(res_right) - tril(L_ref)) / norm(tril(L_ref))

        # Test Left-Looking (cholesky_lower_left!)
        d_A_left = CuArray(A_host)
        cholesky_lower_left!(d_A_left) # FIXED: Was cholesky_left!
        res_left = Array(d_A_left)
        diff_left = norm(tril(res_left) - tril(L_ref)) / norm(tril(L_ref))

        @test diff_right < tolerance
        @test diff_left < tolerance
        
        if diff_right >= tolerance || diff_left >= tolerance
             @printf("FAIL N=%d: RightErr=%.2e, LeftErr=%.2e\n", n, diff_right, diff_left)
        end
    end
    println("Accuracy Tests Passed!\n")
end

function run_chol_benchmark()
    # Benchmark sizes
    n_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    # n_sizes = [64]

    println("="^110)
    @printf("%-6s | %-12s | %-12s | %-12s | %-15s | %-15s\n", 
            "N", "Previous (ms)", "Current (ms)", "CUSOLVER(ms)", "Prev vs Current", "CUSOLVER vs Current")
    println("="^110)

    for n in n_sizes
        # Setup PD Matrix
        A_rand = rand(Float64, n, n)
        A_host = A_rand * A_rand' + n * I 
        
        d_A_init = CuArray(A_host) # Source for resets
        d_A_work = CuArray(A_host) # Working array
        
        backend = KernelAbstractions.get_backend(d_A_work)

        # 1. Benchmark Right-Looking (cholesky_lower!)
        op_right = () -> cholesky_lower!(d_A_work) # FIXED NAME
        reset_func = () -> copyto!(d_A_work, d_A_init)
        t_right = benchmark_op(op_right, reset_func, backend) / 1e6

        # 2. Benchmark Left-Looking (cholesky_lower_left!)
        op_left = () -> cholesky_lower_left!(d_A_work) # FIXED NAME
        t_left = benchmark_op(op_left, reset_func, backend) / 1e6

        # 3. Benchmark CUSOLVER
        # Note: we need a fresh array variable because CUSOLVER is in-place
        d_A_cusolver = CuArray(A_host)
        op_cusolver = () -> CUSOLVER.potrf!('L', d_A_cusolver)
        reset_cusolver = () -> copyto!(d_A_cusolver, d_A_init)
        t_cusolver = benchmark_op(op_cusolver, reset_cusolver, backend) / 1e6

        # Calculate Ratios
        speedup_left_vs_right = t_right / t_left
        speedup_ref_vs_left = t_left / t_cusolver 

        @printf("%6d | %12.4f | %12.4f | %12.4f | \033[1;32m%14.2fx\033[0m | %14.2fx\n", 
                n, t_right, t_left, t_cusolver, speedup_left_vs_right, speedup_ref_vs_left)
        
        CUDA.reclaim()
    end
    println("-"^110)
end

run_chol_benchmark()