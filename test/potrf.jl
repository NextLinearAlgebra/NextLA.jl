using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions
using CUDA.CUSOLVER

function benchmark_op(op, reset_op, backend)
    reset_op()
    op() # Warmup
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:10
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

@testset "Cholesky Accuracy vs CPU (Float64)" begin
    # Note: Since this kernel is O(N) synchronization steps, it doesn't scale 
    # well to massive N, but we test up to 512/1024.
    n_sizes = [32, 64, 128, 256, 512] 
    tolerance = 1e-10

    @testset "Potrf 'L'" begin
        for n in n_sizes
            # Generate Positive Definite Matrix
            # A = Q * Q' + I
            A_rand = rand(Float64, n, n)
            A_host = A_rand * A_rand' + n * I 
            
            d_A = CuArray(A_host)

            cholesky_lower!(d_A)

            # CPU Reference
            L_ref = cholesky(A_host).L
            
            res_gpu = Array(d_A)
            
            # Check Lower Triangle only
            diff_norm = norm(tril(res_gpu) - tril(L_ref))
            ref_norm  = norm(tril(L_ref))
            rel_error = (ref_norm > 0) ? (diff_norm / ref_norm) : diff_norm

            if rel_error > tolerance
                @printf("FAIL: N: %4d | Rel Error: %.3e\n", n, rel_error)
            end
            @test rel_error < tolerance
        end
        println("Accuracy Tests Passed!")
    end
end

function run_chol_benchmark()
    # We test sizes that fit reasonably within the single-block constraint
    n_sizes = [32, 64, 128, 256, 512, 1024, 2048]

    println("="^95)
    @printf("%-6s | %-18s | %-18s | %-15s\n", "N", "Time Custom (ms)", "Time CUSOLVER (ms)", "Speedup (Ref/KA)")
    println("="^95)

    for n in n_sizes
        # Setup PD Matrix
        A_rand = rand(Float64, n, n)
        A_host = A_rand * A_rand' + n * I 
        
        d_A = CuArray(A_host)
        d_A_ref = CuArray(A_host)
        d_A_init = CuArray(A_host) # Save state

        backend = KernelAbstractions.get_backend(d_A)

        # 1. Custom Kernel
        op_custom = () -> cholesky_lower!(d_A)
        reset_custom = () -> copyto!(d_A, d_A_init)
        
        time_custom_ns = benchmark_op(op_custom, reset_custom, backend)
        time_custom_ms = time_custom_ns / 1_000_000

        # 2. CUSOLVER potrf!
        # potrf! computes Cholesky in-place. 
        # Returns info code, but we just benchmark execution time.
        op_cublas = () -> CUDA.CUSOLVER.potrf!('L', d_A_ref)
        reset_cublas = () -> copyto!(d_A_ref, d_A_init)

        time_cublas_ns = benchmark_op(op_cublas, reset_cublas, backend)
        time_cublas_ms = time_cublas_ns / 1_000_000

        ratio = time_cublas_ms / time_custom_ms

        @printf("%6d | %18.4f | %18.4f | %15.4fx\n", 
                n, time_custom_ms, time_cublas_ms, ratio)
        
        CUDA.reclaim()
    end
    println("-"^95)
end

run_chol_benchmark()