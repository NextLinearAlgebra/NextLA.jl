using NextLA
using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

include("benchmark.jl")

@testset "Accuracy Test for SYRK Kernel" begin
    # Matrix sizes to test
    n_sizes = [16, 32, 128, 256, 1024, 250, 275, 300, 325, 350, 750]
    
    # Number of columns/rows in K to test
    k_sizes = [1, 8, 64, 256]

    alpha = 1.5
    beta  = 0.5
    
    tolerance = 1e-10

    @testset "SYRK L/N: C = alpha*A*A' + beta*C" begin
        for n in n_sizes
            for k in k_sizes
                
                A_host = rand(Float64, n, k)
                C_host = rand(Float64, n, n)
                
                C_ref  = copy(C_host)

                d_A = CuArray(A_host)
                d_C = CuArray(C_host)

                SYRK_KERNEL!('L', 'N', alpha, d_A, beta, d_C)
                
                C_ref = alpha * (A_host * A_host') + beta * C_ref

                res_gpu = tril(Array(d_C))
                res_cpu = tril(C_ref)

                diff_norm = norm(res_gpu - res_cpu)
                ref_norm  = norm(res_cpu)
                
                rel_error = (ref_norm > 0) ? (diff_norm / ref_norm) : diff_norm

                println("Size: $n x $n, K size: $k | Result Diff (Relative Error): $rel_error")

                @test rel_error < tolerance
            end
        end
    end
end


function benchmark_op(op, reset_op, backend)
    reset_op()
    op()
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:10
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

function run_syrk_benchmark()
    n_sizes = [32, 64, 128, 256, 512, 1024]

    alpha = 1.5
    beta  = 0.5

    println("="^110)
    @printf("%-6s | %-6s | %-18s | %-18s | %-15s\n", 
            "N", "K", "Time Custom (ms)", "Time CUBLAS (ms)", "Speedup (Ref/KA)")
    println("="^110)

    for n in n_sizes
        k = n
        
        A_host = rand(Float64, n, k)
        C_host = rand(Float64, n, n)
        
        d_A = CuArray(A_host)
        d_C = CuArray(C_host)
        d_C_syrk = CuArray(C_host)
        d_C_init = CuArray(C_host)

        backend = KernelAbstractions.get_backend(d_A)

        op_custom = () -> SYRK_KERNEL!('L', 'N', alpha, d_A, beta, d_C)
        reset_custom = () -> copyto!(d_C, d_C_init)
        
        time_custom_ns = benchmark_op(op_custom, reset_custom, backend)
        time_custom_ms = time_custom_ns / 1_000_000

        op_syrk = () -> CUBLAS.syrk!('L', 'N', alpha, d_A, beta, d_C_syrk)
        reset_syrk = () -> copyto!(d_C_syrk, d_C_init)

        time_syrk_ns = benchmark_op(op_syrk, reset_syrk, backend)
        time_syrk_ms = time_syrk_ns / 1_000_000

        ratio = time_syrk_ms / time_custom_ms

        @printf("%6d | %6d | %18.4f | %18.4f | %15.4fx\n", 
                n, k, time_custom_ms, time_syrk_ms, ratio)
        
        CUDA.reclaim()
    end
    println("-"^110)
end

run_syrk_benchmark()