using NextLA
using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

include("benchmark.jl")

@testset "Accuracy Test for LU Base Kernel" begin
    # Base cases are usually small tiles
    n_sizes = [4, 8, 16, 32]
    
    tolerance = 1e-10

    @testset "P * A = L * U Base Case" begin
        for n in n_sizes
            A_host = rand(Float64, n, n)
            A_orig = copy(A_host)

            d_A = CuArray(A_host)
            
            # Call our base kernel
            d_A, d_P = lu_base!(d_A)

            A_res = Array(d_A)
            P_res = Array(d_P)

            # L has 1s on the diagonal, U has the actual diagonal
            L = UnitLowerTriangular(A_res)
            U = UpperTriangular(A_res)
            
            # The permutation P is formed by swapping rows of I exactly as A's rows were swapped.
            # Thus P * A_orig = L * U, so A_orig = inv(P) * L * U
            A_reconstructed = inv(P_res) * L * U

            diff_norm = norm(A_reconstructed - A_orig)
            ref_norm  = norm(A_orig)
            
            rel_error = (ref_norm > 0) ? (diff_norm / ref_norm) : diff_norm

            println("Size: $n x $n | Result Diff (Relative Error): $rel_error")

            @test rel_error < tolerance
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

function run_lubase_benchmark()
    n_sizes = [4, 8, 16, 32]

    println("="^90)
    @printf("%-6s | %-18s | %-18s | %-15s\n", 
            "N", "Time Custom (ms)", "Time CPU LU (ms)", "Speedup (CPU/KA)")
    println("="^90)

    for n in n_sizes
        A_host = rand(Float64, n, n)
        
        d_A = CuArray(A_host)
        d_A_init = CuArray(A_host)

        backend = KernelAbstractions.get_backend(d_A)

        op_custom = () -> lu_base!(d_A)
        reset_custom = () -> copyto!(d_A, d_A_init)
        
        time_custom_ns = benchmark_op(op_custom, reset_custom, backend)
        time_custom_ms = time_custom_ns / 1_000_000

        # CPU Baseline (since GPU base cases are competing at small sizes)
        A_cpu = copy(A_host)
        A_cpu_init = copy(A_host)
        op_cpu = () -> lu!(A_cpu)
        reset_cpu = () -> copyto!(A_cpu, A_cpu_init)
        
        time_cpu_ns = benchmark_op(op_cpu, reset_cpu, CPU())
        time_cpu_ms = time_cpu_ns / 1_000_000

        ratio = time_cpu_ms / time_custom_ms

        @printf("%6d | %18.4f | %18.4f | %15.4fx\n", 
                n, time_custom_ms, time_cpu_ms, ratio)
        
        CUDA.reclaim()
    end
    println("-"^90)
end

run_lubase_benchmark()
