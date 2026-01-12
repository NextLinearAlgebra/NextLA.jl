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
    for _ in 1:10
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end


@testset "GEMM Accuracy vs CPU Math (Float64)" begin
    
    n_sizes = [32, 64, 128, 256, 512]
    k_sizes = [32, 64, 128, 256, 512]

    tolerance = 1e-10

    @testset "GEMM N/N: C += 1.0 * A * B" begin
        for n in n_sizes
            for k in k_sizes
                A_host = rand(Float64, n, k)
                B_host = rand(Float64, k, n)
                C_host = rand(Float64, n, n)
                
                C_ref = copy(C_host)

                d_A = CuArray(A_host)
                d_B = CuArray(B_host)
                d_C = CuArray(C_host)

                GEMM_ADD!(d_A, d_B, d_C)
                
                C_ref += A_host * B_host

                res_gpu = Array(d_C)
                
                diff_norm = norm(res_gpu - C_ref)
                ref_norm  = norm(C_ref)
                
                rel_error = (ref_norm > 0) ? (diff_norm / ref_norm) : diff_norm

                if rel_error > tolerance
                    @printf("FAIL: N: %4d | K: %4d | Rel Error: %.3e\n", n, k, rel_error)
                end
                @test rel_error < tolerance
            end
        end
        println("Accuracy Tests Passed!")
    end
end

function run_gemm_benchmark()
    n_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    k_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    alpha = 1.0
    beta  = 1.0

    println("="^95)
    @printf("%-6s | %-6s | %-18s | %-18s | %-15s\n", "N", "K", "Time Custom (ms)", "Time CUBLAS (ms)", "Speedup (Ref/KA)")
    println("="^95)

    for n in n_sizes
        for k in k_sizes
            
            A_host = rand(Float64, n, k)
            B_host = rand(Float64, k, n)
            C_host = rand(Float64, n, n)
            
            d_A = CuArray(A_host)
            d_B = CuArray(B_host)
            d_C = CuArray(C_host)
            d_C_ref = CuArray(C_host)
            d_C_init = CuArray(C_host) # Save initial state for reset

            backend = KernelAbstractions.get_backend(d_A)

            op_custom = () -> GEMM_ADD!(d_A, d_B, d_C)
            reset_custom = () -> copyto!(d_C, d_C_init)
            
            time_custom_ns = benchmark_op(op_custom, reset_custom, backend)
            time_custom_ms = time_custom_ns / 1_000_000

            op_cublas = () -> CUBLAS.gemm!('N', 'N', alpha, d_A, d_B, beta, d_C_ref)
            reset_cublas = () -> copyto!(d_C_ref, d_C_init)

            time_cublas_ns = benchmark_op(op_cublas, reset_cublas, backend)
            time_cublas_ms = time_cublas_ns / 1_000_000

            ratio = time_cublas_ms / time_custom_ms

            @printf("%6d | %6d | %18.4f | %18.4f | %15.4fx\n", 
                    n, k, time_custom_ms, time_cublas_ms, ratio)
            
            CUDA.reclaim()
        end
        println("-"^95)
    end
end

run_gemm_benchmark()