using Test
using CUDA
using LinearAlgebra
using Printf

# Make sure your kernel is included
# include("src/syrk.jl")

@testset "SYRK Accuracy vs CPU Math (Float64)" begin
    
    n_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    k_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

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

                @printf("N: %4d | K: %4d | Rel Error: %.3e\n", n, k, rel_error)

                @test rel_error < tolerance
            end
        end
    end
end