using Test
using CUDA
using LinearAlgebra

@testset "Accuracy Test for SYRK Kernel (Lower, NoTrans)" begin
    # N dimensions (Rows/Cols of C)
    n_sizes = [16, 32, 128, 256, 1024, 250, 275]
    
    # K dimensions (Inner dimension, Cols of A)
    k_sizes = [16, 32, 64, 128, 255]

    # Scalars
    alpha = 1.5f0
    beta  = 0.5f0
    
    # Tolerance
    tolerance = 1e-5

    @testset "SYRK L/N: C = alpha*A*A' + beta*C" begin
        for n in n_sizes
            for k in k_sizes
                # -------------------------------------------------
                # 1. Prepare Data
                # -------------------------------------------------
                # A is N x K (since trans='N')
                A_host = rand(Float32, n, k)
                
                # C is N x N. 
                # Note: Real SYRK usually assumes C is symmetric, but 
                # mathematically it just updates the specified triangle.
                # We initialize it randomly.
                C_host = rand(Float32, n, n)

                # Move to GPU
                d_A = CuArray(A_host)
                d_C = CuArray(C_host)       # Will be modified by YOUR kernel
                d_C_ref = CuArray(C_host)   # Will be modified by CUBLAS

                # -------------------------------------------------
                # 2. Run Custom Kernel
                # -------------------------------------------------
                # Signature: (uplo, trans, alpha, A, beta, C)
                SYRK_KERNEL!('L', 'N', alpha, d_A, beta, d_C)

                # -------------------------------------------------
                # 3. Run CUBLAS Reference
                # -------------------------------------------------
                # CUBLAS.syrk!(uplo, trans, alpha, A, beta, C)
                CUBLAS.syrk!('L', 'N', alpha, d_A, beta, d_C_ref)

                # -------------------------------------------------
                # 4. Verify Results
                # -------------------------------------------------
                # SYRK with uplo='L' only updates the lower triangle.
                # We must mask out the upper triangle before comparing,
                # because the kernel might leave garbage or old data there.
                
                res_kernel = tril(Array(d_C))
                res_cublas = tril(Array(d_C_ref))

                # Compute relative error on the lower triangle only
                diff_norm = norm(res_kernel - res_cublas)
                ref_norm  = norm(res_cublas)
                
                # Avoid division by zero if ref is 0 (unlikely with random)
                rel_error = (ref_norm > 0) ? (diff_norm / ref_norm) : diff_norm

                println("Size C: $n x $n, Size A: $n x $k | Rel Error: $rel_error")

                @test rel_error < tolerance
            end
        end
    end
end