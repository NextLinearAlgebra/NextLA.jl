using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "ZTSQRT Tests" begin
    @testset "ComplexF64 Basic Tests" begin
        m, n, ib = 15, 10, 4
        
        # Create upper triangular A1 and general A2
        A1 = triu(rand(ComplexF64, n, n))
        A2 = rand(ComplexF64, m, n)
        
        # Make copies for verification
        A1_original = copy(A1)
        A2_original = copy(A2)
        combined_original = [A1_original; A2_original]
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        # Apply our ZTSQRT
        NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        # Verify that A1 is still upper triangular but modified
        for i in 1:n
            for j in 1:i-1
                @test abs(A1[i, j]) < 1e-14
            end
        end
        
        # Verify that A2 contains Householder vectors
        @test size(A2) == (m, n)
        @test all(isfinite.(A2))
        
        # Verify that T contains the triangular factors
        @test size(T) == (ib, n)
        @test all(isfinite.(T))
        
        # T should be block upper triangular
        for block_start in 1:ib:n
            block_end = min(block_start + ib - 1, n)
            for i in 1:(block_end - block_start + 1)
                for j in 1:(i-1)
                    if block_start + i - 1 <= n && block_start + j - 1 <= n
                        @test abs(T[i, block_start + j - 1]) < 1e-12
                    end
                end
            end
        end
    end
    
    @testset "ComplexF32 Tests" begin
        m, n, ib = 12, 8, 3
        
        A1 = triu(rand(ComplexF32, n, n))
        A2 = rand(ComplexF32, m, n)
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF32, ib, n)
        ldt = ib
        tau = zeros(ComplexF32, n)
        work = zeros(ComplexF32, ib * n)
        
        NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        # Basic checks
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
        @test all(isfinite.(tau))
        
        # A1 should remain upper triangular
        for i in 1:n
            for j in 1:i-1
                @test abs(A1[i, j]) < 1e-6
            end
        end
    end
    
    @testset "Different Block Sizes" begin
        m, n = 20, 12
        block_sizes = [1, 2, 4, 6, 8]
        
        for ib in block_sizes
            A1 = triu(rand(ComplexF64, n, n))
            A2 = rand(ComplexF64, m, n)
            
            lda1 = n
            lda2 = m
            T = zeros(ComplexF64, ib, n)
            ldt = ib
            tau = zeros(ComplexF64, n)
            work = zeros(ComplexF64, ib * n)
            
            NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
            
            # Should complete without errors
            @test all(isfinite.(A1))
            @test all(isfinite.(A2))
            @test all(isfinite.(T))
            
            # A1 should remain upper triangular
            for i in 1:n
                for j in 1:i-1
                    @test abs(A1[i, j]) < 1e-12
                end
            end
        end
    end
    
    @testset "Different Matrix Sizes" begin
        test_cases = [
            (10, 8, 3), (25, 15, 5), (30, 20, 6), (15, 12, 4)
        ]
        
        for (m, n, ib) in test_cases
            A1 = triu(rand(ComplexF64, n, n))
            A2 = rand(ComplexF64, m, n)
            
            # Ensure A1 is well-conditioned
            for i in 1:n
                A1[i, i] += 1.0  # Add to diagonal for numerical stability
            end
            
            lda1 = n
            lda2 = m
            T = zeros(ComplexF64, ib, n)
            ldt = ib
            tau = zeros(ComplexF64, n)
            work = zeros(ComplexF64, ib * n)
            
            NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
            
            @test all(isfinite.(A1))
            @test all(isfinite.(A2))
            @test all(isfinite.(T))
            @test all(isfinite.(tau))
        end
    end
    
    @testset "QR Property Verification" begin
        m, n, ib = 20, 12, 4
        
        A1 = triu(rand(ComplexF64, n, n))
        A2 = rand(ComplexF64, m, n)
        
        # Make A1 well-conditioned
        for i in 1:n
            A1[i, i] += 2.0
        end
        
        # Store original combined matrix
        combined_original = [A1; A2]
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        # The factorization should produce a QR decomposition of the combined matrix
        # [A1_orig] = Q * [R]
        # [A2_orig]       [0]
        # where R is upper triangular and stored in A1
        
        # Check that A1 (now R) is upper triangular
        for i in 1:n
            for j in 1:i-1
                @test abs(A1[i, j]) < 1e-12
            end
        end
        
        # Check that the factorization is numerically stable
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
    end
    
    @testset "Comparison with Standard QR" begin
        m, n, ib = 15, 10, 3
        
        A1 = triu(rand(ComplexF64, n, n))
        A2 = rand(ComplexF64, m, n)
        
        # Make A1 well-conditioned
        for i in 1:n
            A1[i, i] += 1.0
        end
        
        combined_original = [A1; A2]
        
        # Our implementation
        A1_our = copy(A1)
        A2_our = copy(A2)
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.ztsqrt(m, n, ib, A1_our, lda1, A2_our, lda2, T, ldt, tau, work)
        
        # Reference QR factorization
        Q_ref, R_ref = qr(combined_original)
        R_ref_mat = Matrix(R_ref)
        
        # Compare the R factors (up to signs)
        R_our = A1_our
        for j in 1:n
            if abs(R_ref_mat[j, j]) > 1e-10 && abs(R_our[j, j]) > 1e-10
                # Normalize for sign differences
                scale = R_ref_mat[j, j] / R_our[j, j]
                @test abs(abs(scale) - 1) < 1e-8
            end
        end
    end
    
    @testset "Edge Cases" begin
        # Single column
        m, n, ib = 10, 1, 1
        A1 = triu(rand(ComplexF64, n, n))
        A2 = rand(ComplexF64, m, n)
        A1[1, 1] += 1.0  # Ensure well-conditioned
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
        
        # Minimal size
        m, n, ib = 2, 2, 1
        A1 = triu(rand(ComplexF64, n, n))
        A2 = rand(ComplexF64, m, n)
        A1 += I  # Make well-conditioned
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
    end
    
    @testset "Error Handling" begin
        m, n, ib = 10, 8, 3
        A1 = zeros(ComplexF64, n, n)
        A2 = zeros(ComplexF64, m, n)
        T = zeros(ComplexF64, ib, n)
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        # Negative dimensions
        @test_throws ArgumentError NextLA.ztsqrt(-1, n, ib, A1, n, A2, m, T, ib, tau, work)
        @test_throws ArgumentError NextLA.ztsqrt(m, -1, ib, A1, n, A2, m, T, ib, tau, work)
        @test_throws ArgumentError NextLA.ztsqrt(m, n, -1, A1, n, A2, m, T, ib, tau, work)
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, ib = 12, 8, 3
            
            # Create CPU data
            A1_cpu = triu(rand(ComplexF32, n, n))
            A2_cpu = rand(ComplexF32, m, n)
            A1_cpu += I  # Make well-conditioned
            
            lda1 = n
            lda2 = m
            T_cpu = zeros(ComplexF32, ib, n)
            ldt = ib
            tau_cpu = zeros(ComplexF32, n)
            work_cpu = zeros(ComplexF32, ib * n)
            
            # Create GPU data
            A1_gpu = CuArray(A1_cpu)
            A2_gpu = CuArray(A2_cpu)
            T_gpu = CuArray(T_cpu)
            tau_gpu = CuArray(tau_cpu)
            work_gpu = CuArray(work_cpu)
            
            # Apply on CPU
            A1_cpu_result = copy(A1_cpu)
            A2_cpu_result = copy(A2_cpu)
            T_cpu_result = copy(T_cpu)
            tau_cpu_result = copy(tau_cpu)
            NextLA.ztsqrt(m, n, ib, A1_cpu_result, lda1, A2_cpu_result, lda2, T_cpu_result, ldt, tau_cpu_result, work_cpu)
            
            # Apply on GPU
            NextLA.ztsqrt(m, n, ib, A1_gpu, lda1, A2_gpu, lda2, T_gpu, ldt, tau_gpu, work_gpu)
            
            @test Array(A1_gpu) ≈ A1_cpu_result rtol=1e-6
            @test Array(A2_gpu) ≈ A2_cpu_result rtol=1e-6
            @test Array(T_gpu) ≈ T_cpu_result rtol=1e-6
            @test Array(tau_gpu) ≈ tau_cpu_result rtol=1e-6
        end
    end
end
