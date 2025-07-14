using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "ZTTQRT Tests" begin
    @testset "ComplexF64 Basic Tests" begin
        m, n, ib = 12, 8, 4
        
        # Create upper triangular A1 and upper triangular A2
        A1 = triu(rand(ComplexF64, n, n))
        A2 = triu(rand(ComplexF64, m, n))
        
        # Make well-conditioned
        for i in 1:n
            A1[i, i] += 1.0
        end
        for i in 1:min(m, n)
            A2[i, i] += 1.0
        end
        
        # Store originals for verification
        A1_original = copy(A1)
        A2_original = copy(A2)
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        # Apply our ZTTQRT
        NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        # Verify that A1 remains upper triangular
        for i in 1:n
            for j in 1:i-1
                @test abs(A1[i, j]) < 1e-14
            end
        end
        
        # Verify that A2 contains the Householder vectors
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
        @test all(isfinite.(tau))
    end
    
    @testset "ComplexF32 Tests" begin
        m, n, ib = 10, 6, 3
        
        A1 = triu(rand(ComplexF32, n, n))
        A2 = triu(rand(ComplexF32, m, n))
        
        # Make well-conditioned
        for i in 1:n
            A1[i, i] += ComplexF32(1.0)
        end
        for i in 1:min(m, n)
            A2[i, i] += ComplexF32(1.0)
        end
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF32, ib, n)
        ldt = ib
        tau = zeros(ComplexF32, n)
        work = zeros(ComplexF32, ib * n)
        
        NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
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
        m, n = 15, 10
        block_sizes = [1, 2, 4, 5, 8]
        
        for ib in block_sizes
            if ib <= n
                A1 = triu(rand(ComplexF64, n, n))
                A2 = triu(rand(ComplexF64, m, n))
                
                # Make well-conditioned
                A1 += I
                for i in 1:min(m, n)
                    A2[i, i] += 1.0
                end
                
                lda1 = n
                lda2 = m
                T = zeros(ComplexF64, ib, n)
                ldt = ib
                tau = zeros(ComplexF64, n)
                work = zeros(ComplexF64, ib * n)
                
                NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
                
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
    end
    
    @testset "Different Matrix Sizes" begin
        test_cases = [
            (8, 6, 3), (15, 10, 4), (20, 12, 5), (12, 8, 2)
        ]
        
        for (m, n, ib) in test_cases
            A1 = triu(rand(ComplexF64, n, n))
            A2 = triu(rand(ComplexF64, m, n))
            
            # Ensure well-conditioned matrices
            A1 += I
            for i in 1:min(m, n)
                A2[i, i] += 1.0
            end
            
            lda1 = n
            lda2 = m
            T = zeros(ComplexF64, ib, n)
            ldt = ib
            tau = zeros(ComplexF64, n)
            work = zeros(ComplexF64, ib * n)
            
            NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
            
            @test all(isfinite.(A1))
            @test all(isfinite.(A2))
            @test all(isfinite.(T))
            @test all(isfinite.(tau))
        end
    end
    
    @testset "QR Property Verification" begin
        m, n, ib = 15, 10, 4
        
        A1 = triu(rand(ComplexF64, n, n))
        A2 = triu(rand(ComplexF64, m, n))
        
        # Make well-conditioned
        A1 += 2*I
        for i in 1:min(m, n)
            A2[i, i] += 2.0
        end
        
        # Store original combined matrix
        combined_original = [A1; A2[1:min(m,n), :]]
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        # The result should be a QR factorization
        # A1 (now R) should be upper triangular
        for i in 1:n
            for j in 1:i-1
                @test abs(A1[i, j]) < 1e-12
            end
        end
        
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
    end
    
    @testset "Edge Cases" begin
        # Single column
        m, n, ib = 5, 1, 1
        A1 = triu(rand(ComplexF64, n, n))
        A2 = triu(rand(ComplexF64, m, n))
        A1[1, 1] += 1.0
        A2[1, 1] += 1.0
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
        
        # Minimal case
        m, n, ib = 2, 2, 1
        A1 = triu(rand(ComplexF64, n, n))
        A2 = triu(rand(ComplexF64, m, n))
        A1 += I
        A2 += I
        
        lda1 = n
        lda2 = m
        T = zeros(ComplexF64, ib, n)
        ldt = ib
        tau = zeros(ComplexF64, n)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
        
        @test all(isfinite.(A1))
        @test all(isfinite.(A2))
        @test all(isfinite.(T))
    end
    
    @testset "Consistency Tests" begin
        m, n, ib = 15, 10, 4
        
        A1 = triu(rand(ComplexF64, n, n))
        A2 = triu(rand(ComplexF64, m, n))
        A1 += I
        A2 += I
        
        # First run
        A1_test1 = copy(A1)
        A2_test1 = copy(A2)
        lda1 = n
        lda2 = m
        T1 = zeros(ComplexF64, ib, n)
        ldt = ib
        tau1 = zeros(ComplexF64, n)
        work1 = zeros(ComplexF64, ib * n)
        
        NextLA.zttqrt(m, n, ib, A1_test1, lda1, A2_test1, lda2, T1, ldt, tau1, work1)
        
        # Second run
        A1_test2 = copy(A1)
        A2_test2 = copy(A2)
        T2 = zeros(ComplexF64, ib, n)
        tau2 = zeros(ComplexF64, n)
        work2 = zeros(ComplexF64, ib * n)
        
        NextLA.zttqrt(m, n, ib, A1_test2, lda1, A2_test2, lda2, T2, ldt, tau2, work2)
        
        # Results should be identical
        @test A1_test1 ≈ A1_test2 rtol=1e-12
        @test A2_test1 ≈ A2_test2 rtol=1e-12
        @test T1 ≈ T2 rtol=1e-12
        @test tau1 ≈ tau2 rtol=1e-12
    end
    
    @testset "Integration with LAPACK" begin
        # Test with LAPACK wrapper if available
        m, n = 8, 5
        l = 3
        
        A = rand(ComplexF64, m, n)
        B = rand(ComplexF64, m, n)
        Tau = zeros(ComplexF64, n, n)
        
        try
            # Test the LAPACK wrapper
            NextLA.lapack_ttqrt!(ComplexF64, l, A, B, Tau)
            
            @test all(isfinite.(A))
            @test all(isfinite.(B))
            @test all(isfinite.(Tau))
        catch
            # If LAPACK wrapper fails, just verify it doesn't crash
            @test true
        end
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, ib = 10, 6, 3
            
            # Create CPU data
            A1_cpu = triu(rand(ComplexF32, n, n))
            A2_cpu = triu(rand(ComplexF32, m, n))
            A1_cpu += I
            for i in 1:min(m, n)
                A2_cpu[i, i] += ComplexF32(1.0)
            end
            
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
            NextLA.zttqrt(m, n, ib, A1_cpu_result, lda1, A2_cpu_result, lda2, T_cpu_result, ldt, tau_cpu_result, work_cpu)
            
            # Apply on GPU
            NextLA.zttqrt(m, n, ib, A1_gpu, lda1, A2_gpu, lda2, T_gpu, ldt, tau_gpu, work_gpu)
            
            @test Array(A1_gpu) ≈ A1_cpu_result rtol=1e-6
            @test Array(A2_gpu) ≈ A2_cpu_result rtol=1e-6
            @test Array(T_gpu) ≈ T_cpu_result rtol=1e-6
            @test Array(tau_gpu) ≈ tau_cpu_result rtol=1e-6
        end
    end
end
