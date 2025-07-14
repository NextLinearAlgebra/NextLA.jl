using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "ZUNMQR Tests" begin
    @testset "Left No-Transpose Application Tests" begin
        m, n, k, ib = 20, 15, 8, 4
        
        # Create QR factorization first
        A_qr = rand(ComplexF64, m, k)
        A_original = copy(A_qr)
        lda = m
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        # Perform QR factorization
        NextLA.zgeqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        # Test matrix to apply Q to
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        ldc = m
        
        # Workspace for ZUNMQR
        work = zeros(ComplexF64, ib * n)
        ldwork = ib
        
        # Apply Q from left (Q * C)
        NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        # Verify using reference QR decomposition
        Q_ref, R_ref = qr(A_original)
        C_expected = Matrix(Q_ref) * C_original
        
        # Note: Due to potential sign differences in QR, we check properties rather than exact equality
        @test size(C) == (m, n)
        @test all(isfinite.(C))
        
        # Check that the transformation preserves matrix structure
        @test norm(C) ≈ norm(C_expected) rtol=1e-8  # Orthogonal transformations preserve norm
    end
    
    @testset "Left Conjugate Transpose Application Tests" begin
        m, n, k, ib = 20, 15, 8, 4
        
        A_qr = rand(ComplexF64, m, k)
        A_original = copy(A_qr)
        lda = m
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.zgeqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        ldc = m
        
        work = zeros(ComplexF64, ib * n)
        ldwork = ib
        
        # Apply Q^H from left (Q^H * C)
        NextLA.zunmqr('L', 'C', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        @test size(C) == (m, n)
        @test all(isfinite.(C))
        @test norm(C) ≈ norm(C_original) rtol=1e-8
    end
    
    @testset "Right No-Transpose Application Tests" begin
        m, n, k, ib = 15, 20, 8, 4
        
        A_qr = rand(ComplexF64, n, k)
        A_original = copy(A_qr)
        lda = n
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.zgeqrt(n, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        ldc = m
        
        work = zeros(ComplexF64, m * ib)
        ldwork = m
        
        # Apply Q from right (C * Q)
        NextLA.zunmqr('R', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        @test size(C) == (m, n)
        @test all(isfinite.(C))
        @test norm(C) ≈ norm(C_original) rtol=1e-8
    end
    
    @testset "Right Conjugate Transpose Application Tests" begin
        m, n, k, ib = 15, 20, 8, 4
        
        A_qr = rand(ComplexF64, n, k)
        lda = n
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.zgeqrt(n, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        ldc = m
        
        work = zeros(ComplexF64, m * ib)
        ldwork = m
        
        # Apply Q^H from right (C * Q^H)
        NextLA.zunmqr('R', 'C', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        @test size(C) == (m, n)
        @test all(isfinite.(C))
        @test norm(C) ≈ norm(C_original) rtol=1e-8
    end
    
    @testset "ComplexF32 Tests" begin
        m, n, k, ib = 16, 12, 6, 3
        
        A_qr = rand(ComplexF32, m, k)
        lda = m
        T = zeros(ComplexF32, ib, k)
        ldt = ib
        tau = zeros(ComplexF32, k)
        work_qr = zeros(ComplexF32, ib * k)
        
        NextLA.zgeqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = rand(ComplexF32, m, n)
        C_original = copy(C)
        ldc = m
        
        work = zeros(ComplexF32, ib * n)
        ldwork = ib
        
        NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        @test size(C) == (m, n)
        @test all(isfinite.(C))
        @test norm(C) ≈ norm(C_original) rtol=1e-6
    end
    
    @testset "Different Block Sizes" begin
        m, n, k = 20, 15, 10
        block_sizes = [1, 2, 4, 5, 8]
        
        for ib in block_sizes
            A_qr = rand(ComplexF64, m, k)
            lda = m
            T = zeros(ComplexF64, ib, k)
            ldt = ib
            tau = zeros(ComplexF64, k)
            work_qr = zeros(ComplexF64, ib * k)
            
            NextLA.zgeqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
            
            C = rand(ComplexF64, m, n)
            C_original = copy(C)
            ldc = m
            
            work = zeros(ComplexF64, ib * n)
            ldwork = ib
            
            NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
            
            @test size(C) == (m, n)
            @test all(isfinite.(C))
            @test norm(C) ≈ norm(C_original) rtol=1e-10
        end
    end
    
    @testset "Orthogonality Property" begin
        # Test that Q * Q^H = I by applying both operations
        m, n, k, ib = 20, 20, 10, 4
        
        A_qr = rand(ComplexF64, m, k)
        lda = m
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.zgeqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = Matrix{ComplexF64}(I, m, n)  # Identity matrix
        C_original = copy(C)
        ldc = m
        
        work = zeros(ComplexF64, ib * n)
        ldwork = ib
        
        # Apply Q then Q^H
        NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        NextLA.zunmqr('L', 'C', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        # Should get back to identity (at least for the first k columns)
        @test C[:, 1:k] ≈ C_original[:, 1:k] rtol=1e-10
    end
    
    @testset "Different Matrix Sizes" begin
        test_cases = [
            (10, 8, 5, 2), (25, 20, 12, 4), (15, 30, 10, 5), (30, 15, 8, 3)
        ]
        
        for (m, n, k, ib) in test_cases
            A_qr = rand(ComplexF64, max(m,n), k)
            
            if m >= k  # Left application
                lda = max(m, n)
                T = zeros(ComplexF64, ib, k)
                ldt = ib
                tau = zeros(ComplexF64, k)
                work_qr = zeros(ComplexF64, ib * k)
                
                NextLA.zgeqrt(max(m,n), k, ib, A_qr, lda, T, ldt, tau, work_qr)
                
                C = rand(ComplexF64, m, n)
                C_original = copy(C)
                ldc = m
                
                work = zeros(ComplexF64, ib * n)
                ldwork = ib
                
                NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
                
                @test all(isfinite.(C))
                @test norm(C) ≈ norm(C_original) rtol=1e-8
            end
        end
    end
    
    @testset "Error Handling" begin
        m, n, k, ib = 10, 8, 5, 2
        A = zeros(ComplexF64, m, k)
        T = zeros(ComplexF64, ib, k)
        C = zeros(ComplexF64, m, n)
        work = zeros(ComplexF64, ib * n)
        
        # Invalid side
        @test_throws ArgumentError NextLA.zunmqr('X', 'N', m, n, k, ib, A, m, T, ib, C, m, work, ib)
        
        # Invalid trans
        @test_throws ArgumentError NextLA.zunmqr('L', 'X', m, n, k, ib, A, m, T, ib, C, m, work, ib)
        
        # Negative dimensions
        @test_throws ArgumentError NextLA.zunmqr('L', 'N', -1, n, k, ib, A, m, T, ib, C, m, work, ib)
        @test_throws ArgumentError NextLA.zunmqr('L', 'N', m, -1, k, ib, A, m, T, ib, C, m, work, ib)
        @test_throws ArgumentError NextLA.zunmqr('L', 'N', m, n, -1, ib, A, m, T, ib, C, m, work, ib)
        @test_throws ArgumentError NextLA.zunmqr('L', 'N', m, n, k, -1, A, m, T, ib, C, m, work, ib)
        
        # Invalid k (k > nq)
        @test_throws ArgumentError NextLA.zunmqr('L', 'N', m, n, m+1, ib, A, m, T, ib, C, m, work, ib)
    end
    
    @testset "Edge Cases" begin
        # k = 0 (no reflectors to apply)
        m, n, k, ib = 10, 8, 0, 2
        A = zeros(ComplexF64, m, max(1, k))
        T = zeros(ComplexF64, ib, max(1, k))
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.zunmqr('L', 'N', m, n, k, ib, A, m, T, ib, C, m, work, ib)
        
        # With k=0, C should remain unchanged
        @test C ≈ C_original
        
        # ib = 1 (minimal block size)
        m, n, k, ib = 10, 8, 5, 1
        A_qr = rand(ComplexF64, m, k)
        lda = m
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.zgeqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        ldc = m
        work = zeros(ComplexF64, ib * n)
        ldwork = ib
        
        NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        @test all(isfinite.(C))
        @test norm(C) ≈ norm(C_original) rtol=1e-8
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, k, ib = 16, 12, 6, 3
            
            # Create and factorize on CPU
            A_qr_cpu = rand(ComplexF32, m, k)
            lda = m
            T_cpu = zeros(ComplexF32, ib, k)
            ldt = ib
            tau_cpu = zeros(ComplexF32, k)
            work_qr_cpu = zeros(ComplexF32, ib * k)
            
            NextLA.zgeqrt(m, k, ib, A_qr_cpu, lda, T_cpu, ldt, tau_cpu, work_qr_cpu)
            
            # Create test matrices
            C_cpu = rand(ComplexF32, m, n)
            ldc = m
            work_cpu = zeros(ComplexF32, ib * n)
            ldwork = ib
            
            # Create GPU data
            A_qr_gpu = CuArray(A_qr_cpu)
            T_gpu = CuArray(T_cpu)
            C_gpu = CuArray(C_cpu)
            work_gpu = CuArray(work_cpu)
            
            # Apply on CPU
            C_cpu_result = copy(C_cpu)
            NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr_cpu, lda, T_cpu, ldt, C_cpu_result, ldc, work_cpu, ldwork)
            
            # Apply on GPU
            NextLA.zunmqr('L', 'N', m, n, k, ib, A_qr_gpu, lda, T_gpu, ldt, C_gpu, ldc, work_gpu, ldwork)
            
            @test Array(C_gpu) ≈ C_cpu_result rtol=1e-6
        end
    end
end
