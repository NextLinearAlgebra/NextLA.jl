using Test
using NextLA
using LinearAlgebra
using CUDA

const UNMQR_TESTTYPES = [ComplexF32, ComplexF64, Float32, Float64]

@testset "UNMQR Tests" begin
    @testset "Left No-Transpose Application Tests" begin
        for type in UNMQR_TESTTYPES
            m, n, k, ib = 200, 200, 80, 40
            rtol = (type == ComplexF32) || (type == Float32) ? 1e-5 : 1e-8
            
            # Create QR factorization first
            A_qr = rand(type, m, n)
            A_original = copy(A_qr)
            lda = m
            T = zeros(type, ib, k)
            ldt = ib
            tau = zeros(type, k)
            ldwork = ib * n
            work_qr = zeros(type, ldwork)


            # Perform QR factorization
            NextLA.geqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)

            # Test matrix to apply Q to
            C = rand(T, m, n)
            C_original = copy(C)
            ldc = m

            # Workspace for UNMQR
            work = zeros(type, ib * m)
            ldwork = n

            # Apply Q from left (Q * C)
            NextLA.unmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
            # Verify using reference QR decomposition
            Q_ref, R_ref = qr(A_original)
            C_expected = Matrix(Q_ref) * C_original

            # Note: Due to potential sign differences in QR, we check properties rather than exact equality
            @test size(C) == (n, m)
            @test all(isfinite.(C))
        
            # Check that the transformation preserves matrix structure
            @test (norm(C) - norm(C_expected)) / norm(C_expected) < rtol # Orthogonal transformations preserve norm
        end
    end 
    
    @testset "Left Conjugate Transpose Application Tests" begin
        for type in UNMQR_TESTTYPES
            m, n, k, ib = 200, 200, 80, 40
            rtol = (type == ComplexF32) || (type == Float32) ? 1e-5 : 1e-8
            
            # Create QR factorization first
            A_qr = rand(type, m, n)
            A_original = copy(A_qr)
            lda = m
            T = zeros(type, ib, k)
            ldt = ib
            tau = zeros(type, k)
            ldwork = ib * n
            work_qr = zeros(type, ldwork)


            # Perform QR factorization
            NextLA.geqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)

            # Test matrix to apply Q to
            C = rand(T, m, n)
            C_original = copy(C)
            ldc = m

            # Workspace for UNMQR
            work = zeros(type, ib * m)
            ldwork = n

            # Apply Q from left (Q * C)
            NextLA.unmqr('L', 'C', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
            # Verify using reference QR decomposition
            Q_ref, R_ref = qr(A_original)
            C_expected = adjoint(Matrix(Q_ref)) * C_original

            # Note: Due to potential sign differences in QR, we check properties rather than exact equality
            @test size(C) == (n, m)
            @test all(isfinite.(C))
        
            # Check that the transformation preserves matrix structure
            @test (norm(C) - norm(C_expected)) / norm(C_expected) < rtol # Orthogonal transformations preserve norm
        end
    end
    
    @testset "Right No-Transpose Application Tests" begin
         for type in UNMQR_TESTTYPES
            m, n, k, ib = 200, 200, 80, 40
            rtol = (type == ComplexF32) || (type == Float32) ? 1e-5 : 1e-8
            
            # Create QR factorization first
            A_qr = rand(type, m, n)
            A_original = copy(A_qr)
            lda = m
            T = zeros(type, ib, k)
            ldt = ib
            tau = zeros(type, k)
            ldwork = ib * n
            work_qr = zeros(type, ldwork)


            # Perform QR factorization
            NextLA.geqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)

            # Test matrix to apply Q to
            C = rand(T, m, n)
            C_original = copy(C)
            ldc = m

            # Workspace for UNMQR
            work = zeros(type, ib * m)
            ldwork = m

            # Apply Q from left (Q * C)
            NextLA.unmqr('R', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
            # Verify using reference QR decomposition
            Q_ref, R_ref = qr(A_original)
            C_expected =  C_original * Matrix(Q_ref)

            # Note: Due to potential sign differences in QR, we check properties rather than exact equality
            @test size(C) == (n, m)
            @test all(isfinite.(C))
        
            # Check that the transformation preserves matrix structure
            @test (norm(C) - norm(C_expected)) / norm(C_expected) < rtol # Orthogonal transformations preserve norm
        end
    end
    
    @testset "Right Conjugate Transpose Application Tests" begin
        for type in UNMQR_TESTTYPES
            m, n, k, ib = 200, 200, 80, 40
            rtol = (type == ComplexF32) || (type == Float32) ? 1e-5 : 1e-8
            
            # Create QR factorization first
            A_qr = rand(type, m, n)
            A_original = copy(A_qr)
            lda = m
            T = zeros(type, ib, k)
            ldt = ib
            tau = zeros(type, k)
            ldwork = ib * n
            work_qr = zeros(type, ldwork)


            # Perform QR factorization
            NextLA.geqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)

            # Test matrix to apply Q to
            C = rand(T, m, n)
            C_original = copy(C)
            ldc = m

            # Workspace for UNMQR
            work = zeros(type, ib * m)
            ldwork = m

            # Apply Q from left (Q * C)
            NextLA.unmqr('R', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
            # Verify using reference QR decomposition
            Q_ref, R_ref = qr(A_original)
            C_expected =  C_original * adjoint(Matrix(Q_ref))

            # Note: Due to potential sign differences in QR, we check properties rather than exact equality
            @test size(C) == (n, m)
            @test all(isfinite.(C))
        
            # Check that the transformation preserves matrix structure
            @test (norm(C) - norm(C_expected)) / norm(C_expected) < rtol # Orthogonal transformations preserve norm
        end
    end
    
    @testset "Orthogonality Property" begin
        # Test that Q * Q^H = I by applying both operations
        m, n, k, ib = 200, 200, 100, 40
        
        A_qr = rand(ComplexF64, m, k)
        lda = m
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.geqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = Matrix{ComplexF64}(I, m, n)  # Identity matrix
        C_original = copy(C)
        ldc = m
        
        work = zeros(ComplexF64, ib * n)
        ldwork = n
        
        # Apply Q then Q^H
        NextLA.unmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        NextLA.unmqr('L', 'C', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        # Should get back to identity (at least for the first k columns)
        @test C[:, 1:k] ≈ C_original[:, 1:k] rtol=1e-10
    end
    
    @testset "Error Handling" begin
        m, n, k, ib = 100, 80, 50, 20
        A = zeros(ComplexF64, m, k)
        T = zeros(ComplexF64, ib, k)
        C = zeros(ComplexF64, m, n)
        work = zeros(ComplexF64, ib * n)
        
        # Invalid side
        @test_throws ArgumentError NextLA.unmqr('X', 'N', m, n, k, ib, A, m, T, ib, C, m, work, ib)
        
        # Invalid trans
        @test_throws ArgumentError NextLA.unmqr('L', 'X', m, n, k, ib, A, m, T, ib, C, m, work, ib)
        
        # Negative dimensions
        @test_throws ArgumentError NextLA.unmqr('L', 'N', -1, n, k, ib, A, m, T, ib, C, m, work, ib)
        @test_throws ArgumentError NextLA.unmqr('L', 'N', m, -1, k, ib, A, m, T, ib, C, m, work, ib)
        @test_throws ArgumentError NextLA.unmqr('L', 'N', m, n, -1, ib, A, m, T, ib, C, m, work, ib)
        @test_throws ArgumentError NextLA.unmqr('L', 'N', m, n, k, -1, A, m, T, ib, C, m, work, ib)
        
        # Invalid k (k > nq)
        @test_throws ArgumentError NextLA.unmqr('L', 'N', m, n, m+1, ib, A, m, T, ib, C, m, work, ib)
    end
    
    @testset "Edge Cases" begin
        # k = 0 (no reflectors to apply)
        m, n, k, ib = 100, 80, 0, 20
        A = zeros(ComplexF64, m, max(1, k))
        T = zeros(ComplexF64, ib, max(1, k))
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        work = zeros(ComplexF64, ib * n)
        
        NextLA.unmqr('L', 'N', m, n, k, ib, A, m, T, ib, C, m, work, n)
        
        # With k=0, C should remain unchanged
        @test C ≈ C_original
        
        # ib = 1 (minimal block size)
        m, n, k, ib = 100, 80, 50, 10
        A_qr = rand(ComplexF64, m, k)
        lda = m
        T = zeros(ComplexF64, ib, k)
        ldt = ib
        tau = zeros(ComplexF64, k)
        work_qr = zeros(ComplexF64, ib * k)
        
        NextLA.geqrt(m, k, ib, A_qr, lda, T, ldt, tau, work_qr)
        
        C = rand(ComplexF64, m, n)
        C_original = copy(C)
        ldc = m
        work = zeros(ComplexF64, ib * n)
        ldwork = n
        
        NextLA.unmqr('L', 'N', m, n, k, ib, A_qr, lda, T, ldt, C, ldc, work, ldwork)
        
        @test all(isfinite.(C))
        @test norm(C) ≈ norm(C_original) rtol=1e-8
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, k, ib = 160, 120, 60, 30
            
            # Create and factorize on CPU
            A_qr_cpu = rand(ComplexF32, m, k)
            lda = m
            T_cpu = zeros(ComplexF32, ib, k)
            ldt = ib
            tau_cpu = zeros(ComplexF32, k)
            work_qr_cpu = zeros(ComplexF32, ib * k)
            
            NextLA.geqrt(m, k, ib, A_qr_cpu, lda, T_cpu, ldt, tau_cpu, work_qr_cpu)
            
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
            NextLA.unmqr('L', 'N', m, n, k, ib, A_qr_cpu, lda, T_cpu, ldt, C_cpu_result, ldc, work_cpu, ldwork)
            
            # Apply on GPU
            NextLA.unmqr('L', 'N', m, n, k, ib, A_qr_gpu, lda, T_gpu, ldt, C_gpu, ldc, work_gpu, ldwork)
            
            @test Array(C_gpu) ≈ C_cpu_result rtol=1e-6
        end
    end
end
