using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "ZTTMQR Tests" begin
    @testset "Left No-Transpose Application Tests" begin
        m1, n1, m2, n2, k, ib = 12, 10, 8, 10, 6, 3
        
        # Create triangular-triangular QR factorization data
        A1 = triu(rand(ComplexF64, m1, k))
        A2 = triu(rand(ComplexF64, m2, k))
        V = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, ib, k))
        
        # Make well-conditioned and set proper structure
        for i in 1:k
            A1[i, i] += 1.0
            A2[i, i] += 1.0
        end
        
        # Test matrices to apply transformation to
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        lda1 = m1
        lda2 = m2
        ldv = m2
        ldt = ib
        work = zeros(ComplexF64, ib * max(n1, n2))
        ldwork = ib
        
        # Apply our ZTTMQR
        NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        # Basic verification
        @test size(C1) == (m1, n1)
        @test size(C2) == (m2, n2)
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
        
        # Verify orthogonal transformation property (norm preservation)
        combined_norm_original = norm([C1_original; C2_original])
        combined_norm_result = norm([C1; C2])
        @test abs(combined_norm_result - combined_norm_original) < 1e-10
    end
    
    @testset "Right Application Tests" begin
        m1, n1, m2, n2, k, ib = 10, 12, 10, 8, 6, 3
        
        A1 = triu(rand(ComplexF64, n1, k))
        A2 = triu(rand(ComplexF64, n2, k))
        V = rand(ComplexF64, n2, k)
        T = triu(rand(ComplexF64, ib, k))
        
        for i in 1:k
            A1[i, i] += 1.0
            A2[i, i] += 1.0
        end
        
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        lda1 = n1
        lda2 = n2
        ldv = n2
        ldt = ib
        work = zeros(ComplexF64, max(m1, m2) * ib)
        ldwork = max(m1, m2)
        
        NextLA.zttmqr('R', 'N', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
        
        # Check norm preservation for right application
        combined_norm_original = norm([C1_original C2_original])
        combined_norm_result = norm([C1 C2])
        @test abs(combined_norm_result - combined_norm_original) < 1e-10
    end
    
    @testset "Conjugate Transpose Tests" begin
        m1, n1, m2, n2, k, ib = 12, 10, 8, 10, 5, 2
        
        A1 = triu(rand(ComplexF64, m1, k))
        A2 = triu(rand(ComplexF64, m2, k))
        V = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, ib, k))
        
        for i in 1:k
            A1[i, i] += 1.0
            A2[i, i] += 1.0
        end
        
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        lda1 = m1
        lda2 = m2
        ldv = m2
        ldt = ib
        work = zeros(ComplexF64, ib * max(n1, n2))
        ldwork = ib
        
        NextLA.zttmqr('L', 'C', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "ComplexF32 Tests" begin
        m1, n1, m2, n2, k, ib = 10, 8, 6, 8, 4, 2
        
        A1 = triu(rand(ComplexF32, m1, k))
        A2 = triu(rand(ComplexF32, m2, k))
        V = rand(ComplexF32, m2, k)
        T = triu(rand(ComplexF32, ib, k))
        
        for i in 1:k
            A1[i, i] += ComplexF32(1.0)
            A2[i, i] += ComplexF32(1.0)
        end
        
        C1 = rand(ComplexF32, m1, n1)
        C2 = rand(ComplexF32, m2, n2)
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        lda1 = m1
        lda2 = m2
        ldv = m2
        ldt = ib
        work = zeros(ComplexF32, ib * max(n1, n2))
        ldwork = ib
        
        NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
        
        # Check norm preservation with relaxed tolerance for F32
        combined_norm_original = norm([C1_original; C2_original])
        combined_norm_result = norm([C1; C2])
        @test abs(combined_norm_result - combined_norm_original) < 1e-6
    end
    
    @testset "Different Sizes" begin
        test_cases = [
            (8, 6, 5, 6, 3, 2),
            (15, 12, 10, 12, 6, 3),
            (12, 15, 8, 15, 5, 2)
        ]
        
        for (m1, n1, m2, n2, k, ib) in test_cases
            A1 = triu(rand(ComplexF64, m1, k))
            A2 = triu(rand(ComplexF64, m2, k))
            V = rand(ComplexF64, m2, k)
            T = triu(rand(ComplexF64, ib, k))
            
            for i in 1:k
                A1[i, i] += 1.0
                A2[i, i] += 1.0
            end
            
            C1 = rand(ComplexF64, m1, n1)
            C2 = rand(ComplexF64, m2, n2)
            C1_original = copy(C1)
            C2_original = copy(C2)
            
            lda1 = m1
            lda2 = m2
            ldv = m2
            ldt = ib
            work = zeros(ComplexF64, ib * max(n1, n2))
            ldwork = ib
            
            NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                          A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
            
            @test all(isfinite.(C1))
            @test all(isfinite.(C2))
            
            # Verify transformation occurred
            @test !isapprox(C1, C1_original, rtol=1e-10) || !isapprox(C2, C2_original, rtol=1e-10)
        end
    end
    
    @testset "Edge Cases" begin
        # Minimal sizes
        m1, n1, m2, n2, k, ib = 2, 2, 2, 2, 1, 1
        
        A1 = triu(rand(ComplexF64, m1, k))
        A2 = triu(rand(ComplexF64, m2, k))
        V = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, ib, k))
        
        A1[1, 1] += 1.0
        A2[1, 1] += 1.0
        
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        lda1 = m1
        lda2 = m2
        ldv = m2
        ldt = ib
        work = zeros(ComplexF64, ib * max(n1, n2))
        ldwork = ib
        
        NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "Orthogonality Property" begin
        # Test that applying Q then Q^H gives back original
        m1, n1, m2, n2, k, ib = 12, 10, 8, 10, 5, 2
        
        A1 = triu(rand(ComplexF64, m1, k))
        A2 = triu(rand(ComplexF64, m2, k))
        V = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, ib, k))
        
        for i in 1:k
            A1[i, i] += 1.0
            A2[i, i] += 1.0
        end
        
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        lda1 = m1
        lda2 = m2
        ldv = m2
        ldt = ib
        work = zeros(ComplexF64, ib * max(n1, n2))
        ldwork = ib
        
        # Apply Q
        NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        # Apply Q^H
        NextLA.zttmqr('L', 'C', m1, n1, m2, n2, k, ib,
                      A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
        
        # Should get back to original (approximately)
        @test C1 ≈ C1_original rtol=1e-10
        @test C2 ≈ C2_original rtol=1e-10
    end
    
    @testset "Integration Tests" begin
        # Test in context of triangular-triangular QR
        m1, n, m2, ib = 10, 6, 8, 3
        
        # Create initial triangular matrices
        A1_orig = triu(rand(ComplexF64, n, n))
        A2_orig = triu(rand(ComplexF64, m2, n))
        A1_orig += I
        for i in 1:min(m2, n)
            A2_orig[i, i] += 1.0
        end
        
        # Perform ZTTQRT first
        A1_qr = copy(A1_orig)
        A2_qr = copy(A2_orig)
        T_qr = zeros(ComplexF64, ib, n)
        tau = zeros(ComplexF64, n)
        work_qr = zeros(ComplexF64, ib * n)
        
        NextLA.zttqrt(m2, n, ib, A1_qr, n, A2_qr, m2, T_qr, ib, tau, work_qr)
        
        # Now test ZTTMQR with the results
        C1 = rand(ComplexF64, m1, n)
        C2 = rand(ComplexF64, m2, n)
        
        # Use first few columns of the QR result
        k = min(ib, n)
        A1_test = A1_qr[1:m1, 1:k]
        A2_test = A2_qr[1:m2, 1:k]
        V_test = A2_qr[1:m2, 1:k]
        T_test = T_qr[1:ib, 1:k]
        
        work = zeros(ComplexF64, ib * n)
        
        NextLA.zttmqr('L', 'N', m1, n, m2, n, k, ib,
                      A1_test, m1, A2_test, m2, V_test, m2, T_test, ib, work, ib)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m1, n1, m2, n2, k, ib = 10, 8, 6, 8, 4, 2
            
            # Create CPU data
            A1_cpu = triu(rand(ComplexF32, m1, k))
            A2_cpu = triu(rand(ComplexF32, m2, k))
            V_cpu = rand(ComplexF32, m2, k)
            T_cpu = triu(rand(ComplexF32, ib, k))
            
            for i in 1:k
                A1_cpu[i, i] += ComplexF32(1.0)
                A2_cpu[i, i] += ComplexF32(1.0)
            end
            
            C1_cpu = rand(ComplexF32, m1, n1)
            C2_cpu = rand(ComplexF32, m2, n2)
            work_cpu = zeros(ComplexF32, ib * max(n1, n2))
            
            # Create GPU data
            A1_gpu = CuArray(A1_cpu)
            A2_gpu = CuArray(A2_cpu)
            V_gpu = CuArray(V_cpu)
            T_gpu = CuArray(T_cpu)
            C1_gpu = CuArray(C1_cpu)
            C2_gpu = CuArray(C2_cpu)
            work_gpu = CuArray(work_cpu)
            
            # Apply on CPU
            C1_cpu_result = copy(C1_cpu)
            C2_cpu_result = copy(C2_cpu)
            NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                          A1_cpu, m1, A2_cpu, m2, V_cpu, m2, T_cpu, ib, work_cpu, ib)
            
            # Apply on GPU
            NextLA.zttmqr('L', 'N', m1, n1, m2, n2, k, ib,
                          A1_gpu, m1, A2_gpu, m2, V_gpu, m2, T_gpu, ib, work_gpu, ib)
            
            @test Array(C1_gpu) ≈ C1_cpu_result rtol=1e-6
            @test Array(C2_gpu) ≈ C2_cpu_result rtol=1e-6
        end
    end
end
