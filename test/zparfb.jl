using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "ZPARFB Tests" begin
    @testset "Left Forward Column-wise Tests" begin
        m1, n1, m2, n2, k, l = 15, 12, 10, 12, 6, 4
        
        # Create test matrices
        V1 = rand(ComplexF64, m1, k)
        V2 = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, k, k))
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        # Set up proper Householder structure
        for i in 1:k
            V1[1:i-1, i] .= 0.0
            V1[i, i] = 1.0
            V2[1:i-1, i] .= 0.0
        end
        
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        ldv1 = m1
        ldv2 = m2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        # Apply our ZPARFB
        NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        # Basic checks
        @test size(C1) == (m1, n1)
        @test size(C2) == (m2, n2)
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
        
        # Verify orthogonal transformation (norm preservation)
        combined_norm_original = norm([C1_original; C2_original])
        combined_norm_result = norm([C1; C2])
        @test abs(combined_norm_result - combined_norm_original) < 1e-10
    end
    
    @testset "Right Application Tests" begin
        m1, n1, m2, n2, k, l = 12, 15, 12, 10, 6, 4
        
        V1 = rand(ComplexF64, n1, k)
        V2 = rand(ComplexF64, n2, k)
        T = triu(rand(ComplexF64, k, k))
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        # Set up Householder structure
        for i in 1:k
            V1[1:i-1, i] .= 0.0
            V1[i, i] = 1.0
            V2[1:i-1, i] .= 0.0
        end
        
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        ldv1 = n1
        ldv2 = n2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(m1, m2) * k)
        ldwork = max(m1, m2)
        
        NextLA.zparfb('R', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
        
        # Check norm preservation for right application
        combined_norm_original = norm([C1_original C2_original])
        combined_norm_result = norm([C1 C2])
        @test abs(combined_norm_result - combined_norm_original) < 1e-10
    end
    
    @testset "Conjugate Transpose Tests" begin
        m1, n1, m2, n2, k, l = 15, 12, 10, 12, 5, 3
        
        V1 = rand(ComplexF64, m1, k)
        V2 = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, k, k))
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        for i in 1:k
            V1[1:i-1, i] .= 0.0
            V1[i, i] = 1.0
            V2[1:i-1, i] .= 0.0
        end
        
        ldv1 = m1
        ldv2 = m2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        NextLA.zparfb('L', 'C', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "Backward Direction Tests" begin
        m1, n1, m2, n2, k, l = 15, 12, 10, 12, 5, 3
        
        V1 = rand(ComplexF64, m1, k)
        V2 = rand(ComplexF64, m2, k)
        T = tril(rand(ComplexF64, k, k))  # Lower triangular for backward
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        # Different structure for backward direction
        for i in 1:k
            V1[m1-k+i+1:m1, i] .= 0.0
            V1[m1-k+i, i] = 1.0
            V2[m2-k+i+1:m2, i] .= 0.0
        end
        
        ldv1 = m1
        ldv2 = m2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        NextLA.zparfb('L', 'N', 'B', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "Row-wise Storage Tests" begin
        m1, n1, m2, n2, k, l = 12, 15, 10, 15, 5, 3
        
        # V matrices stored row-wise
        V1 = rand(ComplexF64, k, m1)
        V2 = rand(ComplexF64, k, m2)
        T = triu(rand(ComplexF64, k, k))
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        # Row-wise structure
        for i in 1:k
            V1[i, 1:i-1] .= 0.0
            V1[i, i] = 1.0
            V2[i, 1:i-1] .= 0.0
        end
        
        ldv1 = k
        ldv2 = k
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        NextLA.zparfb('L', 'N', 'F', 'R', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "ComplexF32 Tests" begin
        m1, n1, m2, n2, k, l = 12, 10, 8, 10, 4, 2
        
        V1 = rand(ComplexF32, m1, k)
        V2 = rand(ComplexF32, m2, k)
        T = triu(rand(ComplexF32, k, k))
        C1 = rand(ComplexF32, m1, n1)
        C2 = rand(ComplexF32, m2, n2)
        
        for i in 1:k
            V1[1:i-1, i] .= ComplexF32(0.0)
            V1[i, i] = ComplexF32(1.0)
            V2[1:i-1, i] .= ComplexF32(0.0)
        end
        
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        ldv1 = m1
        ldv2 = m2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF32, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
        
        # Check norm preservation with relaxed tolerance for F32
        combined_norm_original = norm([C1_original; C2_original])
        combined_norm_result = norm([C1; C2])
        @test abs(combined_norm_result - combined_norm_original) < 1e-6
    end
    
    @testset "Different Sizes" begin
        test_cases = [
            (10, 8, 6, 8, 4, 2),
            (20, 15, 12, 15, 8, 5),
            (15, 20, 10, 20, 6, 4)
        ]
        
        for (m1, n1, m2, n2, k, l) in test_cases
            V1 = rand(ComplexF64, m1, k)
            V2 = rand(ComplexF64, m2, k)
            T = triu(rand(ComplexF64, k, k))
            C1 = rand(ComplexF64, m1, n1)
            C2 = rand(ComplexF64, m2, n2)
            
            for i in 1:k
                V1[1:i-1, i] .= 0.0
                V1[i, i] = 1.0
                V2[1:i-1, i] .= 0.0
            end
            
            C1_original = copy(C1)
            C2_original = copy(C2)
            
            ldv1 = m1
            ldv2 = m2
            ldt = k
            ldc1 = m1
            ldc2 = m2
            work = zeros(ComplexF64, max(n1, n2) * k)
            ldwork = max(n1, n2)
            
            NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                          V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
            
            @test all(isfinite.(C1))
            @test all(isfinite.(C2))
            
            # Check that transformation occurred
            @test !isapprox(C1, C1_original, rtol=1e-10) || !isapprox(C2, C2_original, rtol=1e-10)
        end
    end
    
    @testset "Orthogonality Property" begin
        # Test that applying H then H^H gives back original
        m1, n1, m2, n2, k, l = 15, 12, 10, 12, 5, 3
        
        V1 = rand(ComplexF64, m1, k)
        V2 = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, k, k))
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        for i in 1:k
            V1[1:i-1, i] .= 0.0
            V1[i, i] = 1.0
            V2[1:i-1, i] .= 0.0
        end
        
        C1_original = copy(C1)
        C2_original = copy(C2)
        
        ldv1 = m1
        ldv2 = m2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        # Apply H
        NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        # Apply H^H
        NextLA.zparfb('L', 'C', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        # Should get back to original (approximately)
        @test C1 ≈ C1_original rtol=1e-10
        @test C2 ≈ C2_original rtol=1e-10
    end
    
    @testset "Edge Cases" begin
        # Minimal sizes
        m1, n1, m2, n2, k, l = 2, 2, 2, 2, 1, 1
        
        V1 = rand(ComplexF64, m1, k)
        V2 = rand(ComplexF64, m2, k)
        T = triu(rand(ComplexF64, k, k))
        C1 = rand(ComplexF64, m1, n1)
        C2 = rand(ComplexF64, m2, n2)
        
        V1[1, 1] = 1.0
        
        ldv1 = m1
        ldv2 = m2
        ldt = k
        ldc1 = m1
        ldc2 = m2
        work = zeros(ComplexF64, max(n1, n2) * k)
        ldwork = max(n1, n2)
        
        NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                      V1, ldv1, T, ldt, V2, ldv2, C1, ldc1, C2, ldc2, work, ldwork)
        
        @test all(isfinite.(C1))
        @test all(isfinite.(C2))
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m1, n1, m2, n2, k, l = 12, 10, 8, 10, 4, 2
            
            # Create CPU data
            V1_cpu = rand(ComplexF32, m1, k)
            V2_cpu = rand(ComplexF32, m2, k)
            T_cpu = triu(rand(ComplexF32, k, k))
            C1_cpu = rand(ComplexF32, m1, n1)
            C2_cpu = rand(ComplexF32, m2, n2)
            work_cpu = zeros(ComplexF32, max(n1, n2) * k)
            
            for i in 1:k
                V1_cpu[1:i-1, i] .= ComplexF32(0.0)
                V1_cpu[i, i] = ComplexF32(1.0)
                V2_cpu[1:i-1, i] .= ComplexF32(0.0)
            end
            
            # Create GPU data
            V1_gpu = CuArray(V1_cpu)
            V2_gpu = CuArray(V2_cpu)
            T_gpu = CuArray(T_cpu)
            C1_gpu = CuArray(C1_cpu)
            C2_gpu = CuArray(C2_cpu)
            work_gpu = CuArray(work_cpu)
            
            # Apply on CPU
            C1_cpu_result = copy(C1_cpu)
            C2_cpu_result = copy(C2_cpu)
            NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                          V1_cpu, m1, T_cpu, k, V2_cpu, m2, C1_cpu_result, m1, C2_cpu_result, m2, work_cpu, max(n1, n2))
            
            # Apply on GPU
            NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                          V1_gpu, m1, T_gpu, k, V2_gpu, m2, C1_gpu, m1, C2_gpu, m2, work_gpu, max(n1, n2))
            
            @test Array(C1_gpu) ≈ C1_cpu_result rtol=1e-6
            @test Array(C2_gpu) ≈ C2_cpu_result rtol=1e-6
        end
    end
end
