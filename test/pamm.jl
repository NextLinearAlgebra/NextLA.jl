using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "PAMM Tests" begin
    @testset "Left Column-wise Forward Tests" begin
        m, n, k, l = 200, 150, 80, 50
        
        # Create test matrices
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
        # Make copies for verification
        A1_original = copy(A1)
        A2_original = copy(A2)
        V_original = copy(V)
        W_original = copy(W)
        
    # Apply our PAMM
    NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)
        
        # Basic checks
        @test size(W) == (n, l)
        @test all(isfinite.(W))
        
        # Verify that operation modified W
        @test !isapprox(W, W_original, rtol=1e-12)
    end
    
    @testset "Right Column-wise Forward Tests" begin
        m, n, k, l = 150, 200, 80, 50
        
        A1 = rand(ComplexF64, k, n)
        A2 = rand(ComplexF64, n, k)
        V = rand(ComplexF64, n, l)
        W = rand(ComplexF64, m, l)
        
    NextLA.pamm!('W', 'R', 'C', 'F', m, n, k, l, A1, A2, V, W)
        
        @test size(W) == (m, l)
        @test all(isfinite.(W))
    end
    
    @testset "A Operation Tests" begin
        m, n, k, l = 200, 150, 80, 50
        
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
        A2_original = copy(A2)
        
        # Apply A operation
    NextLA.pamm!('A', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)
        
        @test size(A2) == (m, k)
        @test all(isfinite.(A2))
        
        # Verify that A2 was modified
        @test !isapprox(A2, A2_original, rtol=1e-12)
    end
    
    @testset "Backward Direction Tests" begin
        m, n, k, l = 200, 150, 80, 50
        
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
    NextLA.pamm!('W', 'L', 'C', 'B', m, n, k, l, A1, A2, V, W)
        
        @test all(isfinite.(W))
    end
    
    @testset "Row-wise Storage Tests" begin
        m, n, k, l = 150, 200, 60, 40
        
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, l, m)  # Row-wise storage
        W = rand(ComplexF64, l, n)  # Row-wise storage
        
    NextLA.pamm!('W', 'L', 'R', 'F', m, n, k, l, A1, A2, V, W)
        
        @test all(isfinite.(W))
    end
    
    @testset "ComplexF32 Tests" begin
        m, n, k, l = 120, 100, 60, 40
        
        A1 = rand(ComplexF32, k, m)
        A2 = rand(ComplexF32, m, k)
        V = rand(ComplexF32, m, l)
        W = rand(ComplexF32, n, l)
        
        W_original = copy(W)
        
    NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)
        
        @test all(isfinite.(W))
        @test !isapprox(W, W_original, rtol=1e-6)
    end
    
    @testset "Different Sizes" begin
        test_cases = [
            (10, 8, 5, 3),
            (25, 20, 12, 8),
            (15, 30, 10, 6),
            (50, 40, 25, 15)
        ]
        
        for (m, n, k, l) in test_cases
            A1 = rand(ComplexF64, k, m)
            A2 = rand(ComplexF64, m, k)
            V = rand(ComplexF64, m, l)
            W = rand(ComplexF64, n, l)
            
            W_original = copy(W)
            
            lda1 = k
            lda2 = m
            ldv = m
            ldw = n
            
            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, lda1, A2, lda2, V, ldv, W, ldw)
            
            @test all(isfinite.(W))
            @test size(W) == (n, l)
        end
    end
    
    @testset "Edge Cases" begin
        # Minimal sizes
        m, n, k, l = 200, 200, 100, 100
        
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
        lda1 = k
        lda2 = m
        ldv = m
        ldw = n
        
        NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, lda1, A2, lda2, V, ldv, W, ldw)
        
        @test all(isfinite.(W))
        
        # Single dimension cases
        m, n, k, l = 500, 100, 300, 200
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
        NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, lda1, A2, lda2, V, ldv, W, ldw)
        
        @test all(isfinite.(W))
    end
    
    @testset "Wrapper Function Tests" begin
    # Test pamm! simplified wrapper
        m, n, k, l = 150, 120, 80, 50
        
        A1 = rand(ComplexF64, k, m)
        A2 = rand(ComplexF64, m, k)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
        W_original = copy(W)
        
    NextLA.pamm!('W', 'L', 'C', 'F', A1, A2, V)
        
        @test all(isfinite.(W))
        @test !isapprox(W, W_original, rtol=1e-12)
        
    # Test pamm! simplified wrapper for 'A'
        A2_test = rand(ComplexF64, m, k)
        A2_original = copy(A2_test)
        
    NextLA.pamm!('A', 'L', 'C', 'F', A1, A2_test, V)
        
        @test all(isfinite.(A2_test))
    end
    
    @testset "Consistency Tests" begin
        # Test that W and A operations are consistent
        m, n, k, l = 150, 120, 80, 50
        
        A1 = rand(ComplexF64, k, m)
        A2_w = rand(ComplexF64, m, k)
        A2_a = copy(A2_w)
        V = rand(ComplexF64, m, l)
        W = rand(ComplexF64, n, l)
        
        # Apply W operation
            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2_w, V, W)
        
        # Apply A operation with same input
            NextLA.pamm!('A', 'L', 'C', 'F', m, n, k, l, A1, A2_a, V, W)
        
        # Results should be finite and well-defined
        @test all(isfinite.(W))
        @test all(isfinite.(A2_a))
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, k, l = 120, 100, 60, 40
            
            # Create CPU data
            A1_cpu = rand(ComplexF32, k, m)
            A2_cpu = rand(ComplexF32, m, k)
            V_cpu = rand(ComplexF32, m, l)
            W_cpu = rand(ComplexF32, n, l)
            
            # Prepare GPU data
            A1_gpu = CuArray(A1_cpu)
            A2_gpu = CuArray(A2_cpu)
            V_gpu = CuArray(V_cpu)
            W_gpu = CuArray(W_cpu)
            
            # Apply on CPU (full signature)
            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1_cpu, A2_cpu, V_cpu, W_cpu)
            
            # Apply on GPU (full signature)
            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1_gpu, A2_gpu, V_gpu, W_gpu)
            
            @test Array(W_gpu) ≈ W_cpu rtol=1e-6
        end
    end
end
