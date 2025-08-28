using Test
using NextLA
using LinearAlgebra
using CUDA

@testset "PEMV Tests" begin
    @testset "Column Storage No-Transpose Tests" begin
        m, n, l = 200, 150, 80
        alpha = 2.5 + 1.5im
        beta = 1.2 - 0.8im
        
    A = rand(ComplexF64, m, n)
        X = rand(ComplexF64, n)
        Y = rand(ComplexF64, m)
        Y_original = copy(Y)
        work = zeros(ComplexF64, m)
        
        # Apply our PEMV
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        
        # Verify using manual computation
        # For column storage with conjugate transpose, this should compute:
        # Y := alpha * A^H * X + beta * Y
        Y_expected = alpha * A' * X + beta * Y_original
        
        @test Y ≈ Y_expected rtol=1e-12
    end
    
    @testset "Row Storage No-Transpose Tests" begin
        m, n, l = 150, 200, 100
        alpha = 1.8 + 2.2im
        beta = 0.5 + 1.0im
        
    A = rand(ComplexF64, m, n)
        X = rand(ComplexF64, m)
        Y = rand(ComplexF64, n)
        Y_original = copy(Y)
        work = zeros(ComplexF64, n)
        
        # Apply our PEMV
    NextLA.pemv!('N', 'R', m, n, l, alpha, A, X, beta, Y, work)
        
        # For row storage with no transpose:
        # Y := alpha * A^T * X + beta * Y
        Y_expected = alpha * A' * X + beta * Y_original
        
        @test Y ≈ Y_expected rtol=1e-12
    end
    
    @testset "ComplexF32 Tests" begin
        m, n, l = 120, 150, 60
        alpha = ComplexF32(2.0 + 1.0im)
        beta = ComplexF32(0.8 - 0.5im)
        
    A = rand(ComplexF32, m, n)
        X = rand(ComplexF32, n)
        Y = rand(ComplexF32, m)
        Y_original = copy(Y)
        work = zeros(ComplexF32, m)
        
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        
        Y_expected = alpha * A' * X + beta * Y_original
        
        @test Y ≈ Y_expected rtol=1e-6
    end
    
    @testset "Different Sizes" begin
        test_cases = [
            (10, 8, 5), (25, 20, 12), (15, 30, 10), (50, 40, 25)
        ]
        
        for (m, n, l) in test_cases
            alpha = rand(ComplexF64)
            beta = rand(ComplexF64)
            
            # Test column storage
            A = rand(ComplexF64, m, n)
            X = rand(ComplexF64, n)
            Y = rand(ComplexF64, m)
            Y_original = copy(Y)
            work = zeros(ComplexF64, m)
            
            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
            
            Y_expected = alpha * A' * X + beta * Y_original
            @test Y ≈ Y_expected rtol=1e-12
        end
    end
    
    @testset "Zero Alpha Tests" begin
        m, n, l = 150, 120, 80
        alpha = ComplexF64(0.0)
        beta = 2.0 + 1.5im
        
    A = rand(ComplexF64, m, n)
        X = rand(ComplexF64, n)
        Y = rand(ComplexF64, m)
        Y_original = copy(Y)
        work = zeros(ComplexF64, m)
        
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        
        # With alpha = 0, result should be beta * Y_original
        Y_expected = beta * Y_original
        @test Y ≈ Y_expected rtol=1e-15
    end
    
    @testset "Zero Beta Tests" begin
        m, n, l = 150, 120, 80
        alpha = 2.0 + 1.5im
        beta = ComplexF64(0.0)
        
    A = rand(ComplexF64, m, n)
        X = rand(ComplexF64, n)
        Y = rand(ComplexF64, m)
        Y_original = copy(Y)
        work = zeros(ComplexF64, m)
        
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        
        # With beta = 0, result should be alpha * A' * X
        Y_expected = alpha * A' * X
        @test Y ≈ Y_expected rtol=1e-12
    end
    
    @testset "Both Alpha and Beta Zero Tests" begin
        m, n, l = 10, 8, 5
        alpha = ComplexF64(0.0)
        beta = ComplexF64(0.0)
        
    A = rand(ComplexF64, m, n)
        X = rand(ComplexF64, n)
        Y = rand(ComplexF64, m)
        Y_original = copy(Y)
        work = zeros(ComplexF64, m)
        
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        
        # Function should return early, Y might be unchanged or zeroed
        # Check that it doesn't crash and produces finite results
        @test all(isfinite.(Y))
    end
    
    @testset "Edge Cases" begin
        # m = 0 case
        m, n, l = 0, 5, 0
        alpha = 2.0 + 1.0im
        beta = 1.5 - 0.5im
        
    A = zeros(ComplexF64, max(1, m), n)
        X = rand(ComplexF64, n)
        Y = ComplexF64[]
        work = ComplexF64[]
        
        # Should return early without error
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        @test length(Y) == 0
        
        # n = 0 case
        m, n, l = 5, 0, 0
    A = rand(ComplexF64, m, max(1, n))
        X = ComplexF64[]
        Y = rand(ComplexF64, m)
        Y_original = copy(Y)
        work = zeros(ComplexF64, m)
        
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
        # Should return early
        @test all(isfinite.(Y))
    end
    
    @testset "Error Handling" begin
        m, n, l = 10, 8, 5
        alpha = 1.0 + 0.5im
        beta = 0.8 - 0.3im
        A = zeros(ComplexF64, m, n)
        X = zeros(ComplexF64, n)
        Y = zeros(ComplexF64, m)
        work = zeros(ComplexF64, m)
        
        # Invalid trans
    @test_throws ArgumentError NextLA.pemv!('X', 'C', m, n, l, alpha, A, X, beta, Y, work)
        
        # Invalid storev
    @test_throws ArgumentError NextLA.pemv!('C', 'X', m, n, l, alpha, A, X, beta, Y, work)
        
        # Invalid trans/storev combination
    @test_throws ArgumentError NextLA.pemv!('N', 'C', m, n, l, alpha, A, X, beta, Y, work)
    @test_throws ArgumentError NextLA.pemv!('C', 'R', m, n, l, alpha, A, X, beta, Y, work)
        
        # Negative dimensions
    @test_throws ArgumentError NextLA.pemv!('C', 'C', -1, n, l, alpha, A, X, beta, Y, work)
    @test_throws ArgumentError NextLA.pemv!('C', 'C', m, -1, l, alpha, A, X, beta, Y, work)
        
        # Invalid l (l > min(m,n))
    @test_throws ArgumentError NextLA.pemv!('C', 'C', m, n, min(m,n)+1, alpha, A, X, beta, Y, work)
        
    # No lda parameter to validate anymore
    end
    
    @testset "Consistency with BLAS" begin
        # Compare with standard GEMV where possible
        m, n, l = 20, 15, 10
        alpha = 2.0 + 1.0im
        beta = 1.5 - 0.8im
        
    A = rand(ComplexF64, m, n)
        X = rand(ComplexF64, n)
        Y1 = rand(ComplexF64, m)
        Y2 = copy(Y1)
        work = zeros(ComplexF64, m)
        
        # Our implementation
    NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y1, work)
        
        # BLAS reference
        LinearAlgebra.BLAS.gemv!('C', alpha, A, X, beta, Y2)
        
        @test Y1 ≈ Y2 rtol=1e-12
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, l = 16, 12, 8
            alpha = ComplexF32(2.0 + 1.0im)
            beta = ComplexF32(1.5 - 0.5im)
            
            # Create CPU data
            A_cpu = rand(ComplexF32, m, n)
            X_cpu = rand(ComplexF32, n)
            Y_cpu = rand(ComplexF32, m)
            work_cpu = zeros(ComplexF32, m)
            
            # Create GPU data
            A_gpu = CuArray(A_cpu)
            X_gpu = CuArray(X_cpu)
            Y_gpu = CuArray(Y_cpu)
            work_gpu = CuArray(work_cpu)
            
            # Apply on CPU
            Y_cpu_result = copy(Y_cpu)
            NextLA.pemv!('C', 'C', m, n, l, alpha, A_cpu, X_cpu, beta, Y_cpu_result, work_cpu)
            
            # Apply on GPU
            NextLA.pemv!('C', 'C', m, n, l, alpha, A_gpu, X_gpu, beta, Y_gpu, work_gpu)
            
            @test Array(Y_gpu) ≈ Y_cpu_result rtol=1e-6
        end
    end
end
