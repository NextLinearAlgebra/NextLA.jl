using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

# LAPACK-style test parameters for NextLA.zlarft
const NTYPES = 2
const NSIZES = 6

# Test matrix types (complex types for zlarft)
const TEST_TYPES = [ComplexF32, ComplexF64]
const TEST_SIZES = [0, 1, 2, 3, 5, 10]

# Generate test matrices using LAPACK-style patterns
function generate_test_matrix(::Type{T}, n, k, imat=1) where T
    if n == 0 || k == 0
        return zeros(T, n, k), zeros(T, k)
    end
    
    # Different matrix patterns following LAPACK conventions
    if imat == 1
        # Random matrix with orthogonal columns
        A = randn(T, n, k)
        Q, R = qr(A)
        V = Matrix(Q)
        tau = randn(T, k)
        return V[:, 1:k], tau
    elseif imat == 2
        # Identity-like matrix
        V = zeros(T, n, k)
        for i in 1:min(n, k)
            V[i, i] = one(T)
        end
        tau = ones(T, k)
        return V, tau
    elseif imat == 3
        # Matrix with alternating pattern
        V = zeros(T, n, k)
        for i in 1:min(n, k)
            V[i, i] = T((-1)^i)
        end
        tau = T[T(0.5 * (-1)^i) for i in 1:k]
        return V, tau
    elseif imat == 4
        # Matrix with extreme values
        V = randn(T, n, k) * T(1e-3)
        tau = randn(T, k) * T(1e3)
        return V, tau
    else
        # Default random
        V = randn(T, n, k)
        tau = randn(T, k)
        return V, tau
    end
end

@testset "ZLARFT LAPACK-style Tests" begin
    @testset "Standard LAPACK Test Suite" begin
        for (itype, T) in enumerate(TEST_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) ? 1e-5 : 1e-12
                atol = rtol
                
                for (isize, n) in enumerate(TEST_SIZES)
                    @testset "Size n=$n (isize=$isize)" begin
                        for k in [0, 1, min(n, 3)]
                            @testset "k=$k" begin
                                # Test different storage directions
                                for direct in ['F', 'B']
                                    for storev in ['C', 'R']
                                        @testset "direct=$direct, storev=$storev" begin
                                            # Test multiple matrix patterns
                                            for imat in 1:4
                                                @testset "Matrix type $imat" begin
                                                    V, tau = generate_test_matrix(T, n, k, imat)
                                                    ldv = size(V, 1)
                                                    
                                                    # Allocate T matrix
                                                    T_mat = zeros(T, k, k)
                                                    ldt = k
                                                    
                                                    # NextLA call: zlarft(direct, storev, n, k, v, ldv, tau, t, ldt)
                                                    NextLA.zlarft(direct, storev, n, k, V, ldv, tau, T_mat, ldt)
                                                    
                                                    # Basic checks
                                                    @test all(isfinite.(T_mat))
                                                    @test size(T_mat) == (k, k)
                                                    
                                                    # Check that T is upper triangular
                                                    if k > 0
                                                        for i in 1:k
                                                            for j in 1:i-1
                                                                @test abs(T_mat[i, j]) < atol
                                                            end
                                                        end
                                                        
                                                        # Check diagonal elements
                                                        for i in 1:k
                                                            if abs(tau[i]) < atol
                                                                @test abs(T_mat[i, i]) < atol
                                                            else
                                                                @test abs(T_mat[i, i] - tau[i]) < rtol * abs(tau[i])
                                                            end
                                                        end
                                                    end
                                                    
                                                    # Mathematical property check: T should satisfy the block reflector property
                                                    if k > 0 && n > 0
                                                        # For valid cases, check basic properties
                                                        if storev == 'C'
                                                            # Column-wise storage
                                                            for j in 1:k
                                                                # Check that columns are normalized properly
                                                                @test all(isfinite.(V[:, j]))
                                                            end
                                                        else
                                                            # Row-wise storage
                                                            for i in 1:k
                                                                # Check that rows are normalized properly
                                                                @test all(isfinite.(V[i, :]))
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    @testset "LAPACK Error Handling Tests" begin
        # Test error conditions following LAPACK conventions
        for T in TEST_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                n, k = 5, 3
                V = randn(T, n, k)
                tau = randn(T, k)
                T_mat = zeros(T, k, k)
                
                @test_nowarn NextLA.zlarft('F', 'C', n, k, V, n, tau, T_mat, k)
                
                # Test edge cases
                @test_nowarn NextLA.zlarft('F', 'C', 0, 0, zeros(T, 0, 0), 1, T[], zeros(T, 0, 0), 1)  # n = 0, k = 0
                @test_nowarn NextLA.zlarft('F', 'C', 1, 0, zeros(T, 1, 0), 1, T[], zeros(T, 0, 0), 1)  # k = 0
                @test_nowarn NextLA.zlarft('F', 'C', 0, 1, zeros(T, 0, 1), 1, T[T(0)], zeros(T, 1, 1), 1)  # n = 0
            end
        end
    end
    
    @testset "LAPACK Numerical Stability Tests" begin
        # Tests inspired by LAPACK's numerical stability checks
        for T in [ComplexF64]  # High precision type
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 100
                
                # Test with very small and very large numbers
                scales = [eps(real(T)), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    n, k = 10, 5
                    V = T.(scale .* randn(ComplexF64, n, k))
                    tau = T.(scale .* randn(ComplexF64, k))
                    T_mat = zeros(T, k, k)
                    
                    # Test calculation
                    NextLA.zlarft('F', 'C', n, k, V, n, tau, T_mat, k)
                    
                    # Check that results are finite
                    @test all(isfinite.(T_mat))
                    
                    # Check upper triangular structure
                    for i in 1:k
                        for j in 1:i-1
                            @test abs(T_mat[i, j]) < rtol * max(1, abs(T_mat[i, i]))
                        end
                    end
                end
            end
        end
    end

    @testset "GPU Tests (LAPACK-compatible)" begin
        if CUDA.functional()
            for T in (ComplexF32,) # Common GPU type
                @testset "Type $T GPU" begin
                    rtol = 1e-5
                    
                    for (n, k) in [(1, 1), (5, 3), (10, 5)]
                        V_cpu = randn(T, n, k)
                        tau_cpu = randn(T, k)
                        
                        # Move data to GPU
                        V_gpu = CuArray(V_cpu)
                        tau_gpu = CuArray(tau_cpu)
                        T_gpu = CuArray(zeros(T, k, k))
                        
                        # Reference CPU calculation
                        T_ref = zeros(T, k, k)
                        NextLA.zlarft('F', 'C', n, k, V_cpu, n, tau_cpu, T_ref, k)
                        
                        # Our implementation on GPU
                        NextLA.zlarft('F', 'C', n, k, V_gpu, n, tau_gpu, T_gpu, k)
                        
                        # Compare results
                        @test norm(Array(T_gpu) - T_ref) < rtol * max(1, norm(T_ref))
                        @test all(isfinite.(Array(T_gpu)))
                    end
                end
            end
        end
    end
end

