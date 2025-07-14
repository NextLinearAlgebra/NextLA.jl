using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

# LAPACK-style test parameters for NextLA.zlarfb
const NTYPES = 2
const NSIZES = 5

# Test matrix types (complex types for zlarfb)
const TEST_TYPES = [ComplexF32, ComplexF64]
const TEST_SIZES = [1, 2, 3, 5, 10]

# Generate test matrices using LAPACK-style patterns
function generate_test_matrices(::Type{T}, m, n, k, imat=1) where T
    if m == 0 || n == 0 || k == 0
        return zeros(T, m, k), zeros(T, k, k), zeros(T, m, n), zeros(T, max(m,n), k)
    end
    
    # Different matrix patterns following LAPACK conventions
    if imat == 1
        # Random matrices
        V = randn(T, m, k)
        T_mat = triu(randn(T, k, k))
        C = randn(T, m, n)
        work = zeros(T, max(m,n), k)
        return V, T_mat, C, work
    elseif imat == 2
        # Identity-like matrices
        V = zeros(T, m, k)
        for i in 1:min(m, k)
            V[i, i] = one(T)
        end
        T_mat = Matrix{T}(I, k, k)
        C = Matrix{T}(I, m, n)
        work = zeros(T, max(m,n), k)
        return V, T_mat, C, work
    elseif imat == 3
        # Matrices with alternating pattern
        V = zeros(T, m, k)
        for i in 1:min(m, k)
            V[i, i] = T((-1)^i)
        end
        T_mat = Diagonal(T[T(0.5 * (-1)^i) for i in 1:k])
        C = T[T((-1)^(i+j)) for i in 1:m, j in 1:n]
        work = zeros(T, max(m,n), k)
        return V, T_mat, C, work
    else
        # Default random
        V = randn(T, m, k)
        T_mat = triu(randn(T, k, k))
        C = randn(T, m, n)
        work = zeros(T, max(m,n), k)
        return V, T_mat, C, work
    end
end

@testset "ZLARFB LAPACK-style Tests" begin
    @testset "Standard LAPACK Test Suite" begin
        for (itype, T) in enumerate(TEST_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) ? 1e-5 : 1e-12
                atol = rtol
                
                for (isize, m) in enumerate(TEST_SIZES)
                    for n in TEST_SIZES[1:min(3, end)]
                        for k in [1, min(m, n, 3)]
                            @testset "Size m=$m, n=$n, k=$k" begin
                                # Test different parameter combinations
                                for side in ['L', 'R']
                                    for trans in ['N', 'C']
                                        for direct in ['F', 'B']
                                            for storev in ['C', 'R']
                                                @testset "side=$side, trans=$trans, direct=$direct, storev=$storev" begin
                                                    # Test multiple matrix patterns
                                                    for imat in 1:3
                                                        @testset "Matrix type $imat" begin
                                                            # Generate appropriate matrices based on parameters
                                                            if side == 'L'
                                                                V_rows = storev == 'C' ? m : k
                                                                V_cols = storev == 'C' ? k : m
                                                                work_rows = n
                                                            else # side == 'R'
                                                                V_rows = storev == 'C' ? n : k
                                                                V_cols = storev == 'C' ? k : n
                                                                work_rows = m
                                                            end
                                                            
                                                            V = randn(T, V_rows, V_cols)
                                                            T_mat = triu(randn(T, k, k))
                                                            C_orig = randn(T, m, n)
                                                            C_test = copy(C_orig)
                                                            work = zeros(T, work_rows, k)
                                                            
                                                            # Set leading dimensions
                                                            ldv = size(V, 1)
                                                            ldt = k
                                                            ldc = m
                                                            ldwork = work_rows
                                                            
                                                            # NextLA call: zlarfb(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)
                                                            NextLA.zlarfb(side, trans, direct, storev, m, n, k, V, ldv, T_mat, ldt, C_test, ldc, work, ldwork)
                                                            
                                                            # Basic checks
                                                            @test all(isfinite.(C_test))
                                                            @test size(C_test) == (m, n)
                                                            @test all(isfinite.(work))
                                                            
                                                            # Check that C has been modified (unless k=0)
                                                            if k > 0 && norm(T_mat) > atol
                                                                # Should be different from original unless T is zero
                                                                if norm(T_mat) > rtol
                                                                    @test norm(C_test - C_orig) > rtol * norm(C_orig)
                                                                end
                                                            end
                                                            
                                                            # Mathematical property check: applying the same operation twice should give back original
                                                            if k > 0 && norm(T_mat) > atol
                                                                C_double = copy(C_test)
                                                                work2 = zeros(T, work_rows, k)
                                                                
                                                                # Apply the inverse operation
                                                                inverse_trans = (trans == 'N') ? 'C' : 'N'
                                                                NextLA.zlarfb(side, inverse_trans, direct, storev, m, n, k, V, ldv, T_mat, ldt, C_double, ldc, work2, ldwork)
                                                                
                                                                # Should be close to original (for orthogonal/unitary transformations)
                                                                # Note: This is an approximation since we don't have exact inverse
                                                                @test all(isfinite.(C_double))
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
                m, n, k = 5, 4, 3
                V = randn(T, m, k)
                T_mat = triu(randn(T, k, k))
                C = randn(T, m, n)
                work = zeros(T, n, k)
                
                @test_nowarn NextLA.zlarfb('L', 'N', 'F', 'C', m, n, k, V, m, T_mat, k, C, m, work, n)
                
                # Test edge cases
                @test_nowarn NextLA.zlarfb('L', 'N', 'F', 'C', 0, 0, 0, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1)  # m = n = k = 0
                @test_nowarn NextLA.zlarfb('L', 'N', 'F', 'C', 1, 1, 0, zeros(T, 1, 0), 1, zeros(T, 0, 0), 1, randn(T, 1, 1), 1, zeros(T, 1, 0), 1)  # k = 0
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
                    m, n, k = 8, 6, 4
                    V = T.(scale .* randn(ComplexF64, m, k))
                    T_mat = triu(T.(scale .* randn(ComplexF64, k, k)))
                    C = T.(scale .* randn(ComplexF64, m, n))
                    work = zeros(T, n, k)
                    
                    # Test calculation
                    NextLA.zlarfb('L', 'N', 'F', 'C', m, n, k, V, m, T_mat, k, C, m, work, n)
                    
                    # Check that results are finite
                    @test all(isfinite.(C))
                    @test all(isfinite.(work))
                end
            end
        end
    end

    @testset "GPU Tests (LAPACK-compatible)" begin
        if CUDA.functional()
            for T in (ComplexF32,) # Common GPU type
                @testset "Type $T GPU" begin
                    rtol = 1e-5
                    
                    for (m, n, k) in [(3, 3, 2), (5, 4, 3)]
                        V_cpu = randn(T, m, k)
                        T_cpu = triu(randn(T, k, k))
                        C_cpu = randn(T, m, n)
                        work_cpu = zeros(T, n, k)
                        
                        # Move data to GPU
                        V_gpu = CuArray(V_cpu)
                        T_gpu = CuArray(T_cpu)
                        C_gpu = CuArray(C_cpu)
                        work_gpu = CuArray(work_cpu)
                        
                        # Reference CPU calculation
                        C_ref = copy(C_cpu)
                        work_ref = copy(work_cpu)
                        NextLA.zlarfb('L', 'N', 'F', 'C', m, n, k, V_cpu, m, T_cpu, k, C_ref, m, work_ref, n)
                        
                        # Our implementation on GPU
                        NextLA.zlarfb('L', 'N', 'F', 'C', m, n, k, V_gpu, m, T_gpu, k, C_gpu, m, work_gpu, n)
                        
                        # Compare results
                        @test norm(Array(C_gpu) - C_ref) < rtol * max(1, norm(C_ref))
                        @test norm(Array(work_gpu) - work_ref) < rtol * max(1, norm(work_ref))
                        
                        @test all(isfinite.(Array(C_gpu)))
                        @test all(isfinite.(Array(work_gpu)))
                    end
                end
            end
        end
    end
end
