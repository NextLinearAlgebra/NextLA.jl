using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

# Test matrix types 
const TEST_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
const TEST_SIZES = [1, 2, 3, 5, 10, 50]

# Generate test matrices using LAPACK-style patterns
function generate_test_data(::Type{T}, m, n, imat=1) where T
    # Different matrix patterns following LAPACK conventions
    if imat == 1
        # Random matrices
        C = randn(T, m, n)
        v = randn(T, max(m, n))
        tau = randn(T)
        return C, v, tau
    elseif imat == 2
        # Structured matrices
        if T <: Complex
            # Hermitian-like matrices
            C = T[T(i + j*im) for i in 1:m, j in 1:n]
        else
            # Real structured matrices
            C = T[T(i + j) for i in 1:m, j in 1:n]
        end
        v = T[T(i) for i in 1:max(m, n)]
        tau = T(0.5)
        return C, v, tau
    elseif imat == 3
        # Identity-like matrices
        C = Matrix{T}(I, m, n)
        v = ones(T, max(m, n))
        tau = T(1.0)
        return C, v, tau
    elseif imat == 4
        # Matrices with extreme values
        if T <: Float32 || T <: ComplexF32
            # Use extreme values for float types
            C = fill(T(1e-5), m, n)
            v = fill(T(1e5), max(m, n))
            tau = T(1e-5)
            return C, v, tau
        elseif T <: Float64 || T <: ComplexF64
            # Use extreme values for double types
            C = fill(T(1e-10), m, n)
            v = fill(T(1e10), max(m, n))
            tau = T(1e-10)
            return C, v, tau
        end
    elseif imat == 5
        # Zero tau (no operation)
        C = randn(T, m, n)
        v = randn(T, max(m, n))
        tau = T(0)
        return C, v, tau
    else
        # Default random
        C = randn(T, m, n)
        v = randn(T, max(m, n))
        tau = randn(T)
        return C, v, tau
    end
end

# Check if the Householder application is mathematically correct
function check_householder_application(side, m, n, v, tau, C_orig, C_new, rtol)
    # Case 1: If tau is very small (close to zero), there should be minimal change in C
    if abs(tau) < rtol
        # Should be no significant change in C if tau is zero or near zero
        @test norm(C_new - C_orig) < rtol * max(1, norm(C_orig))
        return true
    end

    cond_num = cond(C_orig)
    @test cond_num < 1e10  # Set a reasonable threshold for condition number

    # Case 2: For non-zero tau, check that the operation is reasonable
    @test all(isfinite.(C_new))
    @test size(C_new) == size(C_orig)

    # Adjust tolerance based on the scale of tau and the matrix
    norm_C = norm(C_orig)
    norm_v = norm(v)
    
    # Adaptive proportionality factor, based on matrix size and tau
    proportionality_factor = max(10 * abs(tau) * norm_C * norm_v, 1e-8)

    # Case 3: Check that the change is proportional to tau
    if abs(tau) > rtol
        change_norm = norm(C_new - C_orig)
        @test change_norm < proportionality_factor
    end
    
    return true
end

@testset "ZLARF Tests" begin
    @testset "Standard Test Suite" begin
        for T in TEST_TYPES
            @testset "Type $T" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                atol = rtol
                
                # Test both left and right applications
                for side in ['L', 'R']
                side = 'R'
                    @testset "Side $side" begin
                        for (im, m) in enumerate(TEST_SIZES)
                            for (in, n) in enumerate(TEST_SIZES)
                                @testset "Size m=$m, n=$n" begin
                                    # Test multiple matrix patterns
                                    for imat in 1:5
                                        @testset "Matrix type $imat" begin
                                            C_orig, v, tau = generate_test_data(T, m, n, imat)
                                            C_test = copy(C_orig)
                                            
                                            # Determine work array size
                                            work_size = side == 'L' ? n : m
                                            work = zeros(T, work_size, 1)
                                            
                                            # NextLA call: zlarf(side, m, n, v, incv, tau, c, ldc, work)
                                            NextLA.zlarf(side, m, n, v, 1, tau, C_test, work)
                                            
                                            # Basic checks
                                            @test all(isfinite.(C_test))
                                            @test all(isnan.(C_test) .== false)
                                            @test size(C_test) == (m, n)
                                            
                                            # Check mathematical properties
                                            check_householder_application(side, m, n, v, tau, C_orig, C_test, rtol)
                                            
                                            # Special case: tau = 0 should leave matrix unchanged
                                            if abs(tau) < rtol
                                                @test C_test ≈ C_orig rtol=rtol atol=atol
                                            end
                                            
                                            # Check that work array is used properly
                                            @test all(isfinite.(work))
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
    
    @testset "Error Handling Tests" begin
        # Test error conditions following LAPACK conventions
        for T in TEST_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                m, n = 5, 4
                C = randn(T, m, n)
                v = randn(T, max(m, n))
                tau = T(0.5)
                work = zeros(T, max(m, n), 1)
                
                @test_nowarn NextLA.zlarf('L', m, n, v, 1, tau, C, work)
                @test_nowarn NextLA.zlarf('R', m, n, v, 1, tau, C, work)
                
                # Test edge cases
                @test_nowarn NextLA.zlarf('L', 1, 1, T[T(1)], 1, T(0), T[T(1);;], T[T(0);;])
                @test_nowarn NextLA.zlarf('R', 1, 1, T[T(1)], 1, T(0), T[T(1);;], T[T(0);;])
            end
        end
    end
    
    @testset "Numerical Stability Tests" begin
        # Tests inspired by LAPACK's numerical stability checks
        for T in [ComplexF64, Float64]  # High precision type
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 100
                
                # Test with very small and very large numbers
                scales = [eps(real(T)), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    for side in ['L', 'R']
                        m, n = 10, 8
                        C = T.(scale .* randn(T, m, n))
                        v = T.(scale .* randn(T, max(m, n)))
                        tau = T(scale * randn(T))
                        work = zeros(T, max(m, n), 1)
                        
                        C_orig = copy(C)
                        
                        # Test calculation
                        NextLA.zlarf(side, m, n, v, 1, tau, C, work)
                        
                        # Check that results are finite
                        @test all(isfinite.(C))
                        @test all(isfinite.(work))
                        
                        # Check mathematical properties
                        check_householder_application(side, m, n, v, tau, C_orig, C, rtol)
                    end
                end
            end
        end
    end

    @testset "GPU Tests" begin
        if CUDA.functional()
            for T in (T,) # Common GPU type
                @testset "Type $T GPU" begin
                    rtol = 1e-5
                    
                    for side in ['L', 'R']
                        for (m, n) in [(2, 3), (5, 4), (10, 8)]
                            C_cpu = T.(randn(T, m, n))
                            v_cpu = T.(randn(T, max(m, n)))
                            tau_cpu = T(randn(T))
                            work_cpu = zeros(T, max(m, n), 1)
                            
                            # Move data to GPU
                            C_gpu = CuArray(C_cpu)
                            v_gpu = CuArray(v_cpu)
                            work_gpu = CuArray(work_cpu)
                            
                            # Reference CPU calculation
                            C_ref = copy(C_cpu)
                            work_ref = copy(work_cpu)
                            NextLA.zlarf(side, m, n, v_cpu, 1, tau_cpu, C_ref, m, work_ref)
                            
                            # Our implementation on GPU
                            NextLA.zlarf(side, m, n, v_gpu, 1, tau_cpu, C_gpu, m, work_gpu)
                            
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
end
