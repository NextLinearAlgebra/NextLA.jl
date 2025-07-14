using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

# LAPACK-style test parameters adapted for NextLA.axpy!(a, x, y)
const NTYPES = 4
const NSIZES = 7

# Test matrix types (following LAPACK conventions)
const TEST_TYPES = [Float32, Float64, ComplexF32, ComplexF64]
const TEST_SIZES = [0, 1, 2, 3, 10, 100, 1023]

# Generate test matrices using LAPACK-style patterns
function generate_test_vectors(::Type{T}, n, imat=1) where T
    if n == 0
        return T[], T[]
    end
    
    # Different vector patterns following LAPACK conventions
    if imat == 1
        # Random vectors
        return rand(T, n), rand(T, n)
    elseif imat == 2
        # Vectors with known pattern
        x = T[T(i) for i in 1:n]
        y = T[T(2*i) for i in 1:n]
        return x, y
    elseif imat == 3
        # Vectors with alternating signs
        x = T[T((-1)^i * i) for i in 1:n]
        y = T[T((-1)^(i+1) * i) for i in 1:n]
        return x, y
    elseif imat == 4
        # Vectors with extreme values
        x = fill(T(1e-10), n)
        y = fill(T(1e10), n)
        if n > 1
            x[1] = T(1e10)
            y[1] = T(1e-10)
        end
        return x, y
    else
        # Default random
        return rand(T, n), rand(T, n)
    end
end

# Alpha values for testing (including special cases)
function get_alpha_values(::Type{T}) where T
    if T <: Complex
        return T[
            T(0, 0),                    # Zero
            T(1, 0),                    # One
            T(-1, 0),                   # Minus one
            T(0, 1),                    # Pure imaginary
            T(0, -1),                   # Negative imaginary
            T(0.7, 0.3),               # General complex
            T(-0.5, 0.8),              # Another complex
            T(1e-10, 0),               # Very small real
            T(0, 1e-10),               # Very small imaginary
            T(1e3, 0),                 # Large real
        ]
    else
        return T[
            T(0),                       # Zero
            T(1),                       # One
            T(-1),                      # Minus one
            T(0.7),                     # Positive
            T(-0.3),                    # Negative
            T(1e-10),                   # Very small
            T(1e3),                     # Large
            T(eps(T)),                  # Machine epsilon
            T(floatmin(T)),            # Smallest normal
        ]
    end
end

@testset "AXPY Tests" begin
    @testset "Standard Test Suite" begin
        for (itype, T) in enumerate(TEST_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: Float32) || (T <: ComplexF32) ? 1e-6 : 1e-12
                atol = rtol
                
                for (isize, n) in enumerate(TEST_SIZES)
                    @testset "Size n=$n (isize=$isize)" begin
                        # Test multiple matrix patterns
                        for imat in 1:4
                            @testset "Matrix type $imat" begin
                                x, y_orig = generate_test_vectors(T, n, imat)
                                
                                # Test with multiple alpha values
                                for (ialpha, α) in enumerate(get_alpha_values(T))
                                    y_ref = copy(y_orig)
                                    y_test = copy(y_orig)
                                    
                                    # Reference calculation: y = y + α*x
                                    if n > 0
                                        BLAS.axpy!(α, x, y_ref)
                                    end
                                    
                                    # NextLA call: axpy!(a, x, y)
                                    NextLA.axpy!(α, x, y_test)

                                    # Comparison with LAPACK-style error checking
                                    if n == 0 || α == 0
                                        @test y_test == y_orig  # Should be unchanged
                                    else
                                        # Scale tolerance appropriately
                                        scaled_rtol = rtol * max(1, n, abs(α))
                                        scaled_atol = atol * max(1, norm(y_orig), abs(α) * norm(x))
                                        @test y_test ≈ y_ref rtol=scaled_rtol atol=scaled_atol
                                        
                                        # Additional LAPACK-style checks
                                        @test all(isfinite.(y_test))
                                        @test length(y_test) == length(y_ref)
                                        
                                        # Verify the mathematical property: y = y_orig + α*x
                                        if abs(α) > eps(real(T)) * 100
                                            expected = y_orig + α * x
                                            @test y_test ≈ expected rtol=scaled_rtol atol=scaled_atol
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
                n = 10
                x = rand(T, n)
                y = rand(T, n)
                α = one(T)
                
                # Test with valid parameters (should not error)
                @test_nowarn NextLA.axpy!(α, x, y)
                
                # Test edge cases
                @test_nowarn NextLA.axpy!(T(0), T[], T[])  # n = 0
                @test_nowarn NextLA.axpy!(T(0), x, y)  # α = 0
            end
        end
    end
    
    @testset "Numerical Stability Tests" begin
        # Tests inspired by LAPACK's numerical stability checks
        for T in [Float64, ComplexF64]  # High precision types
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 100
                
                # Test with very small and very large numbers
                scales = [eps(real(T)), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    n = 100
                    α = T(scale)
                    x = T.(scale .* randn(n))
                    y_orig = T.(scale .* randn(n))
                    y_test = copy(y_orig)
                    y_ref = copy(y_orig)
                    
                    # Reference and test calculations
                    BLAS.axpy!(α, x, y_ref)
                    NextLA.axpy!(α, x, y_test)
                    
                    # Check relative error
                    if norm(y_ref) > 0
                        rel_err = norm(y_test - y_ref) / norm(y_ref)
                        @test rel_err < rtol * max(1, n)
                    else
                        @test norm(y_test - y_ref) < rtol * max(1, norm(x))
                    end
                    
                    # Check that results are finite
                    @test all(isfinite.(y_test))
                end
            end
        end
    end

    @testset "GPU Tests" begin
        if CUDA.functional()
            for T in (Float32, ComplexF32) # Common GPU types
                @testset "Type $T GPU" begin
                    rtol = (T <: ComplexF32) || (T <:Float32) ? 1e-5 : 1e-6
                    
                    for n in [0, 1, 10, 100]
                        α = T(0.5)
                        x_cpu = rand(T, n)
                        y_cpu = rand(T, n)
                        y_expected = copy(y_cpu)
                        
                        # Move data to GPU
                        x_gpu = CuArray(x_cpu)
                        y_gpu = CuArray(y_cpu)
                        
                        # Reference CPU calculation
                        if n > 0
                            BLAS.axpy!(α, x_cpu, y_expected)
                        end
                        
                        # Our implementation on GPU
                        NextLA.axpy!(α, x_gpu, y_gpu)
                        
                        # Compare results
                        @test Array(y_gpu) ≈ y_expected rtol=rtol
                        @test all(isfinite.(Array(y_gpu)))
                    end
                end
            end
        end
    end
end
