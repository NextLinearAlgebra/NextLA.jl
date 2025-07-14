using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

# LAPACK-style test parameters for GERC (complex rank-1 update)
# Adapted for NextLA.gerc!(alpha, x, y, A)
const GERC_NTYPES = 2  # Only complex types
const GERC_NSIZES = 8

# Test matrix types (only complex for GERC)
const GERC_TYPES = [ComplexF32, ComplexF64]
const GERC_SIZES = [(0,0), (1,1), (2,1), (1,2), (3,3), (5,4), (10,8), (100,50)]

# Generate test matrices using LAPACK-style patterns
function generate_test_matrix(::Type{T}, m, n, imat=1) where T
    if m == 0 || n == 0
        return zeros(T, m, n)
    end
    
    # Different matrix patterns following LAPACK conventions
    if imat == 1
        # Random matrix
        return rand(T, m, n)
    elseif imat == 2
        # Matrix with known pattern
        A = zeros(T, m, n)
        for i in 1:min(m,n)
            A[i,i] = T(i, i/2)
        end
        return A
    elseif imat == 3
        # Nearly singular matrix
        A = rand(T, m, n) * T(1e-6)
        return A
    elseif imat == 4
        # Matrix with extreme values
        A = fill(T(1e-12), m, n)
        if min(m,n) > 0
            A[1,1] = T(1e12)
        end
        return A
    else
        # Default random
        return rand(T, m, n)
    end
end

function generate_test_vectors(::Type{T}, m, n, imat=1) where T
    if m == 0 || n == 0
        return T[], T[]
    end
    
    if imat == 1
        # Random vectors
        return rand(T, m), rand(T, n)
    elseif imat == 2
        # Vectors with known pattern
        x = T[T(i, -i/2) for i in 1:m]
        y = T[T(2*i, i/3) for i in 1:n]
        return x, y
    elseif imat == 3
        # Unit vectors
        x = zeros(T, m)
        y = zeros(T, n)
        if m > 0; x[1] = T(1, 0); end
        if n > 0; y[1] = T(0, 1); end
        return x, y
    else
        return rand(T, m), rand(T, n)
    end
end

@testset "GERC Tests" begin
    @testset "Standard Test Suite" begin
        for (itype, T) in enumerate(GERC_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) ? 1e-6 : 1e-12
                atol = rtol
                
                for (isize, (m, n)) in enumerate(GERC_SIZES)
                    @testset "Size m=$m, n=$n (isize=$isize)" begin
                        # Test multiple matrix/vector patterns
                        for imat in 1:3
                            @testset "Matrix type $imat" begin
                                A_orig = generate_test_matrix(T, m, n, imat)
                                x, y = generate_test_vectors(T, m, n, imat)
                                
                                # Test multiple alpha values
                                alphas = T[
                                    T(0, 0),                    # Zero
                                    T(1, 0),                    # Real one
                                    T(0, 1),                    # Imaginary one
                                    T(-1, 0),                   # Real minus one
                                    T(0, -1),                   # Imaginary minus one
                                    T(0.7, 0.3),               # General complex
                                    T(-0.5, 0.8),              # Another complex
                                    T(1e-6, 0),                # Small real
                                    T(0, 1e-6),                # Small imaginary
                                ]
                                
                                for (ialpha, α) in enumerate(alphas)
                                    A_ref = copy(A_orig)
                                    A_test = copy(A_orig)

                                    # Reference calculation using manual implementation
                                    # A := A + α * x * conj(y)'
                                    LinearAlgebra.BLAS.ger!(α, x, y, A_ref)
                                    # NextLA call: gerc!(alpha, x, y, A)
                                    NextLA.gerc!(α, x, y, A_test)

                                    # Comparison with LAPACK-style error checking
                                    if m == 0 || n == 0 || α == 0
                                        @test A_test == A_orig  # Should be unchanged
                                    else
                                        # Scale tolerance appropriately
                                        scaled_rtol = rtol * max(1, m, n, abs(α))
                                        scaled_atol = atol * max(1, norm(A_orig), abs(α) * norm(x) * norm(y))
                                        @test A_test ≈ A_ref rtol=scaled_rtol atol=scaled_atol
                                        
                                        # Additional LAPACK-style checks
                                        @test all(isfinite.(A_test))
                                        @test size(A_test) == size(A_ref)
                                        # Check the mathematical property: A = A_orig + α * x * conj(y)'
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            for T in (ComplexF32, ComplexF64)
                @testset "Type $T" begin
                    m, n = 50, 40
                    α = T(rand(real(T)) * 2 - 1, rand(real(T)) * 2 - 1)
                    
                    x_cpu = rand(T, m)
                    y_cpu = rand(T, n)
                    A_cpu = rand(T, m, n)
                    A_expected = copy(A_cpu)
                    
                    # Move data to GPU
                    x_gpu = CuArray(x_cpu)
                    y_gpu = CuArray(y_cpu)
                    A_gpu = CuArray(A_cpu)
                    
                    # Reference CPU calculation
                    BLAS.ger!(α, x_cpu, y_cpu, A_expected)
                    
                    # Our implementation on GPU
                    NextLA.gerc!(α, x_gpu, y_gpu, A_gpu)
                    
                    # Compare results
                    rtol = (T <: ComplexF32) ? 1e-5 : 1e-12
                    @test Array(A_gpu) ≈ A_expected rtol=rtol
                end
            end
        end
    end
end
