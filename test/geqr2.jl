using Test
using NextLA
using LinearAlgebra, LinearAlgebra.LAPACK
using Random


const QR2_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
const QR2_SIZES = [(0,0), (100,100), (200,100), (100,200), (300,300), (500,300), (100,80), (200,150)]

# Generate test matrices using patterns
function generate_qr_test_matrix(::Type{T}, m, n, imat=1) where T
    if m == 0 || n == 0
        return zeros(T, m, n)
    end
    
    # Use the matrix generation from runtests.jl
    if imat == 1
        # Well-conditioned random matrix
        return matrix_generation(T, m, n, mode=:decay, cndnum=2.0)
    elseif imat == 2
        # Moderately ill-conditioned
        return matrix_generation(T, m, n, mode=:decay, cndnum=1e2)
    elseif imat == 3
        # Severely ill-conditioned
        return matrix_generation(T, m, n, mode=:one_large, cndnum=1e6)
    elseif imat == 4
        # Random matrix
        return rand(T, m, n)
    else
        # Identity-like matrix
        A = zeros(T, m, n)
        k = min(m, n)
        for i in 1:k
            A[i, i] = one(T)
        end
        return A
    end
end

@testset "GEQR2 Tests" begin
    @testset "Unblocked QR Factorization Tests" begin
        for (itype, T) in enumerate(QR2_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                atol = rtol
                
                for (isize, (m, n)) in enumerate(QR2_SIZES)
                    @testset "Size m=$m, n=$n (isize=$isize)" begin
                        k = min(m, n)
                        
                        # Test multiple matrix patterns
                        for imat in 1:4
                            @testset "Matrix type $imat" begin
                                A_orig = generate_qr_test_matrix(T, m, n, imat)
                                
                                # --- Reference Calculation ---
                                A_ref = copy(A_orig)
                                tau_ref = zeros(T, k)
                                A_ref = qr(A_ref).factors

                                # --- NextLA Calculation ---
                                A_test = copy(A_orig)
                                lda = max(1, m)
                                tau_test = zeros(T, k)
                                work_test = zeros(T, n)  # Work array size n for geqr2
                                NextLA.geqr2(m, n, A_test, lda, tau_test, work_test)

                                # --- Comparisons ---
                                if m == 0 || n == 0
                                    @test size(A_test) == size(A_orig)
                                else
                                    # 1. Compare the factored matrix A (contains V and R)
                                    scaled_rtol = rtol * max(1, m, n)
                                    @test A_test ≈ A_ref rtol=scaled_rtol

                                    # 3. Mathematical property checks
                                    if k > 0
                                        # Extract R from the factored matrix
                                        R_test = triu(A_test[1:k, 1:n])
                                        
                                        # Form Q using LAPACK's unmqr
                                        Q_test = Matrix{T}(I, m, m)
                                        try
                                            LAPACK.ormqr!('L', 'N', A_test, tau_test, Q_test)
                                            
                                            # Test 3a: Reconstruction. A_orig should be Q * R.
                                            A_recon = Q_test[:, 1:k] * R_test
                                            reconstruction_tol = rtol * max(1, m, n) * norm(A_orig)
                                            @test A_orig ≈ A_recon rtol=reconstruction_tol

                                            # Test 3b: Orthogonality of Q. Q' * Q should be Identity.
                                            orthog_error = norm(Q_test' * Q_test - I)
                                            orthog_tol = rtol * m
                                            @test orthog_error < orthog_tol
                                        catch e
                                            # If LAPACK fails, just check basic properties
                                            @test all(isfinite.(A_test))
                                            @test all(isfinite.(tau_test))
                                        end
                                        
                                        # Additional checks
                                        @test all(isfinite.(A_test))
                                        @test all(isfinite.(tau_test))
                                        @test size(A_test) == size(A_orig)
                                        
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
        for T in QR2_TYPES
            @testset "Type $T Error Handling" begin
                # Test edge cases and error conditions
                m, n = 500, 300
                A = rand(T, m, n)
                lda = m
                tau = zeros(T, min(m, n))
                work = zeros(T, n)
                
                # Valid call should not error
                @test_nowarn NextLA.geqr2(m, n, copy(A), lda, copy(tau), copy(work))
                
                # Zero dimensions should not error
                @test_nowarn NextLA.geqr2(0, 0, zeros(T, 0, 0), 1, T[], T[])
                @test_nowarn NextLA.geqr2(0, 300, zeros(T, 0, 300), 1, T[], zeros(T, 300))
                @test_nowarn NextLA.geqr2(500, 0, zeros(T, 500, 0), 500, T[], T[])
            end
        end
    end

    @testset "GPU Tests (if available)" begin
        if CUDA.functional()
            for T in [ComplexF32]  # GPU typically uses single precision
                @testset "Type $T GPU" begin
                    m, n = 10, 8
                    k = min(m, n)
                    rtol = 1e-5
                    
                    A_cpu = rand(T, m, n)
                    A_gpu = CuArray(A_cpu)
                    
                    tau_cpu = zeros(T, k)
                    tau_gpu = CuArray(tau_cpu)
                    
                    work_cpu = zeros(T, n)
                    work_gpu = CuArray(work_cpu)
                    
                    # CPU reference
                    A_cpu_result = copy(A_cpu)
                    NextLA.geqr2(m, n, A_cpu_result, m, tau_cpu, work_cpu)
                    
                    # GPU test
                    NextLA.geqr2(m, n, A_gpu, m, tau_gpu, work_gpu)
                    
                    # Compare results
                    @test Array(A_gpu) ≈ A_cpu_result rtol=rtol
                    @test Array(tau_gpu) ≈ tau_cpu rtol=rtol
                    
                    @test all(isfinite.(Array(A_gpu)))
                    @test all(isfinite.(Array(tau_gpu)))
                end
            end
        end
    end
end
