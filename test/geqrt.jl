using Test
using NextLA
using LinearAlgebra, LinearAlgebra.LAPACK
using Random

# Function signature: geqrt!(m, n, ib, A, lda, T, ldt, tau, work)
const GEQRT_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
const GEQRT_SIZES = [(0,0), (100,100), (200,100), (100,200), (400,300), (800,600), (150,100), (200,150)]
const GEQRT_BLOCKSIZES = [100, 200, 400, 800]

@testset "GEQRT Tests" begin
    @testset "Blocked QR Factorization Tests" begin
        for (itype, T) in enumerate(GEQRT_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                atol = rtol
                for (isize, (m, n)) in enumerate(GEQRT_SIZES)
                    @testset "Size m=$m, n=$n (isize=$isize)" begin
                        k = min(m, n)
                        
                        for ib in GEQRT_BLOCKSIZES
                            if ib > k && k > 0
                                continue  # Skip if block size larger than matrix
                            end
                            
                            @testset "Block size ib=$ib" begin
                                # Test multiple matrix patterns
                                for imat in 1:3
                                    @testset "Matrix type $imat" begin
                                        # Generate test matrix
                                        if imat == 1
                                            A_orig = rand(T, m, n)
                                        elseif imat == 2
                                            A_orig = matrix_generation(T, m, n, mode=:decay, cndnum=1e3)
                                        else
                                            A_orig = zeros(T, m, n)
                                            for i in 1:min(m,n)
                                                A_orig[i,i] = T(i)
                                            end
                                        end
                                        
                                        # --- Reference using unblocked QR ---
                                        A_ref = copy(A_orig)
                                        A_ref = qr(A_ref).factors

                                        # --- NextLA Blocked QR ---
                                        A_test = copy(A_orig)
                                        T_test = zeros(T, max(1,ib), k)  # Block reflector matrix
                                        tau_test = zeros(T, k)
                                        work_test = zeros(T, ib * n)  # Work array
                                        
                                        NextLA.geqrt!(m, n, ib, A_test, T_test, tau_test, work_test)

                                        # --- Test Helper Function ---
                                        A_helper = copy(A_orig)
                                        T_helper = zeros(T, max(1, ib), k)
                                        tau_helper = zeros(T, k)
                                        NextLA.geqrt!(A_helper, T_helper, tau_helper)
                                        
                                        # Verify helper gives same results as kernel (in-place)
                                        if k > 0
                                            @test A_helper ≈ A_test rtol=rtol atol=atol
                                            @test T_helper[1:ib, 1:k] ≈ T_test[1:ib, 1:k] rtol=rtol atol=atol
                                            @test tau_helper ≈ tau_test rtol=rtol atol=atol
                                        end

                                        # --- Comparisons ---
                                        if m == 0 || n == 0
                                            @test size(A_test) == size(A_orig)
                                        else
                                            # Mathematical property checks
                                            if k > 0
                                                # Check basic properties
                                                @test all(isfinite.(A_test))
                                                @test all(isfinite.(T_test))
                                                @test all(isfinite.(tau_test))
                                                @test size(A_test) == size(A_orig)
                                                
                                                
                                                # For small matrices, verify reconstruction
                                                # Extract R from the factored matrix
                                                R_test = triu(A_test[1:k, 1:n])
                                                
                                                # Form Q using LAPACK's unmqr!
                                                Q_test = Matrix{T}(I, m, m)
                                                LAPACK.ormqr!('L', 'N', A_test, tau_test, Q_test)
                                                
                                                # Test 3a: Reconstruction. A_orig should be Q * R.
                                                A_recon = Q_test[:, 1:k] * R_test
                                                reconstruction_tol = rtol * max(1, m, n) * norm(A_orig)
                                                @test A_orig ≈ A_recon rtol=reconstruction_tol
    
                                                # Test 3b: Orthogonality of Q. Q' * Q should be Identity.
                                                orthog_error = norm(adjoint(Q_test) * Q_test - I)
                                                orthog_tol = rtol * m
                                                @test orthog_error < orthog_tol
                                            
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
        end
    end
    
    @testset "Edge Cases" begin
        # Test with ib = 1 (should behave like unblocked QR)
        m, n, ib = 10, 8, 1
        A = rand(ComplexF64, m, n)
        A_original = copy(A)
        T = zeros(ComplexF64, ib, min(m, n))
        tau = zeros(ComplexF64, min(m, n))
        work = zeros(ComplexF64, ib * n)
        
        NextLA.geqrt!(m, n, ib, A, T, tau, work)
        
        # Compare with unblocked version
        A_unblocked = copy(A_original)
        tau_unblocked = zeros(ComplexF64, min(m, n))
        work_unblocked = zeros(ComplexF64, n)
        NextLA.geqr2!(m, n, A_unblocked, tau_unblocked, work_unblocked)
        
        @test A ≈ A_unblocked rtol=1e-10
        
        # Test with very small matrices
        m, n, ib = 3, 2, 1
        A = rand(ComplexF64, m, n)
        T = zeros(ComplexF64, ib, min(m, n))
        tau = zeros(ComplexF64, min(m, n))
        work = zeros(ComplexF64, ib * n)
        
        NextLA.geqrt!(m, n, ib, A, T, tau, work)
        
        # Should not crash
        @test all(isfinite.(A))
        @test all(isfinite.(T))
        @test all(isfinite.(tau))
    end
    
    @testset "Error Handling" begin
        # Test negative dimensions
        @test_throws ArgumentError NextLA.geqrt!(-1, 5, 2, zeros(ComplexF64, 5, 5), zeros(ComplexF64, 2, 5), zeros(ComplexF64, 5), zeros(ComplexF64, 10))
        @test_throws ArgumentError NextLA.geqrt!(5, -1, 2, zeros(ComplexF64, 5, 5), zeros(ComplexF64, 2, 5), zeros(ComplexF64, 5), zeros(ComplexF64, 10))
        
        # Test invalid block size
        @test_throws ArgumentError NextLA.geqrt!(5, 5, -1, zeros(ComplexF64, 5, 5), zeros(ComplexF64, 2, 5), zeros(ComplexF64, 5), zeros(ComplexF64, 10))
        @test_throws ArgumentError NextLA.geqrt!(5, 5, 0, zeros(ComplexF64, 5, 5), zeros(ComplexF64, 2, 5), zeros(ComplexF64, 5), zeros(ComplexF64, 10))
    end
    
    @testset "Consistency Tests" begin
        # Test that multiple applications give same result
        m, n, ib = 20, 15, 4
        A = rand(ComplexF64, m, n)
        
        # First application
        A1 = copy(A)
        T1 = zeros(ComplexF64, ib, min(m, n))
        tau1 = zeros(ComplexF64, min(m, n))
        work1 = zeros(ComplexF64, ib * n)
        NextLA.geqrt!(m, n, ib, A1, T1, tau1, work1)
        
        # Second application
        A2 = copy(A)
        T2 = zeros(ComplexF64, ib, min(m, n))
        tau2 = zeros(ComplexF64, min(m, n))
        work2 = zeros(ComplexF64, ib * n)
        NextLA.geqrt!(m, n, ib, A2, T2, tau2, work2)
        
        @test A1 ≈ A2 rtol=1e-12
        @test T1 ≈ T2 rtol=1e-12
        @test tau1 ≈ tau2 rtol=1e-12
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            m, n, ib = 16, 12, 4
            
            # Create CPU data
            A_cpu = rand(ComplexF32, m, n)
            T_cpu = zeros(ComplexF32, ib, min(m, n))
            tau_cpu = zeros(ComplexF32, min(m, n))
            work_cpu = zeros(ComplexF32, ib * n)
            
            # Create GPU data
            A_gpu = CuArray(A_cpu)
            T_gpu = CuArray(T_cpu)
            tau_gpu = CuArray(tau_cpu)
            work_gpu = CuArray(work_cpu)
            
            # Apply on CPU
            A_cpu_result = copy(A_cpu)
            T_cpu_result = copy(T_cpu)
            tau_cpu_result = copy(tau_cpu)
            NextLA.geqrt!(m, n, ib, A_cpu_result, T_cpu_result, tau_cpu_result, work_cpu)
            
            # Apply on GPU
            NextLA.geqrt!(m, n, ib, A_gpu, T_gpu, tau_gpu, work_gpu)
            
            @test Array(A_gpu) ≈ A_cpu_result rtol=1e-6
            @test Array(T_gpu) ≈ T_cpu_result rtol=1e-6
            @test Array(tau_gpu) ≈ tau_cpu_result rtol=1e-6
        end
    end
end
