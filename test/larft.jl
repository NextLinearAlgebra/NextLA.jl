using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

# Test matrix types (complex types for larft)
const LARFT_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
const LARFT_SIZES = [0, 100, 200, 300, 500, 1000]

function generate_test_matrix(::Type{T}, m::Int, k::Int; imat::Int=1, storev::Char='C') where T
    if m == 0 || k == 0
        if storev == 'C'
            return zeros(T, m, k), zeros(T, k)
        else
            return zeros(T, k, m), zeros(T, k)
        end
    end

    ldv = storev == 'C' ? m : k

    sdv = storev == 'C' ? k : m

    if imat == 1
        # Random matrix with orthogonal columns
        A = randn(T, m, k)
        Q, _ = qr(A)
        V = Matrix(Q[:, 1:sdv])
        tau = randn(T, k)
        return V, tau

    elseif imat == 2
        # Identity-like matrix
        V = zeros(T, ldv, sdv)
        for i in 1:min(m, k)
            V[i, i] = one(T)
        end
        tau = ones(T, k)
        return V, tau

    elseif imat == 3
        # Alternating pattern
        V = zeros(T, ldv, sdv)
        for i in 1:min(m, k)
            V[i, i] = T((-1)^i)
        end
        tau = [T(0.5 * (-1)^i) for i in 1:k]
        return V, tau

    elseif imat == 4
        # Extreme values
        V = randn(T, ldv, sdv) * T(1e-3)
        tau = randn(T, k) * T(1e3)
        return V, tau

    else
        # Default random
        V = randn(T, ldv, sdv)
        tau = randn(T, k)
        return V, tau
    end
end

@testset "LARFT Tests" begin
    @testset "Standard Test Suite" begin
        for (itype, T) in enumerate(LARFT_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                atol = rtol
                
                for (isize, n) in enumerate(LARFT_SIZES)
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
                                                    V, tau = generate_test_matrix(T, n, k, imat=imat, storev=storev)
                                                    ldv = size(V, 1)
                                                    
                                                    # Allocate T matrix
                                                    T_mat = zeros(T, k, k)
                                                    ldt = k
                                                    
                                                    # NextLA call: larft(direct, storev, n, k, v, ldv, tau, t, ldt)
                                                    NextLA.larft(direct, storev, n, k, V, ldv, tau, T_mat, ldt)
                                                    
                                                    # Basic checks
                                                    @test all(isfinite.(T_mat))
                                                    @test size(T_mat) == (k, k)
                                                    if direct == 'F'
                                                        @test T_mat == UpperTriangular(T_mat)
                                                    else
                                                        @test T_mat == LowerTriangular(T_mat)
                                                    end

                                                    if storev == 'C'
                                                        H_compact = I - V * T_mat * adjoint(V)
                                                    else
                                                        H_compact = I - adjoint(V) * T_mat * V
                                                    end

                                                    # === Build H_true = H1 H2 ... Hk or Hk ... H1
                                                    H_true = Matrix(I, n, n)

                                                    js = storev == 'C' ? (1:size(V,2)) : (1:size(V,1))

                                                    for j in js
                                                        # Build Householder vector vj based on storage
                                                        if storev == 'C'
                                                            vj = zeros(T, n)
                                                            if direct == 'F'
                                                                vj[j:end] .= V[j:end, j]  # vj has zeros before j
                                                            else
                                                                vj[j:end] .= V[1:k+j-1, j]  # from row vector to column
                                                            end     # vj has zeros before j
                                                        else
                                                            vj = zeros(T, n)
                                                            if direct == 'F'
                                                                vj[j:end] .= V[j, j:end]  # vj has zeros before j
                                                            else
                                                                vj[j:end] .= V[j, 1:k+j-1]  # from row vector to column
                                                            end     # vj has zeros before j
                                                        end
                                                        Hj = I - tau[j] * (vj * adjoint(vj))
                                                        H_true = Hj * H_true
                                                    end

                                                    if norm(H_true) > 0
                                                        err = norm(H_compact - H_true) / norm(H_true)
                                                    else
                                                        err = norm(H_compact - H_true)
                                                    end
                                                    @test err < rtol
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
    
    @testset "Error Handling Tests" begin
        # Test error conditions 
        for T in LARFT_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                n, k = 500, 300
                V = randn(T, n, k)
                tau = randn(T, k)
                T_mat = zeros(T, k, k)
                
                @test_nowarn NextLA.larft('F', 'C', n, k, V, n, tau, T_mat, k)
                
                # Test edge cases
                @test_nowarn NextLA.larft('F', 'C', 0, 0, zeros(T, 0, 0), 1, T[], zeros(T, 0, 0), 1)  # n = 0, k = 0
                @test_nowarn NextLA.larft('F', 'C', 1, 0, zeros(T, 1, 0), 1, T[], zeros(T, 0, 0), 1)  # k = 0
                @test_nowarn NextLA.larft('F', 'C', 0, 1, zeros(T, 0, 1), 1, T[T(0)], zeros(T, 1, 1), 1)  # n = 0
            end
        end
    end
    
    @testset "Numerical Stability Tests" begin
        for T in [ComplexF64]  # High precision type
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 100
                
                # Test with very small and very large numbers
                scales = [eps(real(T)), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    n, k = 1000, 500
                    V = T.(scale .* randn(ComplexF64, n, k))
                    tau = T.(scale .* randn(ComplexF64, k))
                    T_mat = zeros(T, k, k)
                    
                    # Test calculation
                    NextLA.larft('F', 'C', n, k, V, n, tau, T_mat, k)
                    
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

    @testset "GPU Tests" begin
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
                        NextLA.larft('F', 'C', n, k, V_cpu, n, tau_cpu, T_ref, k)
                        
                        # Our implementation on GPU
                        NextLA.larft('F', 'C', n, k, V_gpu, n, tau_gpu, T_gpu, k)
                        
                        # Compare results
                        @test norm(Array(T_gpu) - T_ref) < rtol * max(1, norm(T_ref))
                        @test all(isfinite.(Array(T_gpu)))
                    end
                end
            end
        end
    end
end

