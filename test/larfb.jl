using Test
using NextLA
using LinearAlgebra
using Random
using CUDA

const LARFB_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
const LARFB_SIZES = [100, 200, 300, 500, 1000]

# Generate test matrices using patterns
function generate_test_matrices(::Type{T}, m, n, k, side, storev, imat=1) where T
    if m == 0 || n == 0 || k == 0
        V_size = storev == 'C' ? (side == 'L' ? m : n, k) : (k, side == 'L' ? m : n)
        return zeros(T, V_size...), zeros(T, k, k), zeros(T, m, n), zeros(T, side == 'L' ? n : m, k)
    end
    
    # Generate V matrix based on side and storev
    if storev == 'C'  # Column-wise storage
        V_rows = side == 'L' ? m : n
        V_cols = k
    else  # Row-wise storage
        V_rows = k
        V_cols = side == 'L' ? m : n
    end
    
    work_rows = side == 'L' ? n : m
    
    # Different matrix patterns
    if imat == 1
        # Random matrices
        V = randn(T, V_rows, V_cols)
        T_mat = triu(randn(T, k, k))
        C = randn(T, m, n)
        work = zeros(T, work_rows, k)
        return V, T_mat, C, work
    elseif imat == 2
        # Identity-like matrices
        V = zeros(T, V_rows, V_cols)
        min_dim = min(V_rows, V_cols)
        for i in 1:min_dim
            V[i, i] = one(T)
        end
        T_mat = Matrix{T}(I, k, k)
        C = Matrix{T}(I, m, n)
        work = zeros(T, work_rows, k)
        return V, T_mat, C, work
    elseif imat == 3
        # Matrices with alternating pattern
        V = zeros(T, V_rows, V_cols)
        min_dim = min(V_rows, V_cols)
        for i in 1:min_dim
            V[i, i] = T((-1)^i)
        end
        T_mat = Diagonal(T[T(0.5 * (-1)^i) for i in 1:k])
        C = T[T((-1)^(i+j)) for i in 1:m, j in 1:n]
        work = zeros(T, work_rows, k)
        return V, T_mat, C, work
    else
        # Default random
        V = randn(T, V_rows, V_cols)
        T_mat = triu(randn(T, k, k))
        C = randn(T, m, n)
        work = zeros(T, work_rows, k)
        return V, T_mat, C, work
    end
end

@testset "LARFB Tests" begin
    @testset "CPU Tests - Block Reflector Application" begin
        for (itype, T) in enumerate(LARFB_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: Union{ComplexF32, Float32}) ? 1e-5 : 1e-12
                atol = rtol
                for (isize, m) in enumerate(LARFB_SIZES[1:4])  # Limit size for comprehensive testing
                    for n in LARFB_SIZES[1:3]
                        @testset "Size m=$m, n=$n" begin
                            # Test different parameter combinations
                            for side in ['L', 'R']
                                for trans in ['N', 'C']
                                    for direct in ['F', 'B']
                                        for storev in ['C', 'R']
                                            @testset "side=$side, trans=$trans, direct=$direct, storev=$storev" begin
                                                # Test multiple matrix patterns
                                                for imat in 1:3
                                                    @testset "Matrix type $imat" begin
                                                        k = side == 'L' ? m : n 
                                                        V, T_mat, C_orig, work = generate_test_matrices(T, m, n, k, side, storev, imat)
                                                        C_test = copy(C_orig)
                                                        C_ref = copy(C_orig)
                                                        
                                                        # Set leading dimensions
                                                        ldv = size(V, 1)
                                                        ldt = k
                                                        ldc = m
                                                        ldwork = size(work, 1)
                                                        
                                                        # NextLA call: larfb(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)
                                                        NextLA.larfb(side, trans, direct, storev, m, n, k, V, ldv, T_mat, ldt, C_test, ldc, work, ldwork)
                                                        
                                                        # Basic checks
                                                        @test all(isfinite.(C_test))
                                                        @test size(C_test) == (m, n)
                                                        @test all(isfinite.(work))
                                                        
                                                        NextLA.larfb('L', 'N', direct, storev, m, n, k, V, ldv, T_mat, ldt, C_ref, ldc, work, ldwork)
                                                        NextLA.larfb('L', 'C', direct, storev, m, n, k, V, ldv, T_mat, ldt, C_ref, ldc, work, ldwork)

                                                        # Mathematical validation
                                                        @test norm(C_ref - C_orig) / norm(C_orig) < rtol
                                                        @test norm(C_ref - C_orig) < atol

                                                        # Special case: if k=0, C should be unchanged
                                                        if k == 0
                                                            @test C_test ≈ C_orig
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
    
    @testset "Error Handling Tests" begin
        # Test error conditions following conventions
        for T in LARFB_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                m, n, k = 500, 400, 300
                V = randn(T, m, k)
                T_mat = triu(randn(T, k, k))
                C = randn(T, m, n)
                work = zeros(T, n, k)
                
                @test_nowarn NextLA.larfb('L', 'N', 'F', 'C', m, n, k, V, m, T_mat, k, C, m, work, n)
                
                # Test edge cases
                @test_nowarn NextLA.larfb('L', 'N', 'F', 'C', 0, 0, 0, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1)  # m = n = k = 0
                @test_nowarn NextLA.larfb('L', 'N', 'F', 'C', 1, 1, 0, zeros(T, 1, 0), 1, zeros(T, 0, 0), 1, randn(T, 1, 1), 1, zeros(T, 1, 0), 1)  # k = 0
                
                # Test different side/storev combinations
                @test_nowarn NextLA.larfb('R', 'N', 'F', 'C', m, n, k, randn(T, n, k), n, T_mat, k, copy(C), m, zeros(T, m, k), m)  # Right side
                @test_nowarn NextLA.larfb('L', 'C', 'B', 'R', m, n, k, randn(T, k, m), k, T_mat, k, copy(C), m, zeros(T, n, k), n)  # Row-wise storage
            end
        end
    end
    
    @testset "Numerical Stability Tests" begin
        for T in [ComplexF64, Float64]  # High precision types
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 100
                
                # Test with very small and very large numbers
                scales = [eps(real(T)), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    m, n, k = 800, 600, 400
                    V = T.(scale .* randn(ComplexF64, m, k))
                    T_mat = triu(T.(scale .* randn(ComplexF64, k, k)))
                    C = T.(scale .* randn(ComplexF64, m, n))
                    work = zeros(T, n, k)
                    
                    # Test calculation
                    NextLA.larfb('L', 'N', 'F', 'C', m, n, k, V, m, T_mat, k, C, m, work, n)
                    
                    # Check that results are finite
                    @test all(isfinite.(C))
                    @test all(isfinite.(work))
                    
                    # Test with different parameter combinations for stability
                    for side in ['L', 'R']
                        for trans in ['N', 'C']
                            C_test = T.(scale .* randn(ComplexF64, m, n))
                            work_test = zeros(T, side == 'L' ? n : m, k)
                            V_test = side == 'L' ? V : T.(scale .* randn(ComplexF64, n, k))
                            
                            NextLA.larfb(side, trans, 'F', 'C', m, n, k, V_test, size(V_test, 1), T_mat, k, C_test, m, work_test, size(work_test, 1))
                            
                            @test all(isfinite.(C_test))
                            @test all(isfinite.(work_test))
                        end
                    end
                end
            end
        end
    end

    @testset "GPU Tests (CUDA and ROCm)" begin
        # Test CUDA support
        if CUDA.functional()
            @testset "CUDA Tests" begin
                for T in (ComplexF32, Float32) # Common GPU types
                    @testset "Type $T CUDA" begin
                        rtol = (T <: ComplexF32) ? 1e-5 : 1e-6
                        
                        for (m, n, k) in [(3, 3, 2), (5, 4, 3), (8, 6, 4)]
                            @testset "Size m=$m, n=$n, k=$k" begin
                                for side in ['L', 'R']
                                    for trans in ['N', 'C']
                                        @testset "side=$side, trans=$trans" begin
                                            # Generate CPU data
                                            V_cpu = randn(T, side == 'L' ? m : n, k)
                                            T_cpu = triu(randn(T, k, k))
                                            C_cpu = randn(T, m, n)
                                            work_cpu = zeros(T, side == 'L' ? n : m, k)
                                            
                                            # Move data to GPU
                                            V_gpu = CuArray(V_cpu)
                                            T_gpu = CuArray(T_cpu)
                                            C_gpu = CuArray(C_cpu)
                                            work_gpu = CuArray(work_cpu)
                                            
                                            # Reference CPU calculation
                                            C_ref = copy(C_cpu)
                                            work_ref = copy(work_cpu)
                                            NextLA.larfb(side, trans, 'F', 'C', m, n, k, V_cpu, size(V_cpu, 1), T_cpu, k, C_ref, m, work_ref, size(work_ref, 1))
                                            
                                            # GPU calculation
                                            NextLA.larfb(side, trans, 'F', 'C', m, n, k, V_gpu, size(V_gpu, 1), T_gpu, k, C_gpu, m, work_gpu, size(work_gpu, 1))
                                            
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
            end
        end
        
        # Test ROCm support (if available)
        if isdefined(Main, :AMDGPU) && Main.AMDGPU.functional()
            @testset "ROCm Tests" begin
                for T in (ComplexF32, Float32) # Common GPU types
                    @testset "Type $T ROCm" begin
                        rtol = (T <: ComplexF32) ? 1e-5 : 1e-6
                        
                        for (m, n, k) in [(3, 3, 2), (5, 4, 3)]
                            @testset "Size m=$m, n=$n, k=$k" begin
                                for side in ['L', 'R']
                                    @testset "side=$side" begin
                                        # Generate CPU data
                                        V_cpu = randn(T, side == 'L' ? m : n, k)
                                        T_cpu = triu(randn(T, k, k))
                                        C_cpu = randn(T, m, n)
                                        work_cpu = zeros(T, side == 'L' ? n : m, k)
                                        
                                        # Move data to ROCm GPU
                                        V_rocm = Main.AMDGPU.ROCArray(V_cpu)
                                        T_rocm = Main.AMDGPU.ROCArray(T_cpu)
                                        C_rocm = Main.AMDGPU.ROCArray(C_cpu)
                                        work_rocm = Main.AMDGPU.ROCArray(work_cpu)
                                        
                                        # Reference CPU calculation
                                        C_ref = copy(C_cpu)
                                        work_ref = copy(work_cpu)
                                        NextLA.larfb(side, 'N', 'F', 'C', m, n, k, V_cpu, size(V_cpu, 1), T_cpu, k, C_ref, m, work_ref, size(work_ref, 1))
                                        
                                        # ROCm calculation
                                        NextLA.larfb(side, 'N', 'F', 'C', m, n, k, V_rocm, size(V_rocm, 1), T_rocm, k, C_rocm, m, work_rocm, size(work_rocm, 1))
                                        
                                        # Compare results
                                        @test norm(Array(C_rocm) - C_ref) < rtol * max(1, norm(C_ref))
                                        @test norm(Array(work_rocm) - work_ref) < rtol * max(1, norm(work_ref))
                                        
                                        @test all(isfinite.(Array(C_rocm)))
                                        @test all(isfinite.(Array(work_rocm)))
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
