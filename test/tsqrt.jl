using Test
using NextLA
using LinearAlgebra
using Random
using CUDA
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc

# LAPACK TPQRT wrapper for reference testing
function lapack_tpqrt!(::Type{T}, m::Int64, n::Int64, l::Int64, nb::Int64, 
    A::AbstractMatrix{T}, lda::Int64, B::AbstractMatrix{T}, ldb::Int64, 
    Tau::AbstractMatrix{T}, ldt::Int64, work) where {T<:Number}
    
    info = Ref{BlasInt}(0)
    
    if T == ComplexF64
        ccall((@blasfunc(ztpqrt_), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{BlasInt}),
            m, n, l, nb, A, lda, B, ldb, Tau, ldt, work, info)
    elseif T == Float64
        ccall((@blasfunc(dtpqrt_), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{BlasInt}),
            m, n, l, nb, A, lda, B, ldb, Tau, ldt, work, info)
    elseif T == ComplexF32
        ccall((@blasfunc(ctpqrt_), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{BlasInt}),
            m, n, l, nb, A, lda, B, ldb, Tau, ldt, work, info)
    else # T = Float32
        ccall((@blasfunc(stpqrt_), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
             Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{BlasInt}),
            m, n, l, nb, A, lda, B, ldb, Tau, ldt, work, info)
    end
    
    chklapackerror(info[])
end

# TSQRT test parameters for NextLA.tsqrt
const TSQRT_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
const TSQRT_SIZES = [
    (100, 80, 30),    # m, n, ib
    (150, 120, 40),   
    (200, 160, 50),
    (250, 200, 60),
    (300, 240, 80)
]

@testset "TSQRT Tests" begin
    @testset "NextLA vs LAPACK comparison" begin
        for (itype, T) in enumerate(TSQRT_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                for (isize, (m, n, ib)) in enumerate(TSQRT_SIZES)
                    @testset "Size m=$m, n=$n, ib=$ib" begin
                        # Generate test matrices
                        A1 = triu(rand(T, n, n))
                        A2 = rand(T, m, n)
                        
                        # Make A1 well-conditioned
                        
                        # Make copies for different implementations
                        A1_nextla = copy(A1)
                        A2_nextla = copy(A2)
                        A1_lapack = copy(A1)
                        A2_lapack = copy(A2)
                        
                        # Prepare workspace and output arrays
                        lda1 = n
                        lda2 = m
                        ldt = ib
                        
                        T_nextla = zeros(T, ib, n)
                        T_lapack = zeros(T, ib, n)
                        tau_nextla = zeros(T, n)
                        tau_lapack = zeros(T, n)
                        work_nextla = zeros(T, ib * n)
                        
                        # Test NextLA implementation
                        NextLA.tsqrt(m, n, ib, A1_nextla, lda1, A2_nextla, lda2, T_nextla, ldt, tau_nextla, work_nextla)
                        
                        # Test LAPACK implementation 
                        work_lapack = zeros(T, ib * n)
                        lapack_tpqrt!(T, m, n, 0, ib, A1_lapack, n, A2_lapack, m, T_lapack, ib, work_lapack)
                        
                        # Compare results
                        @test A1_nextla ≈ A1_lapack rtol=rtol
                        @test A2_nextla ≈ A2_lapack rtol=rtol
                        
                        # Basic sanity checks
                        @test all(isfinite.(A1_nextla))
                        @test all(isfinite.(A2_nextla))
                        @test all(isfinite.(T_nextla))
                        @test all(isfinite.(tau_nextla))
                        
                        
                        # Check that T has the expected block structure
                        @test size(T_nextla) == (ib, n)
                        for block_start in 1:ib:n
                            block_end = min(block_start + ib - 1, n)
                            for i in 1:(block_end - block_start + 1)
                                for j in 1:(i-1)
                                    if block_start + i - 1 <= n && block_start + j - 1 <= n
                                        @test abs(T_nextla[i, block_start + j - 1]) < rtol * 100
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    @testset "QR Property Verification" begin
        for T in TSQRT_TYPES
            @testset "Type $T QR Properties" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                m, n, ib = 200, 150, 40
                
                # Generate well-conditioned test matrices
                A1 = triu(rand(T, n, n))
                A2 = rand(T, m, n)
                
                # Make A1 well-conditioned
                for i in 1:n
                    A1[i, i] += 2 * one(T)
                end
                
                # Store original combined matrix
                combined_original = [A1; A2]
                
                # Apply TSQRT
                A1_result = copy(A1)
                A2_result = copy(A2)
                T_result = zeros(T, ib, n)
                tau_result = zeros(T, n)
                work = zeros(T, ib * n)
                
                NextLA.tsqrt(m, n, ib, A1_result, n, A2_result, m, T_result, ib, tau_result, work)
                
                # Check that A1 (now R) is upper triangular
                for i in 1:n
                    for j in 1:i-1
                        @test abs(A1_result[i, j]) < rtol * 100
                    end
                end
                
                # Check that the factorization is numerically stable
                @test all(isfinite.(A1_result))
                @test all(isfinite.(A2_result))
                @test all(isfinite.(T_result))
                @test all(isfinite.(tau_result))
                
                # Compare with standard QR factorization
                Q_ref, R_ref = qr(combined_original)
                R_ref_mat = Matrix(R_ref)
                
                # Compare the R factors (allowing for sign differences)
                R_our = A1_result
                for j in 1:n
                    if abs(R_ref_mat[j, j]) > rtol * 10 && abs(R_our[j, j]) > rtol * 10
                        # Check that the magnitudes are similar
                        @test abs(abs(R_ref_mat[j, j]) - abs(R_our[j, j])) < rtol * max(abs(R_ref_mat[j, j]), abs(R_our[j, j])) * 100
                    end
                end
            end
        end
    end
    
    @testset "Error Handling Tests" begin
        for T in TSQRT_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                m, n, ib = 100, 80, 30
                A1 = triu(randn(T, n, n))
                A2 = randn(T, m, n)
                T_mat = zeros(T, ib, n)
                tau = zeros(T, n)
                work = zeros(T, ib * n)
                
                @test_nowarn NextLA.tsqrt(m, n, ib, A1, n, A2, m, T_mat, ib, tau, work)
                
                # Test with invalid parameters
                @test_throws ArgumentError NextLA.tsqrt(-1, n, ib, A1, n, A2, m, T_mat, ib, tau, work)
                @test_throws ArgumentError NextLA.tsqrt(m, -1, ib, A1, n, A2, m, T_mat, ib, tau, work)
                @test_throws ArgumentError NextLA.tsqrt(m, n, -1, A1, n, A2, m, T_mat, ib, tau, work)
                
                # Test edge cases
                @test_nowarn NextLA.tsqrt(0, 0, 0, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, T[], T[])
            end
        end
    end
    
    @testset "Numerical Stability Tests" begin
        for T in [ComplexF64, Float64]  # High precision types
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 1000
                
                # Test with different conditioning
                scales = [eps(real(T))^(1/4), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    m, n, ib = 150, 120, 40
                    A1 = triu(T(scale) .* randn(T, n, n))
                    A2 = T(scale) .* randn(T, m, n)

                    # Make A1 well-conditioned
                    for i in 1:n
                        A1[i, i] += T(scale)
                    end
                    
                    T_mat = zeros(T, ib, n)
                    tau = zeros(T, n)
                    work = zeros(T, ib * n)
                    
                    # Test calculation
                    NextLA.tsqrt(m, n, ib, A1, n, A2, m, T_mat, ib, tau, work)
                    
                    # Check that results are finite
                    @test all(isfinite.(A1))
                    @test all(isfinite.(A2))
                    @test all(isfinite.(T_mat))
                    @test all(isfinite.(tau))
                end
            end
        end
    end
    
    @testset "Different Block Sizes" begin
        for T in TSQRT_TYPES
            @testset "Type $T Block Sizes" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                m, n = 200, 160
                block_sizes = [10, 20, 30, 40, 50, 80]
                
                for ib in block_sizes
                    A1 = triu(rand(T, n, n))
                    A2 = rand(T, m, n)
                    
                    # Make A1 well-conditioned
                    for i in 1:n
                        A1[i, i] += one(T)
                    end
                    
                    T_mat = zeros(T, ib, n)
                    tau = zeros(T, n)
                    work = zeros(T, ib * n)
                    
                    NextLA.tsqrt(m, n, ib, A1, n, A2, m, T_mat, ib, tau, work)
                    
                    # Should complete without errors
                    @test all(isfinite.(A1))
                    @test all(isfinite.(A2))
                    @test all(isfinite.(T_mat))
                    @test all(isfinite.(tau))
                    
                    # A1 should remain upper triangular
                    for i in 1:n
                        for j in 1:i-1
                            @test abs(A1[i, j]) < rtol * 100
                        end
                    end
                end
            end
        end
    end
    
    @testset "Edge Cases" begin
        for T in TSQRT_TYPES
            @testset "Type $T Edge Cases" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                # Single column
                m, n, ib = 50, 10, 10
                A1 = triu(rand(T, n, n))
                A2 = rand(T, m, n)
                A1[1, 1] += one(T)  # Ensure well-conditioned
                
                T_mat = zeros(T, ib, n)
                tau = zeros(T, n)
                work = zeros(T, ib * n)
                
                NextLA.tsqrt(m, n, ib, A1, n, A2, m, T_mat, ib, tau, work)
                
                @test all(isfinite.(A1))
                @test all(isfinite.(A2))
                @test all(isfinite.(T_mat))
                @test all(isfinite.(tau))
                
                # Minimal size
                m, n, ib = 20, 20, 10
                A1 = triu(rand(T, n, n))
                A2 = rand(T, m, n)
                A1 += I  # Make well-conditioned
                
                T_mat = zeros(T, ib, n)
                tau = zeros(T, n)
                work = zeros(T, ib * n)
                
                NextLA.tsqrt(m, n, ib, A1, n, A2, m, T_mat, ib, tau, work)
                
                @test all(isfinite.(A1))
                @test all(isfinite.(A2))
                @test all(isfinite.(T_mat))
                @test all(isfinite.(tau))
            end
        end
    end
    
    @testset "GPU Tests" begin
        if CUDA.functional()
            for T in [ComplexF32, Float32]
                @testset "Type $T GPU" begin
                    rtol = 1e-5
                    
                    m, n, ib = 120, 100, 30
                    
                    # Create CPU data
                    A1_cpu = triu(rand(T, n, n))
                    A2_cpu = rand(T, m, n)
                    A1_cpu += I  # Make well-conditioned
                    
                    T_cpu = zeros(T, ib, n)
                    tau_cpu = zeros(T, n)
                    work_cpu = zeros(T, ib * n)
                    
                    # Create GPU data
                    A1_gpu = CuArray(A1_cpu)
                    A2_gpu = CuArray(A2_cpu)
                    T_gpu = CuArray(T_cpu)
                    tau_gpu = CuArray(tau_cpu)
                    work_gpu = CuArray(work_cpu)
                    
                    # Apply on CPU
                    A1_cpu_result = copy(A1_cpu)
                    A2_cpu_result = copy(A2_cpu)
                    T_cpu_result = copy(T_cpu)
                    tau_cpu_result = copy(tau_cpu)
                    NextLA.tsqrt(m, n, ib, A1_cpu_result, n, A2_cpu_result, m, T_cpu_result, ib, tau_cpu_result, work_cpu)
                    
                    # Apply on GPU
                    NextLA.tsqrt(m, n, ib, A1_gpu, n, A2_gpu, m, T_gpu, ib, tau_gpu, work_gpu)
                    
                    # Compare results
                    @test Array(A1_gpu) ≈ A1_cpu_result rtol=rtol
                    @test Array(A2_gpu) ≈ A2_cpu_result rtol=rtol
                    @test Array(T_gpu) ≈ T_cpu_result rtol=rtol
                    @test Array(tau_gpu) ≈ tau_cpu_result rtol=rtol
                    
                    @test all(isfinite.(Array(A1_gpu)))
                    @test all(isfinite.(Array(A2_gpu)))
                    @test all(isfinite.(Array(T_gpu)))
                    @test all(isfinite.(Array(tau_gpu)))
                end
            end
        end
    end
end
