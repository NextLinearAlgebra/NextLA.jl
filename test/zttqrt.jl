using Test
using NextLA
using LinearAlgebra
using Random
using CUDA
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc

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

const ZTTQRT_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
# Format: (m, n, ib) where:
# - m: number of rows in A2 matrix (M >= 0)
# - n: number of columns in A1 and A2 (N >= 0)
# - ib: block size (IB >= 1)
const ZTTQRT_SIZES = [
    (8, 6, 2),   # m=8, n=6, ib=2
    (10, 8, 3),  # m=10, n=8, ib=3
    (12, 10, 4), # m=12, n=10, ib=4
    (6, 6, 2),   # m=6, n=6, ib=2 (square case)
    (15, 12, 3), # m=15, n=12, ib=3
    (20, 16, 4)  # m=20, n=16, ib=4
]

@testset "ZTTQRT Tests" begin
    @testset "NextLA vs LAPACK comparison" begin
        for (itype, T) in enumerate(ZTTQRT_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                for (isize, (m, n, ib)) in enumerate(ZTTQRT_SIZES)
                    @testset "Size m=$m, n=$n, ib=$ib" begin
                        # Test parameter validation
                        if ib > n || ib <= 0
                            continue  # Skip invalid block sizes
                        end

                        A1 = triu(randn(T, n, n))
                        A2 = triu(randn(T, n, n))
                    
                        T_mat = triu(rand(T, ib, n))

                        tau = zeros(T, n)

                        work = zeros(T, n * ib)

                        A1_nextla = copy(A1)
                        A2_nextla = copy(A2)
                        A1_orig = copy(A1)
                        A2_orig = copy(A2)
                        T_mat_nextla = copy(T_mat)
                        work_nextla = copy(work)


                        NextLA.zttqrt(n, n, ib, A1_nextla, n, A2_nextla, m, T_mat_nextla, ib, tau, work_nextla)

                        lapack_tpqrt!(T, n, n, n, ib, A1, n, A2, n, T_mat, ib, work)

                        @test norm(A1_nextla - A1) < rtol * max(1, norm(A1))
                        @test norm(A2_nextla - A2) < rtol * max(1, norm(A2))
                            
                        # Basic sanity checks
                        @test all(isfinite.(A1_nextla))
                        @test all(isfinite.(A2_nextla))
                        @test all(isfinite.(T_mat_nextla))
                        @test all(isfinite.(tau))
                        @test all(isfinite.(work_nextla))
                        
                        # Check that matrices have been modified (unless trivial case)
                        if m > 0 && n > 0 && ib > 0
                            modification_occurred = !isapprox(A1_nextla, A1_orig, rtol=rtol) ||
                                                  !isapprox(A2_nextla, A2_orig, rtol=rtol)
                            @test modification_occurred
                        end
                    end
                end
            end
        end
    end
    
    @testset "Error Handling Tests" begin
        for T in ZTTQRT_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                m, n, ib = 6, 5, 2
                A1 = triu(randn(T, n, n))
                A2 = randn(T, m, n)
                T_matrix = zeros(T, ib, n)
                tau = zeros(T, n)
                work = zeros(T, ib * n)
                
                @test_nowarn NextLA.zttqrt(m, n, ib, A1, n, A2, m, T_matrix, ib, tau, work)
                
                # Test edge cases
                @test_nowarn NextLA.zttqrt(0, 0, 0, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, 
                                         zeros(T, 0, 0), 1, T[], T[])
                
                # Test with minimal size
                @test_nowarn NextLA.zttqrt(1, 1, 1, ones(T, 1, 1), 1, ones(T, 1, 1), 1,
                                         zeros(T, 1, 1), 1, zeros(T, 1), zeros(T, 1))
            end
        end
    end
    
    @testset "Numerical Stability Tests" begin
        for T in [ComplexF64]  # High precision type
            @testset "Type $T Stability" begin
                rtol = eps(real(T)) * 100
                
                # Test with different scales
                scales = [eps(real(T)), one(real(T)), 1/eps(real(T))^(1/4)]
                
                for scale in scales
                    m, n, ib = 8, 6, 2
                    A1 = triu(T.(scale .* randn(ComplexF64, n, n)))
                    A2 = T.(scale .* randn(ComplexF64, m, n))
                    T_matrix = zeros(T, ib, n)
                    tau = zeros(T, n)
                    work = zeros(T, ib * n)
                    
                    # Ensure diagonal elements are non-zero
                    for i in 1:n
                        if abs(A1[i, i]) < rtol
                            A1[i, i] = one(T)
                        end
                    end
                    
                    # Test calculation
                    NextLA.zttqrt(m, n, ib, A1, n, A2, m, T_matrix, ib, tau, work)
                    
                    # Check that results are finite
                    @test all(isfinite.(A1))
                    @test all(isfinite.(A2))
                    @test all(isfinite.(T_matrix))
                    @test all(isfinite.(tau))
                    @test all(isfinite.(work))
                end
            end
        end
    end

    @testset "GPU Tests" begin
        if CUDA.functional()
            for T in (ComplexF32,)
                @testset "Type $T GPU" begin
                    rtol = 1e-5
                    
                    # Test different sizes
                    test_cases = [
                        (8, 6, 2),
                        (10, 8, 3)
                    ]
                    
                    for (m, n, ib) in test_cases
                        A1_cpu = triu(randn(T, n, n))
                        A2_cpu = randn(T, m, n)
                        
                        # Ensure diagonal elements are non-zero
                        for i in 1:n
                            if abs(A1_cpu[i, i]) < rtol
                                A1_cpu[i, i] = one(T)
                            end
                        end
                        
                        T_cpu = zeros(T, ib, n)
                        tau_cpu = zeros(T, n)
                        work_cpu = zeros(T, ib * n)
                        
                        # Move to GPU
                        A1_gpu = CuArray(A1_cpu)
                        A2_gpu = CuArray(A2_cpu)
                        T_gpu = CuArray(T_cpu)
                        tau_gpu = CuArray(tau_cpu)
                        work_gpu = CuArray(work_cpu)
                        
                        # Reference CPU calculation
                        A1_ref = copy(A1_cpu)
                        A2_ref = copy(A2_cpu)
                        T_ref = copy(T_cpu)
                        tau_ref = copy(tau_cpu)
                        work_ref = copy(work_cpu)
                        NextLA.zttqrt(m, n, ib, A1_ref, n, A2_ref, m, T_ref, ib, tau_ref, work_ref)
                        
                        # GPU calculation
                        NextLA.zttqrt(m, n, ib, A1_gpu, n, A2_gpu, m, T_gpu, ib, tau_gpu, work_gpu)
                        
                        # Compare results
                        @test norm(Array(A1_gpu) - A1_ref) < rtol * max(1, norm(A1_ref))
                        @test norm(Array(A2_gpu) - A2_ref) < rtol * max(1, norm(A2_ref))
                        @test norm(Array(T_gpu) - T_ref) < rtol * max(1, norm(T_ref))
                        @test norm(Array(tau_gpu) - tau_ref) < rtol * max(1, norm(tau_ref))
                        @test norm(Array(work_gpu) - work_ref) < rtol * max(1, norm(work_ref))
                        
                        @test all(isfinite.(Array(A1_gpu)))
                        @test all(isfinite.(Array(A2_gpu)))
                        @test all(isfinite.(Array(T_gpu)))
                        @test all(isfinite.(Array(tau_gpu)))
                        @test all(isfinite.(Array(work_gpu)))
                    end
                end
            end
        end
    end
end
