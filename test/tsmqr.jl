using Test
using NextLA
using LinearAlgebra
using Random
using CUDA
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
function lapack_tpmqrt!(::Type{T}, side::Char, trans::Char, l::Int64, V::AbstractMatrix{T}, 
    Tau::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<: Number}
    m1, n1 = size(A)
    m, n = size(B)
    nb, k = size(Tau)
    minmn = min(m, n)

    if nb > minmn
        throw(ArgumentError("block size $nb > $minmn too large"))
    end

    ldv = max(1, stride(V,2))
    ldt = max(1, stride(Tau,2))
    lda = max(1, stride(A,2))
    ldb = max(1, stride(B,2))
    if side == 'L'
        work = zeros(T, nb, n1)
        ldwork = nb
    else
        work = zeros(T, m1, nb)
        ldwork = m1
    end

    if trans == 'C' && T <: Real
        trans = 'T'
    end
    
    info = Ref{BlasInt}()
  
    if n > 0
        if T == ComplexF64
            ccall((@blasfunc(ztpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])
        elseif T == Float64
                
            ccall((@blasfunc(dtpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])

        elseif T == ComplexF32
            ccall((@blasfunc(ctpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])
        else # T = Float32
            ccall((@blasfunc(stpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])
        end
    end
end

const TSMQR_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
# Format: (m1, n1, m2, n2, k, ib) where:
# - For side='L': n1 == n2 and k <= m1
# - For side='R': m1 == m2 and k <= n1
# - ib is the block size (ib <= k)
const TSMQR_SIZES = [
    (600, 500, 400, 500, 300, 200),   # side='L': n1=n2=500, k=300<=m1=600, ib=200<=k=300
    (800, 600, 800, 400, 400, 200),   # side='R': m1=m2=800, k=400<=n1=600, ib=200<=k=400
    (100, 80, 60, 80, 50, 30),  # side='L': n1=n2=800, k=500<=m1=1000, ib=300<=k=500
    (120, 100, 120, 70, 60, 40) # side='R': m1=m2=1200, k=600<=n1=1000, ib=400<=k=600
]

@testset "TSMQR Tests" begin
    @testset "NextLA vs LAPACK comparison" begin
        for (itype, T) in enumerate(TSMQR_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                for (isize, (m1, n1, m2, n2, k, ib)) in enumerate(TSMQR_SIZES)
                    @testset "Size m1=$m1, n1=$n1, m2=$m2, n2=$n2, k=$k, ib=$ib" begin
                        # Test different parameter combinations
                        for side in ['L', 'R']
                            # Skip invalid combinations based on LAPACK TSMQR constraints
                            if side == 'L' && (n1 != n2 || k > m1)
                                continue  # For side='L', n1 must equal n2 and k <= m1
                            elseif side == 'R' && (m1 != m2 || k > n1)
                                continue  # For side='R', m1 must equal m2 and k <= n1
                            end
                            
                            for trans in ['N', 'C']
                                @testset "side=$side, trans=$trans" begin
                                    # Set up matrices according to LAPACK TSMQR specifications
                                    
                                    # A1 matrix (upper triangular part)
                                    A1 = rand(T, m1, n1)
                                    
                                    # A2 matrix (lower part)
                                    A2 = rand(T, m2, n2)
                                    
                                    # V matrix (Householder vectors)
                                    if side == 'L'
                                        V = rand(T, m2, k)
                                        ldv = m2
                                    else
                                        V = rand(T, n2, k)
                                        ldv = n2
                                    end
                                    
                                    # T matrix (triangular factors)
                                    T_mat = triu(rand(T, ib, k))
                                    
                                    # Work array
                                    if side == 'L'
                                        work = zeros(T, ib, n1)
                                        ldwork = ib
                                    else
                                        work = zeros(T, m1, ib)
                                        ldwork = m1
                                    end
                                    
                                    # Make copies for testing
                                    A1_orig = copy(A1)
                                    A2_orig = copy(A2)
                                    A1_nextla = copy(A1)
                                    A2_nextla = copy(A2)
                                    A1_lapack = copy(A1)
                                    A2_lapack = copy(A2)
                                    
                                    # Test NextLA implementation
                                    NextLA.tsmqr(side, trans, m1, n1, m2, n2, k, ib,
                                                  A1_nextla, m1, A2_nextla, m2, V, ldv, T_mat, ib, work, ldwork)
                                    
                                    # Test LAPACK reference
                                    lapack_tpmqrt!(T, side, trans, 0, V, T_mat, A1_lapack, A2_lapack)
                                    
                                    # Compare results
                                    @test norm(A1_nextla - A1_lapack) < rtol * max(1, norm(A1_lapack))
                                    @test norm(A2_nextla - A2_lapack) < rtol * max(1, norm(A2_lapack))
                                    
                                    # Basic sanity checks
                                    @test all(isfinite.(A1_nextla))
                                    @test all(isfinite.(A2_nextla))
                                    @test size(A1_nextla) == size(A1_orig)
                                    @test size(A2_nextla) == size(A2_orig)
                                    @test all(isfinite.(work))
                                    
                                    # Check that matrices have been modified (unless k=0)
                                    if k > 0 && ib > 0 && norm(T_mat) > rtol
                                        modification_occurred = !isapprox(A1_nextla, A1_orig, rtol=rtol) ||
                                                                !isapprox(A2_nextla, A2_orig, rtol=rtol)
                                        @test modification_occurred
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
        for T in TSMQR_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                m1, n1, m2, n2, k, ib = 600, 500, 400, 500, 300, 200
                A1 = randn(T, m1, n1)
                A2 = randn(T, m2, n2)
                V = randn(T, m2, k)
                T_mat = triu(randn(T, ib, k))
                work = zeros(T, ib, n1)
                
                @test_nowarn NextLA.tsmqr('L', 'N', m1, n1, m2, n2, k, ib,
                                           A1, m1, A2, m2, V, m2, T_mat, ib, work, ib)
                
                # Test edge cases
                @test_nowarn NextLA.tsmqr('L', 'N', 0, 0, 0, 0, 0, 0,
                                           zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, 
                                           zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, T[], 1)
                
                # Test with k=0
                @test_nowarn NextLA.tsmqr('L', 'N', 200, 200, 200, 200, 0, 0,
                                           randn(T, 200, 200), 200, randn(T, 200, 200), 200,
                                           zeros(T, 200, 0), 200, zeros(T, 0, 0), 1, T[], 1)
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
                    m1, n1, m2, n2, k, ib = 800, 600, 500, 600, 400, 200
                    A1 = T.(scale .* randn(ComplexF64, m1, n1))
                    A2 = T.(scale .* randn(ComplexF64, m2, n2))
                    V = T.(scale .* randn(ComplexF64, m2, k))
                    T_mat = triu(T.(scale .* randn(ComplexF64, ib, k)))
                    work = zeros(T, ib, n1)
                    
                    # Test calculation
                    NextLA.tsmqr('L', 'N', m1, n1, m2, n2, k, ib,
                                  A1, m1, A2, m2, V, m2, T_mat, ib, work, ib)
                    
                    # Check that results are finite
                    @test all(isfinite.(A1))
                    @test all(isfinite.(A2))
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
                    
                    # Test both sides
                    test_cases = [
                        ('L', 600, 500, 400, 500, 300, 200),
                        ('R', 800, 600, 800, 400, 400, 200)
                    ]
                    
                    for (side, m1, n1, m2, n2, k, ib) in test_cases
                        A1_cpu = randn(T, m1, n1)
                        A2_cpu = randn(T, m2, n2)
                        
                        if side == 'L'
                            V_cpu = randn(T, m2, k)
                            work_cpu = zeros(T, ib * n1)
                            ldv = m2
                            ldwork = ib
                        else
                            V_cpu = randn(T, n2, k)
                            work_cpu = zeros(T, m1 * ib)
                            ldv = n2
                            ldwork = m1
                        end
                        
                        T_cpu = triu(randn(T, ib, k))
                        
                        # Move to GPU
                        A1_gpu = CuArray(A1_cpu)
                        A2_gpu = CuArray(A2_cpu)
                        V_gpu = CuArray(V_cpu)
                        T_gpu = CuArray(T_cpu)
                        work_gpu = CuArray(work_cpu)
                        
                        # Reference CPU calculation
                        A1_ref = copy(A1_cpu)
                        A2_ref = copy(A2_cpu)
                        work_ref = copy(work_cpu)
                        NextLA.tsmqr(side, 'N', m1, n1, m2, n2, k, ib,
                                      A1_ref, m1, A2_ref, m2, V_cpu, ldv, T_cpu, ib, work_ref, ldwork)
                        
                        # GPU calculation
                        NextLA.tsmqr(side, 'N', m1, n1, m2, n2, k, ib,
                                      A1_gpu, m1, A2_gpu, m2, V_gpu, ldv, T_gpu, ib, work_gpu, ldwork)
                        
                        # Compare results
                        @test norm(Array(A1_gpu) - A1_ref) < rtol * max(1, norm(A1_ref))
                        @test norm(Array(A2_gpu) - A2_ref) < rtol * max(1, norm(A2_ref))
                        @test norm(Array(work_gpu) - work_ref) < rtol * max(1, norm(work_ref))
                        
                        @test all(isfinite.(Array(A1_gpu)))
                        @test all(isfinite.(Array(A2_gpu)))
                        @test all(isfinite.(Array(work_gpu)))
                    end
                end
            end
        end
    end
end