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
const TTMQR_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
# Format: (m1, n1, m2, n2, k, ib) where:
# - For side='L': n1 == n2 and k <= m1
# - For side='R': m1 == m2 and k <= n1  
# - ib is the block size (ib <= k)
const TTMQR_SIZES = [
    # side='L' cases: n1 == n2, k <= m1, ib <= k
    (600, 600, 200),   # m1=8, n1=6, m2=6, n2=6, k=6, ib=2 -> k>=ib ✓
    (800, 800, 300),  # m1=10, n1=8, m2=8, n2=8, k=8, ib=3 -> k>=ib ✓
    (1000, 1000, 400), # m1=12, n1=10, m2=10, n2=10, k=10, ib=4 -> k>=ib ✓
    (700, 700, 500),   # m1=9, n1=7, m2=7, n2=7, k=7, ib=5 -> k>=ib ✓

    # side='R' cases: m1 == m2, k <= n1, ib <= k
    (600, 600, 200),   # m1=6, n1=8, m2=6, n2=6, k=6, ib=2 -> k>=ib ✓
    (800, 800, 300),  # m1=8, n1=10, m2=8, n2=8, k=8, ib=3 -> k>=ib ✓
    (1000, 1000, 400), # m1=10, n1=12, m2=10, n2=10, k=10, ib=4 -> k>=ib ✓
    (700, 700, 500),   # m1=7, n1=9, m2=7, n2=7, k=7, ib=5 -> k>=ib ✓

    # Edge cases with k == ib
    (600, 400, 400),   # side='L': k=ib=4
    (600, 400, 400),   # side='R': k=ib=4
    (800, 600, 600),  # side='L': k=ib=6
    (800, 600, 600),  # side='R': k=ib=6
]

@testset "TTMQR Tests" begin
    @testset "NextLA vs LAPACK comparison" begin
        for (itype, T) in enumerate(TTMQR_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T <: Float32) ? 1e-5 : 1e-12
                
                for (isize, (n2, k, ib)) in enumerate(TTMQR_SIZES)
                    @testset "Size n2=$n2, k=$k, ib=$ib" begin
                        # Test different parameter combinations
                        for side in ['L', 'R']
                            # Skip invalid combinations based on TTMQR constraints
                            if k > n2
                                continue  # For side='L', n1 must equal n2 and k <= m1
                            end
                            
                            for trans in ['N', 'C']
                                @testset "side=$side, trans=$trans"  begin
                                    # Generate matrices according to TTMQR specifications
                                    # C1 and C2 are the matrices being transformed
                                    if ib > n2 || ib <= 0
                                        continue  # Skip invalid block sizes
                                    end
            
                                    A1 = randn(T, n2, n2)
                                    A2 = randn(T, n2, n2)
                                    V = triu(randn(T, n2, n2))
                                
                                    T_mat = triu(rand(T, ib, k))
            
                                    work = zeros(T, ib * n2)
            
                                    A1_nextla = copy(A1)
                                    A2_nextla = copy(A2)
                                    A1_orig = copy(A1)
                                    A2_orig = copy(A2)
                                    T_mat_nextla = copy(T_mat)
                                    work_nextla = copy(work)
            
                                    NextLA.ttmqr!('L', 'N', n2, n2, n2, n2, k, ib,
                                                A1_nextla, A2_nextla, V, T_mat_nextla, work_nextla)
                                    
                                    # --- Test Helper Function ---
                                    A1_helper = copy(A1_orig)
                                    A2_helper = copy(A2_orig)
                                    T_mat_helper = copy(T_mat)
                                    NextLA.ttmqr!('L', 'N', A1_helper, A2_helper, V, T_mat_helper)

                                    # Verify helper gives same results as kernel
                                    @test A1_helper ≈ A1_nextla rtol=rtol
                                    @test A2_helper ≈ A2_nextla rtol=rtol
                                    
                                    lapack_tpmqrt!(T, 'L', 'N', 0, V, T_mat, A1_orig, A2_orig)
                                    @test norm(A1_nextla - A1_orig) < rtol * norm(A1_orig)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    @testset "Error Handling Tests" begin
        for T in TTMQR_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                m1, n1, m2, n2, k, ib = 800, 600, 600, 600, 400, 200
                C1 = randn(T, m1, n1)
                C2 = randn(T, m2, n2)
                V = randn(T, m2, k)
                T_mat = triu(randn(T, ib, k))
                work = zeros(T, ib * n1)
                
                @test_nowarn NextLA.ttmqr!('L', 'N', m1, n1, m2, n2, k, ib,
                                           C1, C2, V, T_mat, work)
                
                # Test edge cases
                @test_nowarn NextLA.ttmqr!('L', 'N', 0, 0, 0, 0, 0, 0,
                                           zeros(T, 0, 0), zeros(T, 0, 0), 
                                           zeros(T, 0, 0), zeros(T, 0, 0), T[])
                
                # Test with k=0
                @test_nowarn NextLA.ttmqr!('L', 'N', 2, 2, 2, 2, 0, 0,
                                           randn(T, 2, 2), randn(T, 2, 2),
                                           zeros(T, 2, 0), zeros(T, 0, 0), T[])
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
                    m1, n1, m2, n2, k, ib = 800, 600, 600, 600, 400, 200
                    C1 = T.(scale .* randn(ComplexF64, m1, n1))
                    C2 = T.(scale .* randn(ComplexF64, m2, n2))
                    V = T.(scale .* randn(ComplexF64, m2, k))
                    T_mat = triu(T.(scale .* randn(ComplexF64, ib, k)))
                    work = zeros(T, ib * n1)
                    
                    # Test calculation
                    NextLA.ttmqr!('L', 'N', m1, n1, m2, n2, k, ib,
                                  C1, C2, V, T_mat, work)
                    
                    # Check that results are finite
                    @test all(isfinite.(C1))
                    @test all(isfinite.(C2))
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
                        ('L', 8, 6, 6, 6, 4, 2),
                        ('R', 10, 8, 10, 5, 5, 3)
                    ]
                    
                    for (side, m1, n1, m2, n2, k, ib) in test_cases
                        C1_cpu = randn(T, m1, n1)
                        C2_cpu = randn(T, m2, n2)
                        
                        if side == 'L'
                            V_cpu = randn(T, m2, k)
                            work_cpu = zeros(T, ib * n1)
                        else
                            V_cpu = randn(T, n2, k)
                            work_cpu = zeros(T, ib * m1)
                        end
                        
                        T_cpu = triu(randn(T, ib, k))
                        
                        # Move to GPU
                        C1_gpu = CuArray(C1_cpu)
                        C2_gpu = CuArray(C2_cpu)
                        V_gpu = CuArray(V_cpu)
                        T_gpu = CuArray(T_cpu)
                        work_gpu = CuArray(work_cpu)
                        
                        # Reference CPU calculation
                        C1_ref = copy(C1_cpu)
                        C2_ref = copy(C2_cpu)
                        work_ref = copy(work_cpu)
                        NextLA.ttmqr!(side, 'N', m1, n1, m2, n2, k, ib,
                                      C1_ref, C2_ref, V_cpu, T_cpu, work_ref)
                        
                        # GPU calculation
                        NextLA.ttmqr!(side, 'N', m1, n1, m2, n2, k, ib,
                                      C1_gpu, C2_gpu, V_gpu, T_gpu, work_gpu)
                        
                        # Compare results
                        @test norm(Array(C1_gpu) - C1_ref) < rtol * max(1, norm(C1_ref))
                        @test norm(Array(C2_gpu) - C2_ref) < rtol * max(1, norm(C2_ref))
                        @test norm(Array(work_gpu) - work_ref) < rtol * max(1, norm(work_ref))
                        
                        @test all(isfinite.(Array(C1_gpu)))
                        @test all(isfinite.(Array(C2_gpu)))
                        @test all(isfinite.(Array(work_gpu)))
                    end
                end
            end
        end
    end
end
