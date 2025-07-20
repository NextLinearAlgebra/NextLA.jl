using Test
using NextLA
using LinearAlgebra
using Random
using CUDA
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc

function lapack_tprfb!(::Type{T}, side::AbstractChar, trans::AbstractChar, direct::AbstractChar, storev::AbstractChar,
    l::Int64, V::AbstractMatrix{T}, Tee::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<: Number}

    m,n = size(B)
    ldt, k = size(Tee)
    ldv = max(1, stride(V,2))
    lda = max(1, stride(A,2))
    ldb = max(1,m)

    if side == 'L'
        ldw = k
        work = Array{T}(undef, (ldw,n))
    else
        ldw = m
        work = Array{T}(undef, (ldw,k))
    end

    if m > 0 && n > 0
        if T == ComplexF64
            ccall((@blasfunc(ztprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)

        elseif T == ComplexF32
            ccall((@blasfunc(ctprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)

        elseif T == Float64
            ccall((@blasfunc(dtprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
                side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)
        else # T == Float32
            ccall((@blasfunc(stprfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, l, V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)
        end
    end
    return work
end


# LAPACK-style test parameters for NextLA.zparfb
const ZPARFB_TYPES = [ComplexF32, ComplexF64, Float32, Float64]
# Format: (m1, n1, m2, n2, k, l) where:
# - For side='L': n1 == n2 (same number of columns)
# - For side='R': m1 == m2 (same number of rows)
# - l <= k (lower triangular part constraint)
const ZPARFB_SIZES = [
    (5, 4, 3, 4, 2, 1),   # side='L': n1=n2=4, side='R': m1=5≠m2=3 (only works for side='L')
    (6, 5, 6, 5, 3, 2),   # side='L': n1=n2=5, side='R': m1=m2=6 (works for both sides)
    (8, 6, 4, 6, 3, 2),   # side='L': n1=n2=6, side='R': m1=8≠m2=4 (only works for side='L')
    (10, 8, 10, 7, 4, 3)  # side='L': n1=8≠n2=7, side='R': m1=m2=10 (only works for side='R')
]

@testset "ZPARFB Tests" begin
    @testset "Standard Test Suite" begin
        for (itype, T) in enumerate(ZPARFB_TYPES)
            @testset "Type $T (itype=$itype)" begin
                rtol = (T <: ComplexF32) || (T<:Float32) ? 1e-5 : 1e-12
                atol = rtol
                
                for (isize, (m1, n1, m2, n2, k, l)) in enumerate(ZPARFB_SIZES)
                    @testset "Size m1=$m1, n1=$n1, m2=$m2, n2=$n2, k=$k, l=$l" begin
                        # Test different parameter combinations
                        for side in ['L', 'R']
                            # Skip invalid combinations based on LAPACK TPRFB constraints
                            if side == 'L' && n1 != n2
                                continue  # For side='L', n1 must equal n2
                            elseif side == 'R' && m1 != m2
                                continue  # For side='R', m1 must equal m2
                            end
                            
                            for trans in ['N', 'C']
                                for direct in ['F', 'B']
                                    for storev in ['C', 'R']
                                        @testset "side=$side, trans=$trans, direct=$direct, storev=$storev" begin
                                            # Set up matrices according to LAPACK TPRFB specifications
                                            if side  == 'L' && direct == 'F' && storev == 'R' #This skip is necessary because the LAPACK implementation breaks here because of wrong parameters
                                                continue
                                            end
                                            # A matrix dimensions based on SIDE
                                            if side == 'L'
                                                A1 = rand(T, k, n1)  # A is K-by-N when SIDE='L'
                                                lda1 = k
                                            else
                                                A1 = rand(T, m1, k)  # A is M-by-K when SIDE='R'
                                                lda1 = m1
                                            end
                                            
                                            # B matrix is always M-by-N
                                            A2 = rand(T, m2, n2)  # This is B in LAPACK notation
                                            lda2 = m2
                                            
                                            # V matrix dimensions based on STOREV and SIDE
                                            if storev == 'C'
                                                if side == 'L'
                                                    V = rand(T, m2, k)  # V is M-by-K when STOREV='C' and SIDE='L'
                                                    ldv = m2
                                                else
                                                    V = rand(T, n2, k)  # V is N-by-K when STOREV='C' and SIDE='R'
                                                    ldv = n2
                                                end
                                            else  # storev == 'R'
                                                if side == 'L'
                                                    V = rand(T, k, m2)  # V is K-by-M when STOREV='R' and SIDE='L'
                                                    ldv = k
                                                else
                                                    V = rand(T, k, n2)  # V is K-by-N when STOREV='R' and SIDE='R'
                                                    ldv = k
                                                end
                                            end
                                            
                                            # T matrix is always K-by-K
                                            Tee = rand(T, k, k)
                                            ldt = k
                                            
                                            # Work array dimensions based on SIDE
                                            if side == 'L'
                                                work = rand(T, k, n1)  # WORK is K-by-N when SIDE='L'
                                                ldw = k
                                            else
                                                work = rand(T, m1, k)  # WORK is M-by-K when SIDE='R'
                                                ldw = m1
                                            end
                                            
                                            # Make copies for testing
                                            A1_orig = copy(A1)
                                            A2_orig = copy(A2)
                                            A1_test = copy(A1)
                                            A2_test = copy(A2)
                                            A1_l = deepcopy(A1)
                                            A2_l = deepcopy(A2)
                                            V_test = copy(V)
                                            T_test = copy(Tee)
                                            
                                            work_l = lapack_tprfb!(T, side, trans, direct, storev, l, V, Tee, A1_l, A2_l)
                                            
                                            # NextLA call: zparfb(side, trans, direct, storev, m1, n1, m2, n2, k, l, A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
                                            NextLA.zparfb(side, trans, direct, storev, m1, n1, m2, n2, k, l,
                                                        A1_test, lda1, A2_test, lda2, V_test, ldv, T_test, ldt, work, ldw)

                                                

                                            # Basic checks
                                            @test all(isfinite.(A1_test))
                                            @test all(isfinite.(A2_test))
                                            @test size(A1_test) == size(A1_orig)
                                            @test size(A2_test) == size(A2_orig)
                                            @test all(isfinite.(work))
                                            
                                            # Check that matrices have been modified (unless k=0)
                                            if k > 0 && norm(T_test) > atol
                                                modification_occurred = !isapprox(A1_test, A1_orig, rtol=rtol) ||
                                                                        !isapprox(A2_test, A2_orig, rtol=rtol) ||
                                                                        !isapprox(V_test, V, rtol=rtol) ||
                                                                        !isapprox(T_test, Tee, rtol=rtol)
                                                @test modification_occurred
                                            end
                                            if norm(work_l) > atol
                                                work_error = norm(work - work_l) / norm(work_l)
                                                @test work_error < rtol
                                            end
                                            if norm(A1_l) > atol
                                                a1_error = norm(A1_test - A1_l) / norm(A1_l)
                                                @test a1_error < rtol
                                            end

                                            if norm(A2_l) > atol
                                                a2_error = norm(A2_test - A2_l) / norm(A2_l)
                                                @test a2_error < rtol
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
        for T in ZPARFB_TYPES
            @testset "Type $T Error Handling" begin
                # Test with valid parameters (should not error)
                # Use side='L' case: n1 == n2
                m1, n1, m2, n2, k, l = 6, 5, 4, 5, 3, 2
                A1 = randn(T, m1, n1)
                A2 = randn(T, m2, n2)
                V = randn(T, m2, k)  # For side='L', V has m2 rows
                T_mat = triu(randn(T, k, k))
                work = zeros(T, k, n1)
                
                @test_nowarn NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                                          A1, m1, A2, m2, V, m2, T_mat, k, work, k)
                
                # Test with valid parameters for side='R' case: m1 == m2
                m1, n1, m2, n2, k, l = 6, 5, 6, 4, 3, 2
                A1 = randn(T, m1, n1)
                A2 = randn(T, m2, n2)
                V = randn(T, n2, k)  # For side='R', V has n2 rows
                T_mat = triu(randn(T, k, k))
                work = zeros(T, m1, k)
                
                @test_nowarn NextLA.zparfb('R', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                                          A1, m1, A2, m2, V, n2, T_mat, k, work, m1)
                
                # Test edge cases
                @test_nowarn NextLA.zparfb('L', 'N', 'F', 'C', 0, 0, 0, 0, 0, 0,
                                          zeros(T, 0, 0), 1, zeros(T, 0, 0), 1, zeros(T, 0, 0), 1,
                                          zeros(T, 0, 0), 1, T[], 1)
                
                # Test with k=0 (valid for both sides)
                @test_nowarn NextLA.zparfb('L', 'N', 'F', 'C', 2, 2, 2, 2, 0, 0,
                                          randn(T, 2, 2), 2, randn(T, 2, 2), 2, zeros(T, 2, 0), 2,
                                          zeros(T, 0, 0), 1, T[], 1)
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
                    # Use valid dimensions: side='L' case with n1==n2
                    m1, n1, m2, n2, k, l = 8, 6, 5, 6, 3, 2
                    A1 = T.(scale .* randn(ComplexF64, m1, n1))
                    A2 = T.(scale .* randn(ComplexF64, m2, n2))
                    V = T.(scale .* randn(ComplexF64, m2, k))  # For side='L', V has m2 rows
                    T_mat = triu(T.(scale .* randn(ComplexF64, k, k)))
                    work = zeros(T, k, n1)
                    
                    # Set up proper Householder structure
                    for i in 1:k
                        V[1:i-1, i] .= zero(T)
                        V[i, i] = one(T)
                    end
                    
                    # Test calculation
                    NextLA.zparfb('L', 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                                  A1, m1, A2, m2, V, m2, T_mat, k, work, k)
                    
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
            for T in (ComplexF32,) # Common GPU type
                @testset "Type $T GPU" begin
                    rtol = 1e-5
                    
                    # Use valid dimensions for both sides
                    test_cases = [
                        # side='L': n1==n2
                        ('L', 6, 5, 4, 5, 3, 2),
                        # side='R': m1==m2  
                        ('R', 6, 5, 6, 4, 3, 2)
                    ]
                    
                    for (side, m1, n1, m2, n2, k, l) in test_cases
                        A1_cpu = randn(T, m1, n1)
                        A2_cpu = randn(T, m2, n2)
                        
                        # Set V dimensions based on side
                        if side == 'L'
                            V_cpu = randn(T, m2, k)
                            work_cpu = zeros(T, k * n1)
                            ldv = m2
                            ldwork = k
                        else  # side == 'R'
                            V_cpu = randn(T, n2, k)
                            work_cpu = zeros(T, m1 * k)
                            ldv = n2
                            ldwork = m1
                        end
                        
                        T_cpu = triu(randn(T, k, k))
                        
                        # Set up proper Householder structure
                        for i in 1:k
                            V_cpu[1:i-1, i] .= zero(T)
                            V_cpu[i, i] = one(T)
                        end
                        
                        # Move data to GPU
                        A1_gpu = CuArray(A1_cpu)
                        A2_gpu = CuArray(A2_cpu)
                        V_gpu = CuArray(V_cpu)
                        T_gpu = CuArray(T_cpu)
                        work_gpu = CuArray(work_cpu)
                        
                        # Reference CPU calculation
                        A1_ref = copy(A1_cpu)
                        A2_ref = copy(A2_cpu)
                        work_ref = copy(work_cpu)
                        NextLA.zparfb(side, 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                                      A1_ref, m1, A2_ref, m2, V_cpu, ldv, T_cpu, k, work_ref, ldwork)
                        
                        # Our implementation on GPU
                        NextLA.zparfb(side, 'N', 'F', 'C', m1, n1, m2, n2, k, l,
                                      A1_gpu, m1, A2_gpu, m2, V_gpu, ldv, T_gpu, k, work_gpu, ldwork)
                        
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
