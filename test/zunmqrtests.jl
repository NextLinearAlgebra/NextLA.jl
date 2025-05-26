using Test
using LinearAlgebra


function gen_zunmqr_test_rand(::Type{T}, side, trans, m, n, k, ib) where {T <: Number}
    C = rand(T,m,n)

    if side == 'L'
        A = rand(T,m,k)
        lda = m
        ldw = n
        work = rand(T,n,n)
    else
        A = rand(T,n,k)
        lda = n
        ldw = m
        work = rand(T,m,ib)
    end
    
    Tau = rand(T,ib,k)
    Tau1 = rand(T,k)
    A1 = deepcopy(A)
    LinearAlgebra.LAPACK.geqrt!(A, Tau)
    LinearAlgebra.LAPACK.geqrf!(A1, Tau1)
    D = deepcopy(C)

    zunmqrv0(side, trans, m, n, k, ib, A, lda, Tau, ib, D, m, work, ldw)
    
    if (T == Float64 || T == Float32) && trans == 'C'
        unmqr!(T, side, 'T', A1, Tau1, C)
    else
        unmqr!(T, side, trans, A1, Tau1, C)
    end

    """ previous metrics
    if T == ComplexF64 || T == ComplexF32
        return norm((C-D)./C) / sqrt(2*m*n)
    else
        return norm((C-D)./C) / sqrt(m*n)  
    end
    """

    return norm(C-D) / (norm(C))
end

@testset "unmqr minitests" begin
    @test gen_zunmqr_test_rand(ComplexF64, 'L', 'N', 128, 128, 128, 1) ≈ 0 atol 5e-15 
    @test gen_zunmqr_test_rand(ComplexF64, 'L', 'N', 1024, 1024, 1024, 1) ≈ 0 atol 5e-15 
    @test gen_zunmqr_test_rand(ComplexF64, 'L', 'N', 1024, 1024, 1024, 64) ≈ 0 atol 5e-15 
end


@testset "unmqr datatype = Float64" begin
    for side in ['L', 'R']
        @testset "side=$side, trans=$trans" for trans in ['N', 'C'] 
            for m in [128, 256, 512, 1024, 2048]
                for n in [m, div(m,10)*9, div(m,10)*11]
                
                    if side == 'L'
                        k = m
                    else
                        k = n
                    end
                    
                    @test gen_zunmqr_test_rand(Float64, side, trans, m,n,k,64) ≈ 0 atol=5e-15
                end
            end
        end
    end
end


@testset "unmqr datatype = ComplexF64" begin
    for side in ['L', 'R']
        @testset "side=$side, trans=$trans" for trans in ['N', 'C'] 
            for m in [128, 256, 512, 1024, 2048]
                for n in [m, div(m,10)*9, div(m,10)*11]
                
                    if side == 'L'
                        k = m
                    else
                        k = n
                    end

                    @test gen_zunmqr_test_rand(ComplexF64, side, trans, m,n,k,1) ≈ 0 atol=5e-15
                end
            end
        end
    end
end


@testset "unmqr datatype = ComplexF32" begin
    for side in ['L', 'R']
        @testset "side=$side, trans=$trans" for trans in ['N', 'C'] 
            for m in [128, 256, 512, 1024, 2048]
                for n in [m, div(m,10)*9, div(m,10)*11]
                
                    if side == 'L'
                        k = m
                    else
                        k = n
                    end
                    
                    @test gen_zunmqr_test_rand(ComplexF32, side, trans, m,n,k,1) ≈ 0 atol=5e-7
                end
            end
        end
    end
end


@testset "unmqr datatype = Float32" begin
    for side in ['L', 'R']
        @testset "side=$side, trans=$trans" for trans in ['N', 'C'] 
            for m in [128, 256, 512, 1024, 2048]
                for n in [m, div(m,10)*9, div(m,10)*11]
                
                    if side == 'L'
                        k = m
                    else
                        k = n
                    end
                    
                    @test gen_zunmqr_test_rand(Float32, side, trans, m,n,k,64) ≈ 0 atol=5e-7
                end
            end
        end
    end
end
