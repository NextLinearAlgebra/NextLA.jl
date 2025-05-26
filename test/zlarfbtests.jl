using Test
using LinearAlgebra


function gen_zlarfb_test_rand(::Type{T}, side, trans, storev, direct, m, n, k) where {T<: Number}
    C = rand(T, m, n)
    D = deepcopy(C)
    Tau = rand(T, k, k)

    if side == 'L'
        work = rand(T,n,k)
        ldw = n

        if storev == 'C'
            V = rand(T,m,k)
            ldv = m
            dv = m
        else #storev = R
            V = rand(T,k,m)
            ldv = k
            dv = m
        end

    else #side = 'R'
        work =  rand(T, m, k)
        ldw = m

        if storev == 'C'
            V = rand(T,n,k)
            ldv = n
            dv = n
        else #storev = R
            V = rand(T,k,n)
            ldv = k
            dv = n
        end
    end

    for i in 1:k
        if direct == 'F'
            V[i,i] = 1
        else
            if storev == 'C'
                V[dv - k + i, i] = 1
            else
                V[i, dv - k + i] = 1
            end
        end

        for j in 1:(i-1)
            if direct == 'F' #Tau is upper triangular 
                Tau[i,j] = 0

                if storev == 'C'
                    V[j,i] = 0
                else
                    V[i,j] = 0
                end

            else
                Tau[j,i] = 0

                if storev == 'C'
                    V[dv - k + i, j] = 0
                else
                    V[j, dv - k + i] = 0
                end
            end
        end
    end

    larfb!(T,side,trans,direct,storev,V,Tau,C)
    zlarfb(side, trans, direct, storev, m, n, k, V, ldv,Tau,k,D,m,work,ldw)

    #return norm((C-D)./C)/(sqrt(2*m*n)) # previous value used
    return norm(C-D) / (norm(C)) # new metric
end

@testset "minitest" begin
    @test gen_zlarfb_test_rand(ComplexF64, 'L' , 'N', 'C', 'F', 5,5,3) ≈ 0 atol=5e-15
    @test gen_zlarfb_test_rand(ComplexF64, 'L' , 'N', 'C', 'F', 10,12,8) ≈ 0 atol=5e-15
    @test gen_zlarfb_test_rand(ComplexF64, 'L' , 'N', 'C', 'F', 100,115,85) ≈ 0 atol=5e-15
end

@testset "zlarfb datatype = Float64" begin
    for storev in ['C', 'R']
        for direct in ['F', 'B']
            for side in ['L', 'R']
                @testset "storev=$storev, direct=$direct, side=$side, trans=$trans" for trans in ['N', 'C']
                    for m in [50, 100, 200, 500,1000]
                        for n in [m, div(m,10)*9, div(m,10)*11]
                           # @testset "m=m, n=n, k=k" 
                           for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                @test gen_zlarfb_test_rand(Float64,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-15
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "zlarfb datatype = ComplexF64" begin
    for storev in ['C', 'R']
        for direct in ['F', 'B']
            for side in ['L', 'R']
                @testset "storev=$storev, direct=$direct, side=$side, trans=$trans" for trans in ['N', 'C']
                    for m in [50, 100, 200, 500,1000]
                        for n in [m, div(m,10)*9, div(m,10)*11]
                           # @testset "m=m, n=n, k=k" 
                           for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                @test gen_zlarfb_test_rand(ComplexF64,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-15
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "zlarfb datatype = Float32" begin
    for storev in ['C', 'R']
        for direct in ['F', 'B']
            for side in ['L', 'R']
                @testset "storev=$storev, direct=$direct, side=$side, trans=$trans" for trans in ['N', 'C']
                    for m in [50, 100, 200, 500,1000]
                        for n in [m, div(m,10)*9, div(m,10)*11]
                           # @testset "m=m, n=n, k=k" 
                           for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                @test gen_zlarfb_test_rand(Float32,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-7
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "zlarfb datatype = ComplexF32" begin
    for storev in ['C', 'R']
        for direct in ['F', 'B']
            for side in ['L', 'R']
                @testset "storev=$storev, direct=$direct, side=$side, trans=$trans" for trans in ['N', 'C']
                    for m in [50, 100, 200, 500,1000]
                        for n in [m, div(m,10)*9, div(m,10)*11]
                           # @testset "m=m, n=n, k=k" 
                           for k in [min(m,n), div(min(m,n), 5)*4, div(min(m,n), 5)*3]
                                @test gen_zlarfb_test_rand(ComplexF32,side, trans, storev, direct, m,n,k) ≈ 0 atol=1e-7
                            end
                        end
                    end
                end
            end
        end
    end
end
