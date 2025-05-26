using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools
using Test



#note n >= m
function gen_tsqrt_test(::Type{T}, m, n, ib) where {T<:Number}
    A1 = rand(T, n, n)
    lda1 = n
    A2 = rand(T, m, n)
    lda2 = m
    Tee = rand(T, ib, n)
    ldt = ib
    Tau = rand(T, n)
    work = rand(T, ib * n)

    l = 0
    A = deepcopy(A1)
    B = deepcopy(A2)
    Tee1 = deepcopy(Tee)

    lapack_tsqrt!(T, l, A, B, Tee1)
    ztsqrt(m, n, ib, A1, lda1, A2, lda2, Tee, ldt, Tau, work)

    errA1 = norm(A - A1) / norm(A)
    errA2 = norm(B - A2) / norm(B)

    println("m is ", m, " n is ", n, " ib is ", ib)
    println("error of A1 ", errA1)
    println("error of A2 ", errA2)

    return max(errA1, errA2)
end

# small test for tsqrt
#gen_tsqrt_test(Float64, 2150, 2351, 256)
"""
@testset "datatype=T" for T in [Float64, ComplexF64, Float32, ComplexF32]

    if T == Float64 || T == ComplexF64
        tol = 5e-15
    else
        tol = 5e-6
    end

    @testset "m=m" for m in [256, 512, 1024]
        for n in [m, div(m,5)*6, div(m,10)*11]
            for ib in [64,128]
                @test gen_tsqrt_test(T,m,n,ib) ≈ 0 atol=tol
            end
        end
    end
end
"""

function gen_tsmqr_test(::Type{T}, side, trans, m1, n1, m2, n2, k, ib) where {T<:Number}
    A1 = rand(T, m1, n1)
    A2 = rand(T, m2, n2)

    if side == 'L'
        ldv = m2
        V = rand(T, ldv, m1)
        ldw = n2
        work = rand(T, ldw, ib)

    else # side = 'R'
        ldv = n2
        V = rand(T, ldv, n1)
        ldw = m2
        work = rand(T, ldw, ib)
    end

    ldt = ib
    Tee = rand(T, ib, k)
end

# for ztsmqr only called with side = 'L', trans = 'C', 
# for zparfb only called with direct = 'F', storev = 'C', and l = 0
# issue with storev = R, side = L

function gen_zparfb_test(::Type{T}, side, trans, direct, storev,
    m1, n1, m2, n2, k, l) where {T<:Number}

    A1 = rand(T, m1, n1)
    A2 = rand(T, m2, n2)

    if storev == 'C'
        if side == 'L'
            ldv = m2
            V = rand(T, ldv, k)
            ldw = k
            work = rand(T, ldw, n1)
        else
            ldv = n2
            V = rand(T, ldv, k)
            ldw = m2
            work = rand(T, ldw, k)
        end
    else
        ldv = k
        if side == 'L'
            V = rand(T, ldv, m2)
            ldw = k
            work = rand(T, ldw, n1)
        else
            V = rand(T, ldv, n2)
            ldw = m2
            work = rand(T, ldw, k)
        end
    end

    Tee = rand(T, k, k)
    ldw = m1
    work = rand(T, ldw, n1)

    A1_l = deepcopy(A1)
    A2_l = deepcopy(A2)

    zparfb(side, trans, direct, storev, m1, n1, m2, n2, k, l,
        A1, m1, A2, m2, V, ldv, Tee, k, work, ldw)

    lapack_tprfb!(T, side, trans, direct, storev, l, V, Tee, A1_l, A2_l)

    a1_error = norm(A1 - A1_l) / norm(A1_l)
    a2_error = norm(A2 - A2_l) / norm(A2_l)

    #println("A1 error ", norm(A1 - A1_l) / norm(A1_l))
    #println("A2 error ", norm(A2 - A2_l) / norm(A2_l))
    return max(a1_error, a2_error)
end

# zparfb testing
side = 'L'
trans = 'C'
direct = 'F'
storev = 'C'

@testset "datatype=T" for T in [Float64, ComplexF64, Float32, ComplexF32]
    if T == Float64 || T == ComplexF64
        tol = 1e-16
    else
        tol = 1e-6
    end

    @testset "m1=m1" for m1 in [256, 512, 1024]
        for m2 in [m1, div(m1, 10)*9, div(m1, 10)*11]
            for n in [214, 557, 1012]
                maxk = min(n, min(m1,m2))
                for k in [maxk, div(maxk,5)*4, div(maxk,10)*9]
                    l = k
                    @test gen_zparfb_test(T, side, trans, direct, storev, m1, n, m2, n, k, l) ≈ 0 atol=tol
                    #@test gen_zparfb_test(T, side, trans, direct, storev, n, m1, n, m2, k, l) ≈ 0 atol=tol
                end
            end
        end
    end
end
"""
# small tests for parfb
m1 = 20
m2 = 20
n = 20 # n1 = n2 because left
k = 10
l = 5

gen_zparfb_test(Float64, side, trans, direct, storev, m1, n, m2, n, k, l)
#gen_zparfb_test(Float64, side, trans, direct, storev, n, m1, n, m2, k, l)
"""

