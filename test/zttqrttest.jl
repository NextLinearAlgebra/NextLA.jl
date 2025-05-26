using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools
using Test


# note i realize i can only get stuff 
# where m = n due to def of tpqrt
# so i cant test when m != n
# wait so is m <= n guarentteed? 
# because throws error for n < m?

function gen_ttqrt_test(::Type{T}, m, n, ib) where {T<:Number}
    A1 = rand(T, n, n)
    lda1 = n
    A2 = rand(T, m, n)
    lda2 = n
    Tee = rand(T, ib, n)
    ldt = ib
    Tau = rand(T, n)
    work = rand(T, ib * n)

    l = m
    A = deepcopy(A1)
    B = deepcopy(A2)
    Tee1 = deepcopy(Tee)

    lapack_ttqrt!(T, l, A, B, Tee1)
    zttqrt(m, n, ib, A1, lda1, A2, lda2, Tee, ldt, Tau, work)

    errA1 = norm(A - A1) / norm(A)
    errA2 = norm(B - A2) / norm(B)

    #println("m is ", m, " n is ", n, " ib is ", ib)
    println("error of A1 ", errA1)
    #display(A1)
    #display(A)
    println("error of A2 ", errA2)
    #display(A2)
    #display(B)

    return max(errA1, errA2)
end

# small test for ttqrt
#println(gen_ttqrt_test(ComplexF64, 4, 4, 2))

@testset "datatype=$T" for T in [ComplexF64, ComplexF32]

    if T == Float64 || T == ComplexF64
        tol = 5e-15
    else
        tol = 5e-6
    end

    @testset "n=$n" for n in [256, 512, 1024]
        for m in [n, div(n, 5) * 4, div(n, 10) * 9]
            for ib in [64, 128]
                @test gen_ttqrt_test(T, m, n, ib) ≈ 0 atol = tol
            end
        end
    end
end


function gen_pemv_test(::Type{T}, m, n, l) where {T<:Number}
    storev = 'C'
    trans = 'C'

    alpha = rand(T)
    beta = rand(T)

    A = rand(T, m, n)
    lda = m
    X = rand(T, m)
    Y1 = rand(T, n)
    work = rand(T, l)

    for i in m-l+1:m
        for j in 1:(i-(m-l+1))
            A[i, j] = 0
        end
    end

    Y2 = deepcopy(Y1)

    zpemv(trans, storev, m, n, l, alpha, A, lda, X, beta, Y1, work)
    Y2 .= alpha * (transpose(A)) * X + beta * Y2

    #display(Y2)
    #display(Y1)
    err = norm(Y1 - Y2) / norm(Y2)
    println("error is ", err)
    return err
end

#gen_pemv_test(ComplexF64, 123, 140, 20)

