using Test
using LinearAlgebra
using BenchmarkTools
using Cthulhu


function gen_zlarft_test(::Type{T}, n,k) where {T<: Number}
    direct = 'F'
    storev = 'C'
    ldv = n
    V = rand(T, ldv, k)
    Tau = rand(T, k)
    Tee = rand(T,k,k)
    Tee1 = deepcopy(Tee)

    display(Tee)

    larft!(T, direct, storev, V, Tau, Tee)
    zlarft(direct, storev, n, k, V, ldv, Tau, Tee1, k)

    display(Tau)
    display(Tee)
    display(Tee1)
    
    println(norm(Tee - Tee1) / norm(Tee))
end

"""
n = 8
k = 6
gen_zlarft_test(ComplexF64,n,k)
"""

function gen_zgeqr2_test(::Type{T}, m,n) where{T<: Number}
    A = rand(T,m,n)
    Tau = rand(T,min(m,n))
    work = rand(T, n)

    A1 = deepcopy(A)
    Tau1 = deepcopy(Tau)

    display(A)
    display(Tau)

    geqr2!(T, A, Tau)
     geqr2(m,n,A1,m,Tau1,work) 

    display(A)
    display(A1)

    display(Tau)
    display(Tau1)

    println(norm(A-A1)/norm(A))
    println(norm(Tau-Tau1)/norm(Tau))
end

"""
m = 5
n = 7
gen_zgeqr2_test(ComplexF64, m, n)
"""

function gen_zlarf_test(::Type{T}, side, m,n) where{T<:Number}
    if side == 'L'
        V = rand(T, m)
        work = rand(T,n)
    else
        V = rand(T, n)
        work = rand(T,m)
    end

    incv = 1

    Tau = rand(T)
    C = rand(T,m,n)
    ldc = max(1,m)

    C1 = deepcopy(C)

    @btime larf!(T, side, V, Tau, C) samples = 1 evals = 1
    
    println("initial ", typeof(Tau))
    println(Tau)
    zlarf(side, m, n, V, incv, Tau, C1, ldc, work)
    
    #display(C)
    #display(C1)
    #println(norm(C-C1) / norm(C))
end

"""
side = 'L'
m = 500
n = 500
gen_zlarf_test(ComplexF64, side, m, n)
"""

function gen_zlarfg_test(::Type{T}, n) where{T<:Number}
    X = rand(T, n-1)
    alpha = rand(T)
    incx = 1
    Tau = rand(T)

    X1 = deepcopy(X)
    alpha1 = alpha
    Tau1 = Tau

    alpha2, Tau2 = larfg!(T, alpha, X, Tau)
    alpha1, Tau1 = zlarfg(n, alpha1, X1, incx, Tau1)

    display(X)
    display(X1)
    #println(alpha, " ", alpha1, " " , alpha2)
    #println(Tau, " " , Tau1, " " , Tau2)
    println(norm(X-X1)/ norm(X))
end
"""
n = 6
gen_zlarfg_test(ComplexF64,n)
"""

function gen_zgeqrt_test(::Type{T}, m, n, ib, ver) where {T<:Number}
    lda = m
    ldt = ib

    A = rand(T, m, n)
    Tee = rand(T, ldt, min(m,n))
    Tau = rand(T, n)
    work = rand(T, ib*n)

    A1 = deepcopy(A)
    Tee1 = deepcopy(Tee)

    #display(A)
    #display(Tee)
    
    #println("lapack")
    geqrt!(T, A1, Tee1)

   # println("julia")
    zgeqrt(m,n,ib,A,lda, Tee, ldt, Tau, work)

    #println("m is ", m, " n is ", n , " ib is ", ib)
    #println(norm(A1 - A) / norm(A1))
    #println(norm(Tee1 - Tee) / norm(Tee1))

    if ver == 'A'
        return norm(A1 - A) / norm(A1)
    else
        return norm(Tee1 - Tee) / norm(Tee1)
    end
        
end
"""
m = 7
n = 9
ib = 2

gen_zgeqrt_test(Float64, m, n, ib, 'A')
"""

@testset "datatype = Float64 A accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(Float64, m,n,ib,'A') ≈ 0 atol=1e-14
            end
        end
    end
end

@testset "datatype = Float64 T accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(Float64, m,n,ib,'T') ≈ 0 atol=2e-14
            end
        end
    end
end

@testset "datatype = ComplexF64 A accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(ComplexF64, m,n,ib,'A') ≈ 0 atol=1e-14
            end
        end
    end
end

@testset "datatype = ComplexF64 T accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(ComplexF64, m,n,ib,'T') ≈ 0 atol=1e-14
            end
        end
    end
end

@testset "datatype = Float32 A accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(Float32, m,n,ib,'A') ≈ 0 atol=1e-6
            end
        end
    end
end

@testset "datatype = Float32 T accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(Float32, m,n,ib,'T') ≈ 0 atol=2e-6
            end
        end
    end
end

@testset "datatype = ComplexF32 A accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(ComplexF32, m,n,ib,'A') ≈ 0 atol=1e-6
            end
        end
    end
end

@testset "datatype = ComplexF32 T accuracy" begin
    for m in [512, 1024, 2048, 4096]
        for n in [m, div(m,10)*9, div(m,10)*11]
            for ib in [64, 128]
                @test gen_zgeqrt_test(ComplexF32, m,n,ib,'T') ≈ 0 atol=2e-6
            end
        end
    end
end
