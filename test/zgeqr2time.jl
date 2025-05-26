using BenchmarkTools
using LinearAlgebra


BLAS.set_num_threads(Threads.nthreads())

function gen_geqr2_time(::Type{T}, m, n) where {T<: Number}
    A = rand(T,m,n)
    Tau = rand(T,min(m,n))
    work = rand(T, n)

    A1 = deepcopy(A)
    Tau1 = deepcopy(Tau)

    #println("geqr2")
    tl = @belapsed geqr2!($T, $A, $Tau) evals = 1
    tj = @belapsed zgeqr2($m,$n,$A1,$m,$Tau1,$work) evals = 1

    return tl, tj
end

"""
T = Float64
m = 1024
n = 1024

tl, tj = gen_geqr2_time(T, side, m, n)
println("lapack time: ", tl, " julia time: ", tj)
"""

function gen_zlarf_time(::Type{T}, side, m, n) where {T<: Number}
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

    tl = @belapsed larf!($T, $side, $V, $Tau, $C) evals = 1
    tj = @belapsed zlarf($side, $m, $n, $V, $incv, $Tau, $C1, $ldc, $work) evals = 1

    return tl, tj
end

"""
T = Float64
side = 'L'
m = 16384
n = 16384

tl, tj = gen_zlarf_time(T, side, m, n)
println("lapack time: ", tl, " julia time: ", tj)
"""


function gen_zlarft_test(::Type{T}, n, k) where {T<: Number}
    direct = 'F'
    storev = 'C'
    ldv = n
    V = rand(T, ldv, k)
    Tau = rand(T, k)
    Tee = rand(T,k,k)
    Tee1 = deepcopy(Tee)

    tl = @belapsed larft!($T, $direct, $storev, $V, $Tau, $Tee) evals = 1
    tj = @belapsed zlarft($direct, $storev, $n, $k, $V, $ldv, $Tau, $Tee1, $k) evals = 1

    return tl, tj
end

"""
T = Float64
n = 4096
k = 64
tl, tj = gen_zlarft_test(T, n, k)
println("lapack time: ", tl, " julia time: ", tj)
"""
