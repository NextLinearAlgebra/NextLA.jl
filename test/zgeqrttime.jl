using BenchmarkTools
using LinearAlgebra
using JLD2


function gen_zgeqrt_time(::Type{T}, m, n, ib) where {T<:Number}
    lda = m
    ldt = ib

    A = rand(T, m, n)
    Tee = rand(T, ldt, min(m,n))
    Tau = rand(T, n)
    work = rand(T, ib*n)

    A1 = deepcopy(A)
    Tee1 = deepcopy(Tee)

    #println("lapack")
    tl = @belapsed geqrt!($T, $A1, $Tee1) evals = 1

   # println("julia")
    tj = @belapsed zgeqrt($m,$n,$ib,$A,$lda, $Tee, $ldt, $Tau, $work) evals = 1

    return tl, tj
end

t = Threads.nthreads()
BLAS.set_num_threads(t)
xvals = Int[]
yj = Float64[]
yl = Float64[]
T = Float64
ib = 64

for m in [1024, 2048, 4096, 8192]
    push!(xvals, m)
    n = m
    tl, tj = gen_zgeqrt_time(T, m, n, ib)

    println("m is ", m, " julia time: ", tj, " lapack time: ", tl)
    push!(yj, tj)
    push!(yl, tl)
end

#@save "geqrt time type=$T t=$t ib=$ib.jdl2" xvals yj yl



