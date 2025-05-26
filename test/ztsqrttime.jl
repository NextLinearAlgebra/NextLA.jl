using BenchmarkTools
using LinearAlgebra
using JLD2


#note n >= m
function gen_tsqrt_time(::Type{T}, m, n, ib) where {T <: Number}
    A1 = rand(T, n, n)
    lda1 = n
    A2 = rand(T, m, n)
    lda2 = m
    Tee = rand(T, ib, n)
    ldt = ib
    Tau = rand(T, n)
    work = rand(T, ib*n)
    
    l = 0
    A = deepcopy(A1)
    B = deepcopy(A2)
    Tee1 = deepcopy(Tee)

    tl = @elapsed lapack_tsqrt!(T, l, A, B, Tee1)
    tj = @elapsed ztsqrt(m,n,ib,A1,lda1,A2, lda2, Tee, ldt, Tau, work) 

    return tl, tj
end

t = Threads.nthreads()
BLAS.set_num_threads(t)
println("threads is ", t)
T = Float64

for m in [512, 1024, 2048, 4096]
    for n in [m]
        for ib in [64,128]
            tl, tj = gen_tsqrt_time(T, m, n, ib)
            println("m is ", m, " n is ", n, " ib is ", ib, " julia time: ", tj, " lapack time: ", tl)
        end
    end
end




