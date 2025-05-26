using LinearAlgebra
using BenchmarkTools
using JLD2
using Plots
using StatsPlots


function gen_getrf2_time(::Type{T}, m, n) where {T<: Number}
    A = rand(T,m,n)
    ipiv = ones(Int, min(m,n))
    info = Ref{Int}(0)

    for i in 1:min(m,n)
        ipiv[i] = Int(i)
    end

    B = deepcopy(A)
    bpiv = deepcopy(ipiv)
    C = deepcopy(A)
    cpiv = deepcopy(ipiv)
    D = deepcopy(A)
    dpiv = deepcopy(ipiv)

    tj = @belapsed getrf2!($A, $ipiv, $info) evals = 1
    tl = @belapsed lapack_getrf2!($T, $B, $bpiv) evals = 1
    
    info = Ref{Int}(0)
    mj = @ballocated getrf2!($C, $cpiv, $info) evals = 1
    ml = @ballocated lapack_getrf2!($T, $D, $dpiv) evals = 1

    return tj, tl, mj, ml
end

t = Threads.nthreads()
BLAS.set_num_threads(t)
println("threads should be ", t)

for T in [Float64, Float32, ComplexF64, ComplexF32]
    println("on type ", T)
    xvals = Int[]
    ytj = Float64[] #julia time
    ytl = Float64[] #lapack time
    ymj = Float64[] #julia memory
    yml = Float64[] #lapack memory

    for m in [512, 1024, 2048, 4096, 8192, 16384, 32768]
    #for m in [256, 512, 1024, 2048] # for testing on my personal computer
        push!(xvals, m)
        n = m
        tj, tl, mj, ml = gen_getrf2_time(T, m, n)

        println("m is ", m, " julia time:", tj, " lapack time:", tl, " julia memory:", mj, " lapack memory:", mj)
        push!(ytj, tj)
        push!(ytl, tl)
        push!(ymj, mj / 10^3)
        push!(yml, ml / 10^3)
    end
    
    @save "getrf2 type=$T t=$t.jdl2" xvals ytj ytl ymj yml

    # plotting

    # plot the time with log scale
    # can comment out too if going to do plots later
    p = plot(yaxis=:log)
    #xvals =  ["256", "512", "1024","2048"]
    xvals = ["512", "1024", "2048", "4096", "8192", "16k", "32k"]
    plot!(p, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel="Time (s)")
    plot!(p, xvals, ytl, marker=:circle, color=1, linewidth=2.3, markerstrokewidth = 0,
    linestyle=:dot, markersize=3, label="$T LAPACK")
    plot!(p, xvals, ytj, marker=:star8, color=1, linewidth=1.8, markerstrokewidth = 0,
    label="$T Julia")
    savefig(p, "getrf2 time type=$T t=$t")
end
