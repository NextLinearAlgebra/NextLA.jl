using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools
using JLD2
using Plots
using StatsPlots


function gen_getc2_time(::Type{T}, n) where T
    A = rand(T,n,n)

    ipiv = ones(Int, n)
    jpiv = ones(Int, n)
    info = Ref{Int}(0)

    for i in 1:n
        ipiv[i] = Int(i)
        jpiv[i] = Int(i)
    end

    B = deepcopy(A)
    bipiv = deepcopy(ipiv)
    bjpiv = deepcopy(jpiv)

    C = deepcopy(A)
    cipiv = deepcopy(ipiv)
    cjpiv = deepcopy(jpiv)
    cinfo = Ref{Int}(0)

    D = deepcopy(A)
    dipiv = deepcopy(ipiv)
    djpiv = deepcopy(jpiv)

    tj = @belapsed getc2!($A,$ipiv,$jpiv,$info) evals = 1
    tl = @belapsed lapack_getc2!($B,$bipiv,$bjpiv) evals = 1
    mj = @ballocated getc2!($C,$cipiv,$cjpiv,$cinfo) evals = 1
    ml = @ballocated lapack_getc2!($D,$dipiv,$djpiv) evals = 1

    return tj, tl, mj,ml
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

    #for n in [512, 1024, 2048, 4096, 8192, 16384, 32768]
    for n in [256, 512, 1024, 2048]
        push!(xvals, n)
        tj, tl, mj, ml = gen_getc2_time(T, n)
        println("n is ", n, " julia time: ", tj, " lapack time: ", tl, " julia memory: ", mj, " lapack memory: ", ml)
        push!(ytj, tj)
        push!(ytl, tl)
        push!(ymj, mj / 10^3)
        push!(yml, ml / 10^3)
    end

    @save "getc2 type=$T t=$t.jdl2" xvals ytj ytl ymj yml

    # plotting

    # plot the time with log scale
    # can comment out too if going to do plots later
    p = plot(yaxis=:log)
    xvals =  ["256", "512", "1024", "2048"]
    #xvals = ["512", "1024", "2048", "4096", "8192", "16k", "32k"]
    plot!(p, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel="Time (s)")
    plot!(p, xvals, ytl, marker=:circle, color=1, linewidth=2.3, markerstrokewidth = 0,
    linestyle=:dot, markersize=3, label="$T LAPACK")
    plot!(p, xvals, ytj, marker=:star8, color=1, linewidth=1.8, markerstrokewidth = 0,
    label="$T Julia")
    savefig(p, "getc2 time type=$T t=$t")
end
