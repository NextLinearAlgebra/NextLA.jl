ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"
# To ensure that the plot doesn't try to open a window

using LinearAlgebra
using BenchmarkTools
using Plots
using LinearAlgebra
using NextLA



for T in [Float32, Float64]
    jul = Float64[]
    nat = Float64[]
    starting = -(floatmax(T)/T(1e10))
    ending = (floatmax(T)/T(1e10))
    for i in 1:100
        A = LinearAlgebra.UpperTriangular(starting .+ (ending - starting) .* rand(T,2,2))

        f = A[1,1]
        g = A[1,2]
        h = A[2,2]

        ssmin = Ref{T}()
        ssmax = Ref{T}()


        j = @belapsed NextLA.slas2!($f, $g, $h, $ssmin, $ssmax)
        n = @belapsed LinearAlgebra.svdvals($A)

        push!(jul, j)
        push!(nat, n)
    end

    plt = plot(jul, label="slas2!", yscale=:log10)
    plot!(plt, nat, label="native sdvals")
    savefig(plt, "timings_$(T).png")
end
