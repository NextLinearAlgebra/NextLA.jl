ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"
# To ensure that the plot doesn't try to open a window

using BenchmarkTools
using Plots
using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: libblastrampoline
using NextLA


function slas2_time(f::Float64, g::Float64, h::Float64, ssmin::Ref{Float64}, ssmax::Ref{Float64})
        return @belapsed ccall(
            (@blasfunc(slas2_), libblastrampoline),
                Cvoid, 
                (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
                $f, $g, $h, $ssmin, $ssmax
            )
end
function slas2_time(f::Float32, g::Float32, h::Float32, ssmin::Ref{Float32}, ssmax::Ref{Float32})
        return @belapsed ccall(
            (@blasfunc(slas2_), libblastrampoline),
                Cvoid, 
                (Ref{Float32}, Ref{Float32}, Ref{Float32}, Ref{Float32}, Ref{Float32}),
                $f, $g, $h, $ssmin, $ssmax
            )
end

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


        j = @belapsed NextLA.slas2!($A)

        n = slas2_time(f, g, h, ssmin, ssmax)

        push!(jul, j)
        push!(nat, n)
    end

    plt = plot(jul, label="slas2!", yscale=:log10)
    plot!(plt, nat, label="native sdvals")
    savefig(plt, "timings_$(T).png")
end
