ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"
# To ensure that the plot doesn't try to open a window

using BenchmarkTools
using Plots
using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasInt, libblastrampoline
using NextLA

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

function laed6_time!(kniter::Int64, orgati::bool, rho::Float64, d::AbstractVector{Float64},
                     z::AbstractVector{Float64}, finit::Float64, tau::Ref{Float64}, info::Ref{Int64})
        b =  @benchmarkable begin ccall(
                    (@blasfunc(dlaed6_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt}, Ref{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{BlasInt}),
                        $kniter, $orgati, $rho, $d, $z, $finit, tau, $info
                    )
        end setup = begin
            tau = deepcopy($tau)
        end
        return minimum(run(b, samples=100)).time
end

function laed6_time!(kniter::Int64, orgati::bool, rho::Float32, d::AbstractVector{Float32},
                     z::AbstractVector{Float32}, finit::Float32, tau::Ref{Float32}, info::Ref{Int64})
        b =  @benchmarkable begin ccall(
                    (@blasfunc(slaed6_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt}, Ref{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{BlasInt}),
                        $kniter, $orgati, $rho, $d, $z, $finit, tau, $info
                    )
        end setup = begin
            tau = deepcopy($tau)
        end
        return minimum(run(b, samples=100)).time
end

for T in [Float32, Float64]
    jul = Float64[]
    lapk = Float64[]
    starting = -(floatmax(T)/T(1e10))
    ending = (floatmax(T)/T(1e10))
    for i in 1:10:500
        accum_jul = zero(Float64)
        accum_lapk = zero(Float64)
        for l in 1:10
           kniter = i%2 == 0 ? Int64(1) : Int64(2)
            orgati = i%2 == 0 ? true : false
            d1 = starting + (ending - starting)*rand(T)
            d2 = d1 + (ending - d1)*rand(T)
            d3 = d2 + (ending - d2)*rand(T)
            d = [d1, d2, d3]
            d_copy = deepcopy(d)
            z = one(T) .+ (ending - one(T)).*rand(T, 3)
            z_copy = deepcopy(z)
            rho = starting + (ending - starting)*rand(T)

            finit = rho + sum(z./d)
            tau = T[0]
            tau_ccal = Ref{T}(T(0))
            info_ccal = Ref{Int64}(Int64(0))
            info = Int64[0]


            b = @benchmarkable begin NextLA.laed6!($kniter, $orgati, $rho, $d, $z, $finit, tau, $info)
            end setup = begin
                tau = deepcopy($tau)
            end
            j = minimum(run(b, samples=100)).time
            n = laed6_time!(kniter, orgati, rho, d_copy, z_copy, finit, tau_ccal, info_ccal)
            accum_jul += j
            accum_lapk += n
        end
        push!(jul, accum_jul/10)
        push!(lapk, accum_lapk/10)
    end

    xs = Vector(1:10:500)
    plt = plot(xs, jul, label="laed6!", yscale=:log10)
    plot!(plt, xs, lapk, label="lapack laed6")
    savefig(plt, "../images/laed6_timings_$(T).png")

end
