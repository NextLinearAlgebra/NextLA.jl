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

function slasd4_time!(n::Int64, i::Int64, d::AbstractVector{Float64},
                        z::AbstractVector{Float64},
                delta::AbstractVector{Float64}, rho::Float64, 
                sigma::Ref{Float64},
                work::AbstractVector{Float64}, info::Ref{Int64})
        b =  @benchmarkable begin ccall(
                        (@blasfunc(dlasd4_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{Float64},
                        Ref{Float64}, Ptr{Float64}, Ref{BlasInt}),
                        $n, $i, $d, $z, delta, $rho, sigma,
                        work, $info
                        )
        end setup = begin
            delta = deepcopy($delta)
            sigma = deepcopy($sigma)
            work = deepcopy($work)
        end

        return minimum(run(b, samples=100)).time
end

function slasd4_time!(n::Int64, i::Int64, d::AbstractVector{Float32},
                        z::AbstractVector{Float32},
                delta::AbstractVector{Float32}, rho::Float32, 
                sigma::Ref{Float32},
                work::AbstractVector{Float32}, info::Ref{Int64})
        b =  @benchmarkable begin ccall(
                        (@blasfunc(slasd4_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{Float32},
                        Ref{Float32}, Ptr{Float32}, Ref{BlasInt}),
                        $n, $i, $d, $z, delta, $rho, sigma,
                        work, $info
                        )
        end setup = begin
            delta = deepcopy($delta)
            sigma = deepcopy($sigma)
            work = deepcopy($work)
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
            n = Int64(i)
            i = Int64(trunc(1+rand(T)*(n-1)))
            orgati = i%2 == 0 ? true : false
            d =  (ending).*rand(T, n)
            sort!(d)
            
            d_copy = deepcopy(d)
            z = normalize(starting .+ (ending - starting).*rand(T, n))
            z_copy = deepcopy(z)
            delta = zeros(T, n)
            delta_copy = deepcopy(delta)
            rho = (ending)*rand(T)
            sigma = T[0]
            sigma_copy = Ref{T}(T(0))
            work = zeros(T, n)
            work_copy = deepcopy(work)
            info = Int64[0]
            info_copy = Ref{Int64}(0)


            b = @benchmarkable begin NextLA.lasd4!($n, $i, $d, $z,
                                        delta, 
                                        $rho, sigma, 
                                        work, $info)
            end setup = begin
                delta = deepcopy($delta)
                sigma = deepcopy($sigma)
                work = deepcopy($work)
            end
            j = minimum(run(b, samples=100)).time
            n = slasd4_time!(n, i, (d_copy), (z_copy), (delta_copy), 
                            (rho), (sigma_copy), (work_copy), info_copy)
            accum_jul += j
            accum_lapk += n
        end
        push!(jul, accum_jul/10)
        push!(lapk, accum_lapk/10)
    end

    xs = Vector(1:10:500)
    plt = plot(xs, jul, label="lasd4!", yscale=:log10)
    plot!(plt, xs, lapk, label="lapack lasd4")
    savefig(plt, "../images/lasd4_timings_$(T).png")

end
