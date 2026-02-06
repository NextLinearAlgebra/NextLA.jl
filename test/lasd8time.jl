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
const range = 16:16:3200

function slasd8_time!(icompq::Int64, k::Int64, d::AbstractVector{Float64}, z::AbstractVector{Float64},
                vf::AbstractVector{Float64}, vl::AbstractVector{Float64}, difl::AbstractVector{Float64},
                difr::AbstractMatrix{Float64}, lddifr::Int64,
                dsigma::AbstractVector{Float64}, work::AbstractVector{Float64}, info::Ref{Int64})
        b = @benchmarkable begin ccall(
                    (@blasfunc(dlasd8_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
                        Ref{BlasInt}),
                        $icompq, $k, pointer(d), pointer(z), pointer(vf), 
                        pointer(vl), pointer(difl), pointer(difr), 
                        $lddifr,
                        pointer($dsigma), pointer(work), $info
                    )
        end setup = begin
            d = deepcopy($d)
            z = deepcopy($z)
            vf = deepcopy($vf)
            vl = deepcopy($vl)
            difl = deepcopy($difl)
            difr = deepcopy($difr)
            work = deepcopy($work)
        end

        return minimum(run(b, samples=100)).time
end

function slasd8_time!(icompq::Int64, k::Int64, d::AbstractVector{Float32}, z::AbstractVector{Float32},
                vf::AbstractVector{Float32}, vl::AbstractVector{Float32}, difl::AbstractVector{Float32},
                difr::AbstractMatrix{Float32}, lddifr::Int64,
                dsigma::AbstractVector{Float32}, work::AbstractVector{Float32}, info::Ref{Int64})
        b =  @benchmarkable begin ccall(
                    (@blasfunc(slasd8_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{BlasInt}, Ptr{Float32}, Ptr{Float32},
                        Ref{BlasInt}),
                        $icompq, $k, pointer(d), pointer(z), pointer(vf), 
                        pointer(vl), pointer(difl), pointer(difr), 
                        $lddifr,
                        pointer($dsigma), pointer(work), $info
                    ) 
        end setup = begin
            d = deepcopy($d)
            z = deepcopy($z)
            vf = deepcopy($vf)
            vl = deepcopy($vl)
            difl = deepcopy($difl)
            difr = deepcopy($difr)
            work = deepcopy($work)
        end

        return minimum(run(b, samples=100)).time
end
plt = plot(
    ylabel = "Time (ns)",
    xlabel = "Vector Input Size",
    yscale = :log10
)
for T in [Float32, Float64]
    jul = Float64[]
    lapk = Float64[]
    starting = -(floatmax(T)/T(1e10))
    ending = (floatmax(T)/T(1e10))
    for i in range
        accum_jul = zero(Float64)
        accum_lapk = zero(Float64)
        for l in 1:10
            icompq = 1
            k = i
            d = zeros(T, k)
            d_copy = deepcopy(d)

            z = starting .+ (ending - starting).*rand(T, k)
            z_copy = deepcopy(z)

            vf = starting .+ (ending - starting).*rand(T, k)
            vf_copy = deepcopy(vf)

            vl = starting .+ (ending - starting).*rand(T, k)
            vl_copy = deepcopy(vl)
            lddifr = k

            dsigma =  (ending).*rand(T, k)
            sort!(dsigma)
            
            dsigma_copy = deepcopy(dsigma)
            difl = zeros(T, k+1)
            difl_copy = deepcopy(difl)
            difr = zeros(T, lddifr, 2)
            difr_copy = deepcopy(difr)
            work = zeros(T, 3*k)
            work_copy = deepcopy(work)
            info = Int64[0]
            info_copy = Ref{Int64}(0)

            b = @benchmarkable begin
                 NextLA.lasd8!($icompq, $k, d, z, vf, vl, difl, difr, 
                                        $lddifr, $dsigma, work, $info)
            end setup = begin
                d = deepcopy($d)
                z = deepcopy($z)
                vf = deepcopy($vf)
                vl = deepcopy($vl)
                difl = deepcopy($difl)
                difr = deepcopy($difr)
                work = deepcopy($work)
            end
            j = minimum(run(b, samples=100)).time
            n = slasd8_time!(icompq, k, (d_copy), (z_copy), (vf_copy), 
                            (vl_copy), (difl_copy), (difr_copy), 
                            lddifr,
                            (dsigma_copy), (work_copy), info_copy)
            accum_jul += j
            accum_lapk += n
        end
        push!(jul, accum_jul/10)
        push!(lapk, accum_lapk/10)
    end
    xs = Vector(range)
    plot!(
        plt,
        xs, jul, 
        label="lasd8! $(T)",
        linestyle = (T == Float32 ? :solid : :dot),
        marker = :circle,
        color = :blue
        )
    plot!(
        plt, xs,
        lapk,
        label="lapack lasd8 $(T)",
        linestyle = (T == Float32 ? :dash : :dashdot),
        marker = :circle,
        color = :orange
        )
end
savefig(plt, "../images/lasd8_timings.png")
