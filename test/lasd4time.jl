ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"
# To ensure that the plot doesn't try to open a window

using BenchmarkTools
using Plots
using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasInt, libblastrampoline
using NextLA
using CSV, DataFrames

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

run_gpu = "gpu" in ARGS

start = 10
stop = 10000010
npts = 15
xs = unique(round.(Int, 10 .^ range(log10(start), log10(stop), length=npts)))
jul_f32 = zeros(Float64, 0)
jul_f64 = zeros(Float64, 0)
lapk_f32 = zeros(Float64, 0)
lapk_f64 = zeros(Float64, 0)
gpu_f32 = zeros(Float32, 0)
gpu_f64 = zeros(Float64, 0)

# xs = unique(2 .^ round.(range(log2(start), log2(stop), length=npts)))
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
function slasd4_time_gpu!(n::Int64, i::Int64, d::AbstractVector{Float64},
                        z::AbstractVector{Float64},
                delta::AbstractVector{Float64}, rho::Float64, 
                sigma::AbstractArray{Float64},
                work::AbstractVector{Float64}, info::Ref{Int64})
        b =  @benchmarkable begin 
            NextLA.lasd4_gpu!($n, $i, $d, z_gpu,
                                        delta_gpu, 
                                        $rho, sigma_gpu, 
                                        work_gpu, info_gpu)
        end setup = begin
            d_gpu = CuArray{Float64}($d)
            z_gpu = CuArray{Float64}($z)
            delta_gpu = CuArray{Float64}($delta)
            sigma_gpu = CuArray{Float64}($sigma)
            work_gpu = CuArray{Float64}($work)
            work_gpu = CuArray{Int64}($info)
        end

        return minimum(run(b, samples=100)).time
end
function slasd4_time_gpu!(n::Int64, i::Int64, d::AbstractVector{Float32},
                        z::AbstractVector{Float32},
                delta::AbstractVector{Float32}, rho::Float32, 
                sigma::AbstractArray{Float32},
                work::AbstractVector{Float32}, info::Ref{Int64})
        b =  @benchmarkable begin 
            NextLA.lasd4_gpu!($n, $i, $d, z_gpu,
                                        delta_gpu, 
                                        $rho, sigma_gpu, 
                                        work_gpu, info_gpu)
        end setup = begin
            d_gpu = CuArray{Float32}($d)
            z_gpu = CuArray{Float32}($z)
            delta_gpu = CuArray{Float32}($delta)
            sigma_gpu = CuArray{Float32}($sigma)
            work_gpu = CuArray{Float32}($work)
            work_gpu = CuArray{Int64}($info)
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

plt = plot(
    ylabel = "Time (ns)",
    xlabel = "Vector Input Size",
    yscale = :log10,
    xscale = :log10,
    legend = :outertopright,
    legendfontsize = 12,
    size=(1600, 900),
    guidefontsize = 14,
    tickfontsize = 12,
    xticks = 10 .^ (1:6),
    yticks = 10 .^ (1:9),
    margin = 10Plots.mm
)

for T in [Float32, Float64]
    jul = Float64[]
    lapk = Float64[]
    gpu = Float64[]
    starting = -(floatmax(T)/T(1e10))
    ending = (floatmax(T)/T(1e10))
    for i in xs
        accum_jul = zero(Float64)
        accum_lapk = zero(Float64)
        accum_gpu = zero(Float64)
        for l in 1:10
            # n = (typeof(i) == Int64) ? i : 
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
            if run_gpu
                m = slasd4_time_gpu!(n, i, (d_copy), (z_copy), (delta_copy), 
                                (rho), (sigma_copy), (work_copy), info_copy)
                accum_gpu += m
            end
            accum_jul += j
            accum_lapk += n
        end
        push!(jul, accum_jul/10)
        push!(lapk, accum_lapk/10)
        if run_gpu
            push!(gpu, accum_gpu/10)
        end
    end

    plot!(
        plt,
        xs, jul, 
        label="lasd4! $(T)",
        linestyle = (T == Float32 ? :solid : :dot),
        marker = (T == Float32 ? :circle : :rect),
        markersize = 5,
        color = :blue,
        )
    plot!(
        plt, xs,
        lapk,
        label="lapack lasd4 $(T)",
        linestyle = (T == Float32 ? :dash : :dashdot),
        marker = (T == Float32 ? :star4 : :octagon),
        markersize = 5,
        color = :orange
        )
    if run_gpu
        plot!(
            plt, xs,
            gpu,
            label="GPU lasd4 $(T)",
            linestyle = (T == Float32 ? :dashdotdot : :dashdotdot),
            marker = (T == Float32 ? :diamond : :star5),
            markersize = 5,
            color = :red
            )
    end
    if T == Float32
        append!(jul_f32, jul)
        append!(lapk_f32, lapk)
        if run_gpu 
            append!(gpu_f32, gpu)
        end
    elseif T == Float64
        append!(jul_f64, jul)
        append!(lapk_f64, lapk)
                if run_gpu 
            append!(gpu_f64, gpu)
        end
    end
    
end

results = DataFrame(
    input_size = xs,
    julia_float32 = jul_f32,
    lapack_float32 = lapk_f32,
    julia_float64 = jul_f64,
    lapack_float64 = lapk_f64
    gpu_float32 = gpu_f32
    gpu_float64 = gpu_f64
)

CSV.write("../timing-data/lasd4_timings.csv", results)
savefig(plt, "../images/lasd4_timings.png")
