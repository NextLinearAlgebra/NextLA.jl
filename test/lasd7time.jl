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
const range = 100:10:150
function slasd7_time(icompq::Int64, nl::Int64, nr::Int64, sqre::Int64,
                         k::Ref{Int64}, D::AbstractVector{Float64},
                z::AbstractVector{Float64}, zw::AbstractVector{Float64}, 
                vf::AbstractVector{Float64}, vfw::AbstractVector{Float64},
                vl::AbstractVector{Float64}, vlw::AbstractVector{Float64}, 
                alpha::Ref{Float64}, beta::Ref{Float64}, dsigma::AbstractVector{Float64},
                idx::AbstractVector{Int64}, idxp::AbstractVector{Int64},
                idxq::AbstractVector{Int64}, perm::AbstractVector{Int64},
                givptr::Ref{Int64}, givcol::AbstractMatrix{Int64},
                ldgcol::Int64, givnum::AbstractMatrix{Float64}, ldgnum::Int64,
                c::Ref{Float64}, s::Ref{Float64}, info::Ref{Int64})
        b = @benchmarkable begin
            ccall(
                        (@blasfunc(dlasd7_), libblastrampoline),
                            Cvoid, 
                            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                            Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                            Ptr{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt},
                            Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                            Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ref{Float64}, Ref{BlasInt}),
                            $icompq, $nl, $nr, $sqre, k, D, z, zw,
                            vf, vfw, vl, vlw, $alpha, $beta, dsigma,
                            idx, idxp, idxq, perm, givptr, givcol,
                            $ldgcol, givnum, $ldgnum, c, s, info
                        )
            end setup = begin
                k = Ref{BlasInt}(0)

                D  = deepcopy($D)
                z  = deepcopy($z)
                zw = deepcopy($zw)
                vf = deepcopy($vf)
                vfw = deepcopy($vfw)
                vl = deepcopy($vl)
                vlw = deepcopy($vlw)
                dsigma = deepcopy($dsigma)

                idx  = deepcopy($idx)
                idxp = deepcopy($idxp)
                idxq = deepcopy($idxq)
                perm = deepcopy($perm)

                givptr = Ref{BlasInt}(0)
                givcol = deepcopy($givcol)
                givnum = deepcopy($givnum)

                c = Ref{Float64}(0)
                s = Ref{Float64}(0)
                info = Ref{BlasInt}(0)
            end
            return minimum(run(b, samples=100)).time
end

function slasd7_time(icompq::Int64, nl::Int64, nr::Int64, sqre::Int64,
                         k::Ref{Int64}, D::AbstractVector{Float32},
                z::AbstractVector{Float32}, zw::AbstractVector{Float32}, 
                vf::AbstractVector{Float32}, vfw::AbstractVector{Float32},
                vl::AbstractVector{Float32}, vlw::AbstractVector{Float32}, 
                alpha::Ref{Float32}, beta::Ref{Float32}, dsigma::AbstractVector{Float32},
                idx::AbstractVector{Int64}, idxp::AbstractVector{Int64},
                idxq::AbstractVector{Int64}, perm::AbstractVector{Int64},
                givptr::Ref{Int64}, givcol::AbstractMatrix{Int64},
                ldgcol::Int64, givnum::AbstractMatrix{Float32}, ldgnum::Int64,
                c::Ref{Float32}, s::Ref{Float32}, info::Ref{Int64})
        b = @benchmarkable begin
            ccall(
                        (@blasfunc(slasd7_), libblastrampoline),
                            Cvoid, 
                            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                            Ref{BlasInt}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                            Ptr{Float32}, Ptr{Float32}, Ref{Float32}, Ref{Float32}, Ptr{Float32}, Ptr{BlasInt}, Ptr{BlasInt},
                            Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                            Ptr{Float32}, Ref{BlasInt}, Ref{Float32}, Ref{Float32}, Ref{BlasInt}),
                            $icompq, $nl, $nr, $sqre, k, D, z, zw,
                            vf, vfw, vl, vlw, $alpha, $beta, dsigma,
                            idx, idxp, idxq, perm, givptr, givcol,
                            $ldgcol, givnum, $ldgnum, c, s, info
                        )
        end setup = begin
            k = Ref{BlasInt}(0)

            D  = deepcopy($D)
            z  = deepcopy($z)
            zw = deepcopy($zw)
            vf = deepcopy($vf)
            vfw = deepcopy($vfw)
            vl = deepcopy($vl)
            vlw = deepcopy($vlw)
            dsigma = deepcopy($dsigma)

            idx  = deepcopy($idx)
            idxp = deepcopy($idxp)
            idxq = deepcopy($idxq)
            perm = deepcopy($perm)

            givptr = Ref{BlasInt}(0)
            givcol = deepcopy($givcol)
            givnum = deepcopy($givnum)

            c = Ref{Float32}(0)
            s = Ref{Float32}(0)
            info = Ref{BlasInt}(0)
        end
        return minimum(run(b, samples=100)).time
end

plt = plot(
    ylabel = "Time (ns)",
    xlabel = "Matrix Size (n × n)",
    yscale = :log10
)
for T in [Float32, Float64]
    jul = Float64[]
    lapk = Float64[]
    starting = -T(1e3)
    ending = T(1e3)
    for i in range
        accum_jul = zero(Float64)
        accum_lapk = zero(Float64)
        for l in 1:10
            block_size = i ÷ 2
            nl = block_size - 1
            nr = block_size
            sqre = 0
            k = [0]
            
            n = nl + nr + 1
            m = n + sqre
            
            A = Bidiagonal((starting .+ (ending - starting) .* rand(T, i, i)), :U)
            B1 = A[1:block_size-1, 1:block_size-1]
            B2 = A[block_size+1:end, block_size+1:end]
            U1, D1, V1 = svd(B1)
            U2, D2, V2 = svd(B2)

            D = [D1; 0 ; D2]
            icompq = 1

            z = zeros(T, n)
            zw = zeros(T, m)
            vf = zeros(T, m)
            vf[1:nl] .= V1[1,:]
            vf[nl+1] = 1.5
            vf[nl+2:m] .= V2[1,:]
            vfw = zeros(T,m)
            vl = zeros(T, m)
            vl[1:nl] .= V1[end,:]
            vl[nl+1] = 0.5
            vl[nl+2:m] .= V2[end,:]
            vlw = zeros(T, m)
            
            alpha = rand(T)
            beta = rand(T)
            beta_native = Ref{T}(T(0.5))
            alpha_native = Ref{T}(alpha)
            beta_native = Ref{T}(beta)
            dsigma = zeros(T, n)
            
            idx = zeros(Int64, n)
            idxp = zeros(Int64, n)
            idxq = zeros(Int64, n)
            idxq[1:nl] = reverse(Vector(1:nl))
            idxq[nl+2:end] = reverse(Vector(1:nr))
            perm = zeros(Int64, n)
            givptr = [0]
            ldgcol = n
            ldgnum = n
            givnum = zeros(T, ldgnum, 2)
            givcol = zeros(Int64, ldgcol, 2)
            c = [T(0)]
            s = [T(0)]
            info = [0]
        
            k_native = Ref{BlasInt}(T(0))
            D_native  = deepcopy(D)
            z_native = deepcopy(z)
            zw_native = deepcopy(zw)
            vf_native = deepcopy(vf)
            vfw_native = deepcopy(vfw)
            vl_native = deepcopy(vl)
            vlw_native = deepcopy(vlw)
            dsigma_native = deepcopy(dsigma)
            idx_native = deepcopy(idx)
            idxp_native = deepcopy(idxp)
            idxq_native = deepcopy(idxq)
            perm_native = deepcopy(perm)
            givptr_native = Ref{BlasInt}(0)
            givcol_native = deepcopy(givcol)
            givnum_native = deepcopy(givnum)
            c_native = Ref{T}(T(0))
            s_native = Ref{T}(T(0))
            info_native = Ref{BlasInt}(T(0))

            b = @benchmarkable begin
                NextLA.lasd7!(
                    $icompq, $nl, $nr, $sqre,
                    k, D, z, zw, vf, vfw, vl, vlw,
                    $alpha, $beta, dsigma,
                    idx, idxp, idxq, perm,
                    givptr, givcol, $ldgcol,
                    givnum, $ldgnum, c, s, info
                )
            end setup = begin
                k       = deepcopy($k)
                D       = deepcopy($D)
                z       = deepcopy($z)
                zw      = deepcopy($zw)
                vf      = deepcopy($vf)
                vfw     = deepcopy($vfw)
                vl      = deepcopy($vl)
                vlw     = deepcopy($vlw)
                dsigma  = deepcopy($dsigma)

                idx     = deepcopy($idx)
                idxp    = deepcopy($idxp)
                idxq    = deepcopy($idxq)
                perm    = deepcopy($perm)

                givptr  = deepcopy($givptr)
                givcol  = deepcopy($givcol)
                givnum  = deepcopy($givnum)

                c       = deepcopy($c)
                s       = deepcopy($s)
                info    = deepcopy($info)
            end

            j = minimum(run(b, samples=100)).time

            n = slasd7_time(icompq, nl, nr, sqre, k_native, D_native, z_native, zw_native,
                            vf_native, vfw_native, vl_native, vlw_native, alpha_native, beta_native, dsigma_native,
                            idx_native, idxp_native, idxq_native, perm_native, givptr_native, givcol_native,
                            ldgcol, givnum_native, ldgnum, c_native, s_native, info_native)

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
        label="lasd7! $(T)",
        linestyle = (T == Float32 ? :solid : :dot),
        marker = :circle,
        color = :blue
        )
    plot!(
        plt, xs,
        lapk,
        label="lapack lasd7 $(T)",
        linestyle = (T == Float32 ? :dash : :dashdot),
        marker = :circle,
        color = :orange
        )
        
end
savefig(plt, "../images/lasd7_timings.png")
