using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

@testset "lasd8! test random input" begin

    for T in [Float32, Float64]
        starting = T(-1e3)
        ending = T(1e3)
        for i in 1:200 
            

            icompq = 1
            k = i
            d = zeros(T, k)
            d_copy = deepcopy(d)
            d_copy_copy = Float64.(deepcopy(d))
            d_copy_copy_copy = Float64.(deepcopy(d))
            z = starting .+ (ending - starting).*rand(T, k)
            z_copy = deepcopy(z)
            z_copy_copy = Float64.(deepcopy(z))
            z_copy_copy_copy = Float64.(deepcopy(z))
            vf = starting .+ (ending - starting).*rand(T, k)
            vf_copy = deepcopy(vf)
            vf_copy_copy = Float64.(deepcopy(vf))
            vf_copy_copy_copy = Float64.(deepcopy(vf))
            vl = starting .+ (ending - starting).*rand(T, k)
            vl_copy = deepcopy(vl)
            vl_copy_copy = Float64.(deepcopy(vl))
            vl_copy_copy_copy = Float64.(deepcopy(vl))
            lddifr = k

            dsigma =  (ending).*rand(T, k)
            sort!(dsigma)
            
            dsigma_copy = deepcopy(dsigma)
            dsigma_copy_copy = Float64.(deepcopy(dsigma))
            dsigma_copy_copy_copy = Float64.(deepcopy(dsigma))
            difl = zeros(T, k+1)
            difl_copy = deepcopy(difl)
            difl_copy_copy = Float64.(deepcopy(difl))
            difl_copy_copy_copy = Float64.(deepcopy(difl))
            difr = zeros(T, lddifr, 2)
            difr_copy = deepcopy(difr)
            difr_copy_copy = Float64.(deepcopy(difr))
            difr_copy_copy_copy = Float64.(deepcopy(difr))
            work = zeros(T, 3*k)
            work_copy = deepcopy(work)
            work_copy_copy = Float64.(deepcopy(work))
            work_copy_copy_copy = Float64.(deepcopy(work))
            info = Int64[0]
            info_copy = Ref{Int64}(0) 
            
            if T == Float32
                ccall(
                    (@blasfunc(slasd8_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{BlasInt}, Ptr{Float32}, Ptr{Float32},
                        Ref{BlasInt}),
                        icompq, k, pointer(d_copy), pointer(z_copy), pointer(vf_copy), 
                        pointer(vl_copy), pointer(difl_copy), pointer(difr_copy), 
                        lddifr,
                        pointer(dsigma_copy), pointer(work_copy), info_copy
                    )
            else
                ccall(
                    (@blasfunc(dlasd8_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
                        Ref{BlasInt}),
                        icompq, k, pointer(d_copy), pointer(z_copy), pointer(vf_copy), 
                        pointer(vl_copy), pointer(difl_copy), pointer(difr_copy), 
                        lddifr,
                        pointer(dsigma_copy), pointer(work_copy), info_copy
                    )
            end
            NextLA.lasd8!(icompq, k, d, z, vf, vl, difl, difr, lddifr, dsigma, work, info)

            d[isnan.(d)] .= Inf
            d_copy[isnan.(d_copy)] .= Inf
            z[isnan.(z)] .= Inf
            z_copy[isnan.(z_copy)] .= Inf
            vf[isnan.(vf)] .= Inf
            vf_copy[isnan.(vf_copy)] .= Inf
            vl[isnan.(vl)] .= Inf
            vl_copy[isnan.(vl_copy)] .= Inf
            difl[isnan.(difl)] .= Inf
            difl_copy[isnan.(difl_copy)] .= Inf
            difr[isnan.(difr)] .= Inf
            difr_copy[isnan.(difr_copy)] .= Inf
            work[isnan.(work)] .= Inf
            work_copy[isnan.(work_copy)] .= Inf
            if !isapprox(d, d_copy) && T == Float32
                ccall(
                    (@blasfunc(dlasd8_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
                        Ref{BlasInt}),
                        icompq, k, pointer(d_copy_copy), pointer(z_copy_copy), pointer(vf_copy_copy), 
                        pointer(vl_copy_copy), pointer(difl_copy_copy), pointer(difr_copy_copy), 
                        lddifr,
                        pointer(dsigma_copy_copy), pointer(work_copy_copy), info_copy
                    )
                @test isapprox(d, d_copy_copy)
                @test isapprox(z, z_copy_copy)
                @test isapprox(vf, vf_copy_copy)
                @test isapprox(vl, vl_copy_copy)
                @test isapprox(difl, difl_copy_copy)
                @test isapprox(difr, difr_copy_copy)
                @test isapprox(work, work_copy_copy)
                @test info[1] == info_copy[]
            else
                @test isapprox(d, d_copy)
                @test isapprox(z, z_copy)
                @test isapprox(vf, vf_copy)
                @test isapprox(vl, vl_copy)
                @test isapprox(difl, difl_copy)
                @test isapprox(difr, difr_copy)
                @test isapprox(work, work_copy)
                @test info[1] == info_copy[]
            end

        end
    end

end
