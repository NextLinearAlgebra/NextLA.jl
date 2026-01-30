using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test
const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

@testset "lasd8! test random input" begin

    for T in [Float32, Float64]
    # for T in [Float32,]
    # for T in [Float64,]
        starting = T(-1e3)
        ending = T(1e3)
        for i in 1:1
            

            icompq = 1
            # k = i
            # k = 17
            k = 17
            d = zeros(T, k)
            d_copy = deepcopy(d)
            # z = starting .+ (ending - starting).*rand(T, k)
            # z = T[-959.886, 717.99744, 265.49182, -30.531738, -414.48187, 33.175903, 789.5238, 931.7551, -562.81384, 77.800415, 29.238403, -960.8655, 2.5304565, 177.76477, -824.39636, 615.2169, 975.6184]
            z = T[-762.16876, -127.85303, 80.60205, -168.76508, 983.01526, 347.45764, 193.02112, -565.4286, -15.068665, 905.0591, 61.184326, -124.61975, 297.76526, -823.17676, 831.71045, -342.93256, -568.7636]
            z_copy = deepcopy(z)
            # vf = starting .+ (ending - starting).*rand(T, k)
            # vf = T[120.39636, -47.67981, 208.86206, -119.29962, -676.91266, -552.96497, 358.6162, -46.035156, 905.89685, -776.54175, -88.48761, 132.92444, 105.46741, 850.5596, 942.8429, 717.2937, -910.17865]
            vf = T[-718.2112, -685.78687, 902.95715, 841.5818, 39.50403, -317.62695, 843.52637, -593.4266, -531.8854, 288.29077, -152.20868, 331.30835, -431.97943, -871.33154, 603.4518, 992.00183, -754.64667]
            vf_copy = deepcopy(vf)
            # vl = starting .+ (ending - starting).*rand(T, k)
            # vl = T[516.0597, 239.63855, 328.4585, 143.31982, 154.00806, -86.503845, 923.28955, -311.9662, 986.10974, 618.6725, 790.11914, 874.1074, -720.7264, -306.55548, 269.45422, -818.05493, -733.5788]
            vl = T[131.84741, 120.65222, 843.07056, -976.0705, 672.1792, -496.48334, 450.07166, -693.3106, 61.47119, 212.8103, -75.798035, -260.83252, -999.4618, -511.93558, 732.13574, -247.20404, 988.63306]
            vl_copy = deepcopy(vl)
            lddifr = k

            # dsigma =  (ending).*rand(T, k)
            # dsigma =  T[47.677578, 50.599514, 68.04288, 302.89996, 345.92264, 352.9104, 405.0141, 416.69415, 449.67084, 643.9484, 690.61957, 731.4207, 786.68274, 830.3885, 889.6413, 909.3689, 929.74365]
            dsigma =  T[32.52858, 42.87666, 175.74442, 272.82834, 307.2124, 345.33005, 365.13483, 386.7036, 403.5517, 444.9117, 481.3782, 569.0638, 588.5782, 596.7156, 599.02826, 745.28906, 762.1857]
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

            
            # println("k: $k")
            # println("z: $z")
            # println("vf: $vf")
            # println("vl: $vl")
            # println("dsigma: $dsigma")
            # println()
            
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
            NextLA.slasd8!(icompq, k, d, z, vf, vl, difl, difr, lddifr, dsigma, work, info)
            # println("$T info: $(info)")
            # println("$T info_copy: $(info_copy[])")
            # println("$T work: $(work)")
            # println("$T work_copy: $(work_copy)")
            # println()
            # println("$T d: $(d)")
            # println("$T d_copy: $(d_copy)")

            # println()
            # println()

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
            @test isapprox(d, d_copy)
            @test isapprox(z, z_copy)
            @test isapprox(vf, vf_copy)
            @test isapprox(vl, vl_copy)
            @test isapprox(difl, difl_copy)
            @test isapprox(difr, difr_copy)
            @test isapprox(work, work_copy)
            @test info[1] == info_copy[]
            # @test isapprox(delta, delta_copy)
            # @test isapprox(dsigma[1], dsigma_copy[])
            # @test isapprox(work, work_copy)

        end
    end

end
