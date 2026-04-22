using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

@testset "lasq5! test random input" begin

    for T in [Float32, Float64]
        starting = T(0)
        ending = T(1e3)
        for i in 10:200 
            n = i
            i0 = 1
            n0 = n

            z = zeros(T, 4*n)
            d = starting .+ (ending - starting).*rand(T, n)
            e = starting .+ (ending - starting).*rand(T, n - 1)

            for i in 1:n
                z[2*i-i] = d[i]^2
                if i < n
                    z[2*i] = e[i]^2
                end
            end
            tau =  starting .+ (ending - starting).*rand(T)
            sigma =  starting .+ (ending - starting).*rand(T)
            eps_s =  T(0.5).*rand(T)
            dmin = T[0]
            dmin1 = T[0]
            dmin2 = T[0]
            dn = T[0]
            dnm1 = T[0]
            dnm2 = T[0]
            pp = (i % 2 == 0) ? 1 : 0
            ieee = 0
            if i %4 == 0
                ieee = true
            elseif i % 4 == 1
                ieee = false
            elseif i % 4 == 2
                ieee = true
            elseif i % 4 == 3
                ieee = false
            end
            sigma =  rand(T)

            z_copy = deepcopy(z)
            pp = (i %2 == 0) ? 1 : 0
            n0in = n


            dmin_copy = Ref{T}(0)
            dmin1_copy = Ref{T}(0)
            dmin2_copy = Ref{T}(0)
            dn_copy = Ref{T}(0)
            dnm1_copy = Ref{T}(0)
            dnm2_copy = Ref{T}(0)

            if T == Float32
                ccall(
                    (@blasfunc(slasq5_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ref{BlasInt}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{BlasInt}, Ref{Float32}),
                        i0, n0, z_copy, pp, tau, sigma, dmin_copy,
                        dmin1_copy, dmin2_copy, dn_copy, dnm1_copy,
                        dnm2_copy, ieee, eps_s
                        )
                else
                    ccall(
                        (@blasfunc(dlasq5_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ref{BlasInt}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{BlasInt}, Ref{Float64}),
                        i0, n0, z_copy, pp, tau, sigma, dmin_copy,
                        dmin1_copy, dmin2_copy, dn_copy, dnm1_copy,
                        dnm2_copy, ieee, eps_s
                    )
               
            end
            
            NextLA.lasq5!(i0, n0, z_copy, pp, tau, sigma, dmin,
                        dmin1, dmin2, dn, dnm1,
                        dnm2, ieee, eps_s)

            @test isapprox(dmin[], dmin_copy[])
            @test isapprox(dmin1[], dmin1_copy[])
            @test isapprox(dmin2[], dmin2_copy[])
            @test isapprox(dn[], dn_copy[])
            @test isapprox(dnm1[], dnm1_copy[])
            @test isapprox(dnm2[], dnm2_copy[])
        end
    end

end
