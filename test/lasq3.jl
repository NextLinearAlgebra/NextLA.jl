using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

@testset "lasq3! test random input" begin

    for T in [Float32, Float64]
        starting = T(0)
        ending = T(1e3)
        for i in 10:200 
            n = i
            i0 = 1
            n0 = Int64[n]
            n0_copy = Ref{BlasInt}(n0[])

            z =  starting .+ (ending - starting).*rand(T, 4*n)
            d = starting .+ (ending - starting).*rand(T, n)
            e = starting .+ (ending - starting).*rand(T, n - 1)

            for i in 1:n
                z[2*i-i] = d[i]^2
                if i < n
                    z[2*i] = e[i]^2
                end
            end
            z_copy = deepcopy(z)
            pp = Int64[(i % 3 == 0) ? 1 : ((i % 3 == 1) ? 0 : 2)]
            pp_copy = Ref{BlasInt}(pp[])
            n0in = n


            dmin = T[0]
            dmin_copy = Ref{T}(0)
            sigma = T[0]
            sigma_copy = Ref{T}(0)
            desig = T[0]
            desig_copy = Ref{T}(0)

            qmax = starting + (ending - starting)*rand(T)

            nfail = Int64[0]
            nfail_copy = Ref{BlasInt}(0)

            iter = Int64[0]
            iter_copy = Ref{BlasInt}(0)

            ndiv = Int64[0]
            ndiv_copy = Ref{BlasInt}(0)

            ieee = (i % 2 == 0) ? true : false
            ieee_i = Ref{BlasInt}(ieee ? 1 : 0)

            ttype = Int64[0]
            ttype_copy = Ref{BlasInt}(0)

            dmin1 = T[minimum([d[1:n0[]-1]; d[n0[]+1:end]])]
            dmin1_copy = Ref{T}(dmin1[])
            dmin2 = T[minimum([d[1:n0[]-2]; d[n0[]+1:end]])]
            dmin2_copy = Ref{T}(dmin2[])
            dn = T[d[n]]
            dn_copy = Ref{T}(dn[])
            dn1 = T[d[n-1]]
            dn1_copy = Ref{T}(dn1[])
            dn2 = T[d[n-2]]
            dn2_copy = Ref{T}(dn2[])
            g = T[starting + (ending - starting)*rand(T)]
            g_copy = Ref{T}(g[])
            tau = T[starting + (ending - starting)*rand(T)]
            tau_copy = Ref{T}(tau[])
            
            if T == Float32
                ccall(
                    (@blasfunc(slasq3_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ref{BlasInt}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}),
                        i0, n0_copy, z_copy,
                        pp_copy, dmin_copy, sigma_copy,
                        desig_copy, qmax, nfail_copy,
                        iter_copy, ndiv_copy, ieee_i,
                        ttype_copy, dmin1_copy, dmin2_copy,
                        dn_copy, dn1_copy, dn2_copy,
                        g_copy, tau_copy,
                    )
            else
                ccall(
                    (@blasfunc(dlasq3_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ref{BlasInt}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}),
                        i0, n0_copy, z_copy, pp_copy, dmin_copy, 
                        sigma_copy, desig_copy, qmax, nfail_copy,
                        iter_copy, ndiv_copy, ieee_i, ttype_copy,
                        dmin1_copy, dmin2_copy, dn_copy, dn1_copy,
                        dn2_copy, g_copy, tau_copy,
                    )
               
            end
            
            NextLA.lasq3!(i0, n0, z, pp, dmin, 
                        sigma, desig, qmax, nfail,
                        iter, ndiv, ieee, ttype,
                        dmin1, dmin2, dn, dn1,
                        dn2, g, tau)

            @test isapprox(n0[], n0_copy[])
            @test isapprox(z, z_copy)
            @test isapprox(pp[], pp_copy[])
            @test isapprox(dmin[], dmin_copy[])
            @test isapprox(sigma[], sigma_copy[])
            @test isapprox(desig[], desig_copy[])
            @test isapprox(nfail[], nfail_copy[])
            @test isapprox(ndiv[], ndiv_copy[])
            @test isapprox(ttype[], ttype_copy[])
            @test isapprox(dmin1[], dmin1_copy[])
            @test isapprox(dmin2[], dmin2_copy[])
            @test isapprox(dn[], dn_copy[])
            @test isapprox(dn1[], dn1_copy[])
            @test isapprox(dn2[], dn2_copy[])
            @test isapprox(g[], g_copy[])
            @test isapprox(tau[], tau_copy[])
        end
    end

end
