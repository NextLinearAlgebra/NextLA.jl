using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

@testset "lasq6! test random input" begin

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
            emin = eps(T)
            z[4*n] = emin
            z_copy = deepcopy(z)
            pp = (i %2 == 0) ? 1 : 0
            dmin = T[0]
            dmin1 = T[0]
            dmin2 = T[0]
            dn = T[0]
            dnm1 = T[0]
            dnm2 = T[0]
            dmin_copy = Ref{T}(0)
            dmin1_copy = Ref{T}(0)
            dmin2_copy = Ref{T}(0)
            dn_copy = Ref{T}(0)
            dnm1_copy = Ref{T}(0)
            dnm2_copy = Ref{T}(0)
            
            if T == Float32
                ccall(
                    (@blasfunc(slasq6_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ref{BlasInt}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32},),
                        i0, n0, z_copy, pp, dmin_copy, dmin1_copy,
                        dmin2_copy, dn_copy, dnm1_copy, dnm2_copy
                    )
            else
                ccall(
                    (@blasfunc(dlasq6_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ref{BlasInt}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64},),
                        i0, n0, z_copy, pp, dmin_copy, dmin1_copy,
                        dmin2_copy, dn_copy, dnm1_copy, dnm2_copy
                    )
               
            end
            
            NextLA.lasq6!(i0, n0, z, pp, dmin, dmin1,
                        dmin2, dn, dnm1, dnm2)

            @test isapprox(dmin_copy[], dmin[])
            @test isapprox(dmin1_copy[], dmin1[])
            @test isapprox(dmin2_copy[], dmin2[])
            @test isapprox(dn_copy[], dn[])
            @test isapprox(dnm1_copy[], dnm1[])
            @test isapprox(dnm2_copy[], dnm2[])

        end
    end

end
