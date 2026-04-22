using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"

@testset "lasq4! test random input" begin

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
            n0in = n


            dmin = minimum(d)

            dmin1 = minimum([d[1:n0-1]; d[n0+1:end]])
            dmin2 = minimum([d[1:n0-2]; d[n0+1:end]])
            dn = d[n]
            dn1 = d[n-1]
            dn2 = d[n-2]
            tau = T[0]
            ttype = [0]
            g = T[0]
            tau_copy = Ref{T}(0)
            ttype_copy = Ref{BlasInt}(0)
            g_copy = Ref{T}(0)
            
            if T == Float32
                ccall(
                    (@blasfunc(slasq4_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{BlasInt}, Ref{Float32}),
                        i0, n0, z_copy, pp, n0in, dmin, dmin1,
                        dmin2, dn, dn1, dn2, tau_copy, ttype_copy,
                        g_copy
                    )
            else
                ccall(
                    (@blasfunc(dlasq4_), libblastrampoline),
                        Cvoid,
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{BlasInt}, Ref{Float64}),
                        i0, n0, z_copy, pp, n0in, dmin, dmin1,
                        dmin2, dn, dn1, dn2, tau_copy, ttype_copy,
                        g_copy
                    )
               
            end
            
            NextLA.lasq4!(i0, n0, z_copy, pp, n0in, dmin, dmin1,
                        dmin2, dn, dn1, dn2, tau, ttype,
                        g)

            @test isapprox(tau[], tau_copy[])
            @test isapprox(ttype[], ttype_copy[])
            @test isapprox(g[], g_copy[])
        end
    end

end
