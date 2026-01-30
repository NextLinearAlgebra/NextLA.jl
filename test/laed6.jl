using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test



@testset "laed6! test random input" begin

    for T in [Float32, Float64]
        starting = T(-1e3)
        ending = T(1e3)
        for i in 1:50
            
            kniter = i%2 == 0 ? Int64(1) : Int64(2)
            orgati = i%2 == 0 ? true : false
            d1 = starting + (ending - starting)*rand(T)
            d2 = d1 + (ending - d1)*rand(T)
            d3 = d2 + (ending - d2)*rand(T)
            d = [d1, d2, d3]
            d_copy = deepcopy(d)
            z = one(T) .+ (ending - one(T)).*rand(T, 3)
            z_copy = deepcopy(z)
            rho = starting + (ending - starting)*rand(T)

            finit = rho + sum(z./d)
            tau = T[0]
            tau_ccal = Ref{T}(T(0))
            info_ccal = Ref{Int64}(Int64(0))
            info = Int64[0]

            NextLA.laed6!(kniter, orgati, rho, d, z, finit, tau, info)
            if T == Float32
                ccall(
                    (@blasfunc(slaed6_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt}, Ref{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{BlasInt}),
                        kniter, orgati, rho, d_copy, z_copy, finit, tau_ccal, info_ccal
                    )
            else
                ccall(
                    (@blasfunc(dlaed6_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt}, Ref{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{BlasInt}),
                        kniter, orgati, rho, d_copy, z_copy, finit, tau_ccal, info_ccal
                    )

            end

            @test info[1] == info_ccal[]
            @test tau[1] == tau_ccal[]
        end
    end

end
