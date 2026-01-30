using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

@testset "laed6! test random input" begin

    for T in [Float32, Float64]
        starting = T(-1e3)
        ending = T(1e3)
        for i in 1:100
            
            i = i%2 == 0 ? Int64(1) : Int64(2)
            orgati = i%2 == 0 ? true : false
            d1 = ending*rand(T)
            d2 = d1 + (ending - d1)*rand(T)
            d = [d1, d2]
            d_copy = deepcopy(d)
            z = normalize(one(T) .+ (ending - one(T)).*rand(T, 2))

            z_copy = deepcopy(z)
            delta = zeros(T, 2)
            delta_copy = deepcopy(delta)
            rho = (ending)*rand(T)
            dsigma = T[0]
            dsigma_copy = Ref{T}(T(0))
            work = zeros(T, 2)
            work_copy = deepcopy(work)

            


            NextLA.lasd5!(i, d, z, delta, rho, dsigma, work)
            if T == Float32
                ccall(
                    (@blasfunc(slasd5_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ptr{Float32},Ptr{Float32},
                        Ptr{Float32}, Ref{Float32}, Ref{Float32},
                        Ptr{Float32}),
                        i, d_copy, z_copy, delta_copy, rho, dsigma_copy, work_copy
                    )
            else
                ccall(
                    (@blasfunc(dlasd5_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ptr{Float64},Ptr{Float64},
                        Ptr{Float64}, Ref{Float64}, Ref{Float64},
                        Ptr{Float64}),
                        i, d_copy, z_copy, delta_copy, rho, dsigma_copy, work_copy
                    )

            end

            @test isapprox(delta, delta_copy)
            @test isapprox(dsigma[1], dsigma_copy[])
            @test isapprox(work, work_copy)

        end
    end

end
