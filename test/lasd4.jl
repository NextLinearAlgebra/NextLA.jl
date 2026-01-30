using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

# const lib = "../OpenBLAS/libopenblas_cooperlakep-r0.3.31.dev.so"
const lib = "../OpenBLAS/libopenblas.so"

function isapprox_nan(a, b)
    if all(isnan.(a)) && all(isnan.(b)) && length(a) == length(b)
        return true
    else
        return isapprox(a, b)
    end
end

@testset "laed4! test random input" begin
    for T in [Float32, Float64]
    
    # for T in [Float32, ]
        starting = T(-1e3)
        ending = T(1e3)
        for i in 1:20
            for j in 3:3:50

		        n = Int64(j)
                i = Int64(trunc(1+rand(T)*(n-1)))
                orgati = i%2 == 0 ? true : false
                prevd = 0
                d = T[]
                for k in 1:n
                    newd = prevd + (ending - prevd)*rand(T)
                    push!(d, newd)
                    prevd = newd
                end
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
                


                if T == Float32
                    ccall(
                        (@blasfunc(slasd4_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{Float32},
                        Ref{Float32}, Ptr{Float32}, Ref{BlasInt}),
                        n, i, d_copy, z_copy, delta_copy, rho, sigma_copy, work_copy, info_copy
                  	)
                else
                    ccall(
                        (@blasfunc(dlasd4_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{Float64},
                        Ref{Float64}, Ptr{Float64}, Ref{BlasInt}),
                        n, i, d_copy, z_copy, delta_copy, rho, sigma_copy, work_copy, info_copy
                        )
                end
                NextLA.lasd4!(n, i, d, z, delta, rho, sigma, work, info)
                @test isapprox_nan(delta, delta_copy)
                @test isapprox_nan(sigma[1], sigma_copy[])
                @test isapprox_nan(work, work_copy)
                @test info[1] == info_copy[]
            end
            for j in 3:3:50

		        n = Int64(j)
                i = Int64(j)
                orgati = i%2 == 0 ? true : false
                prevd = 0
                d = T[]
                for k in 1:n
                    newd = prevd + (ending - prevd)*rand(T)
                    push!(d, newd)
                    prevd = newd
                end
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
                


                if T == Float32
		    
                    ccall(
                        (@blasfunc(slasd4_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float32},
                        Ptr{Float32}, Ptr{Float32}, Ref{Float32},
                        Ref{Float32}, Ptr{Float32}, Ref{BlasInt}),
                        n, i, d_copy, z_copy, delta_copy, rho, sigma_copy, work_copy, info_copy
                  	)
                else
                    ccall(
                        (@blasfunc(dlasd4_), libblastrampoline),
                        Cvoid, 
                        (Ref{BlasInt}, Ref{BlasInt},Ptr{Float64},
                        Ptr{Float64}, Ptr{Float64}, Ref{Float64},
                        Ref{Float64}, Ptr{Float64}, Ref{BlasInt}),
                        n, i, d_copy, z_copy, delta_copy, rho, sigma_copy, work_copy, info_copy
                        )
                end
                NextLA.lasd4!(n, i, d, z, delta, rho, sigma, work, info)
                # if info[1] == info_copy[] && info[1] == 0
                @test isapprox_nan(delta, delta_copy)
                @test isapprox_nan(sigma[1], sigma_copy[])
                @test isapprox_nan(work, work_copy)
                @test info[1] == info_copy[]
                # elseif 
            end
        end
        @test isapprox_nan(T[NaN, NaN], T[NaN, NaN])
    end

end
