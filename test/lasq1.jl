using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

@testset "lasq1! test random input" begin
    for T in (Float32, Float64)
        for n in 3:200
            d = rand(T, n) .* T(1e3)
            # LAPACK interface uses E dimension N; only first N-1 are bidiagonal input
            e = zeros(T, n)
            e[1:(n - 1)] .= rand(T, n - 1) .* T(1e3)
            work = zeros(T, 4 * n)

            d_ref = copy(d)
            e_ref = copy(e)
            work_ref = copy(work)
            info = BlasInt[0]
            info_ref = Ref{BlasInt}(0)
            n_ref = Ref{BlasInt}(n)

            if T == Float32
                ccall(
                    (@blasfunc(slasq1_), libblastrampoline),
                    Cvoid,
                    (Ref{BlasInt}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ref{BlasInt}),
                    n_ref, d_ref, e_ref, work_ref, info_ref
                )
            else
                ccall(
                    (@blasfunc(dlasq1_), libblastrampoline),
                    Cvoid,
                    (Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}),
                    n_ref, d_ref, e_ref, work_ref, info_ref
                )
            end

            NextLA.lasq1!(n, d, e, work, info)

            @test isapprox(info[], info_ref[])
            @test isapprox(d, d_ref)
            @test isapprox(e, e_ref)
        end
    end
end
