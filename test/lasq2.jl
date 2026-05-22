using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test

@testset "lasq2! test random input" begin
    for T in (Float32, Float64)
        for n in 3:200
            # Build valid qd input in LAPACK storage:
            # z[1], z[3], ... are q's and z[2], z[4], ... are e's.
            d = rand(T, n) .* T(1e3)
            e = rand(T, n - 1) .* T(1e3)

            z = zeros(T, 4 * n)
            for i in 1:n
                z[2 * i - 1] = d[i]^2
                if i < n
                    z[2 * i] = e[i]^2
                end
            end

            z_copy = deepcopy(z)
            info = BlasInt[0]
            info_copy = Ref{BlasInt}(0)
            n_ref = Ref{BlasInt}(n)

            if T == Float32
                ccall(
                    (@blasfunc(slasq2_), libblastrampoline),
                    Cvoid,
                    (Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt}),
                    n_ref, z_copy, info_copy
                )
            else
                ccall(
                    (@blasfunc(dlasq2_), libblastrampoline),
                    Cvoid,
                    (Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
                    n_ref, z_copy, info_copy
                )
            end

            NextLA.lasq2!(n, z, info)

            @test info[] == info_copy[]
            if iszero(info_copy[])
                # SLASQ2/DLASQ2 only define z[1:n] (eigenvalues) and z[2n+1:2n+5] (stats) on success;
                # z[n+1:2n] and z[2n+6:4n] are scratch. Pure Julia vs Fortran differ in ulp-level
                # eigenvalues and scratch; compare the documented outputs with tolerances.
                rtol = T == Float32 ? 5.0f-5 : 1.0e-10
                atol = T == Float32 ? 5.0f-3 : 1.0e-8
                @test isapprox(z[1:n], z_copy[1:n]; rtol=rtol, atol=atol)
                @test isapprox(z[2 * n + 1], z_copy[2 * n + 1]; rtol=rtol, atol=atol)
                @test isapprox(z[2 * n + 2], z_copy[2 * n + 2]; rtol=rtol, atol=atol)
                # Iteration counters / NFAIL ratio can differ by O(1) when dqds branches differ in ulp;
                # still check they stay in the same ballpark as the reference LAPACK run.
                rtol_diag = T == Float32 ? 3.0f-1 : 2.0e-1
                atol_diag = T == Float32 ? 1.0f1 : 1.0e1
                @test isapprox(z[(2 * n + 3):(2 * n + 5)], z_copy[(2 * n + 3):(2 * n + 5)]; rtol=rtol_diag, atol=atol_diag)
            else
                @test isapprox(z, z_copy; rtol=T(0.05), atol=T(0.05))
            end
        end
    end
end
