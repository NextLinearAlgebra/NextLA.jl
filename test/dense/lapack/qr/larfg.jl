for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "LARFG [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
                rtol = test_rtol(T)

                @testset "n=$n" for n in [1, 2, 5, 10, 50]
                    alpha = randn(T)
                    x = randn(T, n - 1)
                    x_test = ArrayType(copy(x))

                    alpha_out, tau = NextLA.larfg!(n, alpha, x_test, 1, zero(T))
                    synchronize(x_test)

                    orig_norm = sqrt(abs(alpha)^2 + norm(x)^2)
                    @test abs(alpha_out) ≈ orig_norm rtol=rtol

                    if n > 1
                        x_cpu = Array(x_test)
                        v = vcat(one(T), x_cpu)
                        H = Matrix{T}(I, n, n) - tau * (v * v')
                        result = H' * vcat(alpha, x)
                        @test abs(result[1]) ≈ abs(alpha_out) rtol=rtol
                        @test norm(result[2:end]) < rtol * orig_norm + eps(real(T))
                    end

                    @test isfinite(alpha_out)
                    @test isfinite(tau)
                end

                @testset "Convenience interface" begin
                    alpha = randn(T)
                    x = ArrayType(randn(T, 4))
                    alpha2, tau2 = NextLA.larfg!(alpha, x, 1, zero(T))
                    synchronize(x)
                    @test isfinite(alpha2)
                    @test isfinite(tau2)
                end
        end
    end
end

@testset "LARFG Edge cases" begin
    for T in TEST_TYPES
        a, t = NextLA.larfg!(1, T(3), T[], 1, zero(T))
        @test t == zero(T)
        @test a == T(3)
        a, t = NextLA.larfg!(0, T(1), T[], 1, zero(T))
        @test t == zero(T)
        if T <: Real
            a, t = NextLA.larfg!(3, T(5), zeros(T, 2), 1, zero(T))
            @test t == zero(T)
        end
    end
end

for T in (ComplexF32, ComplexF64, Float32, Float64)
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "LARFG LAPACK $T [$backend_name]" begin
                rtol = test_rtol(T)
                for n in [2, 5, 10]
                    alpha = randn(T)
                    x = randn(T, n - 1)
                    x_nla = ArrayType(copy(x))
                    a_nla, tau_nla = NextLA.larfg!(n, alpha, x_nla, 1, zero(T))
                    synchronize(x_nla)

                    lapack_vec = vcat([alpha], x)
                    tau_ref, a_ref = lapack_larfg!(lapack_vec)
                    x_ref = lapack_vec[2:end]

                    @test abs(a_nla) ≈ abs(a_ref) rtol=rtol
                    @test abs(tau_nla) ≈ abs(tau_ref) rtol=rtol
                    if n > 1
                        @test norm(abs.(Array(x_nla)) .- abs.(x_ref)) < rtol * max(1.0, norm(x_ref))
                    end
                end
            end
        end
    end
