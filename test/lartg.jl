@testset "LARTG" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)
        for _ in 1:50
            f = randn(T)
            g = randn(T)
            c, s, r = NextLA.lartg(f, g)
            r_calc = c * f + s * g
            z_calc = -conj(s) * f + c * g
            @test r ≈ r_calc rtol=rtol
            @test abs(z_calc) <= rtol * max(one(real(T)), abs(r)) + eps(real(T))
            @test abs2(r) ≈ (abs2(f) + abs2(g)) rtol=rtol
            @test isfinite(r)
        end

        # different branch coverage
        zeroT = zero(T)
        oneRT = one(real(T))

        c, s, r = NextLA.lartg(zeroT, zeroT)
        @test c == oneRT
        @test s == zeroT
        @test r == zeroT

        f = randn(T)
        c, s, r = NextLA.lartg(f, zeroT)
        @test c == oneRT
        @test s == zeroT
        @test r == f

        g = randn(T)
        c, s, r = NextLA.lartg(zeroT, g)
        @test c == zero(real(T))
        @test abs(r) ≈ abs(g) rtol=rtol
        @test abs(s) ≈ oneRT rtol=rtol
    end
end

for T in (ComplexF32, ComplexF64, Float32, Float64)
    @testset "LARTG LAPACK $T" begin
        rtol = test_rtol(T)
        for _ in 1:50
            f = randn(T)
            g = randn(T)
            c_nla, s_nla, r_nla = NextLA.lartg(f, g)
            c_ref, s_ref, r_ref = lapack_lartg(f, g)

            @test c_nla ≈ c_ref rtol=rtol
            @test s_nla ≈ s_ref rtol=rtol
            @test r_nla ≈ r_ref rtol=rtol
        end
    end
end
