@testset "row-basis coefficient accumulation" begin
    rng = MersenneTwister(906)
    T = Float32
    b, K, rA, rB = 20, 5, 3, 4
    Vrow = randn(rng, T, b, rA, K)
    Wcol = randn(rng, T, b, rB, K)
    Zcol = randn(rng, T, b, rB, K)
    for t in (2, 5)
        P = randn(rng, T, t, rA, K)
        ws = _TLRM.CoefficientWorkspace(Vrow, P, Wcol, t; q=2)
        M = _TLRM.accumulate_row_coefficients!(ws, Vrow, P, Wcol, Zcol)
        reference = zeros(T, b, t)
        for k in 1:K
            reference .+= Zcol[:, :, k] * (P[:, :, k] *
                                             (Vrow[:, :, k]' * Wcol[:, :, k]))'
        end
        @test M ≈ reference rtol=2e-5 atol=2e-5
    end
end
