@testset "standalone global row basis" begin
    rng = MersenneTwister(902)
    T = Float32
    b, K, rA, rank, S = 24, 5, 4, 3, 3
    Qtrue = Matrix(qr(randn(rng, T, b, rank)).Q[:, 1:rank])
    Ptrue = randn(rng, T, rank, K * rA)
    Ubar = Qtrue * Ptrue
    omega = randn(rng, T, K * rA, S)
    gamma = T[1, 0.7, 1.3, 0.5, 1.1]
    ws = _TLRM.RowBasisWorkspace(Ubar, S)

    result = _TLRM.build_row_basis!(ws, Ubar, omega, gamma;
                                    eps_basis=1e-5, tmax=S)
    @test result.t == rank
    @test norm(result.Q' * result.Q - I, Inf) <= 1e-5
    @test norm(result.Q * result.P - Ubar) / norm(Ubar) <= 2e-5
    @test result.residual_sq[1] <= 1e-7 * sum(abs2, Ubar)

    # Weights select the sketch/covariance directions but P itself remains an
    # unweighted coordinate factor for the input row panel.
    weighted = _TLRM.build_row_basis!(ws, Ubar, omega, gamma;
                                      eps_basis=1e-5, tmax=S)
    unweighted = _TLRM.build_row_basis!(ws, Ubar, omega, ones(T, K);
                                        eps_basis=1e-5, tmax=S)
    @test norm(weighted.Q * weighted.P - Ubar) / norm(Ubar) <= 2e-5
    @test norm(unweighted.Q * unweighted.P - Ubar) / norm(Ubar) <= 2e-5

    @test_throws DimensionMismatch _TLRM.build_row_basis!(
        ws, Ubar, omega[:, 1:2], gamma; eps_basis=1e-5,
    )
end

@testset "standalone global row basis on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            rng = MersenneTwister(903)
            T = Float32
            b, K, rA, rank, S = 24, 4, 3, 3, 3
            Qtrue = Matrix(qr(randn(rng, T, b, rank)).Q[:, 1:rank])
            Ptrue = randn(rng, T, rank, K * rA)
            Uhost = Qtrue * Ptrue
            Ubar = ArrayType(Uhost)
            omega = ArrayType(randn(rng, T, K * rA, S))
            gamma = ArrayType(T[1, 0.75, 1.25, 0.5])
            ws = _TLRM.RowBasisWorkspace(Ubar, S)
            result = _TLRM.build_row_basis!(ws, Ubar, omega, gamma;
                                            eps_basis=1e-5, tmax=S)
            synchronize(ws.Q)
            Q = Array(result.Q)
            P = Array(result.P)
            @test result.t == rank
            @test norm(Q' * Q - I, Inf) <= 2e-5
            @test norm(Q * P - Uhost) / norm(Uhost) <= 3e-5
            @test Array(result.residual_sq)[1] <= 1e-7 * sum(abs2, Uhost)
        end
    end
end
