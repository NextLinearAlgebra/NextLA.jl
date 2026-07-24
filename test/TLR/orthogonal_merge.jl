@testset "orthogonal tile merge" begin
    rng = MersenneTwister(907)
    T = Float64
    b, t, rC = 32, 3, 4
    Q = Matrix(qr(randn(rng, T, b, t)).Q[:, 1:t])
    # Make the old basis independent of Q so the residual branch is exercised.
    Uraw = randn(rng, T, b, rC)
    Uraw .-= Q * (Q' * Uraw)
    U = Matrix(qr(Uraw).Q[:, 1:rC])
    M = randn(rng, T, b, t)
    V = randn(rng, T, b, rC)
    alpha, beta = T(1.2), T(-0.7)
    ws = _TLRM.OrthogonalMergeWorkspace(Q, U)
    result = _TLRM.merge_row_basis_tile!(ws, Q, M, U, V;
                                          alpha, beta, eps_sq=0.0,
                                          maxrank=t + rC)
    reference = alpha * Q * M' + beta * U * V'
    @test result.rank == t + rC
    @test norm(result.Q' * result.Q - I, Inf) <= 1e-10
    @test norm(result.Q * result.V' - reference) / norm(reference) <= 2e-10

    # Exact coordinate pruning is safe because the merge produced an
    # orthogonal left factor; the output capacity is authoritative.
    capped = _TLRM.merge_row_basis_tile!(ws, Q, M, U, V;
                                          alpha, beta, eps_sq=0.0,
                                          maxrank=2)
    @test capped.rank == 2
    @test norm(capped.Q' * capped.Q - I, Inf) <= 1e-10
end
