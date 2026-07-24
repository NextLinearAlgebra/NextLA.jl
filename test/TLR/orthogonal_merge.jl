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

# C2a: the batched row merge must match the per-tile reference on identical
# inputs. Slabs deliberately carry different effective old ranks (0, partial,
# full) padded to the shared rcap width with zero tails.
@testset "batched row merge matches per-tile reference" begin
    rng = MersenneTwister(909)
    T = Float64
    bm, bn, t, rcap, g = 32, 24, 3, 4, 3
    Q = Matrix(qr(randn(rng, T, bm, t)).Q[:, 1:t])
    beta = T(-0.7)
    Uolds = Vector{Matrix{T}}(); Volds = Vector{Matrix{T}}()
    for (j, rj) in enumerate((0, 2, 4))
        U = zeros(T, bm, rcap); V = zeros(T, bn, rcap)
        if rj > 0
            Uraw = randn(rng, T, bm, rj)
            Uraw .-= Q * (Q' * Uraw)                    # keep the residual branch busy
            U[:, 1:rj] .= Matrix(qr(Uraw).Q[:, 1:rj])
            V[:, 1:rj] .= randn(rng, T, bn, rj)
        end
        push!(Uolds, U); push!(Volds, V)
    end
    Vm = randn(rng, T, bn, t, g)

    for cap in (t + rcap, 2)
        mws = _TLRM.BatchedMergeWorkspace(Q, rcap, bn, g)
        rvec = zeros(Int32, g); evec = zeros(Float64, g)
        _TLRM.merge_row_block!(mws, Q, Vm, Uolds, Volds, beta, 0.0, false, cap,
                               rvec, evec, _TLRM.default_gemm_compute_mode(T))
        for j in 1:g
            ws = _TLRM.OrthogonalMergeWorkspace(Q, Uolds[j]; bn=bn)
            ref = _TLRM.merge_row_basis_tile!(ws, Q, Vm[:, :, j], Uolds[j], Volds[j];
                                              alpha=one(T), beta, eps_sq=0.0,
                                              maxrank=cap)
            rank = Int(rvec[j])
            @test rank == ref.rank
            recon = mws.Qmerge[:, 1:rank, j] * mws.Vmerge[:, 1:rank, j]'
            @test recon ≈ Matrix(ref.Q) * Matrix(ref.V)' atol=1e-10 rtol=1e-10
            @test evec[j] ≈ _TLRM._merge_error_sq(ws) atol=1e-18 rtol=1e-8
        end
    end
end

# The full-width batching above is only equivalent to per-tile `t + rho_j`
# because the CholQR2 prune zero-pads both factor tails per slab. Pin that
# invariant explicitly.
@testset "cholqr compress zero-pads factor tails per slab" begin
    rng = MersenneTwister(910)
    T = Float64
    bm, rcap, g = 32, 4, 3
    Q = Matrix(qr(randn(rng, T, bm, 2)).Q[:, 1:2])
    mws = _TLRM.BatchedMergeWorkspace(Q, rcap, 32, g)
    for (j, rj) in enumerate((0, 2, 4))
        panel = zeros(T, bm, rcap)
        rj > 0 && (panel[:, 1:rj] .= randn(rng, T, bm, rj))
        copyto!(view(mws.Ures, :, :, j), panel)
    end
    rank_tol = _TLRM.cholqr_rank_rtol_sq(T, _TLRM.tlr_orthogonalization_type(T), bm, rcap)
    fill!(mws.error_sq, 0.0)
    _TLRM.mixed_cholqr2_compress!(mws.chol, mws.ranks, mws.error_sq, rcap, rank_tol)
    for j in 1:g
        rk = Int(mws.ranks[j])
        @test all(iszero, mws.chol.Q[:, (rk + 1):rcap, j])
        @test all(iszero, mws.chol.V[:, (rk + 1):rcap, j])
    end
end
