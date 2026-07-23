# Shared numerical building blocks used by compression and the future
# factor-space GEMM merge.

function cholqr_factor_fixture(::Type{T}, ArrayType; rank_deficient::Bool=false) where {T}
    rng = MersenneTwister(510)
    m, width, count = 48, 8, 3
    X = randn(rng, T, m, width, count)
    if rank_deficient
        X[:, 5:8, :] .= X[:, 1:4, :]
    end

    Thi = _TLRM.tlr_orthogonalization_type(T)
    Q = ArrayType(X)
    V = ArrayType(zeros(T, width, width, count))
    Y_hi = ArrayType(zeros(Thi, m, width, count))
    G_hi = ArrayType(zeros(Thi, width, width, count))
    R1 = ArrayType(zeros(T, width, width, count))
    R2 = ArrayType(zeros(T, width, width, count))
    multipliers = ArrayType(ones(real(Thi), count))
    ws = _TLRM.CholQR2FactorWorkspace(
        Q, V, Y_hi, G_hi, R1, R2, multipliers,
    )
    return (; X, ws, m, width, count, Thi)
end

@testset "TLR fused numerical primitives on CPU" begin
    @test _TLRM.tlr_orthogonalization_type(Float16) == Float32
    @test _TLRM.tlr_orthogonalization_type(Float32) == Float64
    @test _TLRM.tlr_orthogonalization_type(Float64) == Float64

    rng = MersenneTwister(511)
    X = randn(rng, Float32, 7, 5, 4)
    norm_sq = zeros(Float64, size(X, 3))
    @inferred _TLRM.batch_frobenius_norms_sq!(norm_sq, X)
    @test norm_sq ≈ [sum(abs2, X[:, :, k]) for k in axes(X, 3)]

    for T in (Float32, Float64)
        fixture = cholqr_factor_fixture(T, Array)
        @inferred _TLRM.mixed_cholqr2_factor!(fixture.ws)
        for batch in 1:fixture.count
            Q = fixture.ws.Q[:, :, batch]
            V = fixture.ws.V[:, :, batch]
            recon_tol = T == Float32 ? 5e-6 : 5e-13
            orth_tol = T == Float32 ? 5e-6 : 5e-12
            @test norm(Q * adjoint(V) - fixture.X[:, :, batch]) /
                  norm(fixture.X[:, :, batch]) <= recon_tol
            @test norm(adjoint(Q) * Q - I, Inf) <= orth_tol
        end
    end

    # Rank-deficient panels are revealed by coefficient energy, after which the
    # retained basis is orthonormal and the factor identity is preserved.
    for T in (Float32, Float64)
        fixture = cholqr_factor_fixture(T, Array; rank_deficient=true)
        ranks = zeros(Int32, fixture.count)
        error_sq = fill(-1.0, fixture.count)
        rank_rtol_sq = _TLRM.cholqr_rank_rtol_sq(
            T, fixture.Thi, fixture.m, fixture.width,
        )
        @inferred _TLRM.mixed_cholqr2_compress!(
            fixture.ws, ranks, error_sq, fixture.width, rank_rtol_sq,
        )
        @test ranks == fill(Int32(4), fixture.count)
        for batch in 1:fixture.count
            rank = Int(ranks[batch])
            Q = fixture.ws.Q[:, 1:rank, batch]
            V = fixture.ws.V[:, 1:rank, batch]
            recon_tol = T == Float32 ? 5e-6 : 5e-12
            orth_tol = T == Float32 ? 5e-6 : 5e-11
            @test norm(Q * adjoint(V) - fixture.X[:, :, batch]) /
                  norm(fixture.X[:, :, batch]) <= recon_tol
            @test norm(adjoint(Q) * Q - I, Inf) <= orth_tol
            @test all(iszero, fixture.ws.Q[:, rank+1:end, batch])
            @test all(iszero, fixture.ws.V[:, rank+1:end, batch])
        end
    end

    # Exact-coordinate pruning keeps norm calculation, selection, compaction,
    # rank output, and zero padding in one fused kernel.
    energies = [9.0, 0.01, 4.0, 0.04]
    Q = reshape(Matrix{Float64}(I, 4, 4), 4, 4, 1)
    V = zeros(Float64, 4, 4, 1)
    for col in eachindex(energies)
        V[col, col, 1] = sqrt(energies[col])
    end
    ranks = zeros(Int32, 1)
    error_sq = [0.0]
    _TLRM.prune_orthogonal_columns!(
        Q, V, ranks, error_sq, 4, 4, 0.1, false,
    )
    @test ranks == Int32[2]
    @test error_sq[1] ≈ 0.05
    @test all(iszero, Q[:, 3:4, :])
    @test all(iszero, V[:, 3:4, :])

    # Existing error consumes budget; a hard capacity remains authoritative and
    # reports the resulting tolerance overflow through achieved error.
    Q = reshape(Matrix{Float64}(I, 4, 4), 4, 4, 1)
    V = zeros(Float64, 4, 4, 1)
    for col in eachindex(energies)
        V[col, col, 1] = sqrt(energies[col])
    end
    error_sq[1] = 0.08
    _TLRM.prune_orthogonal_columns!(
        Q, V, ranks, error_sq, 4, 1, 0.1, false,
    )
    @test ranks == Int32[1]
    @test error_sq[1] ≈ 4.13
end

@testset "TLR fused numerical primitives on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            fixture = cholqr_factor_fixture(Float32, ArrayType)
            _TLRM.mixed_cholqr2_factor!(fixture.ws)
            synchronize(fixture.ws.Q)
            Q = Array(fixture.ws.Q)
            V = Array(fixture.ws.V)
            for batch in 1:fixture.count
                @test norm(Q[:, :, batch] * adjoint(V[:, :, batch]) -
                           fixture.X[:, :, batch]) /
                      norm(fixture.X[:, :, batch]) <= 1e-5
                @test norm(adjoint(Q[:, :, batch]) * Q[:, :, batch] - I, Inf) <= 1e-5
            end

            norm_sq = ArrayType(zeros(Float64, fixture.count))
            _TLRM.batch_frobenius_norms_sq!(norm_sq, fixture.ws.Q)
            synchronize(norm_sq)
            @test Array(norm_sq) ≈
                  [sum(abs2, Float64.(Q[:, :, k])) for k in axes(Q, 3)] rtol=1e-10

            deficient = cholqr_factor_fixture(
                Float32, ArrayType; rank_deficient=true,
            )
            ranks = ArrayType(zeros(Int32, deficient.count))
            error_sq = ArrayType(zeros(Float64, deficient.count))
            rank_rtol_sq = _TLRM.cholqr_rank_rtol_sq(
                Float32, deficient.Thi, deficient.m, deficient.width,
            )
            _TLRM.mixed_cholqr2_compress!(
                deficient.ws, ranks, error_sq, deficient.width, rank_rtol_sq,
            )
            synchronize(deficient.ws.Q)
            @test Array(ranks) == fill(Int32(4), deficient.count)
            Qd = Array(deficient.ws.Q)
            Vd = Array(deficient.ws.V)
            for batch in 1:deficient.count
                @test norm(
                    Qd[:, 1:4, batch] * adjoint(Vd[:, 1:4, batch]) -
                    deficient.X[:, :, batch],
                ) / norm(deficient.X[:, :, batch]) <= 1e-5
                @test norm(
                    adjoint(Qd[:, 1:4, batch]) * Qd[:, 1:4, batch] - I,
                    Inf,
                ) <= 1e-5
            end
        end
    end
end
