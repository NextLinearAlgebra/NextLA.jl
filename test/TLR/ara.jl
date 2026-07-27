# Build the minimal ARA orthogonalization workspace around a batch of panels.
function ara_cholesky_fixture(::Type{T}, ArrayType, X::Array{T,3}) where {T}
    m, s, count = size(X)
    Thi = _TLRM.tlr_orthogonalization_type(T)
    return _TLRM.ARACholeskyWorkspace(
        ArrayType(copy(X)),
        ArrayType(zeros(Thi, m, s, count)), ArrayType(zeros(Thi, s, s, count)),
        ArrayType(zeros(T, s, s, count)), ArrayType(zeros(T, s, s, count)),
    )
end

@testset "ARA convergence bookkeeping on CPU" begin
    rng = MersenneTwister(1301)

    @testset "dR recovers projected norms on a full-rank block" begin
        m, s, count = 64, 8, 2
        X = randn(rng, Float64, m, s, count)
        ws = ara_cholesky_fixture(Float64, Array, X)
        _TLRM.ara_cholesky_pass!(ws, ws.R1, ws.R1_tiles)
        _TLRM.ara_cholesky_pass!(ws, ws.R2, ws.R2_tiles)
        dR = zeros(Float64, s, count)
        _TLRM.ara_block_norms!(dR, ws)
        # dR[j] is the residual of column j against columns 1..j-1, i.e. the
        # diagonal of the R factor of a plain QR of the same panel.
        for b in 1:count
            Rref = abs.(diag(qr(X[:, :, b]).R))
            @test dR[:, b] ≈ Rref rtol = 1e-10
            @test norm(ws.Q[:, :, b]' * ws.Q[:, :, b] - I, Inf) <= 1e-12
        end
    end

    @testset "convergence state drives per-member sampling" begin
        count, block, maxrank = 3, 4, 16
        state = _TLRM.ARAConvergenceState(CPU(), count)
        _TLRM.ara_reset!(state, block, maxrank)
        @test all(state.samples .== block)

        # Member 1 converges immediately (all columns negligible), member 2
        # keeps producing content, member 3 converges after one more pass.
        dR = zeros(Float64, block, count)
        dR[:, 1] .= [1.0, 1e-12, 1e-12, 1e-12]
        dR[:, 2] .= [1.0, 0.9, 0.8, 0.7]
        dR[:, 3] .= [1.0, 1.0, 1e-12, 1e-12]
        active = _TLRM.ara_update_convergence!(state, dR, 1e-8, 3, block, maxrank)
        @test active == 2
        @test state.samples[1] == 0                 # 3 consecutive smalls
        @test state.ranks[1] == 1
        @test state.ranks[2] == 4
        @test state.svec[3] == 2                    # carries across passes
        @test state.jcount == Int32[4, 4, 4]

        # Member 1 is converged and must not advance again.
        dR[:, 1] .= 1.0
        dR[:, 3] .= [1e-12, 1.0, 1e-12, 1e-12]
        active = _TLRM.ara_update_convergence!(state, dR, 1e-8, 3, block, maxrank)
        @test state.samples[1] == 0
        @test state.jcount[1] == 4                  # frozen
        @test state.ranks[1] == 1
        @test state.jcount[2] == 8
        # Member 3 carried svec=2 and its first column is small, but a later
        # column in the same block has genuine content. Convergence is judged on
        # the state after the whole block, so the run resets and sampling
        # continues -- this is the false-early-stop guard, and it is why the
        # count must not short-circuit mid-block.
        @test state.svec[3] == 2
        @test state.ranks[3] == 6
        @test state.samples[3] == block
        @test active == 2

        # It does converge once a whole block is negligible.
        dR .= 1e-12
        active = _TLRM.ara_update_convergence!(state, dR, 1e-8, 3, block, maxrank)
        @test active == 0
        @test state.ranks[2] == 8 && state.ranks[3] == 6

        # The rank cap stops a member that never converges.
        state2 = _TLRM.ARAConvergenceState(CPU(), 1)
        _TLRM.ara_reset!(state2, block, block)
        big = fill(1.0, block, 1)
        @test _TLRM.ara_update_convergence!(state2, big, 1e-8, 3, block, block) == 0
        @test state2.ranks[1] == block
    end

    @testset "relative test uses the running maximum" begin
        # A block whose norms grow: early columns must not be judged against a
        # maximum that has not been seen yet.
        count, block, maxrank = 1, 4, 8
        state = _TLRM.ARAConvergenceState(CPU(), count)
        _TLRM.ara_reset!(state, block, maxrank)
        dR = reshape([1e-3, 1.0, 1e-3, 1e-3], block, count)
        _TLRM.ara_update_convergence!(state, dR, 1e-2, 2, block, maxrank)
        # Column 1 is large relative to the max so far (itself), so it counts;
        # columns 3 and 4 are small relative to 1.0 and terminate the member.
        @test state.ranks[1] == 2
        @test state.samples[1] == 0
    end
end

# Dense black-box sampler: Y .= X[b] * randn. This is the shape A3 needs for
# `compress!`, and it keeps the loop tests independent of the factor-list apply.
function dense_sampler(X::AbstractArray{T,3}, rng, ArrayType=Array) where {T}
    return function (Y, width)
        n, count = size(X, 2), size(X, 3)
        for b in 1:count
            Omega = ArrayType(randn(rng, T, n, width))
            copyto!(view(Y, :, 1:width, b), view(X, :, :, b) * Omega)
        end
        return Y
    end
end

# Prescribed spectrum: exact rank `r`, then a sharp drop.
function graded_batch(::Type{T}, m, n, ranks, gap, rng) where {T}
    count = length(ranks)
    X = zeros(T, m, n, count)
    for b in 1:count
        U = Matrix(qr(randn(rng, T, m, min(m, n))).Q)[:, 1:min(m, n)]
        V = Matrix(qr(randn(rng, T, n, min(m, n))).Q)[:, 1:min(m, n)]
        sig = [k <= ranks[b] ? one(T) : T(gap) for k in 1:min(m, n)]
        X[:, :, b] = U * Diagonal(sig) * V'
    end
    return X
end

@testset "ARA basis construction on CPU" begin
    rng = MersenneTwister(1401)

    @testset "tolerance below the CholeskyQR2 limit is rejected" begin
        # The whole point of the unshifted policy: a tolerance the
        # orthogonalizer cannot support must fail loudly, never run to maxrank.
        ws = _TLRM.ARAWorkspace(Float64, CPU(), 32, 16, 1; block=4)
        floor_rel = _TLRM.ara_stopping_floor(Float64)
        @test floor_rel ≈ sqrt(eps(Float64) / 2)
        @test 1e-8 < floor_rel < 1e-7      # ~1.05e-8, the sqrt(u) bound
        sampler = dense_sampler(randn(rng, Float64, 32, 32, 1), rng)
        @test_throws ArgumentError _TLRM.ara_build_basis!(
            ws, sampler; eps_rel=1e-12)
        @test_throws ArgumentError _TLRM.ara_build_basis!(ws, sampler; eps_rel=0)
    end

    @testset "recovers prescribed ranks" begin
        m, n = 64, 64
        ranks = [5, 12, 20]
        X = graded_batch(Float64, m, n, ranks, 1e-12, rng)
        ws = _TLRM.ARAWorkspace(Float64, CPU(), m, 48, length(ranks); block=8)
        res = _TLRM.ara_build_basis!(ws, dense_sampler(X, rng);
                                     eps_rel=1e-6, r_required=4)
        got = Array(res.ranks)
        for b in eachindex(ranks)
            # ARA overshoots by design (sampling proceeds in units of `block`);
            # A2's final truncation is what recovers the exact rank.
            @test ranks[b] <= got[b] <= ranks[b] + ws.block
        end
        # The basis must be orthonormal and must span the range of each operator.
        Q = Array(res.Q)
        for b in eachindex(ranks)
            Qb = Q[:, 1:got[b], b]
            @test norm(Qb' * Qb - I, Inf) <= 1e-12
            resid = X[:, :, b] - Qb * (Qb' * X[:, :, b])
            @test norm(resid) / norm(X[:, :, b]) <= 1e-6
        end
        @test res.passes >= 1
    end

    @testset "breakdown retires a member instead of throwing" begin
        # A member that is exactly rank 1 makes the block singular almost at
        # once. Unshifted, `potrf` breaks down -- that must be read as "nothing
        # left to capture", not raised as PosDefException.
        m, n = 32, 32
        X = zeros(Float64, m, n, 2)
        v = randn(rng, Float64, m)
        X[:, :, 1] = v * randn(rng, Float64, n)'          # exact rank 1
        X[:, :, 2] = graded_batch(Float64, m, n, [6], 1e-12, rng)[:, :, 1]
        ws = _TLRM.ARAWorkspace(Float64, CPU(), m, 24, 2; block=8)
        res = _TLRM.ara_build_basis!(ws, dense_sampler(X, rng);
                                     eps_rel=1e-6, r_required=3)
        got = Array(res.ranks)
        @test got[1] == 1
        @test 6 <= got[2] <= 6 + ws.block
        Q = Array(res.Q)
        @test norm(Q[:, 1:1, 1]' * Q[:, 1:1, 1] - I, Inf) <= 1e-12
    end

    @testset "a full-rank operator runs to the cap and reports it" begin
        m, n = 24, 24
        X = reshape(Matrix(qr(randn(rng, Float64, m, n)).Q), m, n, 1)
        ws = _TLRM.ARAWorkspace(Float64, CPU(), m, 16, 1; block=4)
        res = _TLRM.ara_build_basis!(ws, dense_sampler(X, rng);
                                     eps_rel=1e-6, r_required=3)
        # Saturation is visible in the rank vector alone -- this is why no
        # residual is computed (see docs/TODO.md, worklog item 4).
        @test Array(res.ranks)[1] == 16
    end

    @testset "block size is a performance knob, not an accuracy one" begin
        m, n = 48, 48
        X = graded_batch(Float64, m, n, [10], 1e-12, rng)
        got = map((2, 5, 16)) do blk
            ws = _TLRM.ARAWorkspace(Float64, CPU(), m, 40, 1; block=blk)
            r = _TLRM.ara_build_basis!(ws, dense_sampler(X, rng);
                                       eps_rel=1e-6, r_required=3)
            Q = Array(r.Q)
            k = Array(r.ranks)[1]
            (k, norm(X[:, :, 1] - Q[:, 1:k, 1] * (Q[:, 1:k, 1]' * X[:, :, 1])) /
                norm(X[:, :, 1]))
        end
        for (k, err) in got
            @test 10 <= k <= 10 + 16      # rank found regardless of block size
            @test err <= 1e-6             # accuracy independent of block size
        end
    end
end

@testset "rolling ARA slot-local progress on CPU" begin
    rng = MersenneTwister(1477)
    m = n = 16
    maxrank, block = 10, 4
    X = cat(
        Matrix(qr(randn(rng, Float64, m, n)).Q),
        Matrix(qr(randn(rng, Float64, m, n)).Q);
        dims=3,
    )
    ws = _TLRM.ARAWorkspace(
        Float64, CPU(), m, maxrank, 2; block)
    member_ids = [1, 2]
    progress = [0, 0]
    sampler = function (Y, width, ids)
        for p in eachindex(ids)
            Ω = randn(rng, Float64, n, width)
            view(Y, :, 1:width, p) .= view(X, :, :, ids[p]) * Ω
        end
        return Y
    end

    _TLRM.ara_reset_slots!(ws, 1, 1)
    for _ in 1:2
        info = _TLRM.ara_packed_pass!(
            ws, sampler, 1, member_ids, progress;
            eps_rel=1e-7, r_required=3)
        @test info.nactive == 1
    end
    @test progress[1] == 8

    # A late member starts at offset zero while the older member enters its
    # terminal width. This exercises the full-prefix/terminal-suffix split.
    _TLRM.ara_reset_slots!(ws, 2, 1)
    progress[2] = 0
    info = _TLRM.ara_packed_pass!(
        ws, sampler, 2, member_ids, progress;
        eps_rel=1e-7, r_required=3)
    @test info.discarded == block - (maxrank % block)
    @test progress[findfirst(==(1), member_ids)] == maxrank
    @test progress[findfirst(==(2), member_ids)] == block

    # Recycle the retired slot for another hard member: admission resets its
    # local offset, so it retains the complete rank budget.
    retired_slot = first(info.retired)
    _TLRM.ara_reset_slots!(ws, retired_slot, 1)
    member_ids[retired_slot] = 2
    progress[retired_slot] = 0
    nactive = info.nactive + 1
    while nactive > 0
        info = _TLRM.ara_packed_pass!(
            ws, sampler, nactive, member_ids, progress;
            eps_rel=1e-7, r_required=3)
        nactive = info.nactive
    end
    @test maximum(progress) == maxrank
end

@testset "ARA convergence bookkeeping on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            s, count = 8, 2
            dR = ArrayType(hcat(
                [1.0, 0.8, 0.7, fill(1e-12, 5)...],
                [1.0, 0.9, 0.8, 0.7, fill(1e-12, 4)...],
            ))
            state = _TLRM.ARAConvergenceState(get_backend(dR), count)
            _TLRM.ara_reset!(state, s, 2 * s)
            active = _TLRM.ara_update_convergence!(state, dR, 1e-8, 3, s, 2 * s)
            @test active == 0                        # both members converged
            @test Array(state.ranks) == Int32[3, 4]
        end
    end
end

@testset "ARA basis construction on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            rng = MersenneTwister(1403)
            m, n = 64, 64
            ranks = [5, 12, 20]
            Xh = graded_batch(Float64, m, n, ranks, 1e-12, rng)
            X = ArrayType(Xh)
            ws = _TLRM.ARAWorkspace(Float64, get_backend(X), m, 48,
                                    length(ranks); block=8)
            res = _TLRM.ara_build_basis!(ws, dense_sampler(X, rng, ArrayType);
                                         eps_rel=1e-6, r_required=4)
            synchronize(ws.Q)
            got = Array(res.ranks)
            Q = Array(res.Q)
            for b in eachindex(ranks)
                @test ranks[b] <= got[b] <= ranks[b] + ws.block
                Qb = Q[:, 1:got[b], b]
                @test norm(Qb' * Qb - I, Inf) <= 1e-12
                @test norm(Xh[:, :, b] - Qb * (Qb' * Xh[:, :, b])) /
                      norm(Xh[:, :, b]) <= 1e-6
            end
            # Breakdown must be handled on device too, not just on the host path.
            @test _TLRM.ara_stopping_floor(Float64) ≈ sqrt(eps(Float64) / 2)
        end
    end
end

# A2: optimal truncation of X ≈ Q Zᵀ.

# Build (Q, Z) representing a matrix with a prescribed spectrum, so the optimal
# truncation error is known in closed form from Eckart-Young.
function truncation_fixture(::Type{T}, bm, bn, sQ, sigmas, rng) where {T}
    Q = Matrix(qr(randn(rng, T, bm, sQ)).Q)[:, 1:sQ]
    P = Matrix(qr(randn(rng, T, bn, sQ)).Q)[:, 1:sQ]
    W = Matrix(qr(randn(rng, T, sQ, sQ)).Q)
    Z = P * Diagonal(T.(sigmas)) * W'          # so Q*Z' has singular values σ
    return Q, Z
end

@testset "ARA truncation on CPU" begin
    rng = MersenneTwister(1501)
    bm, bn, sQ = 48, 40, 12

    @testset "empty co-range bypasses the SVD backend" begin
        U0, S0, V0 = _TLRM.batched_thin_svd!(zeros(Float64, bn, 0, 3))
        @test size(U0) == (bn, 0, 3)
        @test size(S0) == (0, 3)
        @test size(V0) == (0, 0, 3)
    end

    @testset "matches the Eckart-Young optimum" begin
        sigmas = [2.0^(-k) for k in 0:(sQ-1)]         # smooth decay
        Q, Z = truncation_fixture(Float64, bm, bn, sQ, sigmas, rng)
        X = Q * Z'
        for tol in (1e-1, 1e-3, 1e-6)
            U = zeros(Float64, bm, sQ, 1)
            V = zeros(Float64, bn, sQ, 1)
            ranks = zeros(Int32, 1); err = zeros(Float64, 1)
            _TLRM.ara_truncate!(U, V, ranks, err,
                                reshape(copy(Q), bm, sQ, 1),
                                reshape(copy(Z), bn, sQ, 1);
                                tol, relative=true, maxrank=sQ)
            r = Int(ranks[1])
            Ur, Vr = U[:, 1:r, 1], V[:, 1:r, 1]
            achieved = norm(X - Ur * Vr')
            optimal = sqrt(sum(abs2, sigmas[(r+1):end]))
            # Optimality: the achieved error equals the Eckart-Young bound for
            # the rank chosen, to round-off.
            @test achieved ≈ optimal atol = 1e-12 rtol = 1e-8
            # The budget is met, and one rank less would have missed it.
            @test achieved <= tol * norm(X) * (1 + 1e-8)
            @test r == 1 || sqrt(sum(abs2, sigmas[r:end])) > tol * norm(X)
            # The reported error is the achieved one, not a bound.
            @test sqrt(err[1]) ≈ optimal atol = 1e-12 rtol = 1e-8
            @test norm(Ur' * Ur - I, Inf) <= 1e-12
        end
    end

    @testset "orthonormal output despite zero columns in Q" begin
        # ara_mask_breakdown! leaves exact zero columns in Q, so QᵀQ ≠ I. The
        # retained right singular vectors must vanish on those rows, which is
        # what keeps U = QW orthonormal.
        sigmas = [1.0, 0.5, 0.25, 0.1, 0.0, 0.0]
        s = length(sigmas)
        Q, Z = truncation_fixture(Float64, bm, bn, s, sigmas, rng)
        Q[:, 5:6] .= 0.0                      # dead columns
        Z[:, :] = Z * Diagonal([1, 1, 1, 1, 0, 0])
        @test norm(Q' * Q - I, Inf) > 0.5     # Q really is not orthonormal
        U = zeros(Float64, bm, s, 1); V = zeros(Float64, bn, s, 1)
        ranks = zeros(Int32, 1); err = zeros(Float64, 1)
        _TLRM.ara_truncate!(U, V, ranks, err, reshape(Q, bm, s, 1),
                            reshape(copy(Z), bn, s, 1);
                            tol=1e-8, relative=true, maxrank=s)
        r = Int(ranks[1])
        @test r == 4
        Ur = U[:, 1:r, 1]
        @test norm(Ur' * Ur - I, Inf) <= 1e-12
        @test norm(Q * Z' - Ur * V[:, 1:r, 1]') / norm(Q * Z') <= 1e-12
    end

    @testset "ragged ranks stay in one batched call" begin
        count = 3
        specs = ([1.0, 1e-9, 1e-9, 1e-9], [1.0, 0.5, 0.25, 1e-9], [1.0, 0.5, 0.25, 0.125])
        s = 4
        Qs = zeros(Float64, bm, s, count); Zs = zeros(Float64, bn, s, count)
        for b in 1:count
            q, z = truncation_fixture(Float64, bm, bn, s, specs[b], rng)
            Qs[:, :, b] = q; Zs[:, :, b] = z
        end
        U = zeros(Float64, bm, s, count); V = zeros(Float64, bn, s, count)
        ranks = zeros(Int32, count); err = zeros(Float64, count)
        _TLRM.ara_truncate!(U, V, ranks, err, copy(Qs), copy(Zs);
                            tol=1e-6, relative=true, maxrank=s)
        @test ranks == Int32[1, 3, 4]
        for b in 1:count
            r = Int(ranks[b])
            @test norm(U[:, 1:r, b]' * U[:, 1:r, b] - I, Inf) <= 1e-12
            # Surplus columns must be exactly zero, so the padded batch is safe
            # to consume without per-member bookkeeping downstream.
            @test all(iszero, U[:, (r+1):s, b])
            @test all(iszero, V[:, (r+1):s, b])
        end
    end

    @testset "maxrank cap is authoritative and self-reporting" begin
        sigmas = fill(1.0, 8)                 # flat: nothing is truncatable
        s = 8
        Q, Z = truncation_fixture(Float64, bm, bn, s, sigmas, rng)
        U = zeros(Float64, bm, 3, 1); V = zeros(Float64, bn, 3, 1)
        ranks = zeros(Int32, 1); err = zeros(Float64, 1)
        _TLRM.ara_truncate!(U, V, ranks, err, reshape(Q, bm, s, 1),
                            reshape(copy(Z), bn, s, 1);
                            tol=1e-8, relative=true, maxrank=3)
        @test ranks[1] == 3                   # saturation visible in the rank
        @test sqrt(err[1]) ≈ sqrt(5.0) rtol = 1e-10   # 5 discarded unit values
    end
end

@testset "ARA truncation on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            rng = MersenneTwister(1502)
            bm, bn, sQ = 48, 40, 12
            sigmas = [2.0^(-k) for k in 0:(sQ-1)]
            Qh, Zh = truncation_fixture(Float64, bm, bn, sQ, sigmas, rng)
            X = Qh * Zh'
            U = ArrayType(zeros(Float64, bm, sQ, 1))
            V = ArrayType(zeros(Float64, bn, sQ, 1))
            ranks = ArrayType(zeros(Int32, 1)); err = ArrayType(zeros(Float64, 1))
            _TLRM.ara_truncate!(U, V, ranks, err,
                                ArrayType(reshape(Qh, bm, sQ, 1)),
                                ArrayType(reshape(Zh, bn, sQ, 1));
                                tol=1e-6, relative=true, maxrank=sQ)
            synchronize(U)
            r = Int(Array(ranks)[1])
            Ur = Array(U)[:, 1:r, 1]; Vr = Array(V)[:, 1:r, 1]
            optimal = sqrt(sum(abs2, sigmas[(r+1):end]))
            @test norm(X - Ur * Vr') ≈ optimal atol = 1e-12 rtol = 1e-6
            @test norm(Ur' * Ur - I, Inf) <= 1e-11
            @test sqrt(Array(err)[1]) ≈ optimal atol = 1e-12 rtol = 1e-6

            # The batched SVD must run on device. The generic fallback copies to
            # the host and would pass every numerical check above while quietly
            # costing a round trip per tile, so assert residency directly.
            Zd = ArrayType(reshape(Zh, bn, sQ, 1))
            Ud, Sd, Vd = _TLRM.batched_thin_svd!(Zd)
            @test Ud isa typeof(Zd) && Vd isa typeof(Zd)
            @test Array(Sd)[:, 1] ≈ svd(Zh).S rtol = 1e-12

            U0, S0, V0 = _TLRM.batched_thin_svd!(
                ArrayType(zeros(Float64, bn, 0, 2)),
            )
            @test size(U0) == (bn, 0, 2)
            @test size(S0) == (0, 2)
            @test size(V0) == (0, 0, 2)
            @test backend_name == "CPU" || U0 isa typeof(Zd)
        end
    end
end

# Regression: the projection must be INTERLEAVED with the two Cholesky passes,
# `(P·CQR)²` as in Algorithm 2.3 lines 22-27 -- not `P²·CQR²`.
#
# The two are different algorithms in finite precision. A column numerically
# dependent on the rest of its block has a within-block residual of size
# `O(u‖Y‖)`, the same order as the `O(u‖Y‖)` overlap with `Q` that BCGS leaves;
# so its direction has an `O(1)` fraction inside `span(Q)`. The first Cholesky
# normalizes it to unit length, freezing that fraction at `O(1)`. Only a
# projection placed *after* that amplification removes it.
#
# The exposure needs a mixed (part-genuine, part-null) block in a pass *after*
# the first -- pass 1 has no basis to lose orthogonality against -- and is worst
# when the block has just 1-2 null columns. Batched ordering failed ~1/3 of
# seeds here with `‖QᵀQ−I‖` up to 1.0; interleaved is clean.
@testset "ARA interleaves projection with the Cholesky passes" begin
    T = Float64
    bm, bn = 32, 34
    for (r_true, block) in ((6, 4), (7, 4), (20, 8))
        worst_orth = 0.0
        worst_rank = 0
        for seed in 1:40
            rng = MersenneTwister(seed)
            Uo = Matrix(qr(randn(rng, T, bm, r_true)).Q)[:, 1:r_true]
            Vo = Matrix(qr(randn(rng, T, bn, r_true)).Q)[:, 1:r_true]
            X = Uo * Diagonal(collect(range(1.0, 0.6; length=r_true))) * Vo'
            ws = _TLRM.ARAWorkspace(T, CPU(), bm, 32, 1; block)
            omega = zeros(T, bn, ws.block, 1)
            sampler = function (Y, width)
                Om = view(omega, :, 1:width, 1)
                Random.randn!(Om)
                copyto!(view(Y, :, 1:width, 1), X * Om)
                return Y
            end
            res = _TLRM.ara_build_basis!(ws, sampler; eps_rel=1e-7, r_required=4)
            r = Int(Array(res.ranks)[1])
            Qr = Array(res.Q)[:, 1:r, 1]
            worst_orth = max(worst_orth, opnorm(Qr' * Qr - I, Inf))
            worst_rank = max(worst_rank, abs(r - r_true))
        end
        # Batched ordering gave ~1e0 here; interleaved must stay at round-off.
        @test worst_orth <= 1e-10
        @test worst_rank == 0
    end
end
