# A0: the bookkeeping the blocked ARA loop needs from CholQR2.

# Build a CholQR2 workspace around a batch of panels.
function ara_cholqr_fixture(::Type{T}, ArrayType, X::Array{T,3}) where {T}
    m, s, count = size(X)
    Thi = _TLRM.tlr_orthogonalization_type(T)
    ws = _TLRM.CholQR2FactorWorkspace(
        ArrayType(copy(X)), ArrayType(zeros(T, s, s, count)),
        ArrayType(zeros(Thi, m, s, count)), ArrayType(zeros(Thi, s, s, count)),
        ArrayType(zeros(T, s, s, count)), ArrayType(zeros(T, s, s, count)),
        ArrayType(ones(real(Thi), count)),
    )
    return ws
end

@testset "ARA convergence bookkeeping on CPU" begin
    rng = MersenneTwister(1301)

    @testset "shift floor is the documented relative bound" begin
        # The floor must match sqrt of the shift coefficient exactly: the whole
        # dR correction is built on this identity.
        for (T, m, s) in ((Float64, 256, 32), (Float64, 32, 16), (Float32, 128, 8))
            Thi = _TLRM.tlr_orthogonalization_type(T)
            floor_rel = _TLRM.cholqr2_relative_shift_floor(Thi, m, s)
            @test floor_rel ≈ sqrt(Float64(_TLRM._cholqr_shift_coeff(Thi, m, s)))
        end
        # The hazard this exists to expose: at a realistic block shape the floor
        # sits above tolerances callers will plausibly ask for.
        @test _TLRM.cholqr2_relative_shift_floor(Float64, 256, 32) > 1e-6
    end

    @testset "column norms reproduce max diag(G)" begin
        m, s, count = 40, 6, 3
        X = randn(rng, Float64, m, s, count)
        cn = zeros(Float64, s, count)
        colmax = zeros(Float64, count)
        _TLRM.ara_column_norms_sq!(cn, colmax, X, s)
        for b in 1:count
            G = X[:, :, b]' * X[:, :, b]
            @test cn[:, b] ≈ diag(G)
            @test colmax[b] ≈ maximum(diag(G))
        end
        # A short final pass must ignore the stale tail.
        _TLRM.ara_column_norms_sq!(cn, colmax, X, 2)
        @test colmax ≈ [maximum(sum(abs2, X[:, 1:2, b]; dims=1)) for b in 1:count]
    end

    @testset "dR recovers projected norms on a full-rank block" begin
        m, s, count = 64, 8, 2
        X = randn(rng, Float64, m, s, count)
        ws = ara_cholqr_fixture(Float64, Array, X)
        cn = zeros(Float64, s, count)
        colmax = zeros(Float64, count)
        _TLRM.ara_column_norms_sq!(cn, colmax, ws.Q, s)
        _TLRM.mixed_cholqr2_factor!(ws)
        dR = zeros(Float64, s, count)
        _TLRM.ara_block_norms!(dR, ws, colmax)
        # dR[j] is the residual of column j against columns 1..j-1, i.e. the
        # diagonal of the R factor of a plain QR of the same panel.
        for b in 1:count
            Rref = abs.(diag(qr(X[:, :, b]).R))
            @test dR[:, b] ≈ Rref rtol = 1e-10
        end
    end

    @testset "dR collapses on the redundant columns of a deficient block" begin
        # The generic stopping case: every column has O(1) norm, but the block
        # spans only `rank` directions. Column norms cannot see this; dR must.
        m, s, rank, count = 64, 8, 3, 2
        X = zeros(Float64, m, s, count)
        for b in 1:count
            basis = Matrix(qr(randn(rng, Float64, m, rank)).Q)[:, 1:rank]
            X[:, :, b] = basis * randn(rng, Float64, rank, s)
        end
        ws = ara_cholqr_fixture(Float64, Array, X)
        cn = zeros(Float64, s, count)
        colmax = zeros(Float64, count)
        _TLRM.ara_column_norms_sq!(cn, colmax, ws.Q, s)
        _TLRM.mixed_cholqr2_factor!(ws)
        dR = zeros(Float64, s, count)
        _TLRM.ara_block_norms!(dR, ws, colmax)
        for b in 1:count
            scale = sqrt(colmax[b])
            @test all(dR[1:rank, b] .> 1e-3 * scale)      # genuine directions
            @test all(dR[rank+1:s, b] .< 1e-8 * scale)    # nothing new left
            # Column norms are blind here — this is why dR is the right signal.
            @test all(sqrt.(cn[:, b]) .> 1e-2 * scale)
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

@testset "ARA convergence bookkeeping on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            rng = MersenneTwister(1302)
            m, s, rank, count = 64, 8, 3, 2
            X = zeros(Float64, m, s, count)
            for b in 1:count
                basis = Matrix(qr(randn(rng, Float64, m, rank)).Q)[:, 1:rank]
                X[:, :, b] = basis * randn(rng, Float64, rank, s)
            end
            ws = ara_cholqr_fixture(Float64, ArrayType, X)
            cn = ArrayType(zeros(Float64, s, count))
            colmax = ArrayType(zeros(Float64, count))
            _TLRM.ara_column_norms_sq!(cn, colmax, ws.Q, s)
            _TLRM.mixed_cholqr2_factor!(ws)
            dR = ArrayType(zeros(Float64, s, count))
            _TLRM.ara_block_norms!(dR, ws, colmax)
            synchronize(dR)
            dRh = Array(dR)
            cmh = Array(colmax)
            for b in 1:count
                scale = sqrt(cmh[b])
                @test all(dRh[1:rank, b] .> 1e-3 * scale)
                @test all(dRh[rank+1:s, b] .< 1e-8 * scale)
            end

            state = _TLRM.ARAConvergenceState(get_backend(dR), count)
            _TLRM.ara_reset!(state, s, 2 * s)
            active = _TLRM.ara_update_convergence!(state, dR, 1e-8, 3, s, 2 * s)
            @test active == 0                        # both members converged
            @test Array(state.ranks) == Int32[rank, rank]
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
