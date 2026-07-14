function make_rect_lowrank_tile(::Type{T}, m::Int, n::Int, r::Int; seed::Integer) where {T}
    0 <= r <= min(m, n) || throw(ArgumentError("rank must satisfy 0 <= r <= min(m, n)"))
    rng = MersenneTwister(seed)
    if r == 0
        return zeros(T, m, n)
    end
    qleft = Matrix(qr(randn(rng, T, m, r)).Q)
    qright = Matrix(qr(randn(rng, T, n, r)).Q)
    sigma = r == 1 ? [T(1)] : collect(range(T(2), T(1), length=r))
    return qleft[:, 1:r] * Diagonal(sigma) * qright[:, 1:r]'
end

@testset "shifted CholQR2 policy" begin
    M = NextLA.TLRmodule
    for Tgram in (Float64, ComplexF64)
        m, s = 37, 9
        expected = 11 * (m * s + s * (s + 1)) * (eps(Float64) / 2)
        @test M._cholqr_shift_coeff(Tgram, m, s) == expected
    end

    rng = MersenneTwister(901)
    for Thi in (Float64, ComplexF64)
        m, s, batch = 48, 8, 3
        Y = randn(rng, Thi, m, s, batch)
        Q = copy(Y)
        G = zeros(Thi, s, s, batch)
        R = zeros(Thi, s, s, batch)
        multipliers = ones(Float64, batch)
        M.cholqr2!(Q, Y, G, R, M._batch_views(R), M._batch_views(Q), multipliers)
        floor = M._cholqr_shift_coeff(Thi, m, s)
        for k in 1:batch
            @test norm(Q[:, :, k]' * Q[:, :, k] - I, Inf) <= 2floor
        end

        Z = zeros(Thi, m, s, 1)
        Qz = copy(Z)
        Gz = zeros(Thi, s, s, 1)
        Rz = zeros(Thi, s, s, 1)
        M.cholqr2!(Qz, Z, Gz, Rz, M._batch_views(Rz), M._batch_views(Qz),
            ones(Float64, 1))
        @test all(iszero, Qz)
        @test all(isfinite, Gz)
    end

    # Throughput path: Gram/POTRF are Float64 while both triangular solves and
    # the stored basis remain Float32.
    Q32 = randn(rng, Float32, 48, 8, 2)
    Y64 = zeros(Float64, 48, 8, 2)
    G64 = zeros(Float64, 8, 8, 2)
    R32 = zeros(Float32, 8, 8, 2)
    M.cholqr2!(Q32, Y64, G64, R32, M._batch_views(R32), M._batch_views(Q32),
        ones(Float64, 2))
    @test eltype(Q32) == Float32
    @test all(isfinite, Q32)
    for k in axes(Q32, 3)
        @test norm(Q32[:, :, k]' * Q32[:, :, k] - I, Inf) <= 20eps(Float32)
    end

    # Rank-deficient inputs must remain finite; dependent directions are damped
    # and subsequently removed by energy pruning.
    Y = zeros(Float64, 32, 6, 1)
    Y[:, 1:3, 1] .= randn(rng, 32, 3)
    Y[:, 4:6, 1] .= Y[:, 1:3, 1]
    Q = copy(Y)
    G = zeros(Float64, 6, 6, 1)
    R = zeros(Float64, 6, 6, 1)
    M.cholqr2!(Q, Y, G, R, M._batch_views(R), M._batch_views(Q), ones(Float64, 1))
    @test all(isfinite, Q)
    singular_values = svdvals(Q[:, :, 1])
    @test singular_values[3] > 0.99
    @test singular_values[4] < 1e-3
end

function boundary_dense_fixture(::Type{T}) where {T}
    d11 = make_dense_tile(T, 4; seed=11)
    d22 = make_dense_tile(T, 4; seed=22)
    d33 = make_dense_tile(T, 2; seed=33)

    a12 = make_lowrank_tile(T, 4, 2; seed=101)
    a21 = make_lowrank_tile(T, 4, 3; seed=102)
    a13 = make_rect_lowrank_tile(T, 4, 2, 2; seed=103)
    a23 = make_rect_lowrank_tile(T, 4, 2, 2; seed=104)
    a31 = make_rect_lowrank_tile(T, 2, 4, 2; seed=105)
    a32 = make_rect_lowrank_tile(T, 2, 4, 2; seed=106)

    top = hcat(d11, a12, a13)
    mid = hcat(a21, d22, a23)
    bot = hcat(a31, a32, d33)
    A = vcat(top, mid, bot)
    return (; A, d11, d22, d33, a12, a21, a13, a23, a31, a32)
end

function full_tlr_rectangular_fixture(::Type{T}) where {T}
    tile_size = (3, 4)
    m, n = 8, 11
    maxrank = 3
    A = zeros(T, m, n)
    ranks = Dict{Tuple{Int,Int},Int}()
    for tile_i in 1:cld(m, tile_size[1]), tile_j in 1:cld(n, tile_size[2])
        p0 = (tile_i - 1) * tile_size[1] + 1
        q0 = (tile_j - 1) * tile_size[2] + 1
        tm = min(tile_size[1], m - p0 + 1)
        tn = min(tile_size[2], n - q0 + 1)
        r = min(maxrank, tm, tn, mod(tile_i + 2tile_j, maxrank) + 1)
        tile = make_rect_lowrank_tile(T, tm, tn, r; seed=100tile_i + tile_j)
        A[p0:(p0+tm-1), q0:(q0+tn-1)] .= tile
        ranks[(tile_i, tile_j)] = r
    end
    return (; A, tile_size, maxrank, ranks)
end

@testset "TLR compress! on CPU" begin
    fixture = canonical_dense_fixture(Float64)
    A_uniform = NextLA.TLRDenseDiagMatrix(fixture.A, fixture.b, 16)
    ws_uniform = NextLA.TLRmodule.alloc_workspace(A_uniform)
    NextLA.compress!(A_uniform, fixture.A, ws_uniform; tol=1e-6)

    relerr = norm(reconstruct_tlr(A_uniform) - fixture.A) / norm(fixture.A)
    @test relerr <= 1e-6
    # The all-shifted policy deliberately keeps weak dependent directions when
    # the requested tolerance is at its orthogonality floor.
    assert_tile_rank_and_error(A_uniform, 1, 2, 10, fixture.offdiag12; atol_rank=6, rtol_error=1e-6)
    assert_tile_rank_and_error(A_uniform, 2, 1, 16, fixture.offdiag21; rtol_error=1e-6)

    boundary = boundary_dense_fixture(Float64)
    A_panel = NextLA.TLRDenseDiagMatrix(boundary.A, 4, 3)
    ws_panel = NextLA.TLRmodule.alloc_workspace(A_panel)
    NextLA.compress!(A_panel, boundary.A, ws_panel; tol=1e-6)

    relerr_panel = norm(reconstruct_tlr(A_panel) - boundary.A) / norm(boundary.A)
    @test relerr_panel <= 1e-6
    @test size(A_panel.int_U) == (4, 3, 2)
    @test size(A_panel.right_U) == (4, 3, 2)
    @test size(A_panel.bottom_U) == (2, 3, 2)
    assert_tile_rank_and_error(A_panel, 1, 2, 2, boundary.a12; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 2, 1, 3, boundary.a21; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 1, 3, 2, boundary.a13; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 2, 3, 2, boundary.a23; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 3, 1, 2, boundary.a31; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 3, 2, 2, boundary.a32; atol_rank=1, rtol_error=1e-6)

    full_rect = full_tlr_rectangular_fixture(Float64)
    A_full = NextLA.TLRMatrix(full_rect.A, full_rect.tile_size, full_rect.maxrank)
    ws_full = NextLA.TLRmodule.alloc_workspace(A_full)
    NextLA.compress!(A_full, full_rect.A, ws_full; tol=1e-8)

    relerr_full = norm(reconstruct_tlr(A_full) - full_rect.A) / norm(full_rect.A)
    @test relerr_full <= 1e-8
    @test size(A_full.int_U) == (3, 3, 4)
    @test size(A_full.right_U) == (3, 3, 2)
    @test size(A_full.bottom_U) == (2, 3, 2)
    @test size(A_full.corner_U) == (2, 3, 1)
    for ((tile_i, tile_j), _) in full_rect.ranks
        p0, q0 = NextLA.tile_origin_coords(A_full, tile_i, tile_j)
        tm, tn = NextLA.tile_size(A_full, tile_i, tile_j)
        tile_ref = @view full_rect.A[p0:(p0+tm-1), q0:(q0+tn-1)]
        rank = Int(NextLA.ranks(A_full)[NextLA.TLRmodule._rank_index(A_full, tile_i, tile_j)])
        @test rank <= full_rect.maxrank
        U, V = NextLA.get_factors(A_full, tile_i, tile_j)
        approx = rank == 0 ? zeros(Float64, tm, tn) : Matrix(U) * Matrix(adjoint(V))
        @test norm(tile_ref - approx) / max(norm(tile_ref), eps(Float64)) <= 1e-8
    end
end

@testset "TLR compress! error indicator and FAIL semantics" begin
    b = 64
    maxr = 16

    # (1) Saturated tile: numerical rank ≫ maxrank must be flagged, not
    # silently reported as converged, and the residual estimate must be honest.
    rng = MersenneTwister(7)
    Q1 = Matrix(qr(randn(rng, b, b)).Q)
    Q2 = Matrix(qr(randn(rng, b, b)).Q)
    hard = Q1 * Diagonal(10.0 .^ range(0, -6, length=b)) * Q2'
    easy = make_lowrank_tile(Float64, b, 4; seed=2)
    A = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=1), hard,
        easy, make_dense_tile(Float64, b; seed=3))
    A_tlr = NextLA.TLRDenseDiagMatrix(A, b, maxr)
    NextLA.compress!(A_tlr, A; tol=1e-3)

    ob_hard = NextLA.TLRmodule._rank_index(A_tlr, 1, 2)
    ob_easy = NextLA.TLRmodule._rank_index(A_tlr, 2, 1)
    resid = NextLA.residuals(A_tlr)

    @test Int(NextLA.ranks(A_tlr)[ob_hard]) == maxr
    @test resid[ob_hard] > 1e-3
    U_hard, V_hard = NextLA.get_factors(A_tlr, 1, 2)
    true_err = norm(hard - Matrix(U_hard) * Matrix(adjoint(V_hard)))
    @test isapprox(resid[ob_hard], true_err; rtol=1e-2)

    @test Int(NextLA.ranks(A_tlr)[ob_easy]) == 4
    @test resid[ob_easy] <= 1e-3
    U_easy, V_easy = NextLA.get_factors(A_tlr, 2, 1)
    easy_err = norm(easy - Matrix(U_easy) * Matrix(adjoint(V_easy)))
    @test easy_err < 1e-10

    # (2) Tiny-scale tiles: the relative cholqr shift keeps rank detection
    # working at scale 1e-7 (an absolute shift swamps the Gram matrix here).
    tiny12 = 1e-7 .* make_lowrank_tile(Float64, b, 3; seed=5)
    tiny21 = 1e-7 .* make_lowrank_tile(Float64, b, 5; seed=6)
    At = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=7), tiny12,
        tiny21, make_dense_tile(Float64, b; seed=8))
    At_tlr = NextLA.TLRDenseDiagMatrix(At, b, maxr)
    NextLA.compress!(At_tlr, At; tol=1e-10)
    @test Int(NextLA.ranks(At_tlr)[NextLA.TLRmodule._rank_index(At_tlr, 1, 2)]) == 3
    @test Int(NextLA.ranks(At_tlr)[NextLA.TLRmodule._rank_index(At_tlr, 2, 1)]) == 5
    @test all(NextLA.residuals(At_tlr) .<= 1e-10)

    # (3) Zero tile: exact rank 0 with zero residual, even at tol = 0.
    Az = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=9), zeros(Float64, b, b),
        make_lowrank_tile(Float64, b, 2; seed=10), make_dense_tile(Float64, b; seed=11))
    Az_tlr = NextLA.TLRDenseDiagMatrix(Az, b, maxr)
    NextLA.compress!(Az_tlr, Az; tol=0.0)
    ob_zero = NextLA.TLRmodule._rank_index(Az_tlr, 1, 2)
    @test Int(NextLA.ranks(Az_tlr)[ob_zero]) == 0
    @test NextLA.residuals(Az_tlr)[ob_zero] == 0.0

    # (4) rel=true scales the budget per tile: an absolute tol flattens the
    # small-scale tile to rank 0, the relative one preserves it.
    small = 1e-8 .* make_lowrank_tile(Float64, b, 6; seed=12)
    Am = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=13), make_lowrank_tile(Float64, b, 6; seed=14),
        small, make_dense_tile(Float64, b; seed=15))
    Am_tlr = NextLA.TLRDenseDiagMatrix(Am, b, maxr)
    ob_small = NextLA.TLRmodule._rank_index(Am_tlr, 2, 1)

    NextLA.compress!(Am_tlr, Am; tol=1e-5)
    @test Int(NextLA.ranks(Am_tlr)[ob_small]) == 0
    @test NextLA.residuals(Am_tlr)[ob_small] <= 1e-5   # dropped, but within budget

    NextLA.compress!(Am_tlr, Am; tol=1e-5, rel=true)
    @test Int(NextLA.ranks(Am_tlr)[ob_small]) == 6
    @test NextLA.residuals(Am_tlr)[ob_small] <= 1e-5 * norm(small)
end

@testset "TLR compress! sketch capacity" begin
    b = 64
    r12 = 5
    r21 = 11

    for T in (Float64, Float32)
        tol = T == Float64 ? 1e-6 : 1.0f-3
        A = assemble_block_matrix(
            make_dense_tile(T, b; seed=41), make_lowrank_tile(T, b, r12; seed=42),
            make_lowrank_tile(T, b, r21; seed=43), make_dense_tile(T, b; seed=44))

        # maxrank includes any desired randomized-range buffer and is both the
        # sketch width and output-panel capacity.
        for maxr in (12, 20, 32)
            A_tlr = NextLA.TLRDenseDiagMatrix(A, b, maxr)
            ws = NextLA.alloc_workspace(A_tlr)
            # Seed 2 previously put the Float64 residual a few ulps above
            # the nominal CholQR floor and exposed flaky full-rank retention.
            Random.seed!(T == Float64 && maxr == 12 ? 2 : 1000 + maxr + sizeof(T))
            NextLA.compress!(A_tlr, A, ws; tol=tol, rel=true)

            assert_tile_rank_and_error(A_tlr, 1, 2, r12,
                make_lowrank_tile(T, b, r12; seed=42); rtol_error=2 * tol)
            assert_tile_rank_and_error(A_tlr, 2, 1, r21,
                make_lowrank_tile(T, b, r21; seed=43); rtol_error=2 * tol)

            # Stored rank never exceeds the shared sketch/output capacity.
            mt, nt = NextLA.tilegrid_size(A_tlr)
            for tile_i in 1:mt, tile_j in 1:nt
                tile_i == tile_j && continue
                rank_idx = NextLA.TLRmodule._rank_index(A_tlr, tile_i, tile_j)
                @test Int(NextLA.ranks(A_tlr)[rank_idx]) <= maxr
            end
        end
    end

end

@testset "TLR uncompress! on CPU" begin
    for order in (NextLA.TileColMajor(), NextLA.TileRowMajor())
        @testset "$(order)" begin
            fixture = canonical_dense_fixture(Float64)
            A_uniform = NextLA.TLRDenseDiagMatrix(fixture.A, fixture.b, 16; tile_order=order)
            ws_uniform = NextLA.TLRmodule.alloc_workspace(A_uniform)
            NextLA.compress!(A_uniform, fixture.A, ws_uniform; tol=1e-6)

            dense_uniform = fill(-1.0, size(fixture.A))
            NextLA.uncompress!(dense_uniform, A_uniform)
            relerr_uniform = norm(dense_uniform - fixture.A) / norm(fixture.A)
            @test relerr_uniform <= 1e-6

            for tile in 1:NextLA.ndiag_tiles(A_uniform)
                p0, q0 = NextLA.tile_origin_coords(A_uniform, tile, tile)
                tm, tn = NextLA.tile_size(A_uniform, tile, tile)
                rows = p0:(p0 + tm - 1)
                cols = q0:(q0 + tn - 1)
                @test dense_uniform[rows, cols] == fixture.A[rows, cols]
            end

            boundary = boundary_dense_fixture(Float64)
            A_boundary = NextLA.TLRDenseDiagMatrix(boundary.A, 4, 3; tile_order=order)
            ws_boundary = NextLA.TLRmodule.alloc_workspace(A_boundary)
            NextLA.compress!(A_boundary, boundary.A, ws_boundary; tol=1e-6)

            dense_boundary = fill(-2.0, size(boundary.A))
            NextLA.uncompress!(dense_boundary, A_boundary)
            relerr_boundary = norm(dense_boundary - boundary.A) / norm(boundary.A)
            @test relerr_boundary <= 1e-6

            small = Float64[2 1; -1 3]
            A_small = NextLA.TLRDenseDiagMatrix(small, 4, 2; tile_order=order)
            ws_small = NextLA.TLRmodule.alloc_workspace(A_small)
            NextLA.compress!(A_small, small, ws_small; tol=1e-12)

            dense_small = fill(-5.0, size(small))
            NextLA.uncompress!(dense_small, A_small)
            @test dense_small == small
            @test size(NextLA.dense_diag(A_small), 3) == 0
            @test size(A_small.D_corner) == (2, 2, 1)

            zero_rank = NextLA.TLRDenseDiagMatrix(boundary.A, 4, 3; tile_order=order)
            for tile in 1:NextLA.ndiag_tiles(zero_rank)
                p0, q0 = NextLA.tile_origin_coords(zero_rank, tile, tile)
                tm, tn = NextLA.tile_size(zero_rank, tile, tile)
                rows = p0:(p0 + tm - 1)
                cols = q0:(q0 + tn - 1)
                if tile <= size(zero_rank.D, 3)
                    zero_rank.D[1:tm, 1:tn, tile] .= boundary.A[rows, cols]
                else
                    zero_rank.D_corner[1:tm, 1:tn, 1] .= boundary.A[rows, cols]
                end
            end

            dense_zero_rank = fill(-3.0, size(boundary.A))
            NextLA.uncompress!(dense_zero_rank, zero_rank)

            for linear in 1:prod(NextLA.tilegrid_size(zero_rank))
                tile_i, tile_j = NextLA.TLRmodule.inverse_tile_index(
                    zero_rank.order, NextLA.tilegrid_size(zero_rank)..., linear)
                p0, q0 = NextLA.tile_origin_coords(zero_rank, tile_i, tile_j)
                tm, tn = NextLA.tile_size(zero_rank, tile_i, tile_j)
                rows = p0:(p0 + tm - 1)
                cols = q0:(q0 + tn - 1)
                if tile_i == tile_j
                    @test dense_zero_rank[rows, cols] == boundary.A[rows, cols]
                else
                    @test dense_zero_rank[rows, cols] == zeros(Float64, tm, tn)
                end
            end
        end
    end
end

@testset "TLR compress! on GPU" begin
    boundary = boundary_dense_fixture(Float32)
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name in ("CUDA", "AMDGPU") || continue
        @testset "$backend_name" begin
            dense = ArrayType(boundary.A)
            A_tlr = NextLA.TLRDenseDiagMatrix(dense, 4, 3)
            ws = NextLA.TLRmodule.alloc_workspace(A_tlr)
            NextLA.compress!(A_tlr, dense, ws; tol=1f-4)

            synchronize(A_tlr.int_U)
            relerr = norm(reconstruct_tlr(A_tlr) - boundary.A) / norm(boundary.A)
            @test relerr <= 5f-3
            @test Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._rank_index(A_tlr, 1, 3)]) == 2
            @test Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._rank_index(A_tlr, 3, 1)]) == 2
        end
    end
end

@testset "TLR uncompress! on GPU" begin
    boundary = boundary_dense_fixture(Float32)
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name in ("CUDA", "AMDGPU") || continue
        @testset "$backend_name" begin
            for order in (NextLA.TileColMajor(), NextLA.TileRowMajor())
                dense = ArrayType(boundary.A)
                A_tlr = NextLA.TLRDenseDiagMatrix(dense, 4, 3; tile_order=order)
                ws = NextLA.TLRmodule.alloc_workspace(A_tlr)
                NextLA.compress!(A_tlr, dense, ws; tol=1f-4)

                dense_roundtrip = ArrayType(fill(Float32(-4), size(boundary.A)))
                NextLA.uncompress!(dense_roundtrip, A_tlr)

                synchronize(dense_roundtrip)
                relerr = norm(Array(dense_roundtrip) - boundary.A) / norm(boundary.A)
                @test relerr <= 5f-3
            end
        end
    end
end

@testset "compress_tiles! on a packed tile batch" begin
    M = NextLA.TLRmodule
    rng = MersenneTwister(7)
    b = 6; n = 4; kout = 3
    true_ranks = [2, 2, 3, 3]

    P = zeros(Float64, b, b, n)
    refs = Matrix{Float64}[]
    for k in 1:n
        r = true_ranks[k]
        Tk = randn(rng, b, r) * randn(rng, r, b)
        P[:, :, k] = Tk
        push!(refs, copy(Tk))
    end

    U = zeros(Float64, b, kout, n)
    V = zeros(Float64, b, kout, n)
    ws = M.alloc_tile_workspace(U, V, b, b, kout, n)
    @test parent(ws.Q_T) === U
    @test parent(ws.V_T) === V
    @test parent(ws.norm_err_sq) === ws.G_hi
    @test parent(ws.shift_mult) === parent(ws.V_T)
    M.compress_tiles!(M.PackedTiles(P), ws; eps_sq=1e-12, rel=false)

    ranks = Array(ws.ranks_local)
    for k in 1:n
        @test ranks[k] == true_ranks[k]
        r = Int(ranks[k])
        recon = U[:, 1:r, k] * V[:, 1:r, k]'
        @test norm(recon - refs[k]) / norm(refs[k]) <= 1e-6
    end
end

@testset "in-place fused truncation and compaction" begin
    M = NextLA.TLRmodule

    # Hard-cap pruning keeps columns 1, 3, and 5. The minimum-move map fills
    # hole 2 from source 5, so the compact deterministic order is [1, 5, 3].
    energies = [9.0, 0.01, 4.0, 0.04, 1.0, 0.09]
    S = length(energies)
    U = zeros(Float64, S, S, 1)
    V = zeros(Float64, S, S, 1)
    for j in 1:S
        U[j, j, 1] = j
        V[j, j, 1] = sqrt(energies[j])
    end
    rk = zeros(Int32, 1)
    norm_err_sq = [sum(energies)]

    M.prune_ranks!(U, V, rk, norm_err_sq, S, 3, 0.0, false, 0.0)

    @test rk == Int32[3]
    @test [findfirst(!iszero, U[:, j, 1]) for j in 1:3] == [1, 5, 3]
    @test all(iszero, view(U, :, 4:S, 1))
    @test all(iszero, view(V, :, 4:S, 1))
    @test norm_err_sq[1] ≈ sum(energies[[2, 4, 6]])

    # Equal energies are ordered by source index during pruning. Dropping 1 and
    # 2 leaves sources 3 and 4, compacted reproducibly as [4, 3].
    U .= 0
    V .= 0
    for j in 1:4
        U[j, j, 1] = j
        V[j, j, 1] = 1
    end
    fill!(rk, 0)
    norm_err_sq[1] = 4
    M.prune_ranks!(U, V, rk, norm_err_sq, 4, 2, 0.0, false, 0.0)
    @test rk == Int32[2]
    @test [findfirst(!iszero, U[:, j, 1]) for j in 1:2] == [4, 3]
    @test all(iszero, view(U, :, 3:S, 1))

    arena = M.compress_arena_elems(32, 24, 12, 7)
    @test arena.S == 12
    @test arena.accum == (32 * 12 + 12 * 12) * 7
end
