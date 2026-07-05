function make_rect_lowrank_tile(::Type{T}, m::Int, n::Int, r::Int; seed::Integer) where {T}
    0 <= r <= min(m, n) || throw(ArgumentError("rank must satisfy 0 <= r <= min(m, n)"))
    rng = MersenneTwister(seed)
    if r == 0
        return zeros(T, m, n)
    end
    qleft = Matrix(qr(randn(rng, T, m, r)).Q)
    qright = Matrix(qr(randn(rng, T, n, r)).Q)
    sigma = collect(range(T(2), T(1), length=r))
    return qleft[:, 1:r] * Diagonal(sigma) * qright[:, 1:r]'
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

@testset "TLR compress! on CPU" begin
    fixture = canonical_dense_fixture(Float64)
    A_uniform = NextLA.TLRMatrix(fixture.A, fixture.b, 16)
    ws_uniform = NextLA.TLRmodule.alloc_workspace(A_uniform)
    NextLA.compress!(A_uniform, fixture.A, ws_uniform; tol=1e-6)

    relerr = norm(reconstruct_tlr(A_uniform) - fixture.A) / norm(fixture.A)
    @test relerr <= 1e-6
    assert_tile_rank_and_error(A_uniform, 1, 2, 10, fixture.offdiag12; atol_rank=4, rtol_error=1e-6)
    assert_tile_rank_and_error(A_uniform, 2, 1, 16, fixture.offdiag21; rtol_error=1e-6)

    boundary = boundary_dense_fixture(Float64)
    A_panel = NextLA.TLRMatrix(boundary.A, 4, 3)
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
    A_tlr = NextLA.TLRMatrix(A, b, maxr)
    NextLA.compress!(A_tlr, A; tol=1e-3)

    ob_hard = NextLA.TLRmodule._offdiag_index(A_tlr, 1, 2)
    ob_easy = NextLA.TLRmodule._offdiag_index(A_tlr, 2, 1)
    resid = NextLA.residuals(A_tlr)

    @test Int(NextLA.ranks(A_tlr)[ob_hard]) == maxr
    @test resid[ob_hard] > 1e-3
    true_err = norm(hard - Matrix(NextLA.tile_u(A_tlr, ob_hard)) *
                           Matrix(adjoint(NextLA.tile_v(A_tlr, ob_hard))))
    @test isapprox(resid[ob_hard], true_err; rtol=1e-2)

    @test Int(NextLA.ranks(A_tlr)[ob_easy]) == 4
    @test resid[ob_easy] <= 1e-3
    easy_err = norm(easy - Matrix(NextLA.tile_u(A_tlr, ob_easy)) *
                           Matrix(adjoint(NextLA.tile_v(A_tlr, ob_easy))))
    @test easy_err < 1e-10

    # (2) Tiny-scale tiles: the relative cholqr shift keeps rank detection
    # working at scale 1e-7 (an absolute shift swamps the Gram matrix here).
    tiny12 = 1e-7 .* make_lowrank_tile(Float64, b, 3; seed=5)
    tiny21 = 1e-7 .* make_lowrank_tile(Float64, b, 5; seed=6)
    At = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=7), tiny12,
        tiny21, make_dense_tile(Float64, b; seed=8))
    At_tlr = NextLA.TLRMatrix(At, b, maxr)
    NextLA.compress!(At_tlr, At; tol=1e-10)
    @test Int(NextLA.ranks(At_tlr)[NextLA.TLRmodule._offdiag_index(At_tlr, 1, 2)]) == 3
    @test Int(NextLA.ranks(At_tlr)[NextLA.TLRmodule._offdiag_index(At_tlr, 2, 1)]) == 5
    @test all(NextLA.residuals(At_tlr) .<= 1e-10)

    # (3) Zero tile: exact rank 0 with zero residual, even at tol = 0.
    Az = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=9), zeros(Float64, b, b),
        make_lowrank_tile(Float64, b, 2; seed=10), make_dense_tile(Float64, b; seed=11))
    Az_tlr = NextLA.TLRMatrix(Az, b, maxr)
    NextLA.compress!(Az_tlr, Az; tol=0.0)
    ob_zero = NextLA.TLRmodule._offdiag_index(Az_tlr, 1, 2)
    @test Int(NextLA.ranks(Az_tlr)[ob_zero]) == 0
    @test NextLA.residuals(Az_tlr)[ob_zero] == 0.0

    # (4) rel=true scales the budget per tile: an absolute tol flattens the
    # small-scale tile to rank 0, the relative one preserves it.
    small = 1e-8 .* make_lowrank_tile(Float64, b, 6; seed=12)
    Am = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=13), make_lowrank_tile(Float64, b, 6; seed=14),
        small, make_dense_tile(Float64, b; seed=15))
    Am_tlr = NextLA.TLRMatrix(Am, b, maxr)
    ob_small = NextLA.TLRmodule._offdiag_index(Am_tlr, 2, 1)

    NextLA.compress!(Am_tlr, Am; tol=1e-5)
    @test Int(NextLA.ranks(Am_tlr)[ob_small]) == 0
    @test NextLA.residuals(Am_tlr)[ob_small] <= 1e-5   # dropped, but within budget

    NextLA.compress!(Am_tlr, Am; tol=1e-5, rel=true)
    @test Int(NextLA.ranks(Am_tlr)[ob_small]) == 6
    @test NextLA.residuals(Am_tlr)[ob_small] <= 1e-5 * norm(small)
end

@testset "TLR uncompress! on CPU" begin
    for order in (NextLA.TileColMajor(), NextLA.TileRowMajor())
        @testset "$(order)" begin
            fixture = canonical_dense_fixture(Float64)
            A_uniform = NextLA.TLRMatrix(fixture.A, fixture.b, 16; tile_order=order)
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
            A_boundary = NextLA.TLRMatrix(boundary.A, 4, 3; tile_order=order)
            ws_boundary = NextLA.TLRmodule.alloc_workspace(A_boundary)
            NextLA.compress!(A_boundary, boundary.A, ws_boundary; tol=1e-6)

            dense_boundary = fill(-2.0, size(boundary.A))
            NextLA.uncompress!(dense_boundary, A_boundary)
            relerr_boundary = norm(dense_boundary - boundary.A) / norm(boundary.A)
            @test relerr_boundary <= 1e-6

            small = Float64[2 1; -1 3]
            A_small = NextLA.TLRMatrix(small, 4, 2; tile_order=order)
            ws_small = NextLA.TLRmodule.alloc_workspace(A_small)
            NextLA.compress!(A_small, small, ws_small; tol=1e-12)

            dense_small = fill(-5.0, size(small))
            NextLA.uncompress!(dense_small, A_small)
            @test dense_small == small
            @test size(NextLA.dense_diag(A_small), 3) == 0
            @test size(A_small.D_corner) == (2, 2, 1)

            zero_rank = NextLA.TLRMatrix(boundary.A, 4, 3; tile_order=order)
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
                tile_i, tile_j = NextLA.TLRmodule._inverse_tile_index(zero_rank, linear)
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
            A_tlr = NextLA.TLRMatrix(dense, 4, 3)
            ws = NextLA.TLRmodule.alloc_workspace(A_tlr)
            NextLA.compress!(A_tlr, dense, ws; tol=1f-4)

            synchronize(A_tlr.int_U)
            relerr = norm(reconstruct_tlr(A_tlr) - boundary.A) / norm(boundary.A)
            @test relerr <= 5f-3
            @test Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._offdiag_index(A_tlr, 1, 3)]) == 2
            @test Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._offdiag_index(A_tlr, 3, 1)]) == 2
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
                A_tlr = NextLA.TLRMatrix(dense, 4, 3; tile_order=order)
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
