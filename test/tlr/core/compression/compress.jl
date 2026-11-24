# Dense to TLR and FTLR: constructors, reusable ARA staging, boundary batches and
# packed tile compression, src/tlr/core/compression/compress.jl and workspace.jl.
# Helpers: test/setup/tlr.jl.

function full_tlr_rectangular_fixture(::Type{T}) where {T}
    tile_size = (3, 4)
    m, n, maxrank = 8, 11, 3
    A = zeros(T, m, n)
    for j in 1:cld(n, tile_size[2]), i in 1:cld(m, tile_size[1])
        p0, q0 = (i-1)*tile_size[1]+1, (j-1)*tile_size[2]+1
        tm, tn = min(tile_size[1], m-p0+1), min(tile_size[2], n-q0+1)
        r = min(maxrank, tm, tn, mod(i + 2j, maxrank) + 1)
        A[p0:p0+tm-1, q0:q0+tn-1] .=
            make_rect_lowrank_tile(T, tm, tn, r; seed=100i+j)
    end
    return (; A, tile_size, maxrank)
end

@testset "dense to finalized TLR/FTLR [CPU]" begin
    Random.seed!(3016)
    boundary = boundary_dense_fixture(Float64)
    ws = TileLowRank.FTLRCompressionWorkspace(
        boundary.A, 4; maxrank=3, diagonal=:dense)
    A = TileLowRank.TLRMatrix(
        boundary.A, 4; maxrank=3, tol=1e-7, workspace=ws,
        rank_multiple=8)
    @test TileLowRank.rank_multiple(A) == 8
    @test all(k -> TileLowRank.ranks(A)[expected_storage_slot(A, k, k)] == 0,
              1:TileLowRank.ndiag_tiles(A))
    @test norm(reconstruct_tlr(A) - boundary.A) / norm(boundary.A) <= 1e-7
    assert_tile_rank_and_error(A, 1, 2, 2, boundary.a12; atol_rank=1, rtol_error=1e-7)
    assert_tile_rank_and_error(A, 2, 1, 3, boundary.a21; atol_rank=1, rtol_error=1e-7)

    # The same numerical staging can be reused; the returned packed matrix is
    # newly allocated because its offsets depend on this pass's ranks.
    B = TileLowRank.TLRMatrix(
        boundary.A, 4; maxrank=3, tol=1e-7, workspace=ws)
    @test B !== A
    @test B.offdiag.outer.data !== A.offdiag.outer.data
    @test reconstruct_tlr(B) ≈ boundary.A rtol=1e-7

    full = full_tlr_rectangular_fixture(Float64)
    C = TileLowRank.CompressedFTLRMatrix(
        full.A, full.tile_size; maxrank=full.maxrank, tol=1e-9,
        rank_multiple=8)
    @test TileLowRank.grid_size(C) == (3, 3)
    @test TileLowRank.tile_size(C, 3, 3) == (2, 3)
    @test norm(reconstruct_tlr(C) - full.A) / norm(full.A) <= 1e-9
    @test all(r -> r <= full.maxrank, TileLowRank.ranks(C))

    small = Float64[2 1; -1 3]
    S = TileLowRank.TLRMatrix(small, 4; maxrank=2, tol=1e-12)
    @test size(TileLowRank.dense_diag(S), 3) == 0
    @test size(TileLowRank.dense_diag_corner(S)) == (2, 2, 1)
    @test reconstruct_tlr(S) == small
    @test_throws ArgumentError TileLowRank.TLRMatrix(
        small, 4; maxrank=2, r_required=0)
end

@testset "compression residual and rank-cap semantics [CPU]" begin
    Random.seed!(3017)
    b, maxr = 32, 16
    rng = MersenneTwister(7)
    Q1 = Matrix(qr(randn(rng, b, b)).Q)
    Q2 = Matrix(qr(randn(rng, b, b)).Q)
    hard = Q1 * Diagonal(10.0 .^ range(0, -6, length=b)) * Q2'
    easy = make_lowrank_tile(Float64, b, 4; seed=2)
    dense = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=1), hard,
        easy, make_dense_tile(Float64, b; seed=3))
    A = TileLowRank.TLRMatrix(dense, b; maxrank=maxr, tol=1e-3)
    hard_idx = expected_storage_slot(A, 1, 2)
    easy_idx = expected_storage_slot(A, 2, 1)
    @test Int(TileLowRank.ranks(A)[hard_idx]) == maxr
    @test TileLowRank.residuals(A)[hard_idx] > 1e-3
    @test Int(TileLowRank.ranks(A)[easy_idx]) == 4
    @test TileLowRank.residuals(A)[easy_idx] <= 1e-3

    zero_dense = assemble_block_matrix(
        make_dense_tile(Float64, b; seed=9), zeros(Float64, b, b),
        easy, make_dense_tile(Float64, b; seed=11))
    Z = TileLowRank.TLRMatrix(zero_dense, b; maxrank=maxr, tol=0)
    zero_idx = expected_storage_slot(Z, 1, 2)
    @test TileLowRank.ranks(Z)[zero_idx] == 0
    @test TileLowRank.residuals(Z)[zero_idx] == 0
end

@testset "packed tile compression workspace [CPU]" begin
    Random.seed!(3018)
    rng = MersenneTwister(7)
    b, count, cap = 6, 4, 3
    true_ranks = [2, 2, 3, 3]
    tiles = zeros(Float64, b, b, count)
    for k in 1:count
        r = true_ranks[k]
        tiles[:, :, k] .= randn(rng, b, r) * randn(rng, r, b)
    end
    U = zeros(Float64, b, cap, count)
    V = zeros(Float64, b, cap, count)
    ws = _TLRM.make_category_workspace(
        (b, b), [(k, 1) for k in 1:count], U, V)
    @test parent(ws.Z) === V
    _TLRM.compress_tiles!(
        _TLRM.PackedTiles(tiles), ws; eps_sq=1e-12, rel=false, r_required=4)
    got = Array(ws.ranks)
    @test got == true_ranks
    for k in 1:count
        r = got[k]
        @test norm(tiles[:, :, k] - U[:, 1:r, k] * V[:, 1:r, k]') /
              norm(tiles[:, :, k]) <= 1e-8
        @test sqrt(ws.errors_sq[k]) <= 1e-8 * norm(tiles[:, :, k])
    end
end

# Compression on the device of a matrix whose tiles break down in ARA's
# Cholesky-QR, so the truncation sees zero singular values (see the ARA test
# "ARA truncation with zero singular values"): finite, and as accurate as the CPU.
@testset "compression with breakdown [$backend_name]" for (backend_name, ArrayType, synchronize) in filter(b -> b[1] != "CPU", TLR_BACKENDS)
    Random.seed!(3048)
    n, b, maxrank = 512, 64, 16
    A = [1 / (1 + abs(i - j)) for i in 1:n, j in 1:n] + I
    M = TileLowRank.TLRMatrix(ArrayType(A), b; maxrank, tol=1e-8, rel=true)
    D = ArrayType(zeros(n, n)); TileLowRank.uncompress!(D, M); synchronize(D)
    Random.seed!(3048)
    Mh = TileLowRank.TLRMatrix(A, b; maxrank, tol=1e-8, rel=true)
    Dh = zeros(n, n); TileLowRank.uncompress!(Dh, Mh)
    @test all(isfinite, Array(D))
    @test norm(Array(D) - A) / norm(A) <= 10 * max(norm(Dh - A) / norm(A), 1e-12)
end

@testset "dense constructors [$backend_name]" for (backend_name, ArrayType, synchronize) in filter(b -> b[1] in ("CUDA", "AMDGPU"), TLR_BACKENDS)
    Random.seed!(3019)
    boundary = boundary_dense_fixture(Float32)
    dense = ArrayType(boundary.A)
    A = TileLowRank.TLRMatrix(dense, 4; maxrank=3, tol=1f-4)
    synchronize(A.offdiag.outer.data)
    @test norm(reconstruct_tlr(A) - boundary.A) / norm(boundary.A) <= 5f-3
end
