# CompressedFTLRMatrix, src/tlr/core/compressed_ftlr_matrix.jl.
# Helpers: test/setup/tlr.jl.

@testset "finalized CompressedFTLRMatrix layout [CPU]" begin
    Random.seed!(3003)
    ranks = Int[1 3 0; 2 1 1; 1 1 1]
    A = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 11, (4, 5), ranks;
        rank_multiple=8)

    @test size(A) == (10, 11)
    @test TileLowRank.grid_size(A) == (3, 3)
    @test TileLowRank.nominal_tile_size(A) == (4, 5)
    @test TileLowRank.tail_tile_size(A) == (2, 1)
    @test TileLowRank.rank_multiple(A) == 8
    @test TileLowRank.maxrank(A) == 3
    @test TileLowRank.maximum_storage_rank(A) == 8
    @test A.outer.order isa TileLowRank.TileRowMajor
    @test A.inner.order isa TileLowRank.TileColMajor
    @test length(TileLowRank.get_factors(A, 3, 3)) == 2
    @test size(TileLowRank.get_factors(A, 3, 3)[1]) == (2, 1)
    @test size(TileLowRank.get_factors(A, 3, 3)[2]) == (1, 1)
    @test size(_TLRM.compressed_ftlr_storage_outer(A, 3, 3)) == (2, 8)
    @test size(_TLRM.compressed_ftlr_storage_inner(A, 3, 3)) == (1, 8)

    exact = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 8, 8, 4, Int[1 2; 3 4])
    @test TileLowRank.rank_multiple(exact) == 0
    @test [_TLRM.compressed_ftlr_storage_rank(exact, i, j)
           for i in 1:2, j in 1:2] == Int[1 2; 3 4]

    @test_throws ArgumentError TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 8, 8, 4, ones(Int, 2, 2);
        rank_multiple=-1)
    @test_throws DimensionMismatch TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 8, 8, 4, ones(Int, 3, 2))
end

@testset "CompressedFTLR complementary packing and reconstruction [CPU]" begin
    Random.seed!(3004)
    rank_grid = Int[1 2 0; 2 1 1; 1 1 1]
    A = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 9, 10, (4, 4), rank_grid;
        rank_multiple=8)
    fill_random_tlr!(A, Array; seed=88)

    @test A.outer.offsets[1] == 1
    @test A.inner.offsets[1] == 1
    @test issorted(A.outer.offsets)
    @test issorted(A.inner.offsets)
    @test A.outer.offsets[end] - 1 == length(A.outer.data)
    @test A.inner.offsets[end] - 1 == length(A.inner.data)

    dense = zeros(Float64, size(A))
    TileLowRank.uncompress!(dense, A)
    for j in 1:3, i in 1:3
        p0, q0 = TileLowRank.tile_origin_coords(A, i, j)
        tm, tn = TileLowRank.tile_size(A, i, j)
        U, V = TileLowRank.get_factors(A, i, j)
        reference = isempty(U) ? zeros(tm, tn) : Matrix(U) * Matrix(V)'
        @test dense[p0:p0+tm-1, q0:q0+tn-1] ≈ reference
    end
end

@testset "rank_multiple replaces symbolic execution policies [CPU]" begin
    Random.seed!(3005)
    ranks = Int[1 9; 3 8]
    exact = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks)
    q8 = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks;
        rank_multiple=8)
    q16 = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks;
        rank_multiple=16)
    @test [_TLRM.compressed_ftlr_storage_rank(exact, i, j)
           for i in 1:2, j in 1:2] == ranks
    @test [_TLRM.compressed_ftlr_storage_rank(q8, i, j)
           for i in 1:2, j in 1:2] == Int[8 16; 8 8]
    @test [_TLRM.compressed_ftlr_storage_rank(q16, i, j)
           for i in 1:2, j in 1:2] == fill(16, 2, 2)
end
