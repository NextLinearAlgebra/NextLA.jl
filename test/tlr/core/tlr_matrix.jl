# TLRMatrix and the TileLowRank exports, src/tlr/core/tlr_matrix.jl and src/tlr/core/tlr.jl.
# Helpers: test/setup/tlr.jl.

@testset "top-level TLR API [CPU]" begin
    Random.seed!(3001)
    @test TLRMatrix === TileLowRank.TLRMatrix
    @test CompressedFTLRMatrix === TileLowRank.CompressedFTLRMatrix
    @test uncompress! === TileLowRank.uncompress!
end

@testset "finalized TLRMatrix layout [CPU]" begin
    Random.seed!(3002)
    rank_grid = Int[0 2 1; 1 0 1; 1 1 0]
    A = TileLowRank.TLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, rank_grid;
        rank_multiple=8)
    @test !ismutable(A)
    @test TileLowRank.offdiagonal(A) isa TileLowRank.CompressedFTLRMatrix
    @test TileLowRank.rank_multiple(A) == 8
    @test TileLowRank.maxrank(A) == 2
    @test size(TileLowRank.dense_diag(A)) == (4, 4, 2)
    @test size(TileLowRank.dense_diag_corner(A)) == (2, 2, 1)
    @test length(TileLowRank.get_factors(A, 1, 2)) == 2
    @test size(TileLowRank.get_factors(A, 1, 2)[1]) == (4, 2)
    @test size(TileLowRank.get_factors(A, 3, 1)[1]) == (2, 1)
    @test_throws ArgumentError TileLowRank.get_factors(A, 1, 1)

    bad = copy(rank_grid)
    bad[2, 2] = 1
    @test_throws ArgumentError TileLowRank.TLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, bad)
end
