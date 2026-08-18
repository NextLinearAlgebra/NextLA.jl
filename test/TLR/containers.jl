using KernelAbstractions

@testset "finalized CompressedFTLRMatrix layout" begin
    ranks = Int[1 3 0; 2 1 1; 1 1 1]
    A = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 11, (4, 5), ranks;
        rank_multiple=8)

    @test size(A) == (10, 11)
    @test NextLA.grid_size(A) == (3, 3)
    @test NextLA.nominal_tile_size(A) == (4, 5)
    @test NextLA.tail_tile_size(A) == (2, 1)
    @test NextLA.rank_multiple(A) == 8
    @test NextLA.maxrank(A) == 3
    @test NextLA.maximum_storage_rank(A) == 8
    @test A.outer.order isa NextLA.TileRowMajor
    @test A.inner.order isa NextLA.TileColMajor
    @test length(NextLA.get_factors(A, 3, 3)) == 2
    @test size(NextLA.get_factors(A, 3, 3)[1]) == (2, 1)
    @test size(NextLA.get_factors(A, 3, 3)[2]) == (1, 1)
    @test size(_TLRM.compressed_ftlr_storage_outer(A, 3, 3)) == (2, 8)
    @test size(_TLRM.compressed_ftlr_storage_inner(A, 3, 3)) == (1, 8)

    exact = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 8, 8, 4, Int[1 2; 3 4])
    @test NextLA.rank_multiple(exact) == 0
    @test [_TLRM._compressed_ftlr_storage_rank(exact, i, j)
           for i in 1:2, j in 1:2] == Int[1 2; 3 4]

    @test_throws ArgumentError NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 8, 8, 4, ones(Int, 2, 2);
        rank_multiple=-1)
    @test_throws DimensionMismatch NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 8, 8, 4, ones(Int, 3, 2))
end

@testset "finalized TLRMatrix layout" begin
    rank_grid = Int[0 2 1; 1 0 1; 1 1 0]
    A = NextLA.TLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, rank_grid;
        rank_multiple=8)
    @test !ismutable(A)
    @test NextLA.offdiagonal(A) isa NextLA.CompressedFTLRMatrix
    @test NextLA.rank_multiple(A) == 8
    @test NextLA.maxrank(A) == 2
    @test size(NextLA.dense_diag(A)) == (4, 4, 2)
    @test size(NextLA.dense_diag_corner(A)) == (2, 2, 1)
    @test length(NextLA.get_factors(A, 1, 2)) == 2
    @test size(NextLA.get_factors(A, 1, 2)[1]) == (4, 2)
    @test size(NextLA.get_factors(A, 3, 1)[1]) == (2, 1)
    @test_throws ArgumentError NextLA.get_factors(A, 1, 1)

    bad = copy(rank_grid)
    bad[2, 2] = 1
    @test_throws ArgumentError NextLA.TLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, bad)
end
