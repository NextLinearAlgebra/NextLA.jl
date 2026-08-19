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
    @test [_TLRM.compressed_ftlr_storage_rank(exact, i, j)
           for i in 1:2, j in 1:2] == Int[1 2; 3 4]

    @test_throws ArgumentError NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 8, 8, 4, ones(Int, 2, 2);
        rank_multiple=-1)
    @test_throws DimensionMismatch NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 8, 8, 4, ones(Int, 3, 2))
end

@testset "CompressedFTLR complementary packing and reconstruction" begin
    rank_grid = Int[1 2 0; 2 1 1; 1 1 1]
    A = NextLA.CompressedFTLRMatrix(
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
    NextLA.uncompress!(dense, A)
    for j in 1:3, i in 1:3
        p0, q0 = NextLA.tile_origin_coords(A, i, j)
        tm, tn = NextLA.tile_size(A, i, j)
        U, V = NextLA.get_factors(A, i, j)
        reference = isempty(U) ? zeros(tm, tn) : Matrix(U) * Matrix(V)'
        @test dense[p0:p0+tm-1, q0:q0+tn-1] ≈ reference
    end
end

@testset "rank_multiple replaces symbolic execution policies" begin
    ranks = Int[1 9; 3 8]
    exact = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks)
    q8 = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks;
        rank_multiple=8)
    q16 = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks;
        rank_multiple=16)
    @test [_TLRM.compressed_ftlr_storage_rank(exact, i, j)
           for i in 1:2, j in 1:2] == ranks
    @test [_TLRM.compressed_ftlr_storage_rank(q8, i, j)
           for i in 1:2, j in 1:2] == Int[8 16; 8 8]
    @test [_TLRM.compressed_ftlr_storage_rank(q16, i, j)
           for i in 1:2, j in 1:2] == fill(16, 2, 2)
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
