include("helpers.jl")

@testset "TLR TileMap and geometry" begin
    @testset "column-major index table" begin
        order = NextLA.TileColMajor(3, 3)
        layout = NextLA.TileMap(order, 4, 4, 10, 9)
        expected = [
            (1, 1, 1, 1, 1, 4, 4),
            (2, 1, 2, 5, 1, 4, 4),
            (3, 1, 3, 9, 1, 2, 4),
            (1, 2, 4, 1, 5, 4, 4),
            (2, 2, 5, 5, 5, 4, 4),
            (3, 2, 6, 9, 5, 2, 4),
            (1, 3, 7, 1, 9, 4, 1),
            (2, 3, 8, 5, 9, 4, 1),
            (3, 3, 9, 9, 9, 2, 1),
        ]

        for (i, j, linear, p0, q0, tile_m, tile_n) in expected
            @test NextLA.tile_linear_index(order, i, j) == linear
            @test NextLA.inverse_tile_index(layout, linear) == (i, j)
            @test NextLA.tile_origin_coords(layout, i, j) == (p0, q0)
            @test NextLA.tile_sizes(layout, i, j) == (tile_m, tile_n)
        end

        @test NextLA.TLRmodule.offdiag_batch_index(layout, 2, 1) == 1
        @test NextLA.TLRmodule.offdiag_batch_index(layout, 3, 1) == 2
        @test NextLA.TLRmodule.offdiag_batch_index(layout, 1, 2) == 3
        @test NextLA.TLRmodule.offdiag_batch_index(layout, 1, 3) == 5
        @test NextLA.offdiag_linear_index(layout, 1) == 2
        @test NextLA.offdiag_linear_index(layout, 2) == 3
        @test NextLA.offdiag_linear_index(layout, 3) == 4
        @test NextLA.offdiag_linear_index(layout, 5) == 7
    end

    @testset "row-major index table" begin
        order = NextLA.TileRowMajor(3, 3)
        layout = NextLA.TileMap(order, 4, 4, 10, 9)
        expected = [
            (1, 1, 1, 1, 1, 4, 4),
            (1, 2, 2, 1, 5, 4, 4),
            (1, 3, 3, 1, 9, 4, 1),
            (2, 1, 4, 5, 1, 4, 4),
            (2, 2, 5, 5, 5, 4, 4),
            (2, 3, 6, 5, 9, 4, 1),
            (3, 1, 7, 9, 1, 2, 4),
            (3, 2, 8, 9, 5, 2, 4),
            (3, 3, 9, 9, 9, 2, 1),
        ]

        for (i, j, linear, p0, q0, tile_m, tile_n) in expected
            @test NextLA.tile_linear_index(order, i, j) == linear
            @test NextLA.inverse_tile_index(layout, linear) == (i, j)
            @test NextLA.tile_origin_coords(layout, i, j) == (p0, q0)
            @test NextLA.tile_sizes(layout, i, j) == (tile_m, tile_n)
        end

        @test NextLA.TLRmodule.offdiag_batch_index(layout, 1, 2) == 1
        @test NextLA.TLRmodule.offdiag_batch_index(layout, 1, 3) == 2
        @test NextLA.TLRmodule.offdiag_batch_index(layout, 2, 1) == 3
        @test NextLA.TLRmodule.offdiag_batch_index(layout, 3, 1) == 5
        @test NextLA.offdiag_linear_index(layout, 1) == 2
        @test NextLA.offdiag_linear_index(layout, 2) == 3
        @test NextLA.offdiag_linear_index(layout, 5) == 7
    end
end

@testset "TLRMatrix constructor and storage allocation" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            prototype = ArrayType(zeros(Float32, 32, 32))
            A = NextLA.TLRMatrix(prototype, 16, 16; compress_diag=false)
            Aall = NextLA.TLRMatrix(prototype, 16, 16; compress_diag=true)

            @test size(A) == (32, 32)
            @test A.storage isa NextLA.UVTileStorage
            @test A.layout.order isa NextLA.TileColMajor
            @test NextLA.blocksize(A) == 16
            @test NextLA.maxrank(A) == 16
            @test !NextLA.compress_diag(A)
            @test size(NextLA.left_factors(A)) == (16, 16, 2)
            @test size(NextLA.right_factors(A)) == (16, 16, 2)
            @test size(NextLA.dense_diag(A)) == (16, 16, 2)
            @test size(NextLA.ranks(A)) == (2,)
            @test all(iszero, Array(NextLA.ranks(A)))
            @test expected_storage_slot(A, 1, 2) == 2
            @test expected_storage_slot(A, 2, 1) == 1

            @test NextLA.compress_diag(Aall)
            @test size(NextLA.left_factors(Aall)) == (16, 16, 4)
            @test size(NextLA.right_factors(Aall)) == (16, 16, 4)
            @test size(NextLA.dense_diag(Aall)) == (16, 16, 0)
            @test size(NextLA.ranks(Aall)) == (4,)
            @test all(iszero, Array(NextLA.ranks(Aall)))
            @test NextLA.tile_storage_index(Aall, 1, 1) == 1
            @test NextLA.tile_storage_index(Aall, 1, 2) == 3

            synchronize(NextLA.left_factors(A))
            synchronize(NextLA.right_factors(A))
            synchronize(NextLA.dense_diag(A))
            synchronize(NextLA.left_factors(Aall))
            synchronize(NextLA.right_factors(Aall))
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), -1, 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 0, 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 2, -1)

        A = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
        @test_throws BoundsError NextLA.tile_linear_index(A, 3, 1)
        @test_throws BoundsError NextLA.tile_linear_index(A, 1, 3)
        @test_throws ArgumentError NextLA.tile_storage_index(A, 1, 1)
    end
end
