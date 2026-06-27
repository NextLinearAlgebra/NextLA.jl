include("helpers.jl")

@testset "TLR TileMap and geometry" begin
    @testset "column-major index table" begin
        order = NextLA.TileColMajor(3, 3)
        layout = NextLA.TLRmodule.TileMap(order, 4, 4, 10, 9)
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
        layout = NextLA.TLRmodule.TileMap(order, 4, 4, 10, 9)
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
            # 32×32 with b=16: 2×2 tile grid, no boundary tiles
            A = NextLA.TLRMatrix(prototype, 16, 16; compress_diag=false)

            @test size(A) == (32, 32)
            @test A.layout.order isa NextLA.TileColMajor
            @test NextLA.blocksize(A) == 16
            @test NextLA.maxrank(A) == 16
            @test !NextLA.compress_diag(A)
            @test size(A.int_U) == (16, 16, 2)
            @test size(A.int_V) == (16, 16, 2)
            @test size(A.right_U) == (16, 16, 0)
            @test size(A.right_V) == (16, 16, 0)
            @test size(A.bottom_U) == (16, 16, 0)
            @test size(A.bottom_V) == (16, 16, 0)
            @test size(NextLA.dense_diag(A)) == (16, 16, 2)
            @test size(NextLA.ranks(A)) == (2,)
            @test all(iszero, Array(NextLA.ranks(A)))
            @test length(A.obs_int) == 2
            @test isempty(A.obs_right)
            @test isempty(A.obs_bottom)
            @test expected_storage_slot(A, 1, 2) == 2
            @test expected_storage_slot(A, 2, 1) == 1

            lf = NextLA.left_factors(A)
            rf = NextLA.right_factors(A)
            @test lf.interior === A.int_U
            @test rf.interior === A.int_V

            synchronize(A.int_U)
            synchronize(A.int_V)
            synchronize(A.D)
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), -1, 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 0, 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 2, -1)

        A = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
        @test_throws BoundsError  NextLA.tile_linear_index(A, 3, 1)
        @test_throws BoundsError  NextLA.tile_linear_index(A, 1, 3)
        @test_throws ArgumentError NextLA.tile_storage_index(A, 1, 1)
    end
end

@testset "TLR factor access — boundary tile categories" begin
    # 10×10 with b=4: 3×3 tile grid, tail_m=tail_n=2
    A_tlr = NextLA.TLRMatrix(zeros(Float64, 10, 10), 4, 3)

    internal_slot = NextLA.tile_storage_index(A_tlr, 1, 2)
    bottom_slot   = NextLA.tile_storage_index(A_tlr, 3, 1)
    right_slot    = NextLA.tile_storage_index(A_tlr, 1, 3)

    @test A_tlr.category[internal_slot] == NextLA.TLRmodule._TILE_INT
    @test A_tlr.category[bottom_slot]   == NextLA.TLRmodule._TILE_BOTTOM
    @test A_tlr.category[right_slot]    == NextLA.TLRmodule._TILE_RIGHT

    @test size(A_tlr.int_U)    == (4, 3, 2)   # 2 interior off-diag tiles: (2,1),(1,2)
    @test size(A_tlr.right_U)  == (4, 3, 2)   # 2 right-boundary tiles: (1,3),(2,3)
    @test size(A_tlr.right_V)  == (2, 3, 2)   # tail_n=2
    @test size(A_tlr.bottom_U) == (2, 3, 2)   # 2 bottom-boundary tiles: (3,1),(3,2), tail_m=2
    @test size(A_tlr.bottom_V) == (4, 3, 2)

    # Write data for interior tile
    ki = A_tlr.local_index[internal_slot]
    A_tlr.ranks[internal_slot] = 2
    A_tlr.int_U[:, 1:2, ki] .= reshape(collect(21.0:28.0), 4, 2)
    A_tlr.int_V[:, 1:2, ki] .= reshape(collect(31.0:38.0), 4, 2)

    # Write data for bottom boundary tile
    kb = A_tlr.local_index[bottom_slot]
    A_tlr.ranks[bottom_slot] = 2
    A_tlr.bottom_U[1:2, 1:2, kb] .= [1.0 2.0; 3.0 4.0]
    A_tlr.bottom_V[:, 1:2, kb]   .= reshape(collect(1.0:8.0), 4, 2)

    # Write data for right boundary tile
    kr_idx = A_tlr.local_index[right_slot]
    A_tlr.ranks[right_slot] = 2
    A_tlr.right_U[:, 1:2, kr_idx]   .= reshape(collect(11.0:18.0), 4, 2)
    A_tlr.right_V[1:2, 1:2, kr_idx] .= [5.0 6.0; 7.0 8.0]

    # tile_u/tile_v return correctly shaped views
    @test size(NextLA.tile_u(A_tlr, internal_slot)) == (4, 2)
    @test size(NextLA.tile_v(A_tlr, internal_slot)) == (4, 2)
    @test Matrix(NextLA.tile_u(A_tlr, internal_slot)) == reshape(collect(21.0:28.0), 4, 2)
    @test Matrix(NextLA.tile_v(A_tlr, internal_slot)) == reshape(collect(31.0:38.0), 4, 2)

    @test size(NextLA.tile_u(A_tlr, bottom_slot)) == (2, 2)
    @test size(NextLA.tile_v(A_tlr, bottom_slot)) == (4, 2)
    @test Matrix(NextLA.tile_u(A_tlr, bottom_slot)) == [1.0 2.0; 3.0 4.0]
    @test Matrix(NextLA.tile_v(A_tlr, bottom_slot)) == reshape(collect(1.0:8.0), 4, 2)

    @test size(NextLA.tile_u(A_tlr, right_slot)) == (4, 2)
    @test size(NextLA.tile_v(A_tlr, right_slot)) == (2, 2)
    @test Matrix(NextLA.tile_u(A_tlr, right_slot)) == reshape(collect(11.0:18.0), 4, 2)
    @test Matrix(NextLA.tile_v(A_tlr, right_slot)) == [5.0 6.0; 7.0 8.0]
end
