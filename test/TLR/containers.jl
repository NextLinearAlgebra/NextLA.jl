# Container layer: tile-index geometry, constructors, storage shapes, factor access.

@testset "TLR full-grid geometry" begin
    A = NextLA.TLRMatrix(zeros(Float32, 10, 9), 4, 4)
    gs = NextLA.grid_size(A)
    @test gs == (3, 3)
    @test A.order isa NextLA.TileRowMajor
    for linear in 1:prod(gs)
        i, j = NextLA.TLRmodule.inverse_tile_index(A.order, gs..., linear)
        @test NextLA.TLRmodule.tile_linear_index(A.order, gs..., i, j) == linear
        @test NextLA.tile_origin_coords(A, i, j) == ((i - 1) * 4 + 1, (j - 1) * 4 + 1)
    end
    @test all(k -> NextLA.ranks(A)[NextLA.TLRmodule._rank_index(A, k, k)] == 0, 1:3)
    @test all(k -> NextLA.execution_ranks(A)[NextLA.TLRmodule._rank_index(A, k, k)] == 0, 1:3)
end

@testset "TLRMatrix constructor and storage allocation" begin
    # Rectangular nominal tiles with tails on both axes are represented in one
    # packed grid; there are no boundary-category factor arrays.
    @testset "rectangular nominal tile geometry" begin
        A = NextLA.TLRMatrix(zeros(Float64, 10, 11), (4, 5), 3)

        @test ismutable(A)
        @test size(A) == (10, 11)
        @test NextLA.TLRmodule.tile_order(A) isa NextLA.TileRowMajor
        @test NextLA.nominal_tile_size(A) == (4, 5)
        @test NextLA.nominal_tile_size(A, 1) == 4
        @test NextLA.nominal_tile_size(A, 2) == 5
        @test NextLA.tail_tile_size(A) == (2, 1)
        @test NextLA.tail_tile_size(A, 1) == 2
        @test NextLA.tail_tile_size(A, 2) == 1
        @test NextLA.maxrank(A) == 3
        @test NextLA.grid_size(A) == (3, 3)
        @test NextLA.tile_origin_coords(A, 3, 3) == (9, 11)
        @test NextLA.tile_size(A, 1, 1) == (4, 5)
        @test NextLA.tile_size(A, 3, 3) == (2, 1)

        @test NextLA.offdiagonal(A) isa NextLA.CompressedFTLRMatrix
        @test A.offdiag.outer.order isa NextLA.TileRowMajor
        @test A.offdiag.inner.order isa NextLA.TileColMajor
        @test count(!iszero, NextLA.execution_ranks(A)) == 6
        @test NextLA.execution_rank_policy(A) === :explicit
        @test size(A.D) == (4, 5, 2)
        @test size(A.D_corner) == (2, 1, 1)
        @test NextLA.dense_diag_corner(A) === A.D_corner
    end

    @testset "backend allocation smoke" begin
        for (backend_name, ArrayType, synchronize) in available_backends()
            @testset "$backend_name" begin
                prototype = ArrayType(zeros(Float32, 32, 32))
                # 32×32 with b=16: 2×2 tile grid, no boundary tiles
                A = NextLA.TLRMatrix(prototype, 16, 16)

                @test size(A) == (32, 32)
                @test NextLA.grid_size(A) == (2, 2)
                @test A.offdiag.outer.order isa NextLA.TileRowMajor
                @test A.offdiag.inner.order isa NextLA.TileColMajor
                @test size(NextLA.dense_diag(A)) == (16, 16, 2)
                @test size(NextLA.dense_diag_corner(A)) == (16, 16, 0)
                @test size(NextLA.ranks(A)) == (4,)
                @test all(iszero, NextLA.ranks(A))
                @test Int(NextLA.execution_ranks(A)[NextLA.TLRmodule._rank_index(A, 1, 1)]) == 0
                @test Int(NextLA.execution_ranks(A)[NextLA.TLRmodule._rank_index(A, 1, 2)]) == 16

                synchronize(A.offdiag.outer.data)
                synchronize(A.offdiag.inner.data)
                synchronize(A.D)
                synchronize(A.D_corner)
            end
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), -1, 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 0, 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), (2, 0), 2)
        @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 2, -1)

        A = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
        @test_throws BoundsError  NextLA.TLRmodule.tile_linear_index(A.order, NextLA.grid_size(A)..., 3, 1)
        @test_throws BoundsError  NextLA.TLRmodule.tile_linear_index(A.order, NextLA.grid_size(A)..., 1, 3)
        @test_throws ArgumentError NextLA.get_factors(A, 1, 1)
        @test_throws ArgumentError NextLA.TLRMatrix(
            zeros(Float64, 8, 8), 4, 2; tile_order=NextLA.TileColMajor)

        # Smaller than one tile: the whole matrix is the dense corner.
        A_small = NextLA.TLRMatrix(zeros(Float64, 2, 2), 4, 2)
        @test size(NextLA.dense_diag(A_small)) == (4, 4, 0)
        @test size(NextLA.dense_diag_corner(A_small)) == (2, 2, 1)
    end
end

@testset "PaddedFTLRMatrix constructor and factor access" begin
    A = NextLA.PaddedFTLRMatrix(zeros(Float64, 10, 14), (4, 5), 3)

    @test !ismutable(A)
    @test size(A) == (10, 14)
    @test NextLA.grid_size(A) == (3, 3)
    @test NextLA.nominal_tile_size(A) == (4, 5)
    @test NextLA.tail_tile_size(A) == (2, 4)
    @test NextLA.maxrank(A) == 3
    @test size(NextLA.ranks(A)) == (9,)
    @test size(NextLA.residuals(A)) == (9,)

    # q_m = 2, q_n = 2. Full-size regular tiles include diagonal tiles.
    @test size(A.int_U) == (4, 3, 4)
    @test size(A.int_V) == (5, 3, 4)
    @test size(A.right_U) == (4, 3, 2)
    @test size(A.right_V) == (4, 3, 2)
    @test size(A.bottom_U) == (2, 3, 2)
    @test size(A.bottom_V) == (5, 3, 2)
    @test size(A.corner_U) == (2, 3, 1)
    @test size(A.corner_V) == (4, 3, 1)

    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.grid_size(A)..., 1, 1)] = 1
    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.grid_size(A)..., 1, 3)] = 2
    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.grid_size(A)..., 3, 1)] = 3
    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.grid_size(A)..., 3, 3)] = 2

    U_diag, V_diag = NextLA.get_factors(A, 1, 1)
    @test size(U_diag) == (4, 1)
    @test size(V_diag) == (5, 1)

    U_right, V_right = NextLA.get_factors(A, 1, 3)
    @test size(U_right) == (4, 2)
    @test size(V_right) == (4, 2)

    U_bottom, V_bottom = NextLA.get_factors(A, 3, 1)
    @test size(U_bottom) == (2, 3)
    @test size(V_bottom) == (5, 3)

    U_corner, V_corner = NextLA.get_factors(A, 3, 3)
    @test size(U_corner) == (2, 2)
    @test size(V_corner) == (4, 2)

    # Tile-aligned row-major variant: boundary storage is empty.
    B = NextLA.PaddedFTLRMatrix(zeros(Float32, 8, 10), (4, 5), 2; tile_order=NextLA.TileRowMajor)
    @test NextLA.grid_size(B) == (2, 2)
    @test NextLA.TLRmodule.tile_order(B) isa NextLA.TileRowMajor
    @test size(B.int_U) == (4, 2, 4)
    @test size(B.right_U, 3) == 0
    @test size(B.bottom_U, 3) == 0
    @test size(B.corner_U, 3) == 0

    @test_throws ArgumentError NextLA.PaddedFTLRMatrix(zeros(Float64, 5, 5), 0, 2)
    @test_throws ArgumentError NextLA.PaddedFTLRMatrix(zeros(Float64, 5, 5), (2, 0), 2)
    @test_throws ArgumentError NextLA.PaddedFTLRMatrix(zeros(Float64, 5, 5), 2, -1)
    @test_throws BoundsError NextLA.get_factors(A, 4, 1)
end

@testset "TLR exact-rank factor access includes boundary tiles" begin
    rank_grid = Int[0 2 2; 1 0 1; 2 1 0]
    A = NextLA.TLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, rank_grid;
        execution_rank_policy=:exact)
    @test size(NextLA.get_factors(A, 1, 2)[1]) == (4, 2)
    @test size(NextLA.get_factors(A, 3, 1)[1]) == (2, 2)
    @test size(NextLA.get_factors(A, 1, 3)[2]) == (2, 2)
    @test_throws ArgumentError NextLA.get_factors(A, 1, 1)
end
