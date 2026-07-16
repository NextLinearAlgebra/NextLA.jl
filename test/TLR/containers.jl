# Container layer: tile-index geometry, constructors, storage shapes, factor access.

@testset "TLR geometry" begin
    # The column-major table is the hand-enumerated ground truth for the index
    # arithmetic; the row-major variant only flips the enumeration axis, so it is
    # pinned by the round-trip property plus spot checks below.
    @testset "column-major index table" begin
        A = NextLA.TLRDenseDiagMatrix(zeros(Float32, 10, 9), 4, 4; tile_order=NextLA.TileColMajor)
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
            @test NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., i, j) == linear
            @test NextLA.TLRmodule.inverse_tile_index(A.order, NextLA.tilegrid_size(A)..., linear) == (i, j)
            @test NextLA.tile_origin_coords(A, i, j) == (p0, q0)
            @test NextLA.tile_size(A, i, j) == (tile_m, tile_n)
        end

        @test NextLA.tilegrid_size(A) == (3, 3)
        @test NextLA.TLRmodule._offdiag_index(A.order, NextLA.tilegrid_size(A)..., 2, 1) == 1
        @test NextLA.TLRmodule._offdiag_index(A.order, NextLA.tilegrid_size(A)..., 1, 3) == 5
        @test NextLA.TLRmodule._offdiag_coords(A.order, NextLA.tilegrid_size(A)..., 1) == (2, 1)
        @test NextLA.TLRmodule._offdiag_coords(A.order, NextLA.tilegrid_size(A)..., 5) == (1, 3)
    end

    @testset "row-major round trip and offdiag slots" begin
        A = NextLA.TLRDenseDiagMatrix(zeros(Float32, 10, 9), 4, 4; tile_order=NextLA.TileRowMajor)
        gs = NextLA.tilegrid_size(A)
        @test gs == (3, 3)

        for linear in 1:prod(gs)
            i, j = NextLA.TLRmodule.inverse_tile_index(A.order, gs..., linear)
            @test NextLA.TLRmodule.tile_linear_index(A.order, gs..., i, j) == linear
        end
        # Row-major enumerates j fastest — the transpose of the col-major table.
        @test NextLA.TLRmodule.tile_linear_index(A.order, gs..., 1, 2) == 2
        @test NextLA.TLRmodule.tile_linear_index(A.order, gs..., 2, 1) == 4

        @test NextLA.TLRmodule._offdiag_index(A.order, gs..., 1, 2) == 1
        @test NextLA.TLRmodule._offdiag_index(A.order, gs..., 3, 1) == 5
        @test NextLA.TLRmodule._offdiag_coords(A.order, gs..., 1) == (1, 2)
        @test NextLA.TLRmodule._offdiag_coords(A.order, gs..., 5) == (3, 1)
    end
end

@testset "TLRDenseDiagMatrix constructor and storage allocation" begin
    # Rectangular nominal tiles with tails on both axes: exercises every storage
    # region (interior, right, bottom, dense corner) in one construction.
    @testset "rectangular nominal tile geometry" begin
        A = NextLA.TLRDenseDiagMatrix(zeros(Float64, 10, 11), (4, 5), 3)

        @test !ismutable(A)
        @test size(A) == (10, 11)
        @test NextLA.TLRmodule.tile_order(A) isa NextLA.TileColMajor
        @test NextLA.nominal_tile_size(A) == (4, 5)
        @test NextLA.nominal_tile_size(A, 1) == 4
        @test NextLA.nominal_tile_size(A, 2) == 5
        @test NextLA.tail_tile_size(A) == (2, 1)
        @test NextLA.tail_tile_size(A, 1) == 2
        @test NextLA.tail_tile_size(A, 2) == 1
        @test NextLA.maxrank(A) == 3
        @test NextLA.tilegrid_size(A) == (3, 3)
        @test NextLA.tile_origin_coords(A, 3, 3) == (9, 11)
        @test NextLA.tile_size(A, 1, 1) == (4, 5)
        @test NextLA.tile_size(A, 3, 3) == (2, 1)

        @test size(A.int_U) == (4, 3, 2)
        @test size(A.int_V) == (5, 3, 2)
        @test size(A.right_U) == (4, 3, 2)
        @test size(A.right_V) == (1, 3, 2)
        @test size(A.bottom_U) == (2, 3, 2)
        @test size(A.bottom_V) == (5, 3, 2)
        @test size(A.D) == (4, 5, 2)
        @test size(A.D_corner) == (2, 1, 1)
        @test NextLA.dense_diag_corner(A) === A.D_corner
    end

    @testset "backend allocation smoke" begin
        for (backend_name, ArrayType, synchronize) in available_backends()
            @testset "$backend_name" begin
                prototype = ArrayType(zeros(Float32, 32, 32))
                # 32×32 with b=16: 2×2 tile grid, no boundary tiles
                A = NextLA.TLRDenseDiagMatrix(prototype, 16, 16)

                @test size(A) == (32, 32)
                @test NextLA.tilegrid_size(A) == (2, 2)
                @test size(A.int_U) == (16, 16, 2)
                @test size(A.right_U, 3) == 0   # no right boundary
                @test size(A.bottom_U, 3) == 0  # no bottom boundary
                @test size(NextLA.dense_diag(A)) == (16, 16, 2)
                @test size(NextLA.dense_diag_corner(A)) == (16, 16, 0)
                @test size(NextLA.ranks(A)) == (4,)
                @test Int(NextLA.ranks(A)[NextLA.TLRmodule._rank_index(A, 1, 1)]) == 16
                @test Int(NextLA.ranks(A)[NextLA.TLRmodule._rank_index(A, 1, 2)]) == 0

                synchronize(A.int_U)
                synchronize(A.int_V)
                synchronize(A.D)
                synchronize(A.D_corner)
            end
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError NextLA.TLRDenseDiagMatrix(zeros(Float64, 5, 5), -1, 2)
        @test_throws ArgumentError NextLA.TLRDenseDiagMatrix(zeros(Float64, 5, 5), 0, 2)
        @test_throws ArgumentError NextLA.TLRDenseDiagMatrix(zeros(Float64, 5, 5), (2, 0), 2)
        @test_throws ArgumentError NextLA.TLRDenseDiagMatrix(zeros(Float64, 5, 5), 2, -1)

        A = NextLA.TLRDenseDiagMatrix(zeros(Float64, 8, 8), 4, 2)
        @test_throws BoundsError  NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., 3, 1)
        @test_throws BoundsError  NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., 1, 3)
        @test_throws ArgumentError NextLA.TLRmodule.region_slot(A, 1, 1)

        # Smaller than one tile: the whole matrix is the dense corner.
        A_small = NextLA.TLRDenseDiagMatrix(zeros(Float64, 2, 2), 4, 2)
        @test size(NextLA.dense_diag(A_small)) == (4, 4, 0)
        @test size(NextLA.dense_diag_corner(A_small)) == (2, 2, 1)
    end
end

@testset "TLRMatrix constructor and factor access" begin
    A = NextLA.TLRMatrix(zeros(Float64, 10, 14), (4, 5), 3)

    @test !ismutable(A)
    @test size(A) == (10, 14)
    @test NextLA.tilegrid_size(A) == (3, 3)
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

    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., 1, 1)] = 1
    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., 1, 3)] = 2
    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., 3, 1)] = 3
    A.ranks[NextLA.TLRmodule.tile_linear_index(A.order, NextLA.tilegrid_size(A)..., 3, 3)] = 2

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
    B = NextLA.TLRMatrix(zeros(Float32, 8, 10), (4, 5), 2; tile_order=NextLA.TileRowMajor)
    @test NextLA.tilegrid_size(B) == (2, 2)
    @test NextLA.TLRmodule.tile_order(B) isa NextLA.TileRowMajor
    @test size(B.int_U) == (4, 2, 4)
    @test size(B.right_U, 3) == 0
    @test size(B.bottom_U, 3) == 0
    @test size(B.corner_U, 3) == 0

    @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 0, 2)
    @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), (2, 0), 2)
    @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 2, -1)
    @test_throws BoundsError NextLA.get_factors(A, 4, 1)
end

@testset "TLR factor access — boundary tile categories" begin
    # 10×10 with b=4: 3×3 tile grid, tail_m=tail_n=2
    A_tlr = NextLA.TLRDenseDiagMatrix(zeros(Float64, 10, 10), 4, 3)

    internal_slot = NextLA.TLRmodule._rank_index(A_tlr, 1, 2)
    bottom_slot   = NextLA.TLRmodule._rank_index(A_tlr, 3, 1)
    right_slot    = NextLA.TLRmodule._rank_index(A_tlr, 1, 3)

    # Write data for interior tile
    ki = NextLA.TLRmodule._offdiag_index(A_tlr.order, 2, 2, 1, 2)
    A_tlr.ranks[internal_slot] = 2
    A_tlr.int_U[:, 1:2, ki] .= reshape(collect(21.0:28.0), 4, 2)
    A_tlr.int_V[:, 1:2, ki] .= reshape(collect(31.0:38.0), 4, 2)

    # Write data for bottom boundary tile
    kb = 1
    A_tlr.ranks[bottom_slot] = 2
    A_tlr.bottom_U[1:2, 1:2, kb] .= [1.0 2.0; 3.0 4.0]
    A_tlr.bottom_V[:, 1:2, kb]   .= reshape(collect(1.0:8.0), 4, 2)

    # Write data for right boundary tile
    kr_idx = 1
    A_tlr.ranks[right_slot] = 2
    A_tlr.right_U[:, 1:2, kr_idx]   .= reshape(collect(11.0:18.0), 4, 2)
    A_tlr.right_V[1:2, 1:2, kr_idx] .= [5.0 6.0; 7.0 8.0]

    # get_factors maps global tile coordinates to correctly shaped factor views.
    U_int, V_int = NextLA.get_factors(A_tlr, 1, 2)
    @test size(U_int) == (4, 2)
    @test size(V_int) == (4, 2)
    @test Matrix(U_int) == reshape(collect(21.0:28.0), 4, 2)
    @test Matrix(V_int) == reshape(collect(31.0:38.0), 4, 2)

    U_bottom, V_bottom = NextLA.get_factors(A_tlr, 3, 1)
    @test size(U_bottom) == (2, 2)
    @test size(V_bottom) == (4, 2)
    @test Matrix(U_bottom) == [1.0 2.0; 3.0 4.0]
    @test Matrix(V_bottom) == reshape(collect(1.0:8.0), 4, 2)

    U_right, V_right = NextLA.get_factors(A_tlr, 1, 3)
    @test size(U_right) == (4, 2)
    @test size(V_right) == (2, 2)
    @test Matrix(U_right) == reshape(collect(11.0:18.0), 4, 2)
    @test Matrix(V_right) == [5.0 6.0; 7.0 8.0]

    @test_throws ArgumentError NextLA.get_factors(A_tlr, 1, 1)
end
