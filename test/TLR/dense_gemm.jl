# TLR gemm! to dense: correctness against tile-reconstructed dense references.
#
# Parameter sweeps use pairwise coverage: every value of each dimension (tile
# orders, budget extremes, boundary regimes, transpose flags) appears, and every
# pair of dimensions is exercised at least once, without full cross-products.
# Drivers live in `helpers.jl`.

@testset "TLR gemm! to dense on CPU (dense-diagonal)" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    huge = 128 * 1024 * 1024
    nosync = _ -> nothing

    # (orderA, orderB, budget, n, b, r) — aligned, both boundary regimes
    # (tail ≥ Q and tail < Q), and zero rank.  Keep these grids small: a tiny
    # workspace exercises the same blocked schedule, but its cost grows steeply
    # with the number of tiles.
    cases = (
        (RM, CM, 1,    12, 4,  2),   # aligned grid
        (CM, RM, huge, 14, 4,  3),   # boundary, tail < Q (Q=3, s=2)
        (CM, CM, 1,    15, 4,  3),   # boundary, tail ≥ Q (Q=3, s=3)
        (RM, RM, huge, 14, 5,  0),   # zero rank + boundary
    )
    for (oA, oB, budget, n, b, r) in cases
        @testset "$(oA) * $(oB), n=$n b=$b r=$r budget=$budget" begin
            assert_tlr_gemm_matches_dense(Array, Float64, n, b, r, oA, oB, nosync; budget)
        end
    end
end

@testset "TLR compressed workspace bounds" begin
    A = NextLA.TLRMatrix(zeros(Float64, 35, 35), 8, 3)
    B = NextLA.TLRMatrix(zeros(Float64, 35, 35), 8, 3)
    fill_random_tlr!(A, Array; seed=71)
    fill_random_tlr!(B, Array; seed=72)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)
    for (transA, transB) in (('N', 'N'), ('N', 'T'), ('T', 'N'), ('T', 'T'))
        lo = NextLA.gemm_minimum_workspace_bytes(A, B; transA, transB)
        hi = NextLA.gemm_maximum_workspace_bytes(A, B; transA, transB)
        @test 0 < lo <= hi
    end
    lo = NextLA.gemm_minimum_workspace_bytes(A, B)
    for bytes in unique((lo, NextLA.gemm_maximum_workspace_bytes(A, B)))
        workspace = NextLA.DenseGemmWorkspace(A, B; bytes)
        C = zeros(Float64, size(A, 1), size(B, 2))
        NextLA.TLRmodule.gemm!(C, A, B; workspace)
        @test C ≈ reference
    end
    @test_throws ArgumentError NextLA.TLRmodule.gemm!(
        zeros(Float64, size(A, 1), size(B, 2)), A, B; workspace=lo - 1)
end

@testset "reserved-capacity TLR remains in-place populatable" begin
    A = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
    B = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
    NextLA.ranks(A)[NextLA.TLRmodule._rank_index(A, 1, 2)] = 1
    NextLA.ranks(B)[NextLA.TLRmodule._rank_index(B, 2, 1)] = 1
    UA, VA = NextLA.get_factors(A, 1, 2)
    UB, VB = NextLA.get_factors(B, 2, 1)
    UA .= 1; VA .= 2; UB .= 3; VB .= 4

    C = zeros(Float64, 8, 8)
    workspace = NextLA.gemm_minimum_workspace_bytes(A, B)
    NextLA.TLRmodule.gemm!(C, A, B; workspace)
    @test C ≈ reconstruct_tlr(A) * reconstruct_tlr(B)
end

# Whole-matrix transpose is a relabeling of stored factors (`logical_operands`) plus
# effective-order axis inference — the executors are unchanged.
@testset "whole-matrix logical N/T operands" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    A = NextLA.PaddedFTLRMatrix(zeros(Float64, 11, 14), (4, 3), 2; tile_order=RM)
    At = NextLA.TLRmodule.logical_operand(A, 'T')
    @test size(At) == (14, 11)
    @test NextLA.TLRmodule.nominal_tile_size(At) == (3, 4)
    @test NextLA.TLRmodule.tail_tile_size(At) == (2, 3)
    @test NextLA.TLRmodule.grid_size(At) == reverse(NextLA.grid_size(A))
    @test NextLA.TLRmodule.tile_order(At) isa typeof(CM)
    interior = NextLA.TLRmodule.InteriorRegion()
    right = NextLA.TLRmodule.RightRegion()
    bottom = NextLA.TLRmodule.BottomRegion()
    corner = NextLA.TLRmodule.CornerRegion()
    @test NextLA.TLRmodule.outer_factors(At, interior) === A.int_V
    @test NextLA.TLRmodule.inner_factors(At, interior) === A.int_U
    @test NextLA.TLRmodule.outer_factors(At, right) === A.bottom_V
    @test NextLA.TLRmodule.inner_factors(At, right) === A.bottom_U
    @test NextLA.TLRmodule.outer_factors(At, bottom) === A.right_V
    @test NextLA.TLRmodule.inner_factors(At, bottom) === A.right_U
    @test NextLA.TLRmodule.outer_factors(At, corner) === A.corner_V
    @test NextLA.TLRmodule.inner_factors(At, corner) === A.corner_U

    D = NextLA.TLRMatrix(zeros(Float64, 14, 14), 4, 2)
    Dt = NextLA.TLRmodule.logical_operand(D, 't')
    dref = NextLA.TLRmodule._diag_tile_ref(Dt, 1)
    @test parent(NextLA.TLRmodule._dense_data(dref)) === D.D
    @test NextLA.TLRmodule._dense_op(dref) == 'T'
end

@testset "dense-diagonal boundary transpose" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    huge = 128 * 1024 * 1024

    rows = (
        ('N', 'N', RM, CM, 1),
        ('N', 'T', CM, CM, huge),
        ('T', 'N', RM, RM, huge),
        ('T', 'T', CM, RM, 1),
    )
    for (tA, tB, oA, oB, budget) in rows
        assert_dense_diag_transpose_matches(14, 4, 3, oA, oB, tA, tB; budget)
    end
    # Dense diagonal remains meaningful when every low-rank tile has rank zero.
    assert_dense_diag_transpose_matches(14, 4, 0, RM, CM, 'T', 'T'; budget=1)

end

@testset "full-LR with one dense operand on CPU" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    huge = 128 * 1024 * 1024
    rows = (
        ('N', 'N', RM, 1), ('N', 'T', CM, huge),
        ('T', 'N', CM, 1), ('T', 'T', RM, huge),
    )
    for side in (:tlr_dense, :dense_tlr), (ta, tb, order, budget) in rows
        assert_dense_fulllr_gemm(Array, Float64, side, 14, 11, 13, (4, 3),
                                 order, ta, tb, _ -> nothing; budget)
    end

    A = NextLA.PaddedFTLRMatrix(zeros(Float64, 8, 9), (4, 3), 2)
    @test_throws DimensionMismatch _TLRM.gemm!(
        zeros(8, 7), A, zeros(8, 7); workspace=1)
    @test_throws DimensionMismatch _TLRM.gemm!(
        zeros(7, 8), zeros(7, 8), A; workspace=1)
    @test_throws ArgumentError _TLRM.gemm!(
        zeros(8, 7), A, zeros(9, 7); transB='X', workspace=1)

    Z = NextLA.PaddedFTLRMatrix(zeros(Float64, 8, 9), (4, 3), 0)
    Cz = ones(8, 7); _TLRM.gemm!(
        Cz, Z, zeros(9, 7); beta=-0.5, workspace=1)
    Dz = ones(6, 9); _TLRM.gemm!(
        Dz, zeros(6, 8), Z; beta=0.25, workspace=1)
    @test all(Cz .== -0.5) && all(Dz .== 0.25)
end

@testset "TLR gemm! to dense on CUDA" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
            assert_tlr_gemm_matches_dense(ArrayType, Float32, 12, 4, 2, orders..., synchronize;
                                          budget=1, alpha=Float32(1.2), beta=Float32(0.25),
                                          atol=5f-3, rtol=5f-3)

            @testset "precision policy" begin
                A16 = NextLA.PaddedFTLRMatrix(ArrayType(zeros(Float16, 10, 10)), 4, 2;
                                       tile_order=orders[1])
                B16 = NextLA.PaddedFTLRMatrix(ArrayType(zeros(Float16, 10, 10)), 4, 2;
                                       tile_order=orders[2])
                fill_random_tlr!(A16, ArrayType; seed=501)
                fill_random_tlr!(B16, ArrayType; seed=502)

                Dright = ArrayType(randn(Float16, 10, 7))
                Cright = ArrayType(zeros(Float32, 10, 7))
                NextLA.TLRmodule.gemm!(Cright, A16, Dright; compute=Float32,
                                       workspace=16)
                synchronize(Cright)
                @test isapprox(Array(Cright), Float32.(reconstruct_tlr(A16)) *
                                              Float32.(Array(Dright)); atol=0.2f0, rtol=0.03f0)

                Dleft = ArrayType(randn(Float16, 6, 10))
                Cleft = ArrayType(zeros(Float32, 6, 10))
                NextLA.TLRmodule.gemm!(Cleft, Dleft, B16; compute=Float32,
                                       workspace=16)
                synchronize(Cleft)
                @test isapprox(Array(Cleft), Float32.(Array(Dleft)) *
                                             Float32.(reconstruct_tlr(B16)); atol=0.2f0, rtol=0.03f0)

                A32 = NextLA.PaddedFTLRMatrix(ArrayType(zeros(Float32, 8, 8)), 4, 2)
                fill_random_tlr!(A32, ArrayType; seed=503)
                Ctf32 = ArrayType(zeros(Float32, 8, 8))

                Dtf32 = ArrayType(randn(Float32, 8, 6))
                NextLA.TLRmodule.gemm!(view(Ctf32, :, 1:6), A32, Dtf32;
                                       compute=NextLA.TF32(), workspace=1024)
                synchronize(Ctf32)
                @test isapprox(Array(Ctf32[:, 1:6]), reconstruct_tlr(A32) * Array(Dtf32);
                               atol=0.1f0, rtol=0.03f0)
            end
        end
    end
end

@testset "TF32 backend capability" begin
    for (backend_name, ArrayType, _) in available_backends()
        backend_name == "AMDGPU" || continue
        A = NextLA.TLRMatrix(ArrayType(zeros(Float32, 8, 8)), 4, 2)
        B = NextLA.TLRMatrix(ArrayType(zeros(Float32, 8, 8)), 4, 2)
        C = ArrayType(zeros(Float32, 8, 8))
        @test_throws ArgumentError NextLA.TLRmodule.gemm!(
            C, A, B; compute=NextLA.TF32(), workspace=1)
    end
end
