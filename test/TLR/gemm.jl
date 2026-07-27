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

@testset "full-LR TLR gemm! to dense on CPU" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    huge = 128 * 1024 * 1024
    nosync = _ -> nothing

    # (mA, k, nB, tsA, tsB, r, orderA, orderB, budget, tol) — square/rect grids,
    # independent boundary tails, rectangular tiles, single contraction tile,
    # zero rank; orders and budgets distributed pairwise across rows.
    cases = (
        (12, 12, 12, (4, 4), (4, 4), 2, RM, CM, 1,    1e-10),  # square aligned grid
        (10,  7, 12, (4, 4), (4, 4), 3, CM, RM, huge, 1e-10),  # rectangular grid
        (14, 14, 14, (4, 4), (4, 4), 3, CM, CM, 1,    1e-9),   # boundary square
        (10, 11, 13, (4, 4), (4, 4), 3, RM, CM, huge, 1e-9),   # independent m/k/n tails
        (10,  8,  9, (4, 3), (3, 5), 3, RM, RM, 1,    1e-9),   # rect tiles + tails
        (10,  6,  6, (5, 6), (6, 2), 2, RM, CM, huge, 1e-9),   # tall output tile, q_c=1
        (10, 10, 10, (4, 4), (4, 4), 0, CM, RM, 1,    1e-10),  # zero rank + boundary
    )
    for (mA, k, nB, tsA, tsB, r, oA, oB, budget, tol) in cases
        @testset "$(mA)×$(k)×$(nB) tiles=$(tsA)/$(tsB) r=$r $(oA)*$(oB) budget=$budget" begin
            assert_fulllr_gemm_matches_dense(Array, Float64, mA, k, nB, tsA, tsB, r,
                                             oA, oB, nosync; budget, atol=tol, rtol=tol)
        end
    end
end

# The interior off-diagonal product `O_A O_B` picks its Stage-2/3 association (fold)
# from storage layout so the reduction is a write-once fused Stage 3 without any
# transpose: FoldLeft (stack B's Z) iff B is TileColMajor on a FullGrid, else FoldRight.
@testset "FoldLeft layout-driven fold selection" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    FL = NextLA.TLRmodule.FoldLeft; FR = NextLA.TLRmodule.FoldRight

    @testset "choose_fold truth table (FullGrid)" begin
        for (oa, ob, expect) in ((RM, CM, FL), (CM, CM, FL), (CM, RM, FR), (RM, RM, FR))
            A = NextLA.TLRMatrix(zeros(Float64, 12, 9), (4, 3), 3; tile_order=oa)
            B = NextLA.TLRMatrix(zeros(Float64, 9, 10), (3, 5), 3; tile_order=ob)
            ops = NextLA.TLRmodule.logical_operands(A, B)
            @test NextLA.TLRmodule.choose_fold(ops) isa expect
        end
        # dense-diagonal (SkipDiag) always FoldRight, even with a TileColMajor B
        Ad = NextLA.TLRDenseDiagMatrix(zeros(Float64, 16, 16), 4, 3; tile_order=CM)
        Bd = NextLA.TLRDenseDiagMatrix(zeros(Float64, 16, 16), 4, 3; tile_order=CM)
        ops = NextLA.TLRmodule.logical_operands(Ad, Bd)
        @test NextLA.TLRmodule.choose_fold(ops) isa FR
    end

    # Forced fold reproduces the dense interior product `O_A O_B` (= full A·B on an
    # aligned FullGrid). FoldLeft is valid only for B TileColMajor; FoldRight always.
    function assert_fold_matches(oa, ob, mA, k, nB, tsA, tsB, r, fold, budget)
        A = NextLA.TLRMatrix(zeros(Float64, mA, k), tsA, r; tile_order=oa)
        B = NextLA.TLRMatrix(zeros(Float64, k, nB), tsB, r; tile_order=ob)
        fill_random_tlr!(A, Array; seed=1)
        fill_random_tlr!(B, Array; seed=2)
        C = zeros(Float64, mA, nB)
        NextLA.TLRmodule._offdiag_offdiag_gemm!(C, A, B; alpha=1.0, beta=0.0, budget=budget, fold=fold)
        ref = reconstruct_tlr(A) * reconstruct_tlr(B)
        @test isapprox(C, ref; atol=1e-9, rtol=1e-9)
    end

    @testset "forced fold matches dense" begin
        # Tiny and unrestricted workspace are exercised by the end-to-end sweep;
        # here one representative call pins each association independently.
        assert_fold_matches(RM, CM, 12, 9, 10, (4, 3), (3, 5), 3, FL(), 1)
        assert_fold_matches(CM, CM, 12, 9, 10, (4, 3), (3, 5), 2, FR(), 128 * 1024 * 1024)
    end
end

# Whole-matrix transpose is a relabeling of stored factors (`logical_operands`) plus
# effective-order axis inference — the executors are unchanged.
@testset "whole-matrix logical N/T operands" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    A = NextLA.TLRMatrix(zeros(Float64, 11, 14), (4, 3), 2; tile_order=RM)
    At = NextLA.TLRmodule.logical_operand(A, 'T')
    @test size(At) == (14, 11)
    @test NextLA.TLRmodule.nominal_tile_size(At) == (3, 4)
    @test NextLA.TLRmodule.tail_tile_size(At) == (2, 3)
    @test NextLA.TLRmodule.tilegrid_size(At) == reverse(NextLA.tilegrid_size(A))
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

    D = NextLA.TLRDenseDiagMatrix(zeros(Float64, 14, 14), 4, 2; tile_order=CM)
    Dt = NextLA.TLRmodule.logical_operand(D, 't')
    @test NextLA.TLRmodule.outer_factors(Dt, right) === D.bottom_V
    @test NextLA.TLRmodule.outer_factors(Dt, bottom) === D.right_V
    dref = NextLA.TLRmodule._diag_tile_ref(Dt, 1)
    @test parent(NextLA.TLRmodule._dense_data(dref)) === D.D
    @test NextLA.TLRmodule._dense_op(dref) == 'T'
end

@testset "transpose flags on complete FullGrid operands" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    huge = 128 * 1024 * 1024

    # All four op-combinations, each on a distinct layout pair; square and
    # rectangular op tile sizes split between them.
    @testset "op(A)·op(B), pairwise combos × layouts" begin
        rows = (
            ('N', 'N', RM, CM, (12, 8, 16), (4, 4), (4, 4)),
            ('N', 'T', CM, RM, (12, 8, 16), (4, 4), (4, 4)),
            ('T', 'N', CM, CM, (12, 9, 10), (4, 3), (3, 5)),
            ('T', 'T', RM, RM, (12, 9, 10), (4, 3), (3, 5)),
        )
        for (tA, tB, oA, oB, (m, k, n), tsA, tsB) in rows
            assert_transpose_matches_dense(m, k, n, tsA, tsB, oA, oB, tA, tB)
        end
    end

    @testset "independent boundary tails × budgets" begin
        # Effective A is 14×11 with tiles 4×3; effective B is 11×13 with
        # tiles 3×5. All three dimensions have independent boundary tails.
        assert_transpose_matches_dense(14, 11, 13, (4, 3), (3, 5), RM, CM, 'T', 'N'; budget=1)
        assert_transpose_matches_dense(14, 11, 13, (4, 3), (3, 5), CM, RM, 'N', 'T'; budget=huge)
    end

    @testset "flag and effective-geometry validation" begin
        A = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
        B = NextLA.TLRMatrix(zeros(Float64, 8, 8), 4, 2)
        @test_throws ArgumentError NextLA.TLRmodule.gemm!(
            zeros(8, 8), A, B; transA='X', workspace=1)
        Cn = zeros(8, 8); Ct = zeros(8, 8)
        workspace = NextLA.gemm_minimum_workspace_bytes(A, B)
        NextLA.TLRmodule.gemm!(Cn, A, B; transA='n', transB='n', workspace)
        NextLA.TLRmodule.gemm!(Ct, A, B; transA='N', transB='N', workspace)
        @test Cn == Ct
        @test_throws DimensionMismatch NextLA.TLRmodule.gemm!(
            zeros(7, 8), A, B; workspace=1)

        Bt = NextLA.TLRMatrix(zeros(Float64, 8, 8), (2, 4), 2)
        @test_throws DimensionMismatch NextLA.TLRmodule.gemm!(
            zeros(8, 8), A, Bt; workspace=1)

        Af = NextLA.TLRMatrix(zeros(Float32, 8, 8), 4, 2)
        Bf = NextLA.TLRMatrix(zeros(Float32, 8, 8), 4, 2)
        @test_throws ArgumentError NextLA.TLRmodule.gemm!(
            zeros(Float64, 8, 8), Af, Bf; workspace=1)
        @test_throws ArgumentError NextLA.TLRmodule.gemm!(
            zeros(Float32, 8, 8), Af, Bf; compute=Float64, workspace=1)
        @test_throws ArgumentError NextLA.TLRmodule.gemm!(
            zeros(Float32, 8, 8), Af, Bf; compute=NextLA.TF32(), workspace=1)
    end
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

    Arect = NextLA.TLRDenseDiagMatrix(zeros(Float64, 8, 12), 4, 2)
    Brect = NextLA.TLRDenseDiagMatrix(zeros(Float64, 8, 8), 4, 2)
    @test_throws ArgumentError NextLA.TLRmodule.gemm!(
        zeros(12, 8), Arect, Brect; transA='T', workspace=1)
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

    A = NextLA.TLRMatrix(zeros(Float64, 8, 9), (4, 3), 2)
    @test_throws DimensionMismatch _TLRM.gemm!(
        zeros(8, 7), A, zeros(8, 7); workspace=1)
    @test_throws DimensionMismatch _TLRM.gemm!(
        zeros(7, 8), zeros(7, 8), A; workspace=1)
    @test_throws ArgumentError _TLRM.gemm!(
        zeros(8, 7), A, zeros(9, 7); transB='X', workspace=1)

    Z = NextLA.TLRMatrix(zeros(Float64, 8, 9), (4, 3), 0)
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
            # Representative complete-operand transpose check.  CPU covers the
            # remaining layout/transpose combinations exhaustively.
            assert_transpose_matches_dense(14, 11, 13, (4, 3), (3, 5), orders..., 'T', 'N';
                                             budget=1, ArrayType, synchronize,
                                             atol=1e-8, rtol=1e-8)

            @testset "reusable global arena" begin
                Aarena = NextLA.TLRMatrix(
                    ArrayType(zeros(Float32, 10, 10)), 4, 2;
                    tile_order=orders[1])
                Barena = NextLA.TLRMatrix(
                    ArrayType(zeros(Float32, 10, 10)), 4, 2;
                    tile_order=orders[2])
                fill_random_tlr!(Aarena, ArrayType; seed=491)
                fill_random_tlr!(Barena, ArrayType; seed=492)
                bytes = NextLA.gemm_minimum_workspace_bytes(Aarena, Barena)
                workspace = NextLA.DenseGemmWorkspace(Aarena, Barena; bytes)
                storage = workspace.storage
                reference = reconstruct_tlr(Aarena) * reconstruct_tlr(Barena)
                for _ in 1:2
                    C = ArrayType(zeros(Float32, 10, 10))
                    NextLA.TLRmodule.gemm!(C, Aarena, Barena; workspace)
                    synchronize(C)
                    @test Array(C) ≈ reference rtol=3f-4 atol=3f-4
                    @test workspace.storage === storage
                end
            end

            @testset "precision policy" begin
                A16 = NextLA.TLRMatrix(ArrayType(zeros(Float16, 10, 10)), 4, 2;
                                       tile_order=orders[1])
                B16 = NextLA.TLRMatrix(ArrayType(zeros(Float16, 10, 10)), 4, 2;
                                       tile_order=orders[2])
                fill_random_tlr!(A16, ArrayType; seed=501)
                fill_random_tlr!(B16, ArrayType; seed=502)
                ref16 = Float32.(reconstruct_tlr(A16)) * Float32.(reconstruct_tlr(B16))
                workspace16 = NextLA.gemm_minimum_workspace_bytes(A16, B16)

                C16 = ArrayType(zeros(Float16, 10, 10))
                NextLA.TLRmodule.gemm!(
                    C16, A16, B16; compute=Float32, workspace=workspace16)
                synchronize(C16)
                @test isapprox(Float32.(Array(C16)), ref16; atol=0.2f0, rtol=0.03f0)

                C16native = ArrayType(zeros(Float16, 10, 10))
                NextLA.TLRmodule.gemm!(
                    C16native, A16, B16; compute=Float16, workspace=workspace16)
                synchronize(C16native)
                @test isapprox(Float32.(Array(C16native)), ref16; atol=0.5f0, rtol=0.06f0)

                C32 = ArrayType(zeros(Float32, 10, 10))
                NextLA.TLRmodule.gemm!(
                    C32, A16, B16; compute=Float32, workspace=workspace16)
                synchronize(C32)
                @test isapprox(Array(C32), ref16; atol=0.2f0, rtol=0.03f0)

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

                # GEMM scalars follow compute precision, not FP16 factor storage.
                # The factors make A*B exactly 100I even through FP16 S/T storage,
                # so rounding alpha to FP16 would produce 100 instead of 100.01.
                Aexact = NextLA.TLRMatrix(ArrayType(zeros(Float16, 4, 4)), 4, 4)
                Bexact = NextLA.TLRMatrix(ArrayType(zeros(Float16, 4, 4)), 4, 4)
                I16 = Matrix{Float16}(I, 4, 4)
                Aexact.int_U .= ArrayType(reshape(10 .* I16, 4, 4, 1))
                Aexact.int_V .= ArrayType(reshape(I16, 4, 4, 1))
                Bexact.int_U .= ArrayType(reshape(10 .* I16, 4, 4, 1))
                Bexact.int_V .= ArrayType(reshape(I16, 4, 4, 1))
                Aexact.ranks .= 4
                Bexact.ranks .= 4
                alpha32 = Float32(1.0001)
                beta32 = Float32(0.1234)
                Cscalar = ArrayType(fill(Float32(10), 4, 4))
                NextLA.TLRmodule.gemm!(Cscalar, Aexact, Bexact;
                                       alpha=alpha32, beta=beta32,
                                       workspace=NextLA.gemm_minimum_workspace_bytes(Aexact, Bexact))
                synchronize(Cscalar)
                scalar_ref = alpha32 .* (Float32(100) .* Matrix{Float32}(I, 4, 4)) .+
                             beta32 .* fill(Float32(10), 4, 4)
                @test isapprox(Array(Cscalar), scalar_ref; atol=2f-5, rtol=2f-6)

                A32 = NextLA.TLRMatrix(ArrayType(zeros(Float32, 8, 8)), 4, 2)
                B32 = NextLA.TLRMatrix(ArrayType(zeros(Float32, 8, 8)), 4, 2)
                fill_random_tlr!(A32, ArrayType; seed=503)
                fill_random_tlr!(B32, ArrayType; seed=504)
                Ctf32 = ArrayType(zeros(Float32, 8, 8))
                NextLA.TLRmodule.gemm!(
                    Ctf32, A32, B32; compute=NextLA.TF32(),
                    workspace=NextLA.gemm_minimum_workspace_bytes(A32, B32))
                synchronize(Ctf32)
                ref32 = reconstruct_tlr(A32) * reconstruct_tlr(B32)
                @test isapprox(Array(Ctf32), ref32; atol=0.1f0, rtol=0.03f0)

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
