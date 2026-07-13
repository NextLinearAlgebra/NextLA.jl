function fill_random_tlr!(A_tlr::NextLA.TLRDenseDiagMatrix, ArrayType::Type; seed::Integer)
    rng = MersenneTwister(seed)
    T = eltype(A_tlr)
    A_tlr.D .= ArrayType(randn(rng, T, size(A_tlr.D)))
    A_tlr.D_corner .= ArrayType(randn(rng, T, size(A_tlr.D_corner)))
    A_tlr.int_U .= ArrayType(randn(rng, T, size(A_tlr.int_U)))
    A_tlr.int_V .= ArrayType(randn(rng, T, size(A_tlr.int_V)))
    A_tlr.right_U .= ArrayType(randn(rng, T, size(A_tlr.right_U)))
    A_tlr.right_V .= ArrayType(randn(rng, T, size(A_tlr.right_V)))
    A_tlr.bottom_U .= ArrayType(randn(rng, T, size(A_tlr.bottom_U)))
    A_tlr.bottom_V .= ArrayType(randn(rng, T, size(A_tlr.bottom_V)))
    A_tlr.ranks .= A_tlr.maxrank
    return A_tlr
end

function assert_tlr_gemm_matches_dense(ArrayType::Type, T::Type, n::Int, b::Int, r::Int,
                                       orderA, orderB, synchronize; budget::Int,
                                       alpha=T(1.3), beta=T(-0.4), atol=1e-10, rtol=1e-10)
    A_tlr = NextLA.TLRDenseDiagMatrix(ArrayType(zeros(T, n, n)), b, r; tile_order=orderA)
    B_tlr = NextLA.TLRDenseDiagMatrix(ArrayType(zeros(T, n, n)), b, r; tile_order=orderB)
    fill_random_tlr!(A_tlr, ArrayType; seed=101)
    fill_random_tlr!(B_tlr, ArrayType; seed=202)

    rng = MersenneTwister(303)
    C0_cpu = randn(rng, T, n, n)
    C = ArrayType(C0_cpu)
    NextLA.TLRmodule.gemm!(C, A_tlr, B_tlr; alpha=alpha, beta=beta, max_workspace=budget)
    synchronize(C)

    A_dense = reconstruct_tlr(A_tlr)
    B_dense = reconstruct_tlr(B_tlr)
    C_ref = alpha * A_dense * B_dense + beta * C0_cpu
    @test isapprox(Array(C), C_ref; atol=atol, rtol=rtol)
end

@testset "TLR gemm! to dense on CPU" begin
    orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
    for orderA in orders, orderB in orders
        @testset "$(orderA) * $(orderB), budget=1" begin
            assert_tlr_gemm_matches_dense(Array, Float64, 12, 4, 2, orderA, orderB, _ -> nothing;
                                          budget=1)
        end
        @testset "$(orderA) * $(orderB), large budget" begin
            assert_tlr_gemm_matches_dense(Array, Float64, 16, 4, 3, orderA, orderB, _ -> nothing;
                                          budget=128 * 1024 * 1024)
        end
    end

    @testset "zero rank and one tile" begin
        assert_tlr_gemm_matches_dense(Array, Float64, 8, 4, 0,
                                      NextLA.TileRowMajor(), NextLA.TileColMajor(), _ -> nothing;
                                      budget=1)
        assert_tlr_gemm_matches_dense(Array, Float64, 4, 4, 2,
                                      NextLA.TileColMajor(), NextLA.TileRowMajor(), _ -> nothing;
                                      budget=1)
    end

    # Non-uniform tiling (m % b ≠ 0): exercises the boundary decomposition — interior,
    # right/bottom panels and corner. Covers the tail ≥ Q regime (n=87,b=16 → Q=5,s=7)
    # that the Stage-3 tile-size fix unblocked, and tail < Q (n=35,b=8 → Q=4,s=3).
    @testset "boundary tiling (m % b ≠ 0)" begin
        for orderA in orders, orderB in orders
            for (n, b, r) in ((35, 8, 3), (87, 16, 4), (46, 12, 5))
                @testset "$(orderA) * $(orderB), n=$n b=$b" begin
                    assert_tlr_gemm_matches_dense(Array, Float64, n, b, r, orderA, orderB, _ -> nothing;
                                                  budget=1)
                    assert_tlr_gemm_matches_dense(Array, Float64, n, b, r, orderA, orderB, _ -> nothing;
                                                  budget=128 * 1024 * 1024)
                end
            end
        end
        @testset "boundary with zero rank" begin
            assert_tlr_gemm_matches_dense(Array, Float64, 46, 10, 0,
                                          NextLA.TileColMajor(), NextLA.TileRowMajor(), _ -> nothing;
                                          budget=1)
        end
    end

end

function fill_random_tlr!(A_tlr::NextLA.TLRMatrix, ArrayType::Type; seed::Integer)
    rng = MersenneTwister(seed)
    T = eltype(A_tlr)
    for f in (A_tlr.int_U, A_tlr.int_V, A_tlr.right_U, A_tlr.right_V,
              A_tlr.bottom_U, A_tlr.bottom_V, A_tlr.corner_U, A_tlr.corner_V)
        length(f) == 0 && continue
        f .= ArrayType(randn(rng, T, size(f)))
    end
    A_tlr.ranks .= A_tlr.maxrank
    return A_tlr
end

# Fully low-rank TLR × TLR → dense over a (possibly rectangular) tile-aligned grid.
# `A` is `mA×k`, `B` is `k×nB` (all divisible by `b`), so there are no boundary tiles
# and the whole product is the `FullGrid` interior term.
function assert_fulllr_gemm_matches_dense(ArrayType::Type, T::Type, mA::Int, k::Int, nB::Int,
                                          b::Int, r::Int, orderA, orderB, synchronize; budget::Int,
                                          alpha=T(1.3), beta=T(-0.4), atol=1e-10, rtol=1e-10)
    A_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, mA, k)), b, r; tile_order=orderA)
    B_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, k, nB)), b, r; tile_order=orderB)
    fill_random_tlr!(A_tlr, ArrayType; seed=101)
    fill_random_tlr!(B_tlr, ArrayType; seed=202)

    rng = MersenneTwister(303)
    C0_cpu = randn(rng, T, mA, nB)
    C = ArrayType(C0_cpu)
    NextLA.TLRmodule.gemm!(C, A_tlr, B_tlr; alpha=alpha, beta=beta, max_workspace=budget)
    synchronize(C)

    A_dense = reconstruct_tlr(A_tlr)
    B_dense = reconstruct_tlr(B_tlr)
    C_ref = alpha * A_dense * B_dense + beta * C0_cpu
    @test isapprox(Array(C), C_ref; atol=atol, rtol=rtol)
end

# Rectangular *tiles* (bm ≠ bn): `A` uses tile size `(bm, bk)`, `B` uses `(bk, bn)`, so
# the contraction tiling `bk` aligns but the output tile is `bm × bn`. Only the buffer
# sizes change vs. square tiles; the result must still match the dense product.
function assert_fulllr_gemm_rect_tiles(ArrayType::Type, T::Type, mA::Int, k::Int, nB::Int,
                                       bm::Int, bk::Int, bn::Int, r::Int, orderA, orderB, synchronize;
                                       budget::Int, alpha=T(1.3), beta=T(-0.4), atol=1e-9, rtol=1e-9)
    A_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, mA, k)), (bm, bk), r; tile_order=orderA)
    B_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, k, nB)), (bk, bn), r; tile_order=orderB)
    fill_random_tlr!(A_tlr, ArrayType; seed=101)
    fill_random_tlr!(B_tlr, ArrayType; seed=202)

    rng = MersenneTwister(303)
    C0_cpu = randn(rng, T, mA, nB)
    C = ArrayType(C0_cpu)
    NextLA.TLRmodule.gemm!(C, A_tlr, B_tlr; alpha=alpha, beta=beta, max_workspace=budget)
    synchronize(C)

    A_dense = reconstruct_tlr(A_tlr)
    B_dense = reconstruct_tlr(B_tlr)
    C_ref = alpha * A_dense * B_dense + beta * C0_cpu
    @test isapprox(Array(C), C_ref; atol=atol, rtol=rtol)
end

@testset "full-LR TLR gemm! to dense on CPU" begin
    orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
    for orderA in orders, orderB in orders
        @testset "$(orderA) * $(orderB)" begin
            for budget in (1, 128 * 1024 * 1024)
                # square grid (q = 3)
                assert_fulllr_gemm_matches_dense(Array, Float64, 12, 12, 12, 4, 2,
                                                 orderA, orderB, _ -> nothing; budget=budget)
                # rectangular grid: q_m=3, q_c=2, q_n=4
                assert_fulllr_gemm_matches_dense(Array, Float64, 12, 8, 16, 4, 3,
                                                 orderA, orderB, _ -> nothing; budget=budget)
            end
        end
    end

    # Boundary tiles (n % b ≠ 0): square, equal-size A, B so the right/bottom/corner
    # panels are populated. Exercises all four regions with low-rank corners.
    @testset "boundary tiling (n % b ≠ 0)" begin
        for orderA in orders, orderB in orders
            for (n, b) in ((14, 4), (35, 8), (46, 12)), budget in (1, 128 * 1024 * 1024)
                assert_fulllr_gemm_matches_dense(Array, Float64, n, n, n, b, 3,
                                                 orderA, orderB, _ -> nothing; budget=budget, atol=1e-9, rtol=1e-9)
            end
        end
    end

    # Rectangular boundary: independent tails in m, k, n (A is mA×k, B is k×nB).
    @testset "rectangular boundary" begin
        for orderA in orders, orderB in orders
            for (mA, k, nB, b) in ((14, 10, 18, 4), (22, 14, 10, 4), (35, 19, 27, 8)), budget in (1, 128 * 1024 * 1024)
                assert_fulllr_gemm_matches_dense(Array, Float64, mA, k, nB, b, 3,
                                                 orderA, orderB, _ -> nothing; budget=budget, atol=1e-9, rtol=1e-9)
            end
        end
    end

    # Rectangular tiles (bm ≠ bk ≠ bn): only the intermediate buffer sizes differ.
    @testset "rectangular tiles (bm ≠ bn)" begin
        for orderA in orders, orderB in orders, budget in (1, 128 * 1024 * 1024)
            # aligned grid, no boundary: mA=12(3), k=9(3), nB=10(2); tiles A=(4,3), B=(3,5)
            assert_fulllr_gemm_rect_tiles(Array, Float64, 12, 9, 10, 4, 3, 5, 3,
                                          orderA, orderB, _ -> nothing; budget=budget)
            # with tails in m, k and n: mA=14(tail2), k=11(tail2), nB=13(tail3)
            assert_fulllr_gemm_rect_tiles(Array, Float64, 14, 11, 13, 4, 3, 5, 3,
                                          orderA, orderB, _ -> nothing; budget=budget)
            # tall output tile (bm > bn) and single contraction tile
            assert_fulllr_gemm_rect_tiles(Array, Float64, 15, 6, 8, 5, 6, 2, 2,
                                          orderA, orderB, _ -> nothing; budget=budget)
        end
    end

    @testset "edge cases" begin
        # single contraction tile (q_c = 1): kept by FullGrid, unlike the dense-diag interior
        assert_fulllr_gemm_matches_dense(Array, Float64, 8, 4, 8, 4, 2,
                                         NextLA.TileRowMajor(), NextLA.TileColMajor(), _ -> nothing; budget=1)
        # zero rank (aligned and boundary)
        assert_fulllr_gemm_matches_dense(Array, Float64, 8, 8, 8, 4, 0,
                                         NextLA.TileColMajor(), NextLA.TileRowMajor(), _ -> nothing; budget=1)
        assert_fulllr_gemm_matches_dense(Array, Float64, 14, 14, 14, 4, 0,
                                         NextLA.TileColMajor(), NextLA.TileRowMajor(), _ -> nothing; budget=1)
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
        for budget in (1, 128 * 1024 * 1024)
            for oa in (RM, CM)                        # B TileColMajor ⇒ FoldLeft valid
                assert_fold_matches(oa, CM, 12, 12, 12, (4, 4), (4, 4), 3, FL(), budget)  # square
                assert_fold_matches(oa, CM, 12, 9, 10, (4, 3), (3, 5), 3, FL(), budget)   # rect tiles
                assert_fold_matches(oa, CM, 12, 9, 10, (4, 3), (3, 5), 2, FR(), budget)   # FoldRight sanity
            end
            assert_fold_matches(RM, CM, 15, 6, 8, (5, 6), (6, 2), 2, FL(), budget)        # tall output tile
        end
    end
end

# Phase 1: transpose flags (`op(X) = Xᵀ` when the flag ≠ 'N') on the aligned FullGrid
# interior. A transpose is a relabeling of stored factors (`logical_operands`) plus
# effective-order axis inference — the executors are unchanged. Verify all four
# op-combinations against the dense `op(A)·op(B)`.
function assert_transpose_matches_dense(m, k, n, tsA, tsB, oA, oB, transA, transB;
                                        alpha=1.3, beta=-0.4, atol=1e-9, rtol=1e-9)
    bm, bk = tsA
    bk2, bn = tsB
    @assert bk == bk2
    # tsA/tsB are the *op* tile sizes; the stored matrix/tiles flip on transpose.
    storedA, tileA = transA == 'T' ? ((k, m), (bk, bm)) : ((m, k), (bm, bk))
    storedB, tileB = transB == 'T' ? ((n, k), (bn, bk)) : ((k, n), (bk, bn))
    A = NextLA.TLRMatrix(zeros(Float64, storedA...), tileA, 3; tile_order=oA)
    B = NextLA.TLRMatrix(zeros(Float64, storedB...), tileB, 3; tile_order=oB)
    fill_random_tlr!(A, Array; seed=1)
    fill_random_tlr!(B, Array; seed=2)
    C0 = randn(MersenneTwister(7), Float64, m, n)
    C = copy(C0)
    NextLA.TLRmodule.gemm!(C, A, B; alpha=alpha, beta=beta, transA=transA, transB=transB)
    opd(D, t) = t == 'T' ? permutedims(D) : D
    ref = alpha .* (opd(reconstruct_tlr(A), transA) * opd(reconstruct_tlr(B), transB)) .+ beta .* C0
    @test isapprox(C, ref; atol=atol, rtol=rtol)
end

@testset "transpose flags on aligned FullGrid" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    @testset "op(A)·op(B), all four combos × layouts" begin
        for tA in ('N', 'T'), tB in ('N', 'T'), oA in (RM, CM), oB in (RM, CM)
            assert_transpose_matches_dense(12, 8, 16, (4, 4), (4, 4), oA, oB, tA, tB)   # square tiles
            assert_transpose_matches_dense(12, 9, 10, (4, 3), (3, 5), oA, oB, tA, tB)   # rectangular tiles
        end
    end

    @testset "guard: transpose requires boundary-free operands" begin
        A = NextLA.TLRMatrix(zeros(Float64, 8, 14), (4, 4), 3)   # 14 % 4 ≠ 0 → column tail
        B = NextLA.TLRMatrix(zeros(Float64, 8, 8), (4, 4), 3)
        C = zeros(Float64, 14, 8)                                 # = size(op(A)=Aᵀ, 1) × size(B, 2)
        @test_throws ArgumentError NextLA.TLRmodule.gemm!(C, A, B; transA='T')
    end
end

@testset "TLR gemm! to dense on CUDA" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
            for orderA in orders, orderB in orders
                assert_tlr_gemm_matches_dense(ArrayType, Float32, 12, 4, 2, orderA, orderB, synchronize;
                                              budget=1, alpha=Float32(1.2), beta=Float32(0.25),
                                              atol=5f-3, rtol=5f-3)
            end
        end
    end
end
