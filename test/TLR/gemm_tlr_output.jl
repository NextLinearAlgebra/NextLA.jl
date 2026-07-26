# TLR gemm! to a TLR output (ROADMAP milestone 4): dense-accumulate then compress.
#
# Scoped to the regular grid, beta == 0, and the three row-family layout pairs (every
# tile-order pair except A tile-column-major × B tile-row-major, which is the column
# family and not yet supported). Correctness is against the tile-reconstructed dense
# reference `alpha * A * B`; at full rank capacity the product tiles fit and the
# reconstruction is near-exact. Drivers live in `helpers.jl`.

@testset "TLR gemm! to TLR output (row family)" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    huge = 128 * 1024 * 1024

    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            # (mA, k, nB, tsA, tsB, r, oA, oB, maxrankC, budget) — square/rectangular
            # grids, rectangular tiles, tiny vs huge budget, all three row-family layouts.
            cases = (
                (12, 12, 12, (4, 4), (4, 4), 2, RM, RM, 4, huge),
                (12, 12, 12, (4, 4), (4, 4), 2, RM, CM, 4, 1),
                (12, 12, 12, (4, 4), (4, 4), 2, CM, CM, 4, huge),
                # Rectangular tiles. Dimensions must stay aligned to the tiling:
                # non-regular-grid (tail) tiles are deferred and this path
                # rejects them outright.
                (12,  9, 10, (4, 3), (3, 5), 3, RM, CM, 4, 1),
            )
            for (mA, k, nB, tsA, tsB, r, oA, oB, mr, budget) in cases
                @testset "$(mA)×$(k)×$(nB) tiles=$(tsA)/$(tsB) r=$r $(oA)*$(oB) budget=$budget" begin
                    assert_tlr_output_matches_dense(ArrayType, Float64, mA, k, nB, tsA, tsB, r,
                        oA, oB, synchronize; maxrankC=mr, budget=budget, atol=1e-9)
                end
            end

            # Zero rank operand ⇒ zero product ⇒ every output tile rank 0.
            @testset "zero-rank operand" begin
                A = NextLA.TLRMatrix(ArrayType(zeros(Float64, 12, 12)), (4, 4), 0; tile_order=RM)
                B = NextLA.TLRMatrix(ArrayType(zeros(Float64, 12, 12)), (4, 4), 3; tile_order=CM)
                fill_random_tlr!(B, ArrayType; seed=7)
                C = NextLA.TLRMatrix(ArrayType(zeros(Float64, 12, 12)), (4, 4), 4)
                NextLA.TLRmodule.gemm!(C, A, B; alpha=1.0, beta=0)
                @test all(iszero, NextLA.ranks(C))
            end
        end
    end
end

@testset "TLR output rank overflow reports residual" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            A = NextLA.TLRMatrix(ArrayType(zeros(Float64, 8, 8)), (4, 4), 3; tile_order=RM)
            B = NextLA.TLRMatrix(ArrayType(zeros(Float64, 8, 8)), (4, 4), 3; tile_order=CM)
            fill_random_tlr!(A, ArrayType; seed=1); fill_random_tlr!(B, ArrayType; seed=2)
            C = NextLA.TLRMatrix(ArrayType(zeros(Float64, 8, 8)), (4, 4), 2)   # capacity < achievable rank
            NextLA.TLRmodule.gemm!(C, A, B; alpha=1.0, beta=0, tol=0.0)
            synchronize(C.int_U)
            @test all(<=(2), NextLA.ranks(C))
            @test any(>(1e-8), NextLA.residuals(C))
        end
    end
end

# The TLR-output workspace must be concretely typed so `allocate` infers its element
# types — the same invariant `gemm_core.jl` pins for the dense driver (the `Tin`
# regression). A `DataType` field would collapse this silently.
@testset "TLR output workspace inference" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    T = Float64
    A = NextLA.TLRMatrix(zeros(T, 24, 24), (4, 4), 2; tile_order=RM)
    B = NextLA.TLRMatrix(zeros(T, 24, 24), (4, 4), 2; tile_order=CM)
    fill_random_tlr!(A, Array; seed=1); fill_random_tlr!(B, Array; seed=2)
    C = NextLA.TLRMatrix(zeros(T, 24, 24), (4, 4), 4)
    LA = NextLA.TLRmodule.logical_operand(A, 'N')
    LB = NextLA.TLRmodule.logical_operand(B, 'N')
    mode = NextLA.TLRmodule.default_gemm_compute_mode(T)
    ops = NextLA.TLRmodule.logical_operands(LA, LB)
    geom = NextLA.TLRmodule.interior_geometry(LA, LB)
    fold = NextLA.TLRmodule.choose_fold(ops)
    placement = NextLA.TLRmodule.placement_for_fold(fold, ops)
    ws = @inferred NextLA.TLRmodule._alloc_tlr_output_workspace(
        C, geom, placement, ops, 128 * 1024 * 1024, fold)
    @test isconcretetype(typeof(ws))
    @test isconcretetype(eltype(ws.slab))
    @test isconcretetype(eltype(ws.accum))
end

@testset "TLR output validation" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    # The column-family layout (A col-major × B row-major) and beta != 0 are now
    # both handled by the row-basis path (covered in row_basis/driver.jl); they no
    # longer throw. Only the regular-grid (aligned tiling) requirement is enforced.
    Ab = NextLA.TLRMatrix(zeros(Float64, 10, 10), (4, 4), 2; tile_order=RM)
    Bb = NextLA.TLRMatrix(zeros(Float64, 10, 10), (4, 4), 2; tile_order=RM)
    fill_random_tlr!(Ab, Array; seed=3); fill_random_tlr!(Bb, Array; seed=4)
    Cb = NextLA.TLRMatrix(zeros(Float64, 10, 10), (4, 4), 4)
    @test_throws ArgumentError NextLA.TLRmodule.gemm!(Cb, Ab, Bb; alpha=1.0, beta=0)
end
