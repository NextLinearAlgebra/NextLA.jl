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
    A = random_tlr_matrix(Array, Float64, 35, 8, 3; seed=71)
    B = random_tlr_matrix(Array, Float64, 35, 8, 3; seed=72)
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
        NextLA.gemm!(C, A, B; workspace)
        @test C ≈ reference
    end
    @test_throws ArgumentError NextLA.gemm!(
        zeros(Float64, size(A, 1), size(B, 2)), A, B; workspace=lo - 1)
end

@testset "finalized TLR factors remain value-mutable" begin
    A = NextLA.TLRMatrix(KernelAbstractions.CPU(), Float64, 8, 8, 4, Int[0 1; 0 0])
    B = NextLA.TLRMatrix(KernelAbstractions.CPU(), Float64, 8, 8, 4, Int[0 0; 1 0])
    UA, VA = NextLA.get_factors(A, 1, 2)
    UB, VB = NextLA.get_factors(B, 2, 1)
    UA .= 1; VA .= 2; UB .= 3; VB .= 4

    C = zeros(Float64, 8, 8)
    workspace = NextLA.gemm_minimum_workspace_bytes(A, B)
    NextLA.gemm!(C, A, B; workspace)
    @test C ≈ reconstruct_tlr(A) * reconstruct_tlr(B)
end

# Whole-matrix transpose is a zero-copy container view; the executors consume
# its transposed geometry, factor roles, and tile order directly.
@testset "container-level TLR transpose" begin
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    A = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 11, 14, (4, 3), fill(2, 3, 5))
    At = transpose(A)
    @test size(At) == (14, 11)
    @test NextLA.TLRmodule.nominal_tile_size(At) == (3, 4)
    @test NextLA.TLRmodule.tail_tile_size(At) == (2, 3)
    @test NextLA.TLRmodule.grid_size(At) == reverse(NextLA.grid_size(A))
    @test NextLA.TLRmodule.tile_order(At) isa typeof(CM)
    @test NextLA.TLRmodule.compressed_ftlr_outer_order(At) isa NextLA.TileRowMajor
    @test NextLA.TLRmodule.compressed_ftlr_inner_order(At) isa NextLA.TileColMajor

    D = random_tlr_matrix(Array, Float64, 14, 4, 2; seed=91)
    Dt = transpose(D)
    @test NextLA.offdiagonal(Dt) isa NextLA.TLRmodule.TransposeTLRMatrix
    @test size(NextLA.dense_diag(Dt))[1:2] == reverse(size(NextLA.dense_diag(D))[1:2])
    @test size.(NextLA.get_factors(Dt, 2, 1)) ==
          reverse(size.(NextLA.get_factors(D, 1, 2)))
    dtile, op = NextLA.TLRmodule._diag_tile_ref(Dt, 1)
    @test parent(dtile) === D.D
    @test op == 'T'
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

@testset "dense-diagonal TLR mixed dense GEMM on CPU" begin
    rng = MersenneTwister(404)
    G = random_tlr_matrix(Array, Float64, 9, 4, 2; seed=405)
    Gdense = reconstruct_tlr(G)
    alpha = 1.25
    beta = -0.5

    for side in (:tlr_dense, :dense_tlr),
        trans_tlr in ('N', 'T'), trans_dense in ('N', 'T')
        if side === :tlr_dense
            stored_dense = trans_dense == 'N' ? (9, 7) : (7, 9)
            X = randn(rng, Float64, stored_dense...)
            C0 = randn(rng, Float64, 9, 7)
            C = copy(C0)
            workspace = NextLA.DenseGemmWorkspace(G, 4096)
            NextLA.gemm!(C, G, X; workspace, alpha, beta,
                        transA=trans_tlr, transB=trans_dense)
            opG = trans_tlr == 'N' ? Gdense : transpose(Gdense)
            opX = trans_dense == 'N' ? X : transpose(X)
            reference = alpha .* (opG * opX) .+ beta .* C0
        else
            stored_dense = trans_dense == 'N' ? (6, 9) : (9, 6)
            X = randn(rng, Float64, stored_dense...)
            C0 = randn(rng, Float64, 6, 9)
            C = copy(C0)
            workspace = NextLA.DenseGemmWorkspace(G, 4096)
            NextLA.gemm!(C, X, G; workspace, alpha, beta,
                        transA=trans_dense, transB=trans_tlr)
            opX = trans_dense == 'N' ? X : transpose(X)
            opG = trans_tlr == 'N' ? Gdense : transpose(Gdense)
            reference = alpha .* (opX * opG) .+ beta .* C0
        end
        @test C ≈ reference rtol=1e-10 atol=1e-10
    end

    # A zero-rank off-diagonal still carries a nonzero dense diagonal. This
    # also verifies that the diagonal updates work with a zero-byte workspace.
    D = random_tlr_matrix(Array, Float64, 9, 4, 0; seed=406)
    Ddense = reconstruct_tlr(D)
    Xright = randn(rng, Float64, 9, 5)
    Cright = randn(rng, Float64, 9, 5)
    Cright0 = copy(Cright)
    NextLA.gemm!(Cright, D, Xright; workspace=0, alpha, beta)
    @test Cright ≈ alpha .* (Ddense * Xright) .+ beta .* Cright0

    Xleft = randn(rng, Float64, 5, 9)
    Cleft = randn(rng, Float64, 5, 9)
    Cleft0 = copy(Cleft)
    NextLA.gemm!(Cleft, Xleft, D; workspace=0, alpha, beta)
    @test Cleft ≈ alpha .* (Xleft * Ddense) .+ beta .* Cleft0
end

@testset "TLR gemm! to dense on CUDA" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
            assert_tlr_gemm_matches_dense(ArrayType, Float32, 12, 4, 2, orders..., synchronize;
                                          budget=1, alpha=Float32(1.2), beta=Float32(0.25),
                                          atol=5f-3, rtol=5f-3)
        end
    end
end

@testset "TF32 backend capability" begin
    for (backend_name, ArrayType, _) in available_backends()
        backend_name == "AMDGPU" || continue
        A = random_tlr_matrix(ArrayType, Float32, 8, 4, 2; seed=501)
        B = random_tlr_matrix(ArrayType, Float32, 8, 4, 2; seed=502)
        C = ArrayType(zeros(Float32, 8, 8))
        @test_throws ArgumentError NextLA.gemm!(
            C, A, B; compute=NextLA.TF32(), workspace=1)
    end
end
