# gemm! into a dense C, with TLRMatrix and CompressedFTLRMatrix operands, against
# dense references built from the reconstructed tiles, src/tlr/blas/gemm/dense_accumulation.
# Helpers: test/setup/tlr.jl.

#
# Parameter sweeps use pairwise coverage: every value of each dimension (tile
# orders, budget extremes, boundary regimes, transpose flags) appears, and every
# pair of dimensions is exercised at least once, without full cross-products.

@testset "TLR gemm! to dense (dense-diagonal) [CPU]" begin
    Random.seed!(3020)
    RM = TileLowRank.TileRowMajor(); CM = TileLowRank.TileColMajor()
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

@testset "TLR compressed workspace bounds [CPU]" begin
    Random.seed!(3021)
    A = random_tlr_matrix(Array, Float64, 35, 8, 3; seed=71)
    B = random_tlr_matrix(Array, Float64, 35, 8, 3; seed=72)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)
    for (transA, transB) in (('N', 'N'), ('N', 'T'), ('T', 'N'), ('T', 'T'))
        lo = TileLowRank.gemm_minimum_workspace_bytes(A, B; transA, transB)
        hi = TileLowRank.gemm_maximum_workspace_bytes(A, B; transA, transB)
        @test 0 < lo <= hi
    end
    lo = TileLowRank.gemm_minimum_workspace_bytes(A, B)
    for bytes in unique((lo, TileLowRank.gemm_maximum_workspace_bytes(A, B)))
        workspace = TileLowRank.DenseGemmWorkspace(A, B; bytes)
        C = zeros(Float64, size(A, 1), size(B, 2))
        NextLA.gemm!(C, A, B; workspace)
        @test C ≈ reference
    end
    @test_throws ArgumentError NextLA.gemm!(
        zeros(Float64, size(A, 1), size(B, 2)), A, B; workspace=lo - 1)
end

@testset "finalized TLR factors remain value-mutable [CPU]" begin
    Random.seed!(3022)
    A = TileLowRank.TLRMatrix(KernelAbstractions.CPU(), Float64, 8, 8, 4, Int[0 1; 0 0])
    B = TileLowRank.TLRMatrix(KernelAbstractions.CPU(), Float64, 8, 8, 4, Int[0 0; 1 0])
    UA, VA = TileLowRank.get_factors(A, 1, 2)
    UB, VB = TileLowRank.get_factors(B, 2, 1)
    UA .= 1; VA .= 2; UB .= 3; VB .= 4

    C = zeros(Float64, 8, 8)
    workspace = TileLowRank.gemm_minimum_workspace_bytes(A, B)
    NextLA.gemm!(C, A, B; workspace)
    @test C ≈ reconstruct_tlr(A) * reconstruct_tlr(B)
end

# Whole-matrix transpose is a zero-copy container view; the executors consume
# its transposed geometry, factor roles, and tile order directly.
@testset "container-level TLR transpose [CPU]" begin
    Random.seed!(3023)
    RM = TileLowRank.TileRowMajor(); CM = TileLowRank.TileColMajor()
    A = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 11, 14, (4, 3), fill(2, 3, 5))
    At = transpose(A)
    @test size(At) == (14, 11)
    @test NextLA.TileLowRank.nominal_tile_size(At) == (3, 4)
    @test NextLA.TileLowRank.tail_tile_size(At) == (2, 3)
    @test NextLA.TileLowRank.grid_size(At) == reverse(TileLowRank.grid_size(A))
    @test NextLA.TileLowRank.tile_order(At) isa typeof(CM)
    @test NextLA.TileLowRank.compressed_ftlr_outer_order(At) isa TileLowRank.TileRowMajor
    @test NextLA.TileLowRank.compressed_ftlr_inner_order(At) isa TileLowRank.TileColMajor

    D = random_tlr_matrix(Array, Float64, 14, 4, 2; seed=91)
    Dt = transpose(D)
    @test TileLowRank.offdiagonal(Dt) isa NextLA.TileLowRank.TransposeTLRMatrix
    @test size(TileLowRank.dense_diag(Dt))[1:2] == reverse(size(TileLowRank.dense_diag(D))[1:2])
    @test size.(TileLowRank.get_factors(Dt, 2, 1)) ==
          reverse(size.(TileLowRank.get_factors(D, 1, 2)))
    dtile, op = NextLA.TileLowRank.diag_tile_ref(Dt, 1)
    @test parent(dtile) === D.D
    @test op == 'T'
end

@testset "dense-diagonal boundary transpose [CPU]" begin
    Random.seed!(3024)
    RM = TileLowRank.TileRowMajor(); CM = TileLowRank.TileColMajor()
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

@testset "dense-diagonal TLR mixed dense GEMM [CPU]" begin
    Random.seed!(3025)
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
            workspace = TileLowRank.DenseGemmWorkspace(G, 4096)
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
            workspace = TileLowRank.DenseGemmWorkspace(G, 4096)
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

@testset "TLR gemm! to dense [$backend_name]" for (backend_name, ArrayType, synchronize) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3026)
    orders = (TileLowRank.TileRowMajor(), TileLowRank.TileColMajor())
    assert_tlr_gemm_matches_dense(ArrayType, Float32, 12, 4, 2, orders..., synchronize;
                                  budget=1, alpha=Float32(1.2), beta=Float32(0.25),
                                  atol=5f-3, rtol=5f-3)
end

@testset "TF32 backend capability [$backend_name]" for (backend_name, ArrayType, _) in filter(b -> b[1] == "AMDGPU", TLR_BACKENDS)
    Random.seed!(3027)
    A = random_tlr_matrix(ArrayType, Float32, 8, 4, 2; seed=501)
    B = random_tlr_matrix(ArrayType, Float32, 8, 4, 2; seed=502)
    C = ArrayType(zeros(Float32, 8, 8))
    @test_throws ArgumentError NextLA.gemm!(
        C, A, B; compute=NextLA.TF32(), workspace=1)
end



function _compressed_ftlr_to_backend(A::TileLowRank.CompressedFTLRMatrix, AT)
    qm, qn = TileLowRank.grid_size(A)
    rank_grid = [Int(TileLowRank.ranks(A)[expected_storage_slot(A, i, j)])
                 for i in 1:qm, j in 1:qn]
    B = TileLowRank.CompressedFTLRMatrix(KernelAbstractions.get_backend(AT(zeros(eltype(A), 1))),
                           eltype(A), size(A, 1), size(A, 2),
                           TileLowRank.nominal_tile_size(A),
                           rank_grid;
                           outer_order=typeof(A.outer.order), inner_order=typeof(A.inner.order),
                           rank_multiple=TileLowRank.rank_multiple(A))

    copyto!(B.outer.data, AT(Array(A.outer.data)))
    copyto!(B.inner.data, AT(Array(A.inner.data)))
    return B
end

@testset "dense × compressed aligned row-run sizing [CPU]" begin
    Random.seed!(3028)
    height = _TLRM.dense_compressed_row_run_height
    @test height(25 * 10, 10, 100, Float16) == 24
    @test height(25 * 10, 10, 100, Float32) == 24
    @test height(25 * 10, 10, 100, Float64) == 24
    @test height(8 * 10, 10, 100, Float16) == 8
    @test height(7 * 10, 10, 100, Float16) == 7
    @test height(23 * 10, 10, 23, Float16) == 23
    @test height(10, 10, 100, Float16) == 1
    @test_throws ArgumentError height(100, 0, 100, Float16)
    @test_throws ArgumentError height(100, 10, 0, Float16)

    for (T, quantum) in ((Float16, 8), (Float32, 4), (Float64, 2)),
        capacity in quantum:40
        selected = height(capacity * 3, 3, 100, T)
        @test selected % quantum == 0
        @test selected <= capacity
        @test selected + quantum > capacity
    end
end

@testset "CompressedFTLR mixed dense GEMM [CPU]" begin
    Random.seed!(3029)
    rank_grid = Int[1 2 1; 2 0 1; 1 1 1]
    G = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 9, 9, (4, 4), rank_grid)
    rng = MersenneTwister(2026)
    for j in 1:3, i in 1:3
        U, V = TileLowRank.get_factors(G, i, j)
        U .= randn(rng, Float64, size(U))
        V .= randn(rng, Float64, size(V))
    end
    Gdense = zeros(Float64, 9, 9)
    TileLowRank.uncompress!(Gdense, G)

    for side in (:compressed_dense, :dense_compressed),
        trans_dense in ('N', 'T'), trans_compressed in ('N', 'T')
        X = randn(rng, Float64, 9, 9)
        C0 = randn(rng, Float64, 9, 9)
        C = copy(C0)
        if side === :compressed_dense
            NextLA.gemm!(C, G, X; workspace=4096, alpha=1.25, beta=-0.5,
                         transA=trans_compressed, transB=trans_dense)
            opG = trans_compressed == 'N' ? Gdense : transpose(Gdense)
            opX = trans_dense == 'N' ? X : transpose(X)
            product = opG * opX
        else
            NextLA.gemm!(C, X, G; workspace=4096, alpha=1.25, beta=-0.5,
                         transA=trans_dense, transB=trans_compressed)
            opX = trans_dense == 'N' ? X : transpose(X)
            opG = trans_compressed == 'N' ? Gdense : transpose(Gdense)
            product = opX * opG
        end
        @test C ≈ 1.25 .* product .- 0.5 .* C0 rtol=1e-10 atol=1e-10
    end

    X = randn(rng, Float64, 9, 9)
    for side in (:compressed_dense, :dense_compressed)
        C = zeros(Float64, 9, 9)
        workspace = TileLowRank.DenseGemmWorkspace(G, 4096)
        analysis = side === :compressed_dense ?
            TileLowRank.analyze_compressed_gemm(C, G, X; workspace) :
            TileLowRank.analyze_compressed_gemm(C, X, G; workspace)
        try
            if side === :compressed_dense
                NextLA.gemm!(C, G, X; workspace, analysis)
                @test C ≈ Gdense * X rtol=1e-10 atol=1e-10
            else
                NextLA.gemm!(C, X, G; workspace, analysis)
                @test C ≈ X * Gdense rtol=1e-10 atol=1e-10
            end
        finally
            close(analysis)
        end
    end

    # Below one fused rank stack, the compressed-only tilewise fallback keeps
    # the low-workspace contract without any matrix-shaped staging container.
    for side in (:compressed_dense, :dense_compressed)
        C = zeros(Float64, 9, 9)
        if side === :compressed_dense
            NextLA.gemm!(C, G, X; workspace=2 * sizeof(Float64))
            @test C ≈ Gdense * X rtol=1e-10 atol=1e-10
        else
            NextLA.gemm!(C, X, G; workspace=2 * sizeof(Float64))
            @test C ≈ X * Gdense rtol=1e-10 atol=1e-10
        end
    end

    left_plan = _TLRM.two_stage_rank_plan(G, :left)
    right_plan = _TLRM.two_stage_rank_plan(G, :right)
    @test all(run.fold === :left for run in
              _TLRM.two_stage_schedule(zeros(9, 9), left_plan, 4096, Float64))
    @test all(run.fold === :right for run in
              _TLRM.two_stage_schedule(zeros(9, 9), right_plan, 4096, Float64))
end

@testset "CompressedFTLR clears T only for uncovered rank holes [CPU]" begin
    Random.seed!(3030)
    function make_operand(ranks)
        return TileLowRank.CompressedFTLRMatrix(
            KernelAbstractions.CPU(), Float32, 8, 8, 4, ranks)
    end

    positive = make_operand(ones(Int, 2, 2))
    positive_plan = _TLRM.compressed_ftlr_rank_plan(positive, positive)
    positive_workspace = TileLowRank.DenseGemmWorkspace(
        positive, positive_plan.profile.maximum)
    positive_arena = _TLRM.GemmArena(
        view(positive_workspace.storage, :), 1)
    C = zeros(Float32, 8, 8)

    right = _TLRM.build_compressed_ftlr_foldright_run(
        C, positive, positive, positive_plan, 1:2, 1:2, 1f0, 0f0,
        positive_arena)
    @test !right.needs_zero

    left = _TLRM.build_compressed_ftlr_foldleft_run(
        C, positive, positive, positive_plan, 1:2, 1:2, 1f0, 0f0,
        positive_arena)
    @test !left.needs_zero

    # FoldRight gives every active A rank rows across every output column.
    # A zero B tile therefore leaves a consumed T block unwritten.
    right_B = make_operand(Int[0 1; 1 1])
    right_plan = _TLRM.compressed_ftlr_rank_plan(positive, right_B)
    right_workspace = TileLowRank.DenseGemmWorkspace(
        positive, right_plan.profile.right_byte_prefix[end])
    right_arena = _TLRM.GemmArena(view(right_workspace.storage, :), 1)
    right_hole = _TLRM.build_compressed_ftlr_foldright_run(
        C, positive, right_B, right_plan, 1:2, 1:2, 1f0, 0f0, right_arena)
    @test right_hole.needs_zero

    # FoldLeft gives every physical output row columns for every active B
    # rank. A zero A tile leaves the corresponding rows unwritten.
    left_A = make_operand(Int[0 1; 1 1])
    left_plan = _TLRM.compressed_ftlr_rank_plan(left_A, positive)
    left_workspace = TileLowRank.DenseGemmWorkspace(
        left_A, left_plan.profile.left_byte_prefix[end])
    left_arena = _TLRM.GemmArena(view(left_workspace.storage, :), 1)
    left_hole = _TLRM.build_compressed_ftlr_foldleft_run(
        C, left_A, positive, left_plan, 1:2, 1:2, 1f0, 0f0, left_arena)
    @test left_hole.needs_zero
end

@testset "CompressedFTLR FoldLeft fuses Stage 3 across a row run [CPU]" begin
    Random.seed!(3031)
    rank_grid = Int[1 2; 2 1]
    A = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 8, 8, 4, rank_grid;
        outer_order=TileLowRank.TileColMajor, inner_order=TileLowRank.TileColMajor)
    B = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 8, 8, 4, reverse(rank_grid; dims=1))
    plan = _TLRM.compressed_ftlr_rank_plan(A, B)
    @test plan.profile.right_byte_prefix === nothing
    workspace = TileLowRank.DenseGemmWorkspace(A, plan.profile.maximum)
    arena = _TLRM.GemmArena(view(workspace.storage, :), 1)
    C = zeros(Float32, 8, 8)
    tasks = _TLRM.build_compressed_ftlr_foldleft_run(
        C, A, B, plan, 1:2, 1:2, 1f0, 0f0, arena)

    @test length(tasks.stage3) == TileLowRank.grid_size(B)[2]
    @test all(size(task.C, 1) == size(C, 1) for task in tasks.stage3)
end

@testset "CompressedFTLR dense GEMM with tails [$name]" for (name, AT, sync) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3032)
    ranksA = Int[2 1 1; 1 2 1; 1 1 1]
    ranksB = Int[1 2 1; 2 1 1; 1 1 1]
    Ahost = TileLowRank.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float32, 9, 10, (4, 4), ranksA)
    Bhost = TileLowRank.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float32, 10, 11, (4, 4), ranksB)
    rng = MersenneTwister(419)
    for X in (Ahost, Bhost), j in 1:TileLowRank.grid_size(X)[2], i in 1:TileLowRank.grid_size(X)[1]
        U, V = TileLowRank.get_factors(X, i, j)
        U .= randn(rng, Float32, size(U)); V .= randn(rng, Float32, size(V))
    end
    refA = zeros(Float32, size(Ahost)); refB = zeros(Float32, size(Bhost))
    TileLowRank.uncompress!(refA, Ahost); TileLowRank.uncompress!(refB, Bhost)
    A = _compressed_ftlr_to_backend(Ahost, AT); B = _compressed_ftlr_to_backend(Bhost, AT)
    for bytes in unique((TileLowRank.gemm_minimum_workspace_bytes(A, B),
                         TileLowRank.gemm_maximum_workspace_bytes(A, B)))
        C0 = rand(rng, Float32, 9, 11); C = AT(copy(C0))
        NextLA.gemm!(C, A, B; workspace=bytes, alpha=1.25f0, beta=-0.5f0)
        sync(C)
        @test Array(C) ≈ 1.25f0 .* (refA * refB) .- 0.5f0 .* C0 rtol=3f-4 atol=3f-4
    end
end

@testset "CompressedFTLR logical transpose combinations [$name]" for (name, AT, sync) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3033)
    Ahost = compressed_ftlr_fixture(Float32, Int[1 2; 3 1])
    Bhost = compressed_ftlr_fixture(Float32, Int[2 1; 1 3])
    referenceA = zeros(Float32, size(Ahost)); referenceB = zeros(Float32, size(Bhost))
    TileLowRank.uncompress!(referenceA, Ahost); TileLowRank.uncompress!(referenceB, Bhost)
    A = _compressed_ftlr_to_backend(Ahost, AT)
    B = _compressed_ftlr_to_backend(Bhost, AT)
    for transA in ('N', 'T'), transB in ('N', 'T')
        opA = transA == 'N' ? referenceA : transpose(referenceA)
        opB = transB == 'N' ? referenceB : transpose(referenceB)
        minbytes = TileLowRank.gemm_minimum_workspace_bytes(A, B; transA, transB)
        maxbytes = TileLowRank.gemm_maximum_workspace_bytes(A, B; transA, transB)
        for bytes in unique((minbytes, maxbytes))
            C0 = rand(Float32, size(opA, 1), size(opB, 2))
            C = AT(copy(C0))
            NextLA.gemm!(C, A, B; workspace=bytes, alpha=1.25f0, beta=-0.5f0,
                         transA, transB)
            sync(C)
            @test Array(C) ≈ 1.25f0 .* (opA * opB) .- 0.5f0 .* C0 rtol=2f-4 atol=2f-4
        end
    end
end

@testset "CompressedFTLR dense GEMM, Float32 [CPU]" begin
    Random.seed!(3034)
    Ahost = compressed_ftlr_fixture(Float32)
    Bhost = compressed_ftlr_fixture(Float32, Int[1 0; 2 1; 3 1])
    referenceAhost = zeros(Float32, size(Ahost))
    referenceBhost = zeros(Float32, size(Bhost))
    TileLowRank.uncompress!(referenceAhost, Ahost)
    TileLowRank.uncompress!(referenceBhost, Bhost)
    Chost = zeros(Float32, 8, 8)
    NextLA.gemm!(Chost, Ahost, Bhost;
                workspace=TileLowRank.gemm_minimum_workspace_bytes(Ahost, Bhost))
    @test Chost ≈ referenceAhost * referenceBhost rtol=2f-4 atol=2f-4
end

@testset "CompressedFTLR dense GEMM [$name]" for (name, AT, sync) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3041)
    Ahost = compressed_ftlr_fixture(Float32)
    Bhost = compressed_ftlr_fixture(Float32, Int[1 0; 2 1; 3 1])
    A = _compressed_ftlr_to_backend(Ahost, AT)
    B = _compressed_ftlr_to_backend(Bhost, AT)
    referenceA = zeros(Float32, size(A)); referenceB = zeros(Float32, size(B))
    TileLowRank.uncompress!(referenceA, Ahost); TileLowRank.uncompress!(referenceB, Bhost)
    reconstructed = AT(zeros(Float32, size(A)))
    TileLowRank.uncompress!(reconstructed, A)
    sync(reconstructed)
    @test Array(reconstructed) ≈ referenceA
    minbytes = TileLowRank.gemm_minimum_workspace_bytes(A, B)
    maxbytes = TileLowRank.gemm_maximum_workspace_bytes(A, B)
    for bytes in unique((minbytes, maxbytes))
        C0 = rand(Float32, size(A, 1), size(B, 2))
        C = AT(copy(C0))
        NextLA.gemm!(C, A, B; workspace=bytes, alpha=1.5f0, beta=-0.25f0)
        sync(C)
        @test Array(C) ≈ 1.5f0 .* (referenceA * referenceB) .- 0.25f0 .* C0 rtol=2f-4 atol=2f-4
    end
end

# Both two-stage folds slice their T workspace at each output's rank prefix, so
# a member's operands are 16-byte aligned only when every preceding rank is a
# multiple of the backend's alignment quantum. Quantized ranks therefore stay
# fully grouped, while exact ranks route the unaligned members through ordinary
# GEMMEx -- same results either way, so both paths are covered here.
@testset "CompressedFTLR mixed grouped lowering and analysis [$name]" for (name, AT, sync) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3035)
    rank_grid = Int[1 2 1; 2 0 1; 1 1 1]
    quantum = NextLA.gemm_alignment_quantum(Float32)
    for rank_multiple in (quantum, 0)
        H = TileLowRank.CompressedFTLRMatrix(
            KernelAbstractions.CPU(), Float32, 9, 9, (4, 4), rank_grid;
            rank_multiple)
        rng = MersenneTwister(2026)
        for j in 1:3, i in 1:3
            U, V = TileLowRank.get_factors(H, i, j)
            U .= randn(rng, Float32, size(U))
            V .= randn(rng, Float32, size(V))
        end
        Hdense = zeros(Float32, 9, 9)
        TileLowRank.uncompress!(Hdense, H)

        G = _compressed_ftlr_to_backend(H, AT)
        for side in (:compressed_dense, :dense_compressed),
            trans_dense in ('N', 'T'), trans_compressed in ('N', 'T')
            Xhost = randn(rng, Float32, 9, 9)
            X = AT(Xhost)
            C0 = randn(rng, Float32, 9, 9)
            C = AT(C0)
            workspace = TileLowRank.DenseGemmWorkspace(G, 4096)
            if side === :compressed_dense
                analysis = TileLowRank.analyze_compressed_gemm(
                    C, G, X; workspace,
                    transA=trans_compressed, transB=trans_dense)
                @test analysis.has_fallback == iszero(rank_multiple)
                NextLA.gemm!(
                    C, G, X; workspace, alpha=1.25f0, beta=-0.5f0,
                    transA=trans_compressed, transB=trans_dense, analysis)
                opG = trans_compressed == 'N' ? Hdense : transpose(Hdense)
                opX = trans_dense == 'N' ? Xhost : transpose(Xhost)
            else
                analysis = TileLowRank.analyze_compressed_gemm(
                    C, X, G; workspace,
                    transA=trans_dense, transB=trans_compressed)
                @test analysis.has_fallback == iszero(rank_multiple)
                NextLA.gemm!(
                    C, X, G; workspace, alpha=1.25f0, beta=-0.5f0,
                    transA=trans_dense, transB=trans_compressed, analysis)
                opX = trans_dense == 'N' ? Xhost : transpose(Xhost)
                opG = trans_compressed == 'N' ? Hdense : transpose(Hdense)
            end
            sync(C)
            product = side === :compressed_dense ? opG * opX : opX * opG
            @test Array(C) ≈ 1.25f0 .* product .- 0.5f0 .* C0 rtol=3f-4 atol=3f-4
            close(analysis)
        end
    end
end

function _compressed_dense32(A)
    dense = zeros(Float32, size(A))
    qm, qn = TileLowRank.grid_size(A)
    for j in 1:qn, i in 1:qm
        rows = _TLRM.tile_axis_range(A, i, 1)
        cols = _TLRM.tile_axis_range(A, j, 2)
        U, V = TileLowRank.get_factors(A, i, j)
        dense[rows, cols] .= Float32.(U) * transpose(Float32.(V))
    end
    return dense
end

@testset "CompressedFTLR arbitrary-rank FP16 and TF32 execution [$name]" for (name, AT, sync) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3036)
    rank_grid = Int[1 2; 3 1]
    for (T, compute, tolerance) in (
        (Float16, Float32, 2f-2),
        (Float32, NextLA.TF32(), 5f-3),
    )
        q = T === Float16 ? 8 : compute isa NextLA.TF32 ? 4 : 0
        Ahost = compressed_ftlr_fixture(T, rank_grid; rank_multiple=q)
        Bhost = compressed_ftlr_fixture(
            T, reverse(rank_grid; dims=1); rank_multiple=q)
        reference = _compressed_dense32(Ahost) * _compressed_dense32(Bhost)
        A = _compressed_ftlr_to_backend(Ahost, AT)
        B = _compressed_ftlr_to_backend(Bhost, AT)
        exact_host = compressed_ftlr_fixture(T, rank_grid)
        exact = _compressed_ftlr_to_backend(exact_host, AT)
        @test_throws ArgumentError NextLA.gemm!(
            AT(zeros(T, size(exact, 1), size(exact, 2))), exact, exact;
            workspace=4096, compute)
        workspace = TileLowRank.DenseGemmWorkspace(
            A, B; bytes=TileLowRank.gemm_maximum_workspace_bytes(A, B))
        C = AT(zeros(T, size(A, 1), size(B, 2)))
        analysis = TileLowRank.analyze_compressed_gemm(C, A, B; workspace, compute)
        NextLA.gemm!(C, A, B; workspace, compute, analysis)
        sync(C)
        @test norm(Float32.(Array(C)) - reference) / norm(reference) < tolerance
        # Same result through the transient (no-analysis) entry point.
        fill!(C, zero(T))
        NextLA.gemm!(C, A, B; workspace, compute)
        sync(C)
        @test norm(Float32.(Array(C)) - reference) / norm(reference) < tolerance
        close(analysis)
    end
end
