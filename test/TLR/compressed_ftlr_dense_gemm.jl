using Test
using KernelAbstractions

function _compressed_ftlr_to_backend(A::NextLA.CompressedFTLRMatrix, AT)
    qm, qn = NextLA.grid_size(A)
    rank_grid = [Int(NextLA.ranks(A)[_TLRM._rank_index(A, i, j)]) for i in 1:qm, j in 1:qn]
    B = NextLA.CompressedFTLRMatrix(KernelAbstractions.get_backend(AT(zeros(eltype(A), 1))),
                           eltype(A), size(A, 1), size(A, 2),
                           NextLA.nominal_tile_size(A),
                           rank_grid;
                           outer_order=typeof(A.outer.order), inner_order=typeof(A.inner.order))
    copyto!(B.outer.data, AT(Array(A.outer.data)))
    copyto!(B.inner.data, AT(Array(A.inner.data)))
    return B
end

@testset "dense × compressed aligned row-run sizing" begin
    height = _TLRM._dense_compressed_row_run_height
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

@testset "CompressedFTLR mixed dense GEMM on CPU" begin
    rank_grid = Int[1 2 1; 2 0 1; 1 1 1]
    G = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 9, 9, (4, 4), rank_grid)
    rng = MersenneTwister(2026)
    for j in 1:3, i in 1:3
        U, V = NextLA.get_factors(G, i, j)
        U .= randn(rng, Float64, size(U))
        V .= randn(rng, Float64, size(V))
    end
    Gdense = zeros(Float64, 9, 9)
    NextLA.uncompress!(Gdense, G)

    for side in (:compressed_dense, :dense_compressed),
        trans_dense in ('N', 'T'), trans_compressed in ('N', 'T')
        X = randn(rng, Float64, 9, 9)
        C0 = randn(rng, Float64, 9, 9)
        C = copy(C0)
        if side === :compressed_dense
            _TLRM.gemm!(C, G, X; workspace=4096, alpha=1.25, beta=-0.5,
                         transA=trans_compressed, transB=trans_dense)
            opG = trans_compressed == 'N' ? Gdense : transpose(Gdense)
            opX = trans_dense == 'N' ? X : transpose(X)
            product = opG * opX
        else
            _TLRM.gemm!(C, X, G; workspace=4096, alpha=1.25, beta=-0.5,
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
        workspace = NextLA.DenseGemmWorkspace(G, 4096)
        analysis = side === :compressed_dense ?
            NextLA.analyze_compressed_gemm(C, G, X; workspace) :
            NextLA.analyze_compressed_gemm(C, X, G; workspace)
        try
            if side === :compressed_dense
                _TLRM.gemm!(C, G, X; workspace, analysis)
                @test C ≈ Gdense * X rtol=1e-10 atol=1e-10
            else
                _TLRM.gemm!(C, X, G; workspace, analysis)
                @test C ≈ X * Gdense rtol=1e-10 atol=1e-10
            end
        finally
            close(analysis)
        end
    end

    # Below one fused rank stack, the compressed-only tilewise fallback keeps
    # the low-workspace contract without reintroducing PaddedFTLR machinery.
    for side in (:compressed_dense, :dense_compressed)
        C = zeros(Float64, 9, 9)
        if side === :compressed_dense
            _TLRM.gemm!(C, G, X; workspace=2 * sizeof(Float64))
            @test C ≈ Gdense * X rtol=1e-10 atol=1e-10
        else
            _TLRM.gemm!(C, X, G; workspace=2 * sizeof(Float64))
            @test C ≈ X * Gdense rtol=1e-10 atol=1e-10
        end
    end

    left_plan = _TLRM._two_stage_rank_plan(_TLRM.logical_operand(G), :left)
    right_plan = _TLRM._two_stage_rank_plan(_TLRM.logical_operand(G), :right)
    @test all(run.fold === :left for run in
              _TLRM._two_stage_schedule(zeros(9, 9), left_plan, 4096, Float64))
    @test all(run.fold === :right for run in
              _TLRM._two_stage_schedule(zeros(9, 9), right_plan, 4096, Float64))
end

@testset "CompressedFTLR clears T only for uncovered rank holes" begin
    function make_operand(ranks)
        return NextLA.CompressedFTLRMatrix(
            KernelAbstractions.CPU(), Float32, 8, 8, 4, ranks;
            execution_rank_policy=:exact)
    end

    positive = make_operand(ones(Int, 2, 2))
    positive_plan = _TLRM._compressed_ftlr_rank_plan(positive, positive)
    positive_workspace = NextLA.DenseGemmWorkspace(
        positive, positive_plan.profile.maximum)
    positive_arena = _TLRM.GemmArena(
        view(positive_workspace.storage, :), 1)
    C = zeros(Float32, 8, 8)

    right = _TLRM._build_compressed_ftlr_foldright_run(
        C, positive, positive, positive_plan, 1:2, 1:2, 1f0, 0f0,
        positive_arena)
    @test !right.needs_zero

    left = _TLRM._build_compressed_ftlr_foldleft_run(
        C, positive, positive, positive_plan, 1:2, 1:2, 1f0, 0f0,
        positive_arena)
    @test !left.needs_zero

    # FoldRight gives every active A rank rows across every output column.
    # A zero B tile therefore leaves a consumed T block unwritten.
    right_B = make_operand(Int[0 1; 1 1])
    right_plan = _TLRM._compressed_ftlr_rank_plan(positive, right_B)
    right_workspace = NextLA.DenseGemmWorkspace(
        positive, sum(right_plan.profile.right_row_bytes))
    right_arena = _TLRM.GemmArena(view(right_workspace.storage, :), 1)
    right_hole = _TLRM._build_compressed_ftlr_foldright_run(
        C, positive, right_B, right_plan, 1:2, 1:2, 1f0, 0f0, right_arena)
    @test right_hole.needs_zero

    # FoldLeft gives every physical output row columns for every active B
    # rank. A zero A tile leaves the corresponding rows unwritten.
    left_A = make_operand(Int[0 1; 1 1])
    left_plan = _TLRM._compressed_ftlr_rank_plan(left_A, positive)
    left_workspace = NextLA.DenseGemmWorkspace(
        left_A, sum(left_plan.profile.left_row_bytes))
    left_arena = _TLRM.GemmArena(view(left_workspace.storage, :), 1)
    left_hole = _TLRM._build_compressed_ftlr_foldleft_run(
        C, left_A, positive, left_plan, 1:2, 1:2, 1f0, 0f0, left_arena)
    @test left_hole.needs_zero
end

@testset "CompressedFTLR FoldLeft fuses Stage 3 across a row run" begin
    rank_grid = Int[1 2; 2 1]
    A = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 8, 8, 4, rank_grid;
        outer_order=NextLA.TileColMajor, inner_order=NextLA.TileColMajor)
    B = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 8, 8, 4, reverse(rank_grid; dims=1))
    plan = _TLRM._compressed_ftlr_rank_plan(A, B)
    @test plan.profile.right_row_bytes === nothing
    workspace = NextLA.DenseGemmWorkspace(A, plan.profile.maximum)
    arena = _TLRM.GemmArena(view(workspace.storage, :), 1)
    C = zeros(Float32, 8, 8)
    tasks = _TLRM._build_compressed_ftlr_foldleft_run(
        C, A, B, plan, 1:2, 1:2, 1f0, 0f0, arena)

    @test length(tasks.stage3) == NextLA.grid_size(B)[2]
    @test all(size(task.C, 1) == size(C, 1) for task in tasks.stage3)
end

@testset "CompressedFTLR CUDA dense GEMM with tails" begin
    ranksA = Int[2 1 1; 1 2 1; 1 1 1]
    ranksB = Int[1 2 1; 2 1 1; 1 1 1]
    Ahost = NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float32, 9, 10, (4, 4), ranksA)
    Bhost = NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float32, 10, 11, (4, 4), ranksB)
    rng = MersenneTwister(419)
    for X in (Ahost, Bhost), j in 1:NextLA.grid_size(X)[2], i in 1:NextLA.grid_size(X)[1]
        U, V = NextLA.get_factors(X, i, j)
        U .= randn(rng, Float32, size(U)); V .= randn(rng, Float32, size(V))
    end
    refA = zeros(Float32, size(Ahost)); refB = zeros(Float32, size(Bhost))
    NextLA.uncompress!(refA, Ahost); NextLA.uncompress!(refB, Bhost)
    for (name, AT, sync) in backends
        name == "CUDA" || continue
        A = _compressed_ftlr_to_backend(Ahost, AT); B = _compressed_ftlr_to_backend(Bhost, AT)
        for bytes in unique((NextLA.gemm_minimum_workspace_bytes(A, B),
                             NextLA.gemm_maximum_workspace_bytes(A, B)))
            C0 = rand(rng, Float32, 9, 11); C = AT(copy(C0))
            _TLRM.gemm!(C, A, B; workspace=bytes, alpha=1.25f0, beta=-0.5f0)
            sync(C)
            @test Array(C) ≈ 1.25f0 .* (refA * refB) .- 0.5f0 .* C0 rtol=3f-4 atol=3f-4
        end
    end
end

@testset "CompressedFTLR CUDA logical transpose combinations" begin
    Ahost = _compressed_ftlr_fixture(Float32, Int[1 2; 3 1])
    Bhost = _compressed_ftlr_fixture(Float32, Int[2 1; 1 3])
    referenceA = zeros(Float32, size(Ahost)); referenceB = zeros(Float32, size(Bhost))
    NextLA.uncompress!(referenceA, Ahost); NextLA.uncompress!(referenceB, Bhost)
    for (name, AT, sync) in backends
        name == "CUDA" || continue
        A = _compressed_ftlr_to_backend(Ahost, AT)
        B = _compressed_ftlr_to_backend(Bhost, AT)
        for transA in ('N', 'T'), transB in ('N', 'T')
            opA = transA == 'N' ? referenceA : transpose(referenceA)
            opB = transB == 'N' ? referenceB : transpose(referenceB)
            minbytes = NextLA.gemm_minimum_workspace_bytes(A, B; transA, transB)
            maxbytes = NextLA.gemm_maximum_workspace_bytes(A, B; transA, transB)
            for bytes in unique((minbytes, maxbytes))
                C0 = rand(Float32, size(opA, 1), size(opB, 2))
                C = AT(copy(C0))
                _TLRM.gemm!(C, A, B; workspace=bytes, alpha=1.25f0, beta=-0.5f0,
                             transA, transB)
                sync(C)
                @test Array(C) ≈ 1.25f0 .* (opA * opB) .- 0.5f0 .* C0 rtol=2f-4 atol=2f-4
            end
        end
    end
end

@testset "CompressedFTLR CUDA dense GEMM" begin
    Ahost = _compressed_ftlr_fixture(Float32)
    Bhost = _compressed_ftlr_fixture(Float32, Int[1 0; 2 1; 3 1])
    referenceAhost = zeros(Float32, size(Ahost))
    referenceBhost = zeros(Float32, size(Bhost))
    NextLA.uncompress!(referenceAhost, Ahost)
    NextLA.uncompress!(referenceBhost, Bhost)
    Chost = zeros(Float32, 8, 8)
    _TLRM.gemm!(Chost, Ahost, Bhost;
                workspace=NextLA.gemm_minimum_workspace_bytes(Ahost, Bhost))
    @test Chost ≈ referenceAhost * referenceBhost rtol=2f-4 atol=2f-4
    for (name, AT, sync) in backends
        name == "CUDA" || continue
        A = _compressed_ftlr_to_backend(Ahost, AT)
        B = _compressed_ftlr_to_backend(Bhost, AT)
        referenceA = zeros(Float32, size(A)); referenceB = zeros(Float32, size(B))
        NextLA.uncompress!(referenceA, Ahost); NextLA.uncompress!(referenceB, Bhost)
        reconstructed = AT(zeros(Float32, size(A)))
        NextLA.uncompress!(reconstructed, A)
        sync(reconstructed)
        @test Array(reconstructed) ≈ referenceA
        minbytes = NextLA.gemm_minimum_workspace_bytes(A, B)
        maxbytes = NextLA.gemm_maximum_workspace_bytes(A, B)
        for bytes in unique((minbytes, maxbytes))
            C0 = rand(Float32, size(A, 1), size(B, 2))
            C = AT(copy(C0))
            _TLRM.gemm!(C, A, B; workspace=bytes, alpha=1.5f0, beta=-0.25f0)
            sync(C)
            @test Array(C) ≈ 1.5f0 .* (referenceA * referenceB) .- 0.25f0 .* C0 rtol=2f-4 atol=2f-4
        end
    end
end

@testset "CompressedFTLR CUDA mixed grouped lowering and analysis" begin
    rank_grid = Int[1 2 1; 2 0 1; 1 1 1]
    H = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 9, 9, (4, 4), rank_grid)
    rng = MersenneTwister(2026)
    for j in 1:3, i in 1:3
        U, V = NextLA.get_factors(H, i, j)
        U .= randn(rng, Float32, size(U))
        V .= randn(rng, Float32, size(V))
    end
    Hdense = zeros(Float32, 9, 9)
    NextLA.uncompress!(Hdense, H)
    for (name, AT, sync) in backends
        name == "CUDA" || continue
        G = _compressed_ftlr_to_backend(H, AT)
        for side in (:compressed_dense, :dense_compressed),
            trans_dense in ('N', 'T'), trans_compressed in ('N', 'T')
            Xhost = randn(rng, Float32, 9, 9)
            X = AT(Xhost)
            C0 = randn(rng, Float32, 9, 9)
            C = AT(C0)
            workspace = NextLA.DenseGemmWorkspace(G, 4096)
            if side === :compressed_dense
                analysis = NextLA.analyze_compressed_gemm(
                    C, G, X; workspace,
                    transA=trans_compressed, transB=trans_dense)
                @test !analysis.has_fallback
                _TLRM.gemm!(
                    C, G, X; workspace, alpha=1.25f0, beta=-0.5f0,
                    transA=trans_compressed, transB=trans_dense, analysis)
                opG = trans_compressed == 'N' ? Hdense : transpose(Hdense)
                opX = trans_dense == 'N' ? Xhost : transpose(Xhost)
            else
                analysis = NextLA.analyze_compressed_gemm(
                    C, X, G; workspace,
                    transA=trans_dense, transB=trans_compressed)
                @test !analysis.has_fallback
                _TLRM.gemm!(
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
    qm, qn = NextLA.grid_size(A)
    for j in 1:qn, i in 1:qm
        rows = _TLRM._compressed_ftlr_axis_range(A, i, 1)
        cols = _TLRM._compressed_ftlr_axis_range(A, j, 2)
        U, V = NextLA.get_factors(A, i, j)
        dense[rows, cols] .= Float32.(U) * transpose(Float32.(V))
    end
    return dense
end

@testset "CompressedFTLR arbitrary-rank FP16 and TF32 execution" begin
    rank_grid = Int[1 2; 3 1]
    for (T, compute, tolerance) in (
        (Float16, Float32, 2f-2),
        (Float32, NextLA.TF32(), 5f-3),
    )
        Ahost = _compressed_ftlr_fixture(T, rank_grid)
        Bhost = _compressed_ftlr_fixture(T, reverse(rank_grid; dims=1))
        reference = _compressed_dense32(Ahost) * _compressed_dense32(Bhost)
        for (name, AT, sync) in backends
            name == "CUDA" || continue
            A = _compressed_ftlr_to_backend(Ahost, AT)
            B = _compressed_ftlr_to_backend(Bhost, AT)
            workspace = NextLA.DenseGemmWorkspace(
                A, B; bytes=NextLA.gemm_maximum_workspace_bytes(A, B))
            C = AT(zeros(T, size(A, 1), size(B, 2)))
            analysis = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
            _TLRM.gemm!(C, A, B; workspace, compute, analysis)
            sync(C)
            @test norm(Float32.(Array(C)) - reference) / norm(reference) < tolerance
            # Same result through the transient (no-analysis) entry point.
            fill!(C, zero(T))
            _TLRM.gemm!(C, A, B; workspace, compute)
            sync(C)
            @test norm(Float32.(Array(C)) - reference) / norm(reference) < tolerance
            close(analysis)
        end
    end
end
