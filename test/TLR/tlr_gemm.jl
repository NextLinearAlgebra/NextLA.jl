_canonical_tlr_backend(ArrayType::Type) =
    KernelAbstractions.get_backend(ArrayType(zeros(Float64, 1)))

function _canonical_tlr_fixture(::Type{T}, ArrayType;
                                rA=3, rB=4, rC=16, seed=1200) where {T}
    n, b = 48, 16
    qm = qn = cld(n, b)
    backend = _canonical_tlr_backend(ArrayType)
    A = NextLA.CompressedFTLRMatrix(backend, T, n, n, b, fill(rA, qm, qn);
                                    execution_rank_policy=:exact)
    B = NextLA.CompressedFTLRMatrix(backend, T, n, n, b, fill(rB, qm, qn);
                                    execution_rank_policy=:exact)
    C = NextLA.CompressedFTLRMatrix(backend, T, n, n, b, zeros(Int, qm, qn);
                                    execution_ranks=fill(rC, qm, qn))
    fill_random_tlr!(A, ArrayType; seed=seed + 1)
    fill_random_tlr!(B, ArrayType; seed=seed + 2)
    return A, B, C
end

function _check_canonical_tlr_gemm(ArrayType, synchronize;
                                   transA='N', transB='N',
                                   rA=3, rB=4, beta=0.0, seed=1200,
                                   workspace=nothing)
    T = Float64
    A, B, C = _canonical_tlr_fixture(T, ArrayType; rA, rB, seed)
    if !iszero(beta)
        fill_random_tlr!(C, ArrayType; seed=seed + 3)
    end
    Ad, Bd, C0 = reconstruct_tlr(A), reconstruct_tlr(B), reconstruct_tlr(C)
    op(X, t) = uppercase(t) == 'T' ? transpose(X) : X
    alpha = T(1.2)
    ref = alpha * op(Ad, transA) * op(Bd, transB) + T(beta) * C0
    actual_workspace = workspace === :minimum ?
        NextLA.tlr_gemm_minimum_workspace_bytes(
            C, A, B; transA, transB, block=4) : workspace

    NextLA.gemm!(
        C, A, B; alpha, beta=T(beta), transA, transB,
        tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4,
        workspace=actual_workspace,
    )
    synchronize(C.outer.data)
    got = reconstruct_tlr(C)
    @test norm(got - ref) / max(norm(ref), eps(T)) <= 2e-6
    @test all(NextLA.ranks(C) .<= NextLA.execution_ranks(C))
    return C, A, B
end

@testset "canonical row-major TLR-result gemm!" begin
    A, B, C = _canonical_tlr_fixture(Float64, Array)
    @test _TLRM.compressed_ftlr_outer_order(A) isa NextLA.TileRowMajor
    @test _TLRM.compressed_ftlr_inner_order(A) isa NextLA.TileColMajor
    @test _TLRM.compressed_ftlr_outer_order(B) isa NextLA.TileRowMajor
    @test _TLRM.compressed_ftlr_inner_order(B) isa NextLA.TileColMajor
    @test _TLRM.compressed_ftlr_outer_order(C) isa NextLA.TileRowMajor
    @test _TLRM.compressed_ftlr_inner_order(C) isa NextLA.TileColMajor

    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='N')
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='N',
                              beta=-0.3, seed=1210)
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='T',
                              rA=2, rB=6, seed=1220) # rank cost chooses right
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='T',
                              rA=6, rB=2, seed=1230) # rank cost chooses left
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='T',
                              seed=1240)
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='T',
                              beta=0.25, seed=1245)
    # TN: unreachable under PaddedFTLRMatrix's single fixed physical layout
    # (neither side had a zero-copy stack); complementary packing makes both
    # sides always zero-copy, so this is now an ordinary cost-based choice
    # like NT, not a special case.
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='N',
                              seed=1246)
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='N',
                              rA=2, rB=6, seed=1247)
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='N',
                              rA=6, rB=2, seed=1248, beta=0.4)

    LA = _TLRM.logical_operand(A, 'N')
    LAT = _TLRM.logical_operand(A, 'T')
    LB = _TLRM.logical_operand(B, 'N')
    LBT = _TLRM.logical_operand(B, 'T')
    @test _TLRM.choose_tlr_sampling_side(LA, LBT, 16, 4, 2, 6) == :right
    @test _TLRM.choose_tlr_sampling_side(LA, LBT, 16, 4, 6, 2) == :left
    # The cost formula has no transpose dependence once both sides are always
    # zero-copy, so TN reproduces the same :right/:left split as NT for the
    # same rank ratio.
    @test _TLRM.choose_tlr_sampling_side(LAT, LB, 16, 4, 2, 6) == :right
    @test _TLRM.choose_tlr_sampling_side(LAT, LB, 16, 4, 6, 2) == :left

    qm = qn = cld(48, 16)
    Acol = NextLA.CompressedFTLRMatrix(
        _canonical_tlr_backend(Array), Float64, 48, 48, 16, fill(3, qm, qn);
        outer_order=NextLA.TileColMajor, execution_rank_policy=:exact)
    @test_throws ArgumentError NextLA.gemm!(C, Acol, B; tol=1e-6)
end

@testset "canonical TLR-result reusable workspace" begin
    T = Float64
    A, B, C = _canonical_tlr_fixture(T, Array; rA=3, rB=4, seed=1248)
    ref = T(1.2) * reconstruct_tlr(A) * reconstruct_tlr(B)
    minimum = NextLA.tlr_gemm_minimum_workspace_bytes(C, A, B; block=4)
    maximum = NextLA.tlr_gemm_maximum_workspace_bytes(C, A, B; block=4)
    workspace = NextLA.TLRGemmWorkspace(C, A, B; block=4)
    @test sizeof(workspace) == maximum
    @test workspace.key.capacity == 3
    storage = (
        workspace.arena.persistent_t.storage,
        workspace.arena.phase_t.storage,
        workspace.arena.phase_thi.storage,
        workspace.U, workspace.V,
    )
    for _ in 1:2
        NextLA.gemm!(
            C, A, B; alpha=1.2, tol=1e-7, rel=true, eps_rel=1e-7,
            r_required=3, block=4, workspace)
        @test norm(reconstruct_tlr(C) - ref) / norm(ref) <= 2e-6
        @test all(a === b for (a, b) in zip(storage, (
            workspace.arena.persistent_t.storage,
            workspace.arena.phase_t.storage,
            workspace.arena.phase_thi.storage,
            workspace.U, workspace.V,
        )))
    end
    minimum_ws = NextLA.TLRGemmWorkspace(
        C, A, B; bytes=minimum, block=4)
    @test sizeof(minimum_ws) == minimum
    @test minimum_ws.key.capacity == 1
    NextLA.gemm!(
        C, A, B; alpha=1.2, tol=1e-7, rel=true, eps_rel=1e-7,
        r_required=3, block=4, workspace=minimum_ws)
    @test norm(reconstruct_tlr(C) - ref) / norm(ref) <= 2e-6

    middle = minimum
    while middle < maximum &&
          NextLA.TLRGemmWorkspace(C, A, B; bytes=middle, block=4).key.capacity == 1
        middle += max((maximum - minimum) ÷ 16, 1)
    end
    middle_ws = NextLA.TLRGemmWorkspace(
        C, A, B; bytes=min(middle, maximum), block=4)
    @test 1 <= middle_ws.key.capacity <= 3

    capped = NextLA.TLRGemmWorkspace(
        C, A, B; bytes=maximum + 1_000_000, block=4)
    @test capped.key.capacity == 3
    @test sizeof(capped) == maximum
    @test_throws ArgumentError NextLA.gemm!(
        C, A, B; block=4, workspace=minimum - 1)
    # A workspace's key is keyed on shape (rA, rB, grid, family, ...), not
    # transA/transB directly -- under complementary packing, `family`/`side`
    # are a pure function of rank/grid, so a transA/transB change alone need
    # not change the key (unlike the old PaddedFTLRMatrix regime, where NN
    # always meant :row_right and TT always meant :row_left). A genuinely
    # different-shaped operand pair still must be rejected.
    A2, B2, _ = _canonical_tlr_fixture(T, Array; rA=2, rB=4, seed=1249)
    @test_throws ArgumentError NextLA.gemm!(
        C, A2, B2; block=4, workspace)
end

@testset "rolling capacity preserves canonical operations" begin
    cases = [
        (transA='N', transB='N', rA=3, rB=4, beta=-0.2, seed=1280),
        (transA='N', transB='T', rA=2, rB=6, beta=0.0, seed=1290),
        (transA='N', transB='T', rA=6, rB=2, beta=0.15, seed=1300),
        (transA='T', transB='T', rA=3, rB=4, beta=-0.1, seed=1310),
        (transA='T', transB='N', rA=2, rB=6, beta=0.2, seed=1315),
    ]
    for kw in cases
        A, B, C = _canonical_tlr_fixture(
            Float64, Array; rA=kw.rA, rB=kw.rB, seed=kw.seed)
        minimum = NextLA.tlr_gemm_minimum_workspace_bytes(
            C, A, B; transA=kw.transA, transB=kw.transB, block=4)
        _check_canonical_tlr_gemm(
            Array, _ -> nothing; kw..., workspace=minimum)
    end
end

@testset "canonical row-major TLR-result gemm! on CUDA" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='N', seed=1250)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='T',
                rA=2, rB=6, seed=1260)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='T', transB='T', seed=1270)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='T', transB='N', seed=1275)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='N', seed=1320,
                workspace=:minimum)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='T',
                rA=2, rB=6, seed=1330, workspace=:minimum)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='T',
                rA=6, rB=2, seed=1340, workspace=:minimum)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='T', transB='T', seed=1350,
                workspace=:minimum)
        end
    end
end
