# Behaviour gates for the direct TLR GEMM core. End-to-end layout, transpose, boundary,
# precision, and budget sweeps live in `gemm.jl` and `gemm_budget.jl`.

@testset "direct GEMM factor operands" begin
    T = Float64
    b, r = 8, 3
    A = NextLA.TLRMatrix(zeros(T, b * 4 + 3, b * 5 + 2), b, r;
                         tile_order=NextLA.TileRowMajor())
    fill_random_tlr!(A, Array; seed=17)

    @testset "zero-copy boundary and transpose mapping" begin
        LA = _TLRM.logical_operand(A, 'N')
        Aright = _TLRM._right_pair(LA)
        Abottom = _TLRM._bottom_pair(LA)
        Acorner = _TLRM._corner_pair(LA)
        @test parent(_TLRM.tilefactor(Aright[1], 2, 1)) === A.right_U
        @test parent(_TLRM.tilefactor(Abottom[2], 1, 2)) === A.bottom_V
        @test parent(_TLRM.tilefactor(Acorner[1], 1, 1)) === A.corner_U

        LT = _TLRM.logical_operand(A, 'T')
        Tright = _TLRM._right_pair(LT)
        Tbottom = _TLRM._bottom_pair(LT)
        Tcorner = _TLRM._corner_pair(LT)
        @test Tright[1].data === A.bottom_V
        @test Tright[2].data === A.bottom_U
        @test Tbottom[1].data === A.right_V
        @test Tbottom[2].data === A.right_U
        @test Tcorner[1].data === A.corner_V
        @test Tcorner[2].data === A.corner_U
    end
end

@testset "direct regular execution" begin
    T = Float64
    b, r, nt = 8, 3, 5
    α, β = T(1.3), T(-0.4)

    function aligned(order, seed)
        X = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, r; tile_order=order)
        fill_random_tlr!(X, Array; seed)
        return X
    end

    @testset "layout selects both traversal families" begin
        for (oa, ob, family) in (
                (NextLA.TileRowMajor(), NextLA.TileRowMajor(), _TLRM.KAsGemmK),
                (NextLA.TileColMajor(), NextLA.TileRowMajor(), _TLRM.KAsSerialLoop))
            A = aligned(oa, 31)
            B = aligned(ob, 32)
            LA = _TLRM.logical_operand(A)
            LB = _TLRM.logical_operand(B)
            ops = _TLRM.logical_operands(LA, LB)
            fold = _TLRM.choose_fold(ops)
            @test _TLRM.placement_for_fold(fold, ops) isa family
        end
    end

    @testset "row and column drivers match dense reference" begin
        for (oa, ob) in ((NextLA.TileRowMajor(), NextLA.TileColMajor()),
                         (NextLA.TileColMajor(), NextLA.TileRowMajor()))
            A = aligned(oa, 41)
            B = aligned(ob, 42)
            LA = _TLRM.logical_operand(A)
            LB = _TLRM.logical_operand(B)
            ops = _TLRM.logical_operands(LA, LB)
            geom = _TLRM.interior_geometry(LA, LB)
            C0 = randn(T, size(A, 1), size(B, 2))
            C = copy(C0)
            _TLRM.execute_lowrank_gemm!(C, LA, LB, ops, geom, 1, 1;
                alpha=α, beta=β, budget=1, compute=_TLRM.default_gemm_compute_mode(T))
            @test C ≈ α * reconstruct_tlr(A) * reconstruct_tlr(B) + β * C0
        end
    end

    @testset "Stage 1 is output-independent" begin
        A = aligned(NextLA.TileRowMajor(), 51)
        B = aligned(NextLA.TileColMajor(), 52)
        LA = _TLRM.logical_operand(A)
        LB = _TLRM.logical_operand(B)
        ops = _TLRM.logical_operands(LA, LB)
        geom = _TLRM.interior_geometry(LA, LB)
        fold = _TLRM.choose_fold(ops)
        placement = _TLRM.placement_for_fold(fold, ops)
        C = zeros(T, size(A, 1), size(B, 2))
        ws = _TLRM.allocate_workspace(placement, geom, ops, C, 4096, fold)
        run = first(_TLRM.runs(placement, geom, 4096, fold))
        _TLRM.prepare_run!(placement, run, ws)
        _TLRM.execute_stage1!(placement, run, ops, ws,
                              _TLRM.default_gemm_compute_mode(T))
        @test any(x -> !iszero(x), ws.S.data)
        @test all(iszero, C)
    end
end

@testset "concrete regular workspace" begin
    T = Float64
    b, r, nt = 8, 4, 6
    A = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, r;
                         tile_order=NextLA.TileRowMajor())
    B = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, r;
                         tile_order=NextLA.TileColMajor())
    fill_random_tlr!(A, Array; seed=61)
    fill_random_tlr!(B, Array; seed=62)
    LA = _TLRM.logical_operand(A)
    LB = _TLRM.logical_operand(B)
    ops = _TLRM.logical_operands(LA, LB)
    geom = @inferred _TLRM.interior_geometry(LA, LB)
    @test geom isa _TLRM.RegularGeometry{T}
    @test eltype(geom) === T

    C = zeros(T, size(A, 1), size(B, 2))
    for (placement, fold) in ((_TLRM.KAsGemmK{:j}(), _TLRM.FoldRight()),
                              (_TLRM.KAsGemmK{:k}(), _TLRM.FoldLeft()),
                              (_TLRM.KAsSerialLoop{:j}(), _TLRM.FoldRight()))
        ws = _TLRM.allocate_workspace(placement, geom, ops, C, 4096, fold)
        @test isconcretetype(only(Base.return_types(_TLRM.allocate_workspace,
            (typeof(placement), typeof(geom), typeof(ops), typeof(C), Int, typeof(fold)))))
        @test eltype(ws.S.data) === T
        @test eltype(ws.T.data) === T
        @test all(buf -> isconcretetype(eltype(buf)), values(ws.batches))
    end
end

@testset "empty product and beta behaviour" begin
    T = Float64
    b, nt = 8, 4
    A = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, 0;
                         tile_order=NextLA.TileRowMajor())
    B = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, 0;
                         tile_order=NextLA.TileRowMajor())
    LA = _TLRM.logical_operand(A)
    LB = _TLRM.logical_operand(B)
    ops = _TLRM.logical_operands(LA, LB)
    geom = _TLRM.interior_geometry(LA, LB)
    C = fill(T(2), size(A, 1), size(B, 2))
    _TLRM.execute_lowrank_gemm!(C, LA, LB, ops, geom, 1, 1;
        alpha=T(3), beta=T(-0.5), budget=1,
        compute=_TLRM.default_gemm_compute_mode(T))
    @test all(C .== -one(T))
end
