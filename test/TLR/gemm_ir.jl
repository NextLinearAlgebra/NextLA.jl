# Durable gates for the contraction IR (ROADMAP milestone 3).
#
# These pin the invariants that outlive the term-by-term migration:
#
#   * leaves are zero-copy and stay canonical `outer * inner'` under transpose —
#     the invariant that lets executors drop their transpose branches;
#   * the eight `(i,k,j) ∈ {regular, boundary}³` domains partition the tile-triple
#     space — falsifiable without reference to the code they schedule (a missing
#     corner is a gap, a miscounted span an overlap);
#   * semantic construction (`ContractOp`) is pure — folds, workspace, and backend
#     state appear only at `lower`, mutation only at `execute!`;
#   * the scheduler's promoted workspace is concretely typed. Correctness tests
#     cannot catch this class of regression (the `Tin` incident allocated ~1.4 KB
#     per run through dynamic dispatch while all correctness tests passed), and
#     `isconcretetype(typeof(x))` is vacuous — `typeof` of a runtime value is
#     always concrete. Only `@inferred` / `Base.return_types` ask the compiler.
#
# End-to-end correctness of every term lives in `gemm.jl`; budget compliance in
# `gemm_budget.jl`.

@testset "contraction IR leaves" begin
    b, r, nt = 8, 3, 5
    T = Float64

    # Boundary-tiled and rectangular, so interior/right/bottom/corner are all
    # non-empty and the row/column extents are distinguishable.
    mA, kk = b * nt + 3, b * nt + 5

    function fulllr(m, n, order)
        X = NextLA.TLRMatrix(zeros(T, m, n), b, r; tile_order=order)
        fill_random_tlr!(X, Array; seed=17)
        return X
    end

    @testset "leaves are zero-copy views" begin
        A = fulllr(mA, kk, NextLA.TileRowMajor())
        LA = _TLRM.logical_operand(A, 'N')

        @test parent(_TLRM.tilefactor(_TLRM.right_panel_leaf(LA).outer, 2, 1)) === A.right_U
        @test parent(_TLRM.tilefactor(_TLRM.bottom_panel_leaf(LA).inner, 1, 2)) === A.bottom_V
        @test parent(_TLRM.tilefactor(_TLRM.corner_leaf(LA).outer, 1, 1)) === A.corner_U
        @test parent(_TLRM.tilefactor(_TLRM.interior_leaf(_TLRM.FullGrid(), LA).outer, 2, 3)) === A.int_U
    end

    # Under `op = 'T'` the logical right panel is the physical bottom panel, and the
    # two factors swap so the leaf is still `outer * inner'`. A leaf therefore never
    # carries a transpose flag.
    @testset "transpose maps right↔bottom and swaps factors" begin
        A = fulllr(mA, kk, NextLA.TileRowMajor())
        LT = _TLRM.logical_operand(A, 'T')

        rp = _TLRM.right_panel_leaf(LT)
        @test rp.outer.data === A.bottom_V      # logical outer ← physical bottom inner
        @test rp.inner.data === A.bottom_U
        @test rp.outer.axis isa _TLRM.PanelRowAxis

        bp = _TLRM.bottom_panel_leaf(LT)
        @test bp.outer.data === A.right_V
        @test bp.inner.data === A.right_U
        @test bp.outer.axis isa _TLRM.PanelColAxis

        cn = _TLRM.corner_leaf(LT)
        @test cn.outer.data === A.corner_V
        @test cn.inner.data === A.corner_U

        # The interior leaf's grid extents follow the transposed geometry.
        li = _TLRM.interior_leaf(_TLRM.FullGrid(), LT)
        @test (li.outer.qm, li.outer.qn) == reverse(NextLA.TLRmodule.regular_tilegrid_size(
            _TLRM.logical_operand(A, 'N')))
    end

    # Dense-diagonal containers reach the interior through `SkipDiag` and carry their
    # diagonal as a dense leaf rather than a rank-deficient low-rank one.
    @testset "dense-diagonal container leaves" begin
        n = b * nt + 3
        A = NextLA.TLRDenseDiagMatrix(zeros(T, n, n), b, r; tile_order=NextLA.TileRowMajor())
        fill_random_tlr!(A, Array; seed=23)
        LA = _TLRM.logical_operand(A, 'N')

        leaf = _TLRM.interior_leaf(_TLRM.SkipDiag(), LA)
        @test leaf.outer isa _TLRM.InteriorOperand{_TLRM.SkipDiag}
        ops = _TLRM.logical_operands(LA, LA)
        @test _TLRM.tilefactor(leaf.outer, 1, 2) == _TLRM.tilefactor(ops.au, 1, 2)

        dl = _TLRM.dense_leaf(_TLRM._diag_tile_ref(LA, 1))
        @test _TLRM.leafop(dl) == 'N'
        @test _TLRM.leafdata(dl) == view(A.D, :, :, 1)
        @test _TLRM.leafop(_TLRM.dense_leaf(_TLRM._diag_tile_ref(_TLRM.logical_operand(A, 'T'), 1))) == 'T'
    end
end

# Constructing `ContractOp` cannot choose a fold or allocate workspace; `lower` adds
# those decisions in a concrete scheduled type; only `execute!` mutates the output.
@testset "interior ContractOp lowering" begin
    T = Float64
    b, r, nt = 8, 3, 5
    α, β = T(1.3), T(-0.4)

    function aligned_tlr(order; seed)
        X = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, r; tile_order=order)
        fill_random_tlr!(X, Array; seed)
        return X
    end

    @testset "operation contains semantics, lowering contains schedule" begin
        A = aligned_tlr(NextLA.TileRowMajor(); seed=61)
        B = aligned_tlr(NextLA.TileColMajor(); seed=62)
        LA = _TLRM.logical_operand(A, 'N')
        LB = _TLRM.logical_operand(B, 'N')
        C = zeros(T, size(A, 1), size(B, 2))

        op = @inferred _TLRM.interior_contract(C, LA, LB, α, _TLRM.ScaleExisting(β))
        @test op isa _TLRM.ContractOp
        @test op.domain === _TLRM.contract_domains(LA, LB).interior
        @test op.output.data === C
        @test op.alpha === α
        @test _TLRM.init_beta(op.init) === β
        @test op.left.outer.data === A.int_U
        @test op.left.inner.data === A.int_V
        @test op.right.outer.data === B.int_U
        @test op.right.inner.data === B.int_V
        @test !hasproperty(op, :placement)
        @test !hasproperty(op, :workspace)

        mode = _TLRM.default_gemm_compute_mode(T)
        # Automatic choice has a deliberate two-type inferred union: when both folds are
        # write-once, `choose_fold` compares runtime tile/rank sizes. Once a side is
        # selected, the scheduled object is fully concrete and lowering is inferable.
        scheduled = _TLRM._lower_contract(op, mode, 1024, nothing)
        forced = @inferred _TLRM._lower_contract(op, mode, 1024, _TLRM.FoldRight())
        @test scheduled isa _TLRM.ScheduledLowRankContract
        @test forced isa _TLRM.ScheduledLowRankContract
        @test scheduled.op === op
        @test scheduled.reassociation isa _TLRM.FoldSide
        @test scheduled.placement isa _TLRM.KAxisSchedule
        @test scheduled.geometry == _TLRM.geometry(op.domain, op.left, op.right)
        @test scheduled.operands.au === op.left.outer
        @test scheduled.operands.av === op.left.inner
        @test scheduled.operands.bw === op.right.outer
        @test scheduled.operands.bz === op.right.inner
        @test scheduled.budget == 1024
        @test all(iszero, C)                  # construction and lowering are pure
    end

    # One representative correctness row per lowering entry point — the full
    # layout × budget sweeps live in `gemm.jl` end-to-end.
    @testset "automatic lowering matches the dense reference" begin
        A = aligned_tlr(NextLA.TileColMajor(); seed=63)
        B = aligned_tlr(NextLA.TileRowMajor(); seed=64)
        LA = _TLRM.logical_operand(A, 'N')
        LB = _TLRM.logical_operand(B, 'N')
        C0 = randn(T, size(A, 1), size(B, 2))
        C = copy(C0)
        op = _TLRM.interior_contract(C, LA, LB, α, _TLRM.ScaleExisting(β))
        scheduled = _TLRM.lower(op; compute=_TLRM.default_gemm_compute_mode(T), budget=1)

        @test _TLRM.execute!(scheduled) === C
        reference = α * reconstruct_tlr(A) * reconstruct_tlr(B) + β * C0
        @test C ≈ reference rtol=1e-10 atol=1e-10
    end

    # ReassociateLeft stacks B's inner factors across k and is only a legal write-once
    # lowering when B's effective storage is TileColMajor. Pin that path independently
    # of `choose_fold`'s size tie-break.
    @testset "forced ReassociateLeft on a legal layout" begin
        A = aligned_tlr(NextLA.TileRowMajor(); seed=65)
        B = aligned_tlr(NextLA.TileColMajor(); seed=66)
        LA = _TLRM.logical_operand(A, 'N')
        LB = _TLRM.logical_operand(B, 'N')
        C0 = randn(T, size(A, 1), size(B, 2))
        C = copy(C0)
        op = _TLRM.interior_contract(C, LA, LB, α, _TLRM.ScaleExisting(β))
        scheduled = _TLRM.lower(op; compute=_TLRM.default_gemm_compute_mode(T), budget=1024,
                                reassociation=_TLRM.FoldLeft())
        _TLRM.execute!(scheduled)
        @test C ≈ α * reconstruct_tlr(A) * reconstruct_tlr(B) + β * C0 rtol=1e-10 atol=1e-10
    end

    @testset "empty contraction still lowers init" begin
        A = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, 0;
                             tile_order=NextLA.TileRowMajor())
        B = NextLA.TLRMatrix(zeros(T, b * nt, b * nt), b, 0;
                             tile_order=NextLA.TileRowMajor())
        C = ones(T, b * nt, b * nt)
        op = _TLRM.interior_contract(C, _TLRM.logical_operand(A), _TLRM.logical_operand(B),
                                    α, _TLRM.ScaleExisting(β))
        scheduled = _TLRM.lower(op; compute=_TLRM.default_gemm_compute_mode(T), budget=1)
        _TLRM.execute!(scheduled)
        @test all(C .== β)
    end
end

# The full-LR container uses the shared three-stage lowering for corner leaves; the
# dense-diagonal container dispatches to specialized two-stage LR×Dense / Dense×LR
# forms. Boundary-slice correctness lives in `gemm.jl`.
@testset "corner leaf pairs select their lowering family" begin
    T = Float64
    b, r = 8, 3
    α, β = T(1.25), T(-0.5)
    n = 4b + 3

    lrA = NextLA.TLRMatrix(zeros(T, n, n), b, r; tile_order=NextLA.TileRowMajor())
    lrB = NextLA.TLRMatrix(zeros(T, n, n), b, r; tile_order=NextLA.TileColMajor())
    ddA = NextLA.TLRDenseDiagMatrix(zeros(T, n, n), b, r; tile_order=NextLA.TileRowMajor())
    ddB = NextLA.TLRDenseDiagMatrix(zeros(T, n, n), b, r; tile_order=NextLA.TileColMajor())
    for (seed, X) in enumerate((lrA, lrB, ddA, ddB))
        fill_random_tlr!(X, Array; seed=100 + seed)
    end
    C = zeros(T, n, n)
    mode = _TLRM.default_gemm_compute_mode(T)

    lr_right = _TLRM.rpanel_by_corner_contract(
        C, _TLRM.logical_operand(lrA), _TLRM.logical_operand(lrB), α,
        _TLRM.ScaleExisting(β))
    @test lr_right.left isa _TLRM.LowRankLeaf
    @test lr_right.right isa _TLRM.LowRankLeaf
    @test _TLRM.lower(lr_right; compute=mode, budget=1024) isa
          _TLRM.ScheduledLowRankContract

    dense_right = @inferred _TLRM.rpanel_by_corner_contract(
        C, _TLRM.logical_operand(ddA), _TLRM.logical_operand(ddB), α,
        _TLRM.ScaleExisting(β))
    @test dense_right.left isa _TLRM.LowRankLeaf
    @test dense_right.right isa _TLRM.DenseLeaf
    @test (@inferred _TLRM._lower_contract(dense_right, mode, 1024, nothing)) isa
          _TLRM.ScheduledLowRankDenseContract

    dense_bottom = @inferred _TLRM.corner_by_bpanel_contract(
        C, _TLRM.logical_operand(ddA), _TLRM.logical_operand(ddB), α,
        _TLRM.ScaleExisting(β))
    @test dense_bottom.left isa _TLRM.DenseLeaf
    @test dense_bottom.right isa _TLRM.LowRankLeaf
    @test (@inferred _TLRM._lower_contract(dense_bottom, mode, 1024, nothing)) isa
          _TLRM.ScheduledDenseLowRankContract
end

# The eight terms are the eight corners of the `(i,k,j) ∈ {regular, boundary}³` cube.
# Stated that way it is falsifiable without reference to the scheduled code: the eight
# domains must partition the full tile-triple space for both containers and for every
# combination of tile-aligned and tailed axes.
# The geometric enumeration of a domain. This is the test's own specification of what a
# domain *means*; `src` carries no such iterator because nothing schedules through one —
# executors loop `span_range` directly and choose their own order.
_triples(d) = Set((i, k, j) for i in NextLA.TLRmodule.span_range(d.i),
                                k in NextLA.TLRmodule.span_range(d.k),
                                j in NextLA.TLRmodule.span_range(d.j))

@testset "contraction IR domains" begin
    b, r, nt = 8, 3, 5
    T = Float64

    function fulllr_pair(mA, kk, nB; oa=NextLA.TileRowMajor(), ob=NextLA.TileRowMajor())
        A = NextLA.TLRMatrix(zeros(T, mA, kk), b, r; tile_order=oa)
        B = NextLA.TLRMatrix(zeros(T, kk, nB), b, r; tile_order=ob)
        fill_random_tlr!(A, Array; seed=11)
        fill_random_tlr!(B, Array; seed=22)
        return _TLRM.logical_operand(A, 'N'), _TLRM.logical_operand(B, 'N')
    end

    # (label, mA, kk, nB) — which axes carry a tail
    geometries = (
        ("all axes tailed",      b * nt + 3, b * nt + 5, b * nt + 2),
        ("tile-aligned",         b * nt,     b * nt,     b * nt),
        ("k aligned, i/j tailed", b * nt + 3, b * nt,     b * nt + 2),
        ("only k tailed",        b * nt,     b * nt + 5, b * nt),
    )

    @testset "eight domains partition the tile-triple space — $label" for
            (label, mA, kk, nB) in geometries
        LA, LB = fulllr_pair(mA, kk, nB)
        doms = _TLRM.contract_domains(LA, LB)
        @test length(doms) == 8

        mtA, ktA = _TLRM.tilegrid_size(LA)
        ktB, ntB = _TLRM.tilegrid_size(LB)
        @test ktA == ktB

        all_triples = Set((i, k, j) for i in 1:mtA, k in 1:ktA, j in 1:ntB)
        sets = [_triples(d) for d in values(doms)]

        @test union(sets...) == all_triples                 # no gaps
        @test sum(length, sets) == length(all_triples)      # no overlaps
        for a in 1:8, c in (a + 1):8
            @test isempty(intersect(sets[a], sets[c]))
        end
    end

    # Emptiness is derived from geometry, not guarded per term: a tile-aligned axis has
    # no tail, so every corner pinning it drops out. Tile-aligned on all axes leaves the
    # interior alone — which is exactly the aligned `gemm!` path.
    @testset "tile-aligned axes empty the corners that pin them" begin
        LA, LB = fulllr_pair(b * nt, b * nt, b * nt)
        doms = _TLRM.contract_domains(LA, LB)
        @test !isempty(doms.interior)
        for name in (:int_by_rpanel, :bpanel_by_int, :rpanel_by_bpanel, :rpanel_by_corner,
                     :corner_by_bpanel, :bpanel_by_rpanel, :corner_by_corner)
            @test isempty(doms[name])
        end

        # Only the contraction axis tailed. A tail on `k` means A gains a *right* panel
        # (column tail) and B a *bottom* panel (row tail), so the term contracting over
        # the boundary `k` — `rpanel_by_bpanel` = u_A v_B' — switches ON. Nothing else
        # does: `i` and `j` stay aligned, so C is entirely interior.
        LA2, LB2 = fulllr_pair(b * nt, b * nt + 5, b * nt)
        d2 = _TLRM.contract_domains(LA2, LB2)
        @test !isempty(d2.interior)
        @test !isempty(d2.rpanel_by_bpanel)          # (1:q_m, bnd, 1:q_n) — u_A v_B'
        @test isempty(d2.bpanel_by_rpanel)           # needs A row tail + B col tail
        for name in (:int_by_rpanel, :bpanel_by_int, :rpanel_by_corner,
                     :corner_by_bpanel, :corner_by_corner)
            @test isempty(d2[name])
        end

        # A domain is non-empty exactly when the operand's region has tiles.
        @test isempty(d2.rpanel_by_bpanel) ==
              (_TLRM.region_tile_count(LA2, _TLRM._RIGHT) == 0 ||
               _TLRM.region_tile_count(LB2, _TLRM._BOTTOM) == 0)
        @test isempty(d2.bpanel_by_rpanel) ==
              (_TLRM.region_tile_count(LA2, _TLRM._BOTTOM) == 0 ||
               _TLRM.region_tile_count(LB2, _TLRM._RIGHT) == 0)
    end

    # The term tuple must be compile-time shaped: eight distinct concrete types, so a
    # loop over it unrolls and each term specialises. A `Vector{Any}` of domains would
    # put every term behind dynamic dispatch and regress the interior (see ROADMAP).
    @testset "contract_domains is inferable and concretely typed" begin
        LA, LB = fulllr_pair(b * nt + 3, b * nt + 5, b * nt + 2)
        doms = @inferred _TLRM.contract_domains(LA, LB)
        @test doms isa NamedTuple
        @test all(isconcretetype, map(typeof, values(doms)))
        @test isconcretetype(typeof(doms))
    end

    @testset "mismatched contraction grids are rejected" begin
        A = NextLA.TLRMatrix(zeros(T, b * 4, b * 4), b, r; tile_order=NextLA.TileRowMajor())
        B = NextLA.TLRMatrix(zeros(T, b * 6, b * 6), b, r; tile_order=NextLA.TileRowMajor())
        @test_throws DimensionMismatch _TLRM.contract_domains(
            _TLRM.logical_operand(A, 'N'), _TLRM.logical_operand(B, 'N'))
    end
end

# The init policy names the β rule: the row family writes each tile once so β folds
# into that write, while the column family loops the reduction and must pre-scale.
@testset "output init policy" begin
    @testset "policy values" begin
        @test _TLRM.init_beta(_TLRM.ScaleExisting(2.5)) == 2.5
        @test _TLRM.init_beta(_TLRM.AccumulateExisting(Float64)) === 1.0
        @test _TLRM.is_accumulate(_TLRM.AccumulateExisting(Float64))
        @test _TLRM.is_accumulate(_TLRM.ScaleExisting(1.0))     # β=1 is materially an accumulate
        @test !_TLRM.is_accumulate(_TLRM.ScaleExisting(2.5))
    end

    @testset "row family folds β into the single write" begin
        region = ones(Float64, 4, 4)
        β = _TLRM.lower_init!(region, _TLRM.ScaleExisting(2.5), _TLRM.KAsGemmK{:j}())
        @test β == 2.5                       # handed to the terminal GEMM
        @test all(region .== 1.0)            # region untouched — no pre-scale
    end

    @testset "column family pre-scales, then accumulates" begin
        region = ones(Float64, 4, 4)
        β = _TLRM.lower_init!(region, _TLRM.ScaleExisting(2.5), _TLRM.KAsSerialLoop{:k}())
        @test β == 1.0                       # writes accumulate
        @test all(region .== 2.5)            # β applied up front instead

        # β = 1 needs no scaling under either placement.
        region2 = ones(Float64, 4, 4)
        @test _TLRM.lower_init!(region2, _TLRM.ScaleExisting(1.0), _TLRM.KAsSerialLoop{:k}()) == 1.0
        @test all(region2 .== 1.0)
    end

    @testset "accumulate never scales" begin
        for placement in (_TLRM.KAsGemmK{:j}(), _TLRM.KAsSerialLoop{:k}())
            region = fill(3.0, 4, 4)
            @test _TLRM.lower_init!(region, _TLRM.AccumulateExisting(Float64), placement) === 1.0
            @test all(region .== 3.0)
        end
    end
end

# The `Tin` regression gate (ROADMAP milestone 3, *Measurement*). Scratch is
# `allocate(backend, eltype(geom), ...)`, inferable only while the operand type reaches
# the compiler as `ContractGeometry{T}`'s parameter. When it was a `Tin::DataType`
# field instead, inference collapsed to abstract batch buffers and the staged loops
# allocated ~1.4 KB per run — with all correctness tests passing. The gate was
# verified by injecting that bug: correctness stayed green while these failed.
@testset "scheduler promotes concretely-typed workspace" begin
    T = Float64
    b, r = 8, 4

    function mk(m, n, order)
        X = NextLA.TLRMatrix(zeros(T, m, n), b, r; tile_order=order)
        fill_random_tlr!(X, Array; seed=51)
        return X
    end

    @testset "geometry is inferable" begin
        A = mk(b * 5 + 3, b * 6 + 2, NextLA.TileRowMajor())
        B = mk(b * 6 + 2, b * 4 + 1, NextLA.TileRowMajor())
        LA = _TLRM.logical_operand(A, 'N'); LB = _TLRM.logical_operand(B, 'N')
        la, lb = _TLRM.interior_leaves(LA, LB)
        dom = _TLRM.contract_domains(LA, LB).interior
        # `@inferred` throws unless the inferred return type is concrete — the operand
        # type must reach the compiler as `ContractGeometry`'s parameter, not a field.
        g = @inferred _TLRM.geometry(dom, la, lb)
        @test g isa _TLRM.ContractGeometry{T}
        @test eltype(g) === T
    end

    @testset "$oa × $ob" for (oa, ob) in
            ((NextLA.TileRowMajor(), NextLA.TileColMajor()),
             (NextLA.TileColMajor(), NextLA.TileRowMajor()))
        A = mk(b * 8, b * 8, oa)
        B = mk(b * 8, b * 8, ob)
        LA = _TLRM.logical_operand(A, 'N')
        LB = _TLRM.logical_operand(B, 'N')
        C = zeros(T, b * 8, b * 8)
        ops = _TLRM.logical_operands(LA, LB)
        geom = _TLRM.interior_geometry(LA, LB)
        f = _TLRM.choose_fold(ops)
        placement = _TLRM.placement_for_fold(f, ops)
        ws = _TLRM.allocate_workspace(placement, geom, ops, C, 128 * 1024 * 1024, f)

        @test eltype(geom) === T

        # The real check: does the compiler know the workspace type? This is what a
        # vacuous `isconcretetype(typeof(ws))` fails to ask.
        @test isconcretetype(only(Base.return_types(_TLRM.allocate_workspace,
            (typeof(placement), typeof(geom), typeof(ops), typeof(C), Int, typeof(f)))))

        # eltype is the property under test; `ndims` legitimately differs by family (the
        # row family carries an `i` axis the column family folds into its `k` loop).
        @test eltype(ws.S.data) === T
        @test eltype(ws.T.data) === T
        # A Vector's declared eltype can genuinely be abstract — this one is not vacuous.
        for buf in values(ws.batches)
            @test isconcretetype(eltype(buf))
        end
        @test (@inferred _TLRM.runs(placement, geom, 128 * 1024 * 1024)) !== nothing
    end

    # Explicit per-family coverage: pin both reduction families directly rather than
    # relying on `choose_fold` to keep routing one combo each way (DESIGN §11).
    @testset "both reduction families promote concretely" begin
        A = mk(b * 8, b * 8, NextLA.TileColMajor())
        B = mk(b * 8, b * 8, NextLA.TileRowMajor())
        LA = _TLRM.logical_operand(A, 'N'); LB = _TLRM.logical_operand(B, 'N')
        C = zeros(T, b * 8, b * 8)
        ops = _TLRM.logical_operands(LA, LB)
        geom = _TLRM.interior_geometry(LA, LB)

        for placement in (_TLRM.KAsGemmK{:j}(), _TLRM.KAsSerialLoop{:j}())
            ws = _TLRM.allocate_workspace(placement, geom, ops, C, 128 * 1024 * 1024,
                                          _TLRM.FoldRight())
            @test isconcretetype(only(Base.return_types(_TLRM.allocate_workspace,
                (typeof(placement), typeof(geom), typeof(ops), typeof(C), Int,
                 typeof(_TLRM.FoldRight())))))
            @test eltype(ws.S.data) === T && eltype(ws.T.data) === T
            for buf in values(ws.batches)
                @test isconcretetype(eltype(buf))
            end
        end
    end

    # Both folds must promote concretely; FoldLeft has its own workspace shape.
    @testset "FoldLeft workspace is concrete" begin
        A = mk(b * 8, b * 8, NextLA.TileRowMajor())
        B = mk(b * 8, b * 8, NextLA.TileColMajor())
        LA = _TLRM.logical_operand(A, 'N'); LB = _TLRM.logical_operand(B, 'N')
        C = zeros(T, b * 8, b * 8)
        ops = _TLRM.logical_operands(LA, LB)
        geom = _TLRM.interior_geometry(LA, LB)
        ws = _TLRM.allocate_workspace(_TLRM.KAsGemmK{:k}(), geom, ops, C,
                                      128 * 1024 * 1024, _TLRM.FoldLeft())
        @test eltype(ws.S.data) === T
        @test isconcretetype(only(Base.return_types(_TLRM.allocate_workspace,
            (typeof(_TLRM.KAsGemmK{:k}()), typeof(geom), typeof(ops), typeof(C), Int,
             typeof(_TLRM.FoldLeft())))))
        for buf in values(ws.batches)
            @test isconcretetype(eltype(buf))
        end
    end
end
