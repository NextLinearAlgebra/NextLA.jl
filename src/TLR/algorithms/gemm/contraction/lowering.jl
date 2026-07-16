# Scheduling, execution, and workspace lowering for structured contractions.

"""
The four factor operands expected by the existing staged executor, reconstructed from
the algebra leaves. This is a lowering adapter, not a second transpose interpretation:
the leaves are already canonical `outer * inner'` views.
"""
@inline scheduled_operands(left::LowRankLeaf, right::LowRankLeaf) =
    LogicalTLROperands(left.inner, right.outer, right.inner, left.outer)

"""
    ScheduledLowRankContract

A low-rank × low-rank `ContractOp` after reassociation and reduction mapping have been
selected. All fields affecting dispatch are concrete type parameters. Runtime run sizes
remain values because they are budget-dependent.
"""
struct ScheduledLowRankContract{O,P<:KAxisSchedule,F<:FoldSide,G,Ops,M}
    op::O
    placement::P
    reassociation::F
    geometry::G
    operands::Ops
    compute::M
    budget::Int
end

"""Specialized two-stage lowering of `LowRankLeaf × DenseLeaf`."""
struct ScheduledLowRankDenseContract{O,M}
    op::O
    compute::M
    budget::Int
end

"""Specialized two-stage lowering of `DenseLeaf × LowRankLeaf`."""
struct ScheduledDenseLowRankContract{O,M}
    op::O
    compute::M
    budget::Int
end

"""Specialized one-stage lowering of `DenseLeaf × DenseLeaf`."""
struct ScheduledDenseDenseContract{O,M}
    op::O
    compute::M
end

# A bottom panel × right panel has singleton output axes and a non-trivial reduction.
# Although both factor stacks are contiguous in `k`, mapping the complete reduction to
# one GEMM K dimension would make its workspace unbounded in the tile-grid size. Keep
# `k` as the serial, budget-blockable loop and K-stack only within each run.
@inline _contract_fold(::LowRankLeaf{<:PanelOperand{PanelColAxis}},
                       ::LowRankLeaf{<:PanelOperand{PanelRowAxis}}, ops) = FoldRight()
@inline _contract_fold(::LowRankLeaf, ::LowRankLeaf, ops) = choose_fold(ops)

@inline _contract_placement(::LowRankLeaf{<:PanelOperand{PanelColAxis}},
                            ::LowRankLeaf{<:PanelOperand{PanelRowAxis}},
                            ::FoldRight, ops) = KAsSerialLoop{:k}()
@inline _contract_placement(::LowRankLeaf, ::LowRankLeaf, fold::FoldSide, ops) =
    placement_for_fold(fold, ops)

@inline function _lower_contract(op::ContractOp{<:ContractDomain,<:LowRankLeaf,<:LowRankLeaf},
                                 compute, budget::Int, reassociation::FoldSide)
    ops = scheduled_operands(op.left, op.right)
    geom = geometry(op.domain, op.left, op.right)
    placement = _contract_placement(op.left, op.right, reassociation, ops)
    return ScheduledLowRankContract(op, placement, reassociation, geom, ops, compute, budget)
end

@inline _lower_contract(op::ContractOp{<:ContractDomain,<:LowRankLeaf,<:DenseLeaf},
                        compute, budget::Int, ::Nothing) =
    ScheduledLowRankDenseContract(op, compute, budget)

@inline _lower_contract(op::ContractOp{<:ContractDomain,<:DenseLeaf,<:LowRankLeaf},
                        compute, budget::Int, ::Nothing) =
    ScheduledDenseLowRankContract(op, compute, budget)

@inline _lower_contract(op::ContractOp{<:ContractDomain,<:DenseLeaf,<:DenseLeaf},
                        compute, ::Int, ::Nothing) =
    ScheduledDenseDenseContract(op, compute)

@inline function _lower_contract(op::ContractOp{<:ContractDomain,<:LowRankLeaf,<:LowRankLeaf},
                                 compute, budget::Int, ::Nothing)
    ops = scheduled_operands(op.left, op.right)
    reassociation = _contract_fold(op.left, op.right, ops)
    geom = geometry(op.domain, op.left, op.right)
    placement = _contract_placement(op.left, op.right, reassociation, ops)
    return ScheduledLowRankContract(op, placement, reassociation, geom, ops, compute, budget)
end

"""
    lower(op; compute, budget, reassociation=nothing)

Choose a low-rank contraction's reassociation, reduction mapping, and budget-shaped
geometry. Lowering is pure: workspace promotion and library calls happen in `execute!`.
"""
@inline lower(op::ContractOp; compute, budget::Int,
              reassociation::Union{Nothing,FoldSide}=nothing) =
    _lower_contract(op, compute, budget, reassociation)

"""
    execute!(scheduled) -> output

Promote typed S/T workspace for the scheduled run width and execute the unchanged
three-stage low-rank contraction. The init policy is lowered only now because its
implementation depends on whether reduction was mapped to GEMM K or a serial loop.
"""
function execute!(scheduled::ScheduledLowRankContract)
    op = scheduled.op
    geom = scheduled.geometry
    C = op.output.data
    # No output coordinates means there is no destination region to initialize. An empty
    # reduction with a non-empty destination is different: it still applies β below.
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    region = output_region(op.output)

    # No scheduling primitive may see a zero-sized reduction/run. The algebraic product
    # is empty, but the operation remains responsible for its output init policy.
    if isempty(op.domain.k) || geom.perA_row == 0 || geom.rA == 0 || geom.rB == 0
        is_accumulate(op.init) || _scale_output!(region, init_beta(op.init))
        return C
    end

    beta_stage = lower_init!(region, op.init, scheduled.placement)
    ws = allocate_workspace(scheduled.placement, geom, scheduled.operands, C,
                            scheduled.budget, scheduled.reassociation)
    @inbounds for run in runs(scheduled.placement, geom, scheduled.budget)
        prepare_run!(scheduled.placement, run, ws)
        execute_stage!(stage1(scheduled.placement, run, scheduled.operands, ws,
                              scheduled.reassociation, scheduled.compute))
        execute_stage!(stage2(scheduled.placement, run, scheduled.operands, ws,
                              scheduled.reassociation, scheduled.compute))
        execute_stage!(stage3(scheduled.placement, run, scheduled.operands, ws, op.output,
                              op.alpha, beta_stage, scheduled.reassociation,
                              scheduled.compute))
    end
    return C
end


@inline function _empty_product!(op::ContractOp)
    (isempty(op.domain.i) || isempty(op.domain.j)) && return op.output.data
    is_accumulate(op.init) || _scale_output!(output_region(op.output), init_beta(op.init))
    return op.output.data
end

"""Execute the specialized `LowRank × Dense` two-stage lowering."""
function execute!(scheduled::ScheduledLowRankDenseContract)
    op = scheduled.op
    C = op.output.data
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    (isempty(op.domain.k) || rankdim(op.left) == 0) && return _empty_product!(op)

    T = eltype(op.left.inner.data)
    ni = span_extent(op.domain.i)
    k = first(span_range(op.domain.k))
    bn = size(output_tile(op.output, 1, 1), 2)
    rA = rankdim(op.left)
    maxI = clamp(div(scheduled.budget, max(rA * bn * sizeof(T), 1)), 1, ni)
    work = allocate(get_backend(op.left.inner.data), T, rA, bn, maxI)
    dense = leafdata(op.right)

    s1v = _batchvec(tilefactor(op.left.inner, 1, k), maxI)
    s1d = _batchvec(dense, maxI)
    s1t = _batchvec(view(work, :, :, 1), maxI)
    s2u = _batchvec(tilefactor(op.left.outer, 1, k), maxI)
    s2c = _batchvec(output_tile(op.output, 1, 1), maxI)
    beta = init_beta(op.init)

    @inbounds for irange in Iterators.partition(1:ni, maxI)
        i0 = first(irange)
        _clear_batches!(s1v, s1d, s1t, s2u, s2c)
        for i in irange
            il = i - i0 + 1
            push!(s1v, tilefactor(op.left.inner, i, k))
            push!(s1d, dense)
            push!(s1t, view(work, :, :, il))
            push!(s2u, tilefactor(op.left.outer, i, k))
            push!(s2c, output_tile(op.output, i, 1))
        end
        precision_gemm_batched!('T', leafop(op.right), one(T), s1v, s1d,
                                zero(T), s1t, scheduled.compute)
        precision_gemm_batched!('N', 'N', op.alpha, s2u, s1t, beta, s2c,
                                scheduled.compute)
    end
    return C
end

"""Execute the specialized `Dense × LowRank` two-stage lowering."""
function execute!(scheduled::ScheduledDenseLowRankContract)
    op = scheduled.op
    C = op.output.data
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    (isempty(op.domain.k) || rankdim(op.right) == 0) && return _empty_product!(op)

    T = eltype(op.right.outer.data)
    nj = span_extent(op.domain.j)
    k = first(span_range(op.domain.k))
    bm = size(output_tile(op.output, 1, 1), 1)
    rB = rankdim(op.right)
    maxJ = clamp(div(scheduled.budget, max(bm * rB * sizeof(T), 1)), 1, nj)
    work = allocate(get_backend(op.right.outer.data), T, bm, rB, maxJ)
    dense = leafdata(op.left)

    s1d = _batchvec(dense, maxJ)
    s1w = _batchvec(tilefactor(op.right.outer, k, 1), maxJ)
    s1t = _batchvec(view(work, :, :, 1), maxJ)
    s2z = _batchvec(tilefactor(op.right.inner, k, 1), maxJ)
    s2c = _batchvec(output_tile(op.output, 1, 1), maxJ)
    beta = init_beta(op.init)

    @inbounds for jrange in Iterators.partition(1:nj, maxJ)
        j0 = first(jrange)
        _clear_batches!(s1d, s1w, s1t, s2z, s2c)
        for j in jrange
            jl = j - j0 + 1
            push!(s1d, dense)
            push!(s1w, tilefactor(op.right.outer, k, j))
            push!(s1t, view(work, :, :, jl))
            push!(s2z, tilefactor(op.right.inner, k, j))
            push!(s2c, output_tile(op.output, 1, j))
        end
        precision_gemm_batched!(leafop(op.left), 'N', one(T), s1d, s1w,
                                zero(T), s1t, scheduled.compute)
        precision_gemm_batched!('N', 'T', op.alpha, s1t, s2z, beta, s2c,
                                scheduled.compute)
    end
    return C
end


"""Execute the specialized `Dense × Dense` one-stage lowering."""
function execute!(scheduled::ScheduledDenseDenseContract)
    op = scheduled.op
    C = op.output.data
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    isempty(op.domain.k) && return _empty_product!(op)

    left = leafdata(op.left)
    right = leafdata(op.right)
    destination = output_tile(op.output, 1, 1)
    precision_gemm_batched!(leafop(op.left), leafop(op.right), op.alpha,
                            [left], [right], init_beta(op.init), [destination],
                            scheduled.compute)
    return C
end


# ─── Whole-program workspace query ───────────────────────────────────────────

@inline function _logical_dense_size(leaf::DenseLeaf)
    m, n = size(leafdata(leaf))
    return leafop(leaf) == 'N' ? (m, n) : (n, m)
end

@inline function _full_workspace_bytes(domain::ContractDomain,
        left::LowRankLeaf, right::LowRankLeaf)
    (isempty(domain.i) || isempty(domain.k) || isempty(domain.j) ||
     rankdim(left) == 0 || rankdim(right) == 0) && return 0
    ops = scheduled_operands(left, right)
    geom = geometry(domain, left, right)
    fold = _contract_fold(left, right, ops)
    placement = _contract_placement(left, right, fold, ops)
    per = placement isa KAsGemmK ? geom.perA_row : geom.perA_col
    return _slice_bytes(geom.rA, per, geom.rB, geom.bn, eltype(geom)) *
           _full_run_tiles(placement, geom)
end

@inline function _full_workspace_bytes(domain::ContractDomain,
        left::LowRankLeaf, right::DenseLeaf)
    (isempty(domain.i) || isempty(domain.k) || isempty(domain.j) ||
     rankdim(left) == 0) && return 0
    _, bn = _logical_dense_size(right)
    return span_extent(domain.i) * rankdim(left) * bn *
           sizeof(eltype(left.inner.data))
end

@inline function _full_workspace_bytes(domain::ContractDomain,
        left::DenseLeaf, right::LowRankLeaf)
    (isempty(domain.i) || isempty(domain.k) || isempty(domain.j) ||
     rankdim(right) == 0) && return 0
    bm, _ = _logical_dense_size(left)
    return span_extent(domain.j) * bm * rankdim(right) *
           sizeof(eltype(right.outer.data))
end

@inline _full_workspace_bytes(::ContractDomain, ::DenseLeaf, ::DenseLeaf) = 0

"""The eight semantic leaf-pair contractions emitted for a logical operand pair."""
@inline _emitted_contractions(A::LogicalTLROperand, B::LogicalTLROperand) =
    map((d, lr) -> (d, lr[1], lr[2]),
        values(contract_domains(A, B)), values(_term_leaves(A, B)))

"""
    gemm_workspace_bytes(A, B; transA='N', transB='N') -> Int

Smallest `max_workspace` value at which every emitted contraction in
`gemm!(C, A, B; transA, transB, max_workspace=…)` runs at full width. The same
per-operation budget is passed to all four output regions, so this is the maximum of
their individual promoted-workspace requirements. Dense × Dense corner work needs no
intermediate storage. Passing less remains correct and partitions budgeted operations
into smaller runs.
"""
function gemm_workspace_bytes(A::AbstractTLRMatrix, B::AbstractTLRMatrix;
                              transA::Char='N', transB::Char='N')
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match: size(op(A),2) == size(op(B),1)"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("op(A)'s column tile size must equal op(B)'s row tile size (contraction tiling)"))
    requirements = map(_emitted_contractions(LA, LB)) do spec
        domain, left, right = spec
        _full_workspace_bytes(domain, left, right)
    end
    return maximum(requirements)
end
