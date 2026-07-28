# Direct low-rank contraction drivers.
#
# A caller supplies canonical factor operands, local run geometry, and the destination
# tile origin.  Stage 1 is output-independent (`execute_stage1!` in `stages.jl`); only
# the terminal stage knows whether it writes a dense matrix or an M4 accumulation slab.

@inline lowrank_operands(left_outer, left_inner, right_outer, right_inner) =
    LogicalTLROperands(left_inner, right_outer, right_inner, left_outer)

@inline function _interior_pair(A::LogicalTLROperand)
    qm, qn = regular_grid_size(A)
    kind = interior_grid_kind(A)
    order = tile_order(A)
    return (interior_operand(kind, outer_factors(A, _INTERIOR), order, qm, qn),
            interior_operand(kind, inner_factors(A, _INTERIOR), order, qm, qn))
end

@inline _right_pair(A::LogicalTLROperand) =
    (panel_operand(PanelRowAxis(), outer_factors(A, _RIGHT)),
     panel_operand(PanelRowAxis(), inner_factors(A, _RIGHT)))

@inline _bottom_pair(A::LogicalTLROperand) =
    (panel_operand(PanelColAxis(), outer_factors(A, _BOTTOM)),
     panel_operand(PanelColAxis(), inner_factors(A, _BOTTOM)))

@inline _corner_pair(A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix}) =
    (corner_operand(outer_factors(A, _CORNER)),
     corner_operand(inner_factors(A, _CORNER)))

@inline function _lowrank_term_operands(Apair, Bpair)
    Aouter, Ainner = Apair
    Bouter, Binner = Bpair
    return lowrank_operands(Aouter, Ainner, Bouter, Binner)
end

@inline function _output_block_view(C, A, B, i0::Int, i1::Int, j0::Int, j1::Int)
    p0 = (i0 - 1) * nominal_tile_size(A, 1) + 1
    q0 = (j0 - 1) * nominal_tile_size(B, 2) + 1
    p1 = (i1 - 1) * nominal_tile_size(A, 1) + tile_size(A, i1, 1)[1]
    q1 = (j1 - 1) * nominal_tile_size(B, 2) + tile_size(B, 1, j1)[2]
    return view(C, p0:p1, q0:q1)
end

@inline _output_region(C, A, B, i0, qm, j0, qn) =
    _output_block_view(C, A, B, i0, i0 + qm - 1, j0, j0 + qn - 1)

function execute_dense_stage3!(::KAsGemmK, ::FoldRight, run::RowRun, ops,
                               ws::RowWorkspace, C, A, B, i0::Int, j0::Int,
                               alpha, beta, compute)
    bn = size(ws.T.data, 3)
    noff = size(ws.T.data, 2)
    rA = size(ws.T.data, 1)
    Jw = run_width(run)
    vb = ws.batches
    _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        Tstack = reshape(view(ws.T.data, :, :, :, 1:Jw, il), noff * rA, Jw * bn)
        push!(vb.s3u, view(ws.Ustacked, :, :, i))
        push!(vb.s3t, Tstack)
        gi = i0 + i - 1
        gj0 = j0 + run.j0 - 1
        gj1 = j0 + run.j1 - 1
        push!(vb.s3c, _output_block_view(C, A, B, gi, gi, gj0, gj1))
    end
    return precision_gemm_batched!('N', 'N', alpha, vb.s3u, vb.s3t, beta,
                                   vb.s3c, compute)
end

function execute_dense_stage3!(::KAsGemmK, ::FoldLeft, run::RowRun, ops,
                               ws::RowWorkspace, C, A, B, i0::Int, j0::Int,
                               alpha, beta, compute)
    bm = blockdim(ops.au)
    rB = size(ws.T.data, 2)
    noff = size(ws.T.data, 3)
    Zall = ws.Ustacked
    vb = ws.batches
    _clear_batches!(vb.s3t, vb.s3z, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for (jl, j) in enumerate(run.j0:run.j1)
            Tstack = reshape(view(ws.T.data, :, :, :, jl, il), bm, rB * noff)
            push!(vb.s3t, Tstack)
            push!(vb.s3z, view(Zall, :, :, j))
            push!(vb.s3c, _output_tile_view(C, A, B, i0 + i - 1, j0 + j - 1))
        end
    end
    return precision_gemm_batched!('N', 'T', alpha, vb.s3t, vb.s3z, beta,
                                   vb.s3c, compute)
end

function execute_dense_stage3!(::KAsSerialLoop, ::FoldRight, run::ColumnRun, ops,
                               ws::ColumnWorkspace, C, A, B, i0::Int, j0::Int,
                               alpha, beta, compute)
    vb = ws.batches
    noff = size(ws.Ufactored, 3)
    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
        for li in 1:noff
            i = panel_col(ops.au, k, li)
            for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
                j = panel_col(ops.bu, k, jpos)
                push!(vb.s3u, view(ws.Ufactored, :, :, li, k))
                push!(vb.s3t, view(ws.T.data, :, li, :, jx, kx))
                push!(vb.s3c, _output_tile_view(C, A, B, i0 + i - 1, j0 + j - 1))
            end
        end
        precision_gemm_batched!('N', 'N', alpha, vb.s3u, vb.s3t, beta,
                                vb.s3c, compute)
    end
    return nothing
end

@inline _default_fold(ops) = choose_fold(ops)

"""
    execute_lowrank_gemm!(C, A, B, ops, geometry, i0, j0; ...)

Execute one budgeted low-rank contraction directly into dense output. `ops` contains
canonical `V/W/Z/U` factor accessors and `geometry` contains only run/workspace sizes.
Local output coordinates start at the logical destination tile `(i0,j0)`.
"""
function execute_lowrank_gemm!(C, A, B, ops, geom::RegularGeometry, i0::Int, j0::Int;
        alpha, beta, budget::Int, compute, fold::Union{Nothing,FoldSide}=nothing,
        placement::Union{Nothing,KAxisSchedule}=nothing, arena=nothing)
    (geom.qm == 0 || geom.qn == 0) && return C
    region = _output_region(C, A, B, i0, geom.qm, j0, geom.qn)
    if geom.qk == 0 || geom.perA_row == 0 || geom.rA == 0 || geom.rB == 0
        _scale_output!(region, beta)
        return C
    end

    chosen_fold = fold === nothing ? _default_fold(ops) : fold
    chosen_placement = placement === nothing ? placement_for_fold(chosen_fold, ops) : placement
    return _execute_lowrank_gemm!(C, A, B, ops, geom, i0, j0, chosen_placement,
                                  chosen_fold, alpha, beta, budget, compute, region,
                                  arena)
end

# This dispatch boundary keeps the run loop fully concrete even when automatic fold
# selection has a two-type inferred union.
function _execute_lowrank_gemm!(C, A, B, ops, geom, i0::Int, j0::Int,
        chosen_placement::KAxisSchedule, chosen_fold::FoldSide,
        alpha, beta, budget::Int, compute, region, arena)
    beta_stage = if chosen_placement isa KAsGemmK
        beta
    else
        _scale_output!(region, beta)
        one(beta)
    end
    ws = allocate_workspace(chosen_placement, geom, ops, C, budget, chosen_fold;
                            arena)
    @inbounds for run in runs(chosen_placement, geom, budget, chosen_fold)
        prepare_run!(chosen_placement, run, ws)
        execute_stage1!(chosen_placement, run, ops, ws, compute)
        execute_stage2!(chosen_placement, chosen_fold, run, ops, ws, compute)
        execute_dense_stage3!(chosen_placement, chosen_fold, run, ops, ws,
                              C, A, B, i0, j0, alpha, beta_stage, compute)
    end
    return C
end

@inline function execute_lowrank_term!(C, A, B, Apair, Bpair,
        qm::Int, qk::Int, qn::Int, i0::Int, j0::Int;
        alpha, beta, budget::Int, compute,
        fold::Union{Nothing,FoldSide}=nothing,
        placement::Union{Nothing,KAxisSchedule}=nothing, arena=nothing)
    ops = _lowrank_term_operands(Apair, Bpair)
    geom = regular_geometry(qm, qk, qn, ops)
    return execute_lowrank_gemm!(C, A, B, ops, geom, i0, j0;
                                 alpha, beta, budget, compute, fold, placement,
                                 arena)
end

@inline function full_workspace_bytes(geom::RegularGeometry, ops;
        fold::Union{Nothing,FoldSide}=nothing,
        placement::Union{Nothing,KAxisSchedule}=nothing)
    (geom.qm == 0 || geom.qk == 0 || geom.qn == 0 ||
     geom.rA == 0 || geom.rB == 0) && return 0
    chosen_fold = fold === nothing ? _default_fold(ops) : fold
    chosen_placement = placement === nothing ? placement_for_fold(chosen_fold, ops) : placement
    per = chosen_placement isa KAsGemmK ? geom.perA_row : geom.perA_col
    per == 0 && return 0
    per_slice = chosen_placement isa KAsGemmK ?
                _row_slice_bytes(geom, chosen_fold) :
                _slice_bytes(geom.rA, geom.perA_col, geom.rB, geom.bn, eltype(geom))
    return per_slice * _full_run_tiles(chosen_placement, geom)
end

@inline function minimum_workspace_bytes(geom::RegularGeometry, ops;
        fold::Union{Nothing,FoldSide}=nothing,
        placement::Union{Nothing,KAxisSchedule}=nothing)
    (geom.qm == 0 || geom.qk == 0 || geom.qn == 0 ||
     geom.rA == 0 || geom.rB == 0) && return 0
    chosen_fold = fold === nothing ? _default_fold(ops) : fold
    chosen_placement = placement === nothing ? placement_for_fold(chosen_fold, ops) : placement
    per = chosen_placement isa KAsGemmK ? geom.perA_row : geom.perA_col
    per == 0 && return 0
    return chosen_placement isa KAsGemmK ?
           _row_slice_bytes(geom, chosen_fold) :
           _slice_bytes(geom.rA, geom.perA_col, geom.rB, geom.bn, eltype(geom))
end

function execute_lowrank_dense_term!(C, A, B, left_outer, left_inner,
        dense::LogicalDenseTile, qm::Int, i0::Int, j0::Int;
        alpha, beta, budget::Int, compute, arena=nothing)
    qm == 0 && return C
    region = _output_region(C, A, B, i0, qm, j0, 1)
    rA = rankdim(left_inner)
    if rA == 0
        _scale_output!(region, beta)
        return C
    end
    T = eltype(left_inner.data)
    bn = size(_output_tile_view(C, A, B, i0, j0), 2)
    maxI = clamp(div(budget, max(rA * bn * sizeof(T), 1)), 1, qm)
    _arena_reset!(arena)
    work = _workspace_array!(
        arena, get_backend(left_inner.data), T, rA, bn, maxI)
    data = _dense_data(dense)
    s1v = _batchvec(tilefactor(left_inner, 1, 1), maxI)
    s1d = _batchvec(data, maxI)
    s1t = _batchvec(view(work, :, :, 1), maxI)
    s2u = _batchvec(tilefactor(left_outer, 1, 1), maxI)
    s2c = _batchvec(_output_tile_view(C, A, B, i0, j0), maxI)
    @inbounds for irange in Iterators.partition(1:qm, maxI)
        il0 = first(irange)
        _clear_batches!(s1v, s1d, s1t, s2u, s2c)
        for i in irange
            il = i - il0 + 1
            push!(s1v, tilefactor(left_inner, i, 1))
            push!(s1d, data)
            push!(s1t, view(work, :, :, il))
            push!(s2u, tilefactor(left_outer, i, 1))
            push!(s2c, _output_tile_view(C, A, B, i0 + i - 1, j0))
        end
        precision_gemm_batched!('T', _dense_op(dense), one(T), s1v, s1d,
                                zero(T), s1t, compute)
        precision_gemm_batched!('N', 'N', alpha, s2u, s1t, beta, s2c, compute)
    end
    return C
end

function execute_dense_lowrank_term!(C, A, B, dense::LogicalDenseTile,
        right_outer, right_inner, qn::Int, i0::Int, j0::Int;
        alpha, beta, budget::Int, compute, arena=nothing)
    qn == 0 && return C
    region = _output_region(C, A, B, i0, 1, j0, qn)
    rB = rankdim(right_outer)
    if rB == 0
        _scale_output!(region, beta)
        return C
    end
    T = eltype(right_outer.data)
    bm = size(_output_tile_view(C, A, B, i0, j0), 1)
    maxJ = clamp(div(budget, max(bm * rB * sizeof(T), 1)), 1, qn)
    _arena_reset!(arena)
    work = _workspace_array!(
        arena, get_backend(right_outer.data), T, bm, rB, maxJ)
    data = _dense_data(dense)
    s1d = _batchvec(data, maxJ)
    s1w = _batchvec(tilefactor(right_outer, 1, 1), maxJ)
    s1t = _batchvec(view(work, :, :, 1), maxJ)
    s2z = _batchvec(tilefactor(right_inner, 1, 1), maxJ)
    s2c = _batchvec(_output_tile_view(C, A, B, i0, j0), maxJ)
    @inbounds for jrange in Iterators.partition(1:qn, maxJ)
        jl0 = first(jrange)
        _clear_batches!(s1d, s1w, s1t, s2z, s2c)
        for j in jrange
            jl = j - jl0 + 1
            push!(s1d, data)
            push!(s1w, tilefactor(right_outer, 1, j))
            push!(s1t, view(work, :, :, jl))
            push!(s2z, tilefactor(right_inner, 1, j))
            push!(s2c, _output_tile_view(C, A, B, i0, j0 + j - 1))
        end
        precision_gemm_batched!(_dense_op(dense), 'N', one(T), s1d, s1w,
                                zero(T), s1t, compute)
        precision_gemm_batched!('N', 'T', alpha, s1t, s2z, beta, s2c, compute)
    end
    return C
end

@inline full_workspace_bytes_lowrank_dense(qm::Int, rA::Int, bn::Int, ::Type{T}) where {T} =
    qm == 0 || rA == 0 ? 0 : qm * rA * bn * sizeof(T)

@inline full_workspace_bytes_dense_lowrank(qn::Int, bm::Int, rB::Int, ::Type{T}) where {T} =
    qn == 0 || rB == 0 ? 0 : qn * bm * rB * sizeof(T)

@inline minimum_workspace_bytes_lowrank_dense(qm::Int, rA::Int, bn::Int, ::Type{T}) where {T} =
    qm == 0 || rA == 0 ? 0 : rA * bn * sizeof(T)

@inline minimum_workspace_bytes_dense_lowrank(qn::Int, bm::Int, rB::Int, ::Type{T}) where {T} =
    qn == 0 || rB == 0 ? 0 : bm * rB * sizeof(T)

@inline function _lowrank_term_workspace(Apair, Bpair, qm, qk, qn, sizing;
                                         fold=nothing, placement=nothing)
    (qm == 0 || qk == 0 || qn == 0) && return 0
    ops = _lowrank_term_operands(Apair, Bpair)
    geom = regular_geometry(qm, qk, qn, ops)
    return sizing(geom, ops; fold, placement)
end

@inline _has_row_tail(A) = grid_size(A)[1] > regular_grid_size(A)[1]
@inline _has_col_tail(A) = grid_size(A)[2] > regular_grid_size(A)[2]

function _gemm_region_workspace_bound(A::AbstractTLRMatrix, B::AbstractTLRMatrix,
                                      sizing, lowrank_dense_sizing,
                                      dense_lowrank_sizing;
                                      transA::Char='N', transB::Char='N')
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match: size(op(A),2) == size(op(B),1)"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("op(A)'s column tile size must equal op(B)'s row tile size (contraction tiling)"))

    qm, qk = regular_grid_size(LA)
    qkB, qn = regular_grid_size(LB)
    qk == qkB ||
        throw(DimensionMismatch("op(A)'s column tile grid must equal op(B)'s row tile grid"))
    has_i = _has_row_tail(LA)
    has_k = _has_col_tail(LA) && _has_row_tail(LB)
    has_j = _has_col_tail(LB)

    Aint = _interior_pair(LA)
    Bint = _interior_pair(LB)
    Aright = _right_pair(LA)
    Abottom = _bottom_pair(LA)
    Bright = _right_pair(LB)
    Bbottom = _bottom_pair(LB)

    interior = Int[
        _lowrank_term_workspace(Aint, Bint, qm, qk, qn, sizing),
        has_k ? _lowrank_term_workspace(Aright, Bbottom, qm, 1, qn, sizing) : 0,
    ]
    right = Int[
        has_j ? _lowrank_term_workspace(Aint, Bright, qm, qk, 1, sizing) : 0,
    ]
    bottom = Int[
        has_i ? _lowrank_term_workspace(Abottom, Bint, 1, qk, qn, sizing) : 0,
    ]
    corner = Int[
        (has_i && has_j) ? _lowrank_term_workspace(
            Abottom, Bright, 1, qk, 1, sizing;
            fold=FoldRight(), placement=KAsSerialLoop{:k}()) : 0,
    ]

    if has_k && has_j
        if physical(LB) isa PaddedFTLRMatrix
            push!(right, _lowrank_term_workspace(
                Aright, _corner_pair(LB), qm, 1, 1, sizing))
        else
            push!(right, lowrank_dense_sizing(
                qm, maxrank(LA), tail_tile_size(LB, 2), eltype(LA)))
        end
    end
    if has_i && has_k
        if physical(LA) isa PaddedFTLRMatrix
            push!(bottom, _lowrank_term_workspace(
                _corner_pair(LA), Bbottom, 1, 1, qn, sizing))
        else
            push!(bottom, dense_lowrank_sizing(
                qn, tail_tile_size(LA, 1), maxrank(LB), eltype(LB)))
        end
    end
    if has_i && has_k && has_j && physical(LA) isa PaddedFTLRMatrix && physical(LB) isa PaddedFTLRMatrix
        push!(corner, _lowrank_term_workspace(
            _corner_pair(LA), _corner_pair(LB), 1, 1, 1, sizing))
    end

    # Dense-diagonal terms use complete, presently unpartitioned batches. They
    # are sequential with the regular term in their region, so each contributes
    # through a regional maximum rather than a sum.
    if physical(LA) isa TLRMatrix && physical(LB) isa TLRMatrix
        T = eltype(LA)
        n_int = size(outer_factors(LA, _INTERIOR), 3)
        diag_interior = n_int * nominal_tile_size(LA, 1) *
                        max(maxrank(LA), maxrank(LB)) * sizeof(T)
        push!(interior, diag_interior)
        has_j && push!(right, qm * nominal_tile_size(LA, 1) *
                              maxrank(LB) * sizeof(T))
        has_i && push!(bottom, qk * maxrank(LA) *
                               nominal_tile_size(LB, 2) * sizeof(T))
    end
    return (
        interior=maximum(interior; init=0),
        right=maximum(right; init=0),
        bottom=maximum(bottom; init=0),
        corner=maximum(corner; init=0),
    )
end

@inline _auxiliary_workspace(regions) =
    max(regions.right, regions.bottom, regions.corner)

function _gemm_workspace_regions(A, B, which::Symbol;
                                 transA::Char='N', transB::Char='N')
    if which === :minimum
        return _gemm_region_workspace_bound(
            A, B, minimum_workspace_bytes,
            minimum_workspace_bytes_lowrank_dense,
            minimum_workspace_bytes_dense_lowrank;
            transA, transB,
        )
    elseif which === :maximum
        return _gemm_region_workspace_bound(
            A, B, full_workspace_bytes,
            full_workspace_bytes_lowrank_dense,
            full_workspace_bytes_dense_lowrank;
            transA, transB,
        )
    end
    throw(ArgumentError("unknown workspace bound $which"))
end

"""
    gemm_minimum_workspace_bytes(A, B; transA='N', transB='N') -> Int

Return the smallest global numerical workspace for two-stream dense-output TLR
GEMM. It is the interior minimum plus the largest minimum of the serialized
right, bottom, and corner regions.
"""
function gemm_minimum_workspace_bytes(A::AbstractTLRMatrix, B::AbstractTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    regions = _gemm_workspace_regions(A, B, :minimum; transA, transB)
    return regions.interior + _auxiliary_workspace(regions)
end

"""
    gemm_maximum_workspace_bytes(A, B; transA='N', transB='N') -> Int

Return the smallest global numerical workspace for which every dense-output
TLR GEMM term can execute at full run width under the two-stream split.
"""
function gemm_maximum_workspace_bytes(A::AbstractTLRMatrix, B::AbstractTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    regions = _gemm_workspace_regions(A, B, :maximum; transA, transB)
    return regions.interior + _auxiliary_workspace(regions)
end

function _gemm_workspace_split(A, B, bytes::Int,
                               ::InteriorFirstWorkspace;
                               transA::Char='N', transB::Char='N')
    minimum = _gemm_workspace_regions(A, B, :minimum; transA, transB)
    maximum = _gemm_workspace_regions(A, B, :maximum; transA, transB)
    global_min = minimum.interior + _auxiliary_workspace(minimum)
    global_max = maximum.interior + _auxiliary_workspace(maximum)
    bytes >= global_min || throw(ArgumentError(
        "workspace has $bytes bytes; at least $global_min bytes are required"))
    usable = min(bytes, global_max)
    auxiliary_min = _auxiliary_workspace(minimum)
    interior = min(maximum.interior, usable - auxiliary_min)
    auxiliary = usable - interior
    return (; interior, auxiliary, usable, minimum=global_min, maximum=global_max)
end
