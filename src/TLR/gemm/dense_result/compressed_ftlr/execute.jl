@inline function _compressed_ftlr_row_outer_stack(A, i::Int, rho::Int)
    qm, qk = grid_size(A)
    1 <= i <= qm || throw(BoundsError(A, (i, :)))
    f = _compressed_ftlr_outer_storage(A)
    fi, fj = _compressed_ftlr_logical_coords(A, i, 1)
    li, lj = _compressed_ftlr_logical_coords(A, i, qk)
    first_slot = _compressed_ftlr_slot(f, fi, fj)
    last_slot = _compressed_ftlr_slot(f, li, lj)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(_compressed_ftlr_axis_range(A, i, 1))
    rho == 0 && return reshape(view(f.data, 1:0), rows, 0)
    axis = f.dimension_axis === :row ? fi : fj
    packed = reshape(view(f.data, first:last), f.leading_dimensions[axis], rho)
    return view(packed, 1:rows, :)
end

@inline function _compressed_ftlr_row_w_stack(B, k::Int, rho::Int)
    _, qn = grid_size(B)
    f = _compressed_ftlr_outer_storage(B)
    fi, fj = _compressed_ftlr_logical_coords(B, k, 1)
    li, lj = _compressed_ftlr_logical_coords(B, k, qn)
    first_slot = _compressed_ftlr_slot(f, fi, fj)
    last_slot = _compressed_ftlr_slot(f, li, lj)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(_compressed_ftlr_axis_range(B, k, 1))
    rho == 0 && return reshape(view(f.data, 1:0), rows, 0)
    axis = f.dimension_axis === :row ? fi : fj
    packed = reshape(view(f.data, first:last), f.leading_dimensions[axis], rho)
    return view(packed, 1:rows, :)
end

"""
Vertically stack A's inner factor `V_ik` over the run's rows `i0:i1`, for fixed
`k`. A's inner factor is column-packed across tile rows (`TileColMajor`), so
consecutive rows at a fixed `k` are adjacent in the packed array — this is a
zero-copy view, the same contiguity `_compressed_ftlr_row_w_stack` /
`_compressed_ftlr_col_z_stack` already rely on for their own stacking, just
walking a run-local row range instead of the full grid.

Fuses Stage 1 across every row in a scheduled run: `V_{i0:i1,k}' * W_{k,:}`
becomes one GEMM instead of one per row, reducing Stage-1 task count from
`(run length) * qk` to `qk`. Rows with zero rank at this `k` occupy zero bytes
in the packed storage, so they vanish from the span automatically — no
special-casing needed, matching the existing stacking helpers' behavior.
"""
@inline function _compressed_ftlr_run_v_stack(A, i0::Int, i1::Int, k::Int, rho::Int)
    _, qk = grid_size(A)
    1 <= k <= qk || throw(BoundsError(A, (:, k)))
    f = _compressed_ftlr_inner_storage(A)
    fi0, fk0 = _compressed_ftlr_logical_coords(A, i0, k)
    fi1, fk1 = _compressed_ftlr_logical_coords(A, i1, k)
    first_slot = _compressed_ftlr_slot(f, fi0, fk0)
    last_slot = _compressed_ftlr_slot(f, fi1, fk1)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(_compressed_ftlr_axis_range(A, k, 2))
    rho == 0 && return reshape(view(f.data, 1:0), rows, 0)
    axis = f.dimension_axis === :row ? fi0 : fk0
    packed = reshape(view(f.data, first:last), f.leading_dimensions[axis], rho)
    return view(packed, 1:rows, :)
end

@inline function _compressed_ftlr_col_z_stack(B, j::Int, gamma::Int)
    qk, qn = grid_size(B)
    1 <= j <= qn || throw(BoundsError(B, (:, j)))
    f = _compressed_ftlr_inner_storage(B)
    fi, fj = _compressed_ftlr_logical_coords(B, 1, j)
    li, lj = _compressed_ftlr_logical_coords(B, qk, j)
    first_slot = _compressed_ftlr_slot(f, fi, fj)
    last_slot = _compressed_ftlr_slot(f, li, lj)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(_compressed_ftlr_axis_range(B, j, 2))
    gamma == 0 && return reshape(view(f.data, 1:0), rows, 0)
    axis = f.dimension_axis === :row ? fi : fj
    packed = reshape(view(f.data, first:last), f.leading_dimensions[axis], gamma)
    return view(packed, 1:rows, :)
end

@inline function _ragged_view(data, offset::Int, rows::Int, cols::Int)
    cols == 0 && return reshape(view(data, 1:0), rows, 0)
    return reshape(view(data, offset:(offset + rows * cols - 1)), rows, cols)
end

"""Run-local task views. Building and submitting them are deliberately separate."""
struct CompressedFTLRRunTasks{S1,S2,S3,TD}
    stage1::S1
    stage2::S2
    stage3::S3
    tdata::TD
    needs_tdata_zero::Bool
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

"""
Stage 2 emits a GEMM for `(i,k,j)` only when `rA_ik` and `rB_kj` are both
nonzero, but each fold sizes its `T` arena from one side alone: FoldRight
reserves rows for every active `rA_ik` across all output columns, FoldLeft
reserves columns for every active `rB_kj` across all output rows. A rank hole
on the *opposite* side therefore leaves reserved space unwritten, and a stale
workspace value would reach Stage 3.

These detect exactly that, so the `T` clear -- `q` extra kernel launches -- is
skipped whenever no hole exists, which is the common all-positive-rank case.
Both reduce to a per-contraction-row test, since a hole needs only *some* `i`
in the run and *some* `j` to disagree:

    FoldRight: ∃k with (some active rA_ik) and (some rB_kj == 0)
    FoldLeft:  ∃k with (some rA_ik == 0)   and (some active rB_kj)

`rho_k[k] > 0` already answers "some active `rA_ik`" (ranks are non-negative,
so the run's sum is positive iff a term is), and `b_row_ranks` /
`b_row_has_zero` answer both B-side questions at plan level -- so FoldRight
costs O(qk) with no operand access at all.
"""
@inline _compressed_ftlr_foldright_needs_tdata_zero(plan, rho_k) =
    any(k -> rho_k[k] > 0 && plan.b_row_has_zero[k], eachindex(rho_k))

@inline _compressed_ftlr_foldleft_needs_tdata_zero(A, plan, irange) =
    any(eachindex(plan.b_row_ranks)) do k
        plan.b_row_ranks[k] > 0 &&
            any(i -> _compressed_ftlr_execution_rank(A, i, k) == 0, irange)
    end

"""
Stage-1 arena layout for one row run: `S` is one dense `(rho_k × σ_k)` block per
active `k`, fused across every row in the run (see `_compressed_ftlr_run_v_stack`).
`row_off[ii,k]` is where row `ii`'s contribution starts within block `k`;
`koff[k]` is that block's start offset in the run-local S arena. Identical for
FoldRight and FoldLeft — Stage 1 computes the shared `S_ikj = V_ik'W_kj`
regardless of which bracketing Stage 2/3 use downstream.
"""
function _compressed_ftlr_stage1_layout(A, irange, plan::CompressedFTLRRankPlan)
    _, qk = grid_size(A)
    nr = length(irange)
    rho_k = Base.zeros(Int, qk)
    row_off = Base.zeros(Int, nr, qk)
    koff = Base.zeros(Int, qk + 1)
    koff[1] = 1
    @inbounds for k in 1:qk
        acc = 0
        for (ii, i) in enumerate(irange)
            row_off[ii, k] = acc
            acc += _compressed_ftlr_execution_rank(A, i, k)
        end
        rho_k[k] = acc
        koff[k + 1] = koff[k] + acc * plan.b_row_ranks[k]
    end
    return rho_k, row_off, koff, koff[end] - 1
end

"""One grouped Stage-1 task per active `k` — was one task per `(i,k)`; fusing
the run's rows into a single `V`-stack (`_compressed_ftlr_run_v_stack`) cuts
Stage-1 task count from `(run length)·qk` down to `qk`."""
function _compressed_ftlr_stage1_tasks(A, B, irange, plan::CompressedFTLRRankPlan, rho_k, koff, sdata)
    T = eltype(A)
    i0, i1 = first(irange), last(irange)
    _, qk = grid_size(A)
    s1 = nothing
    @inbounds for k in 1:qk
        rBsum = plan.b_row_ranks[k]
        (rho_k[k] == 0 || rBsum == 0) && continue
        task = GroupedGemmTask('T', 'N', one(T),
                               _compressed_ftlr_run_v_stack(A, i0, i1, k, rho_k[k]),
                               _compressed_ftlr_row_w_stack(B, k, rBsum), zero(T),
                               _ragged_view(sdata, koff[k], rho_k[k], rBsum))
        if s1 === nothing; s1 = GroupedGemmTask[task]; else; push!(s1, task); end
    end
    return s1
end

"""`k`'s fused dense Stage-1 block, shaped `(rho_k × σ_k)`. Individual `S_ikj`
are strided views into it: rows `row_off[ii,k] .+ (1:rA)`, columns
`b_row_k_prefix[k,j] .+ (1:rB)`."""
@inline _compressed_ftlr_sblock(sdata, koff, rho_k, plan::CompressedFTLRRankPlan, k) =
    reshape(view(sdata, koff[k]:(koff[k + 1] - 1)), rho_k[k], plan.b_row_ranks[k])

"""
Execute one FoldRight CompressedFTLR row run: `T^R_ikj = S_ikj Z_kj'`, then
`C_i += [U_i1 ... U_iqk]·[T^R_i1j;...;T^R_iqkj]` — concatenating over `k` on the
left. `T` is packed `(i,j,k)`-order so each row is already the Stage-3
`rho_i × (qn*bn)` stack, letting Stage 3 issue one wide GEMM per row.
"""
function _build_compressed_ftlr_foldright_run(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                      alpha, beta, arena)
    T = eltype(A)
    _, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    nr = length(irange)
    scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
    sizehint!(scale_targets, nr)

    rho_k, row_off, koff, s_total = _compressed_ftlr_stage1_layout(A, irange, plan)
    # A row's terminal FoldRight operand is a dense `rho_i × (qn*bn)`
    # column-major matrix. Its individual `(k,j)` pieces are strided views
    # into that matrix, not contiguous blocks: raw concatenation by k would
    # place a block's second column after the next k block's first column.
    tbase = Base.zeros(Int, nr + 1)
    tbase[1] = 1
    @inbounds for (ii, i) in enumerate(irange)
        rho = plan.a_k_prefix[i, end]
        tbase[ii + 1] = tbase[ii] + rho * plan.output_col_prefix[end]
    end
    t_total = tbase[end] - 1

    # No active tile pair can contribute to this output-row run.
    if s_total == 0
        @inbounds for i in irange
            push!(scale_targets, (_compressed_ftlr_axis_range(A, i, 1), 1:size(C, 2)))
        end
        return CompressedFTLRRunTasks(
            nothing, nothing, nothing, nothing, false, scale_targets)
    end

    _arena_reset!(arena)
    backend = get_backend(A)
    sdata = _workspace_array!(arena, backend, T, s_total)
    tdata = _workspace_array!(arena, backend, T, t_total)

    s1 = _compressed_ftlr_stage1_tasks(A, B, irange, plan, rho_k, koff, sdata)

    # Stage 2: each S_ikj is multiplied by Z_kj'. Nested explicitly rather than
    # as one flat (i,k,j) loop so each view is built at the level it actually
    # depends on -- `tstack` on i, `Sblock` on (i,k) -- instead of being rebuilt
    # qk*qn and qn times respectively. Zero-rank i and (i,k) also skip whole
    # bands here rather than being rejected per innermost iteration.
    s2 = nothing
    @inbounds for (ii, i) in enumerate(irange)
        rho = plan.a_k_prefix[i, end]
        rho == 0 && continue
        tstack = _ragged_view(tdata, tbase[ii], rho, plan.output_col_prefix[end])
        for k in 1:qk
            rA = _compressed_ftlr_execution_rank(A, i, k)
            rA == 0 && continue
            rho_before_k = plan.a_k_prefix[i, k]
            Sblock = _compressed_ftlr_sblock(sdata, koff, rho_k, plan, k)
            Srows = (row_off[ii, k] + 1):(row_off[ii, k] + rA)
            for j in 1:qn
                rB = _compressed_ftlr_execution_rank(B, k, j)
                rB == 0 && continue
                Sview = view(Sblock, Srows,
                             (plan.b_row_k_prefix[k, j] + 1):(plan.b_row_k_prefix[k, j] + rB))
                task = GroupedGemmTask('N', 'T', one(T),
                                       Sview,
                                       compressed_ftlr_execution_inner(B, k, j), zero(T),
                                       view(tstack, (rho_before_k + 1):(rho_before_k + rA),
                                            (plan.output_col_prefix[j] + 1):plan.output_col_prefix[j + 1]))
                if s2 === nothing; s2 = GroupedGemmTask[task]; else; push!(s2, task); end
            end
        end
    end

    # Stage 3: one wide output GEMM per row, grouped across the run's rows.
    s3 = nothing
    @inbounds for (ii, i) in enumerate(irange)
        rho = plan.a_k_prefix[i, end]
        Crow = view(C, _compressed_ftlr_axis_range(A, i, 1), :)
        if rho == 0
            push!(scale_targets, (_compressed_ftlr_axis_range(A, i, 1), 1:size(C, 2)))
            continue
        end
        task = GroupedGemmTask('N', 'N', alpha, _compressed_ftlr_row_outer_stack(A, i, rho),
                               _ragged_view(tdata, tbase[ii], rho, plan.output_col_prefix[end]), beta, Crow)
        if s3 === nothing; s3 = GroupedGemmTask[task]; else; push!(s3, task); end
    end
    return CompressedFTLRRunTasks(
        s1, s2, s3, tdata,
        _compressed_ftlr_foldright_needs_tdata_zero(plan, rho_k), scale_targets)
end

"""
FoldLeft companion: `T^L_ikj = U_ik S_ikj`, then
`C_ij += [T^L_i1j ... T^L_iqkj]·[Z_1j';...;Z_qkj']` — concatenating over `k` on
the right. For every output column `j`, the `T` arena stacks all rows in the
run vertically into one `run_height × gamma_j` matrix, letting Stage 3 share
`Z_j` across the run and issue one GEMM per `j` rather than one per `(i,j)`.
"""
function _build_compressed_ftlr_foldleft_run(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                     alpha, beta, arena)
    T = eltype(A)
    _, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    nr = length(irange)
    scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
    sizehint!(scale_targets, qn)

    rho_k, row_off, koff, s_total = _compressed_ftlr_stage1_layout(A, irange, plan)
    i0, i1 = first(irange), last(irange)
    run_row_prefix = Base.zeros(Int, nr + 1)
    @inbounds for (ii, i) in enumerate(irange)
        run_row_prefix[ii + 1] = run_row_prefix[ii] + plan.output_row_heights[i]
    end
    run_height = run_row_prefix[end]
    tbase_cols = Base.zeros(Int, qn + 1)
    tbase_cols[1] = 1
    @inbounds for j in 1:qn
        tbase_cols[j + 1] = tbase_cols[j] + run_height * plan.b_col_ranks[j]
    end
    t_total = tbase_cols[end] - 1
    if s_total == 0
        @inbounds for i in irange
            push!(scale_targets, (_compressed_ftlr_axis_range(A, i, 1), 1:size(C, 2)))
        end
        return CompressedFTLRRunTasks(
            nothing, nothing, nothing, nothing, false, scale_targets)
    end
    _arena_reset!(arena)
    backend = get_backend(A)
    sdata = _workspace_array!(arena, backend, T, s_total)
    tdata = _workspace_array!(arena, backend, T, t_total)

    s1 = _compressed_ftlr_stage1_tasks(A, B, irange, plan, rho_k, koff, sdata)

    # Same nesting rationale as FoldRight's Stage 2: `Sblock` depends on (i,k)
    # and `Tj` on j, so neither is rebuilt in the innermost loop.
    s2 = nothing
    @inbounds for (ii, i) in enumerate(irange)
        Trows = (run_row_prefix[ii] + 1):run_row_prefix[ii + 1]
        for k in 1:qk
            rA = _compressed_ftlr_execution_rank(A, i, k)
            rA == 0 && continue
            Sblock = _compressed_ftlr_sblock(sdata, koff, rho_k, plan, k)
            Srows = (row_off[ii, k] + 1):(row_off[ii, k] + rA)
            Uik = compressed_ftlr_execution_outer(A, i, k)
            for j in 1:qn
                rB = _compressed_ftlr_execution_rank(B, k, j)
                rB == 0 && continue
                Sview = view(Sblock, Srows,
                             (plan.b_row_k_prefix[k, j] + 1):(plan.b_row_k_prefix[k, j] + rB))
                Tj = _ragged_view(tdata, tbase_cols[j], run_height, plan.b_col_ranks[j])
                task = GroupedGemmTask('N', 'N', one(T), Uik, Sview, zero(T),
                                       view(Tj, Trows,
                                            (plan.b_col_k_prefix[j, k] + 1):
                                                (plan.b_col_k_prefix[j, k] + rB)))
                if s2 === nothing; s2 = GroupedGemmTask[task]; else; push!(s2, task); end
            end
        end
    end

    s3 = nothing
    output_rows = (first(_compressed_ftlr_axis_range(A, i0, 1)):
                   last(_compressed_ftlr_axis_range(A, i1, 1)))
    @inbounds for j in 1:qn
        gamma = plan.b_col_ranks[j]
        output_cols = _compressed_ftlr_axis_range(B, j, 2)
        if gamma == 0
            push!(scale_targets, (output_rows, output_cols))
            continue
        end
        task = GroupedGemmTask('N', 'T', alpha,
                               _ragged_view(tdata, tbase_cols[j], run_height, gamma),
                               _compressed_ftlr_col_z_stack(B, j, gamma), beta,
                               view(C, output_rows, output_cols))
        if s3 === nothing; s3 = GroupedGemmTask[task]; else; push!(s3, task); end
    end
    return CompressedFTLRRunTasks(
        s1, s2, s3, tdata,
        _compressed_ftlr_foldleft_needs_tdata_zero(A, plan, irange), scale_targets)
end

function _execute_compressed_ftlr_run_tasks!(C, A, tasks::CompressedFTLRRunTasks, beta, mode)
    tasks.needs_tdata_zero && fill!(tasks.tdata, zero(eltype(A)))
    @inbounds for (rows, cols) in tasks.scale_targets
        _scale_output!(view(C, rows, cols), beta)
    end
    tasks.stage1 === nothing || precision_gemm_grouped!(tasks.stage1, mode)
    tasks.stage2 === nothing || precision_gemm_grouped!(tasks.stage2, mode)
    tasks.stage3 === nothing || precision_gemm_grouped!(tasks.stage3, mode)
    return C
end

function _execute_compressed_ftlr_foldright_run!(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                                  alpha, beta, mode, arena)
    tasks = _build_compressed_ftlr_foldright_run(C, A, B, plan, irange, alpha, beta, arena)
    return _execute_compressed_ftlr_run_tasks!(C, A, tasks, beta, mode)
end

function _execute_compressed_ftlr_foldleft_run!(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                                 alpha, beta, mode, arena)
    tasks = _build_compressed_ftlr_foldleft_run(C, A, B, plan, irange, alpha, beta, arena)
    return _execute_compressed_ftlr_run_tasks!(C, A, tasks, beta, mode)
end
