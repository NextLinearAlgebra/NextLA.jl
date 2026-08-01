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
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

"""
Execute one FoldRight CompressedFTLR row run. `S` is packed in `(i,k,j)` order for the
Stage-1 W-panel fusion; `T` is packed in `(i,j,k)` order so each row is already
the Stage-3 `rho_i × (qn*bn)` stack.
"""
function _build_compressed_ftlr_foldright_run(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                      alpha, beta, arena)
    T = eltype(A)
    qm, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    nr = length(irange)
    scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
    sizehint!(scale_targets, nr)

    # Host offsets into one run-local S/T arena. S is now ONE dense (rho_k x
    # rBsum_k) block per active k, fused across every row in the run (Stage 1
    # becomes one GEMM per k instead of one per (i,k)); row_off[ii,k] is where
    # row ii's contribution starts within that block, and
    # plan.b_row_k_prefix[k,j] (plan-level, independent of the run since B's
    # structure doesn't depend on i) gives the column start for j.
    i0, i1 = first(irange), last(irange)
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
    s_total = koff[end] - 1
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
        return CompressedFTLRRunTasks(nothing, nothing, nothing, nothing, scale_targets)
    end

    _arena_reset!(arena)
    backend = get_backend(A)
    sdata = _workspace_array!(arena, backend, T, s_total)
    tdata = _workspace_array!(arena, backend, T, t_total)

    # Stage 1: ONE task per active k, fusing every row in the run into one
    # GEMM (V's column-packing makes the row-range concatenation zero-copy;
    # see _compressed_ftlr_run_v_stack). Was one task per (i,k); this cuts
    # Stage-1 task count from (run length)*qk down to qk.
    s1 = nothing
    @inbounds for k in 1:qk
        rBsum = plan.b_row_ranks[k]
        (rho_k[k] == 0 || rBsum == 0) && continue
        task = GroupedGemmTask('T', 'N', one(T),
                               _compressed_ftlr_run_v_stack(A, i0, i1, k, rho_k[k]),
                               _compressed_ftlr_row_w_stack(B, k, rBsum), zero(T),
                               _ragged_view(sdata, koff[k], rho_k[k], rBsum))
        if s1 === nothing
            s1 = GroupedGemmTask[task]
        else
            push!(s1, task)
        end
    end

    # Stage 2: each S_ikj is multiplied by Z_kj'. S_ikj is now a strided view
    # into k's fused dense (rho_k x rBsum_k) block instead of its own ragged
    # offset -- the arithmetic is unchanged, only how the view is built.
    s2 = nothing
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk, j in 1:qn
        rA = _compressed_ftlr_execution_rank(A, i, k); rB = _compressed_ftlr_execution_rank(B, k, j)
        (rA == 0 || rB == 0) && continue
        rho_before_k = plan.a_k_prefix[i, k]
        rho = plan.a_k_prefix[i, end]
        tstack = _ragged_view(tdata, tbase[ii], rho, plan.output_col_prefix[end])
        Sblock = reshape(view(sdata, koff[k]:(koff[k + 1] - 1)), rho_k[k], plan.b_row_ranks[k])
        Sview = view(Sblock, (row_off[ii, k] + 1):(row_off[ii, k] + rA),
                     (plan.b_row_k_prefix[k, j] + 1):(plan.b_row_k_prefix[k, j] + rB))
        task = GroupedGemmTask('N', 'T', one(T),
                               Sview,
                               compressed_ftlr_execution_inner(B, k, j), zero(T),
                               view(tstack, (rho_before_k + 1):(rho_before_k + rA),
                                    (plan.output_col_prefix[j] + 1):plan.output_col_prefix[j + 1]))
        if s2 === nothing
            s2 = GroupedGemmTask[task]
        else
            push!(s2, task)
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
        if s3 === nothing
            s3 = GroupedGemmTask[task]
        else
            push!(s3, task)
        end
    end
    return CompressedFTLRRunTasks(s1, s2, s3, tdata, scale_targets)
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

"""FoldLeft companion: the T' arena is packed directly into each `(i,j)` K-stack."""
function _build_compressed_ftlr_foldleft_run(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                     alpha, beta, arena)
    T = eltype(A)
    _, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    nr = length(irange)
    scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
    sizehint!(scale_targets, nr * qn)
    # Same fused Stage-1 arena as FoldRight -- see the comment there.
    i0, i1 = first(irange), last(irange)
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
    s_total = koff[end] - 1
    tbase_rows = Base.zeros(Int, nr + 1); tbase_rows[1] = 1
    @inbounds for (ii, i) in enumerate(irange)
        tbase_rows[ii + 1] = tbase_rows[ii] + plan.output_row_heights[i] * plan.b_total_rank
    end
    t_total = tbase_rows[end] - 1
    if s_total == 0
        @inbounds for i in irange
            push!(scale_targets, (_compressed_ftlr_axis_range(A, i, 1), 1:size(C, 2)))
        end
        return CompressedFTLRRunTasks(nothing, nothing, nothing, nothing, scale_targets)
    end
    _arena_reset!(arena)
    backend = get_backend(A)
    sdata = _workspace_array!(arena, backend, T, s_total)
    tdata = _workspace_array!(arena, backend, T, t_total)

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

    s2 = nothing
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk, j in 1:qn
        rA = _compressed_ftlr_execution_rank(A, i, k); rB = _compressed_ftlr_execution_rank(B, k, j)
        (rA == 0 || rB == 0) && continue
        bm = plan.output_row_heights[i]
        tbase = tbase_rows[ii] + bm * plan.b_col_prefix[j]
        toff = tbase + bm * plan.b_col_k_prefix[j, k]
        Sblock = reshape(view(sdata, koff[k]:(koff[k + 1] - 1)), rho_k[k], plan.b_row_ranks[k])
        Sview = view(Sblock, (row_off[ii, k] + 1):(row_off[ii, k] + rA),
                     (plan.b_row_k_prefix[k, j] + 1):(plan.b_row_k_prefix[k, j] + rB))
        task = GroupedGemmTask('N', 'N', one(T), compressed_ftlr_execution_outer(A, i, k),
                               Sview, zero(T),
                               _ragged_view(tdata, toff, bm, rB))
        if s2 === nothing; s2 = GroupedGemmTask[task]; else; push!(s2, task); end
    end

    s3 = nothing
    @inbounds for (ii, i) in enumerate(irange), j in 1:qn
        gamma = plan.b_col_ranks[j]
        bm = plan.output_row_heights[i]
        Cij = view(C, _compressed_ftlr_axis_range(A, i, 1), _compressed_ftlr_axis_range(B, j, 2))
        if gamma == 0
            push!(scale_targets, (_compressed_ftlr_axis_range(A, i, 1),
                                  _compressed_ftlr_axis_range(B, j, 2)))
            continue
        end
        tbase = tbase_rows[ii] + bm * plan.b_col_prefix[j]
        task = GroupedGemmTask('N', 'T', alpha,
                               _ragged_view(tdata, tbase, bm, gamma),
                               _compressed_ftlr_col_z_stack(B, j, gamma), beta, Cij)
        if s3 === nothing; s3 = GroupedGemmTask[task]; else; push!(s3, task); end
    end
    return CompressedFTLRRunTasks(s1, s2, s3, tdata, scale_targets)
end
