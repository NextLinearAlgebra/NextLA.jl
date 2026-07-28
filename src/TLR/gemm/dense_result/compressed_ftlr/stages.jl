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

@inline function _ragged_view(data, offset::Int, rows::Int, cols::Int)
    cols == 0 && return reshape(view(data, 1:0), rows, 0)
    return reshape(view(data, offset:(offset + rows * cols - 1)), rows, cols)
end

"""
Execute one FoldRight CompressedFTLR row run. `S` is packed in `(i,k,j)` order for the
Stage-1 W-panel fusion; `T` is packed in `(i,j,k)` order so each row is already
the Stage-3 `rho_i × (qn*bn)` stack.
"""
function _execute_compressed_ftlr_foldright_run!(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                       alpha, beta, mode, arena)
    T = eltype(A)
    qm, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    nr = length(irange)

    # Host offsets into one run-local S/T arena.
    soff = Base.zeros(Int, nr, qk, qn)
    s_total = 0
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk, j in 1:qn
        rA = _compressed_ftlr_rank(A, i, k); rB = _compressed_ftlr_rank(B, k, j)
        rA == 0 || rB == 0 || begin
            soff[ii, k, j] = s_total + 1
            s_total += rA * rB
        end
    end
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
            _scale_output!(view(C, _compressed_ftlr_axis_range(A, i, 1), :), beta)
        end
        return C
    end

    _arena_reset!(arena)
    backend = get_backend(A)
    sdata = _workspace_array!(arena, backend, T, s_total)
    tdata = _workspace_array!(arena, backend, T, t_total)
    fill!(tdata, zero(T))

    # Stage 1: one W-row panel per (i,k), so j is folded into N.
    s1 = nothing
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk
        rA = _compressed_ftlr_rank(A, i, k)
        rBsum = plan.b_row_ranks[k]
        (rA == 0 || rBsum == 0) && continue
        firstj = plan.b_first_active_col[k]
        task = GroupedGemmTask('T', 'N', one(T), compressed_ftlr_inner(A, i, k),
                               _compressed_ftlr_row_w_stack(B, k, rBsum), zero(T),
                               _ragged_view(sdata, soff[ii, k, firstj], rA, rBsum))
        if s1 === nothing
            s1 = typeof(task)[task]
        else
            push!(s1, task)
        end
    end
    s1 === nothing || precision_gemm_grouped!(s1, mode)

    # Stage 2: each S_ikj is multiplied by Z_kj'.
    s2 = nothing
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk, j in 1:qn
        rA = _compressed_ftlr_rank(A, i, k); rB = _compressed_ftlr_rank(B, k, j)
        (rA == 0 || rB == 0) && continue
        rho_before_k = plan.a_k_prefix[i, k]
        rho = plan.a_k_prefix[i, end]
        tstack = _ragged_view(tdata, tbase[ii], rho, plan.output_col_prefix[end])
        task = GroupedGemmTask('N', 'T', one(T),
                               _ragged_view(sdata, soff[ii, k, j], rA, rB),
                               compressed_ftlr_inner(B, k, j), zero(T),
                               view(tstack, (rho_before_k + 1):(rho_before_k + rA),
                                    (plan.output_col_prefix[j] + 1):plan.output_col_prefix[j + 1]))
        if s2 === nothing
            s2 = typeof(task)[task]
        else
            push!(s2, task)
        end
    end
    s2 === nothing || precision_gemm_grouped!(s2, mode)

    # Stage 3: one wide output GEMM per row, grouped across the run's rows.
    s3 = nothing
    @inbounds for (ii, i) in enumerate(irange)
        rho = plan.a_k_prefix[i, end]
        Crow = view(C, _compressed_ftlr_axis_range(A, i, 1), :)
        if rho == 0
            _scale_output!(Crow, beta)
            continue
        end
        task = GroupedGemmTask('N', 'N', alpha, _compressed_ftlr_row_outer_stack(A, i, rho),
                               _ragged_view(tdata, tbase[ii], rho, plan.output_col_prefix[end]), beta, Crow)
        if s3 === nothing
            s3 = typeof(task)[task]
        else
            push!(s3, task)
        end
    end
    s3 === nothing || precision_gemm_grouped!(s3, mode)
    return C
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
function _execute_compressed_ftlr_foldleft_run!(C, A, B, plan::CompressedFTLRRankPlan, irange,
                                      alpha, beta, mode, arena)
    T = eltype(A)
    _, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    nr = length(irange)
    soff = Base.zeros(Int, nr, qk, qn)
    s_total = 0
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk, j in 1:qn
        rA = _compressed_ftlr_rank(A, i, k); rB = _compressed_ftlr_rank(B, k, j)
        rA == 0 || rB == 0 || begin
            soff[ii, k, j] = s_total + 1
            s_total += rA * rB
        end
    end
    tbase_rows = Base.zeros(Int, nr + 1); tbase_rows[1] = 1
    @inbounds for (ii, i) in enumerate(irange)
        tbase_rows[ii + 1] = tbase_rows[ii] + plan.output_row_heights[i] * plan.b_total_rank
    end
    t_total = tbase_rows[end] - 1
    if s_total == 0
        @inbounds for i in irange
            _scale_output!(view(C, _compressed_ftlr_axis_range(A, i, 1), :), beta)
        end
        return C
    end
    _arena_reset!(arena)
    backend = get_backend(A)
    sdata = _workspace_array!(arena, backend, T, s_total)
    tdata = _workspace_array!(arena, backend, T, t_total)
    fill!(tdata, zero(T))

    s1 = nothing
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk
        rA = _compressed_ftlr_rank(A, i, k)
        rBsum = plan.b_row_ranks[k]
        (rA == 0 || rBsum == 0) && continue
        firstj = plan.b_first_active_col[k]
        task = GroupedGemmTask('T', 'N', one(T), compressed_ftlr_inner(A, i, k),
                               _compressed_ftlr_row_w_stack(B, k, rBsum), zero(T),
                               _ragged_view(sdata, soff[ii, k, firstj], rA, rBsum))
        if s1 === nothing; s1 = typeof(task)[task]; else; push!(s1, task); end
    end
    s1 === nothing || precision_gemm_grouped!(s1, mode)

    s2 = nothing
    @inbounds for (ii, i) in enumerate(irange), k in 1:qk, j in 1:qn
        rA = _compressed_ftlr_rank(A, i, k); rB = _compressed_ftlr_rank(B, k, j)
        (rA == 0 || rB == 0) && continue
        bm = plan.output_row_heights[i]
        tbase = tbase_rows[ii] + bm * plan.b_col_prefix[j]
        toff = tbase + bm * plan.b_col_k_prefix[j, k]
        task = GroupedGemmTask('N', 'N', one(T), compressed_ftlr_outer(A, i, k),
                               _ragged_view(sdata, soff[ii, k, j], rA, rB), zero(T),
                               _ragged_view(tdata, toff, bm, rB))
        if s2 === nothing; s2 = typeof(task)[task]; else; push!(s2, task); end
    end
    s2 === nothing || precision_gemm_grouped!(s2, mode)

    s3 = nothing
    @inbounds for (ii, i) in enumerate(irange), j in 1:qn
        gamma = plan.b_col_ranks[j]
        bm = plan.output_row_heights[i]
        Cij = view(C, _compressed_ftlr_axis_range(A, i, 1), _compressed_ftlr_axis_range(B, j, 2))
        if gamma == 0
            _scale_output!(Cij, beta)
            continue
        end
        tbase = tbase_rows[ii] + bm * plan.b_col_prefix[j]
        task = GroupedGemmTask('N', 'T', alpha,
                               _ragged_view(tdata, tbase, bm, gamma),
                               _compressed_ftlr_col_z_stack(B, j, gamma), beta, Cij)
        if s3 === nothing; s3 = typeof(task)[task]; else; push!(s3, task); end
    end
    s3 === nothing || precision_gemm_grouped!(s3, mode)
    return C
end
