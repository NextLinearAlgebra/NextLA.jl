@inline function compressed_ftlr_row_outer_stack(A, i::Int, rho::Int)
    qm, qk = grid_size(A)
    1 <= i <= qm || throw(BoundsError(A, (i, :)))

    f = compressed_ftlr_outer_storage(A)
    fi, fj = compressed_ftlr_logical_coords(A, i, 1)
    li, lj = compressed_ftlr_logical_coords(A, i, qk)
    first_slot = tile_linear_index(f.order, f.qm, f.qn, fi, fj)
    last_slot = tile_linear_index(f.order, f.qm, f.qn, li, lj)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(tile_axis_range(A, i, 1))

    rho == 0 && return reshape(view(f.data, 1:0), rows, 0)
    packed = reshape(view(f.data, first:last), length(first:last) ÷ rho, rho)
    return view(packed, 1:rows, :)
end

"""
Zero-copy horizontal stack of B's row-packed outer factors for fixed `k` and
columns `j0:j1`, preserving fused Stage 1 for column-restricted runs.
"""
@inline function _compressed_ftlr_row_w_stack(B, k::Int, j0::Int, j1::Int, rho::Int)
    _, qn = grid_size(B)
    (1 <= j0 <= j1 <= qn) || throw(BoundsError(B, (k, j0:j1)))

    f = compressed_ftlr_outer_storage(B)
    fi, fj = compressed_ftlr_logical_coords(B, k, j0)
    li, lj = compressed_ftlr_logical_coords(B, k, j1)
    first_slot = tile_linear_index(f.order, f.qm, f.qn, fi, fj)
    last_slot = tile_linear_index(f.order, f.qm, f.qn, li, lj)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(tile_axis_range(B, k, 1))

    rho == 0 && return reshape(view(f.data, 1:0), rows, 0)
    packed = reshape(view(f.data, first:last), length(first:last) ÷ rho, rho)
    return view(packed, 1:rows, :)
end

"""
Zero-copy vertical stack of A's column-packed inner factors for fixed `k` and
rows `i0:i1`. It fuses Stage 1 to one task per `k`; zero-rank rows occupy no
bytes and vanish from the span.
"""
@inline function _compressed_ftlr_run_v_stack(A, i0::Int, i1::Int, k::Int, rho::Int)
    _, qk = grid_size(A)
    1 <= k <= qk || throw(BoundsError(A, (:, k)))

    f = compressed_ftlr_inner_storage(A)
    fi0, fk0 = compressed_ftlr_logical_coords(A, i0, k)
    fi1, fk1 = compressed_ftlr_logical_coords(A, i1, k)
    first_slot = tile_linear_index(f.order, f.qm, f.qn, fi0, fk0)
    last_slot = tile_linear_index(f.order, f.qm, f.qn, fi1, fk1)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(tile_axis_range(A, k, 2))

    rho == 0 && return reshape(view(f.data, 1:0), rows, 0)
    packed = reshape(view(f.data, first:last), length(first:last) ÷ rho, rho)
    return view(packed, 1:rows, :)
end

@inline function compressed_ftlr_col_z_stack(B, j::Int, gamma::Int)
    qk, qn = grid_size(B)
    1 <= j <= qn || throw(BoundsError(B, (:, j)))

    f = compressed_ftlr_inner_storage(B)
    fi, fj = compressed_ftlr_logical_coords(B, 1, j)
    li, lj = compressed_ftlr_logical_coords(B, qk, j)
    first_slot = tile_linear_index(f.order, f.qm, f.qn, fi, fj)
    last_slot = tile_linear_index(f.order, f.qm, f.qn, li, lj)
    first = f.offsets[first_slot]
    last = f.offsets[last_slot + 1] - 1
    rows = length(tile_axis_range(B, j, 2))

    gamma == 0 && return reshape(view(f.data, 1:0), rows, 0)
    packed = reshape(view(f.data, first:last), length(first:last) ÷ gamma, gamma)
    return view(packed, 1:rows, :)
end

@inline function _ragged_view(data, offset::Int, rows::Int, cols::Int)
    cols == 0 && return reshape(view(data, 1:0), rows, 0)
    return reshape(view(data, offset:(offset + rows * cols - 1)), rows, cols)
end

# Stage 2 skips zero-rank pairs, but each fold sizes its `T` arena from only one
# operand. Opposite-side rank holes therefore leave reserved entries unwritten:
# FoldRight when some `rA_ik > 0` meets an `rB_kj == 0`, and FoldLeft for the
# converse. `rho_k` and B prefix tables detect these holes in O(qk), allowing
# the `T` clear to be skipped for all-positive ranks.
"""
Stage-1 layout with one fused `(rho_k × σ_k)` block per active `k`.
`row_off[ii,k]` locates a row contribution and `koff[k]` locates the block.
"""
function _compressed_ftlr_stage1_layout(A, irange, jrange, plan)
    _, qk = grid_size(A)
    nr = length(irange)
    rho_k = Base.zeros(Int, qk)
    row_off = Base.zeros(Int, nr, qk)
    koff = Base.zeros(Int, qk + 1)
    koff[1] = 1

    # contraction-row blocks
    @inbounds for k in 1:qk
        acc = 0
        for (ii, i) in enumerate(irange)
            row_off[ii, k] = acc
            acc += compressed_ftlr_storage_rank(A, i, k)
        end
        rho_k[k] = acc
        koff[k + 1] = koff[k] + acc * compressed_ftlr_row_rank(plan, k, jrange)
    end

    return rho_k, row_off, koff, koff[end] - 1
end

"""One grouped Stage-1 task per active `k`, with zero-copy row and column stacks."""
function _compressed_ftlr_stage1_tasks(A, B, irange, jrange, plan,
                                       rho_k, koff, sdata)
    T = eltype(A)
    i0, i1 = first(irange), last(irange)
    j0, j1 = first(jrange), last(jrange)
    _, qk = grid_size(A)
    s1 = GroupedGemmTask[]

    # active contraction rows
    @inbounds for k in 1:qk
        rBsum = compressed_ftlr_row_rank(plan, k, jrange)
        (rho_k[k] == 0 || rBsum == 0) && continue
        task = GroupedGemmTask('T', 'N', one(T),
                               _compressed_ftlr_run_v_stack(A, i0, i1, k, rho_k[k]),
                               _compressed_ftlr_row_w_stack(B, k, j0, j1, rBsum), zero(T),
                               _ragged_view(sdata, koff[k], rho_k[k], rBsum))
        push!(s1, task)
    end

    return isempty(s1) ? nothing : s1
end

"""`k`'s fused dense Stage-1 block, shaped `(rho_k × σ_k)`. Individual `S_ikj`
are strided views into it: rows `row_off[ii,k] .+ (1:rA)`, columns
`b_row_k_prefix[k,j] .+ (1:rB)`."""
@inline _compressed_ftlr_sblock(sdata, koff, rho_k, plan, k, jrange) =
    reshape(view(sdata, koff[k]:(koff[k + 1] - 1)),
            rho_k[k], compressed_ftlr_row_rank(plan, k, jrange))

"""
Execute one FoldRight CompressedFTLR row run: `T^R_ikj = S_ikj Z_kj'`, then
`C_i += [U_i1 ... U_iqk]·[T^R_i1j;...;T^R_iqkj]`, concatenating over `k` on the
left. `T` is packed `(i,j,k)`-order so each row is already the Stage-3
`rho_i × (qn*bn)` stack, letting Stage 3 issue one wide GEMM per row.
"""
function build_compressed_ftlr_foldright_run(C, A, B, plan,
                                      irange, jrange, alpha, beta, arena)
    T = eltype(A)
    _, qk = grid_size(A)
    qkB, _ = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))

    nr = length(irange)
    scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
    sizehint!(scale_targets, nr)
    width = compressed_ftlr_width(plan, jrange)
    output_cols = compressed_ftlr_output_cols(B, jrange)

    # stage-1 layout and terminal row offsets
    rho_k, row_off, koff, s_total = _compressed_ftlr_stage1_layout(A, irange, jrange, plan)

    # column-major terminal rows with strided `(k,j)` pieces
    tbase = Base.zeros(Int, nr + 1)
    tbase[1] = 1
    @inbounds for (ii, i) in enumerate(irange)
        tbase[ii + 1] = tbase[ii] + plan.a_k_prefix[i, end] * width
    end
    t_total = tbase[end] - 1

    # inactive run
    if s_total == 0
        @inbounds for i in irange
            push!(scale_targets, (tile_axis_range(A, i, 1), output_cols))
        end
        return DenseAccumulationRunTasks(
            nothing, nothing, nothing, 0, nothing, false, scale_targets)
    end

    arena_reset!(arena)
    backend = get_backend(A)
    sdata = workspace_array!(arena, backend, T, s_total)
    tdata = workspace_array!(arena, backend, T, t_total)

    s1 = _compressed_ftlr_stage1_tasks(A, B, irange, jrange, plan, rho_k, koff, sdata)

    # stage-2 tasks with loop-local `tstack` and `Sblock` views
    s2 = GroupedGemmTask[]
    @inbounds for (ii, i) in enumerate(irange)
        rho = plan.a_k_prefix[i, end]
        rho == 0 && continue
        tstack = _ragged_view(tdata, tbase[ii], rho, width)
        for k in 1:qk
            rA = compressed_ftlr_storage_rank(A, i, k)
            rA == 0 && continue
            rho_before_k = plan.a_k_prefix[i, k]
            Sblock = _compressed_ftlr_sblock(sdata, koff, rho_k, plan, k, jrange)
            Srows = (row_off[ii, k] + 1):(row_off[ii, k] + rA)
            for j in jrange
                rB = compressed_ftlr_storage_rank(B, k, j)
                rB == 0 && continue
                scol = compressed_ftlr_row_rank_offset(plan, k, j, jrange)
                tcol = plan.output_col_prefix[j] -
                       plan.output_col_prefix[first(jrange)]
                Sview = view(Sblock, Srows, (scol + 1):(scol + rB))
                task = GroupedGemmTask('N', 'T', one(T),
                                       Sview,
                                       compressed_ftlr_storage_inner(B, k, j), zero(T),
                                       view(tstack, (rho_before_k + 1):(rho_before_k + rA),
                                            (tcol + 1):(tcol + plan.output_col_widths[j])))
                push!(s2, task)
            end
        end
    end

    # stage-3 row tasks
    s3 = GroupedGemmTask[]
    @inbounds for (ii, i) in enumerate(irange)
        rho = plan.a_k_prefix[i, end]
        rows = tile_axis_range(A, i, 1)
        if rho == 0
            push!(scale_targets, (rows, output_cols))
            continue
        end
        task = GroupedGemmTask('N', 'N', alpha, compressed_ftlr_row_outer_stack(A, i, rho),
                               _ragged_view(tdata, tbase[ii], rho, width), beta,
                               view(C, rows, output_cols))
        push!(s3, task)
    end
    return DenseAccumulationRunTasks(
        s1, isempty(s2) ? nothing : s2, isempty(s3) ? nothing : s3, 3, tdata,
        any(k -> rho_k[k] > 0 &&
                 plan.b_row_nonzero_prefix[k, last(jrange) + 1] -
                 plan.b_row_nonzero_prefix[k, first(jrange)] < length(jrange),
            eachindex(rho_k)),
        scale_targets)
end

"""
FoldLeft companion: `T^L_ikj = U_ik S_ikj`, then
`C_ij += [T^L_i1j ... T^L_iqkj]·[Z_1j';...;Z_qkj']`, concatenating over `k` on
the right. For every output column `j`, the `T` arena stacks all rows in the
run vertically into one `run_height × gamma_j` matrix, letting Stage 3 share
`Z_j` across the run and issue one GEMM per `j` rather than one per `(i,j)`.
"""
function build_compressed_ftlr_foldleft_run(C, A, B, plan,
                                     irange, jrange, alpha, beta, arena)
    T = eltype(A)
    _, qk = grid_size(A)
    qkB, _ = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))

    nr = length(irange)
    j0 = first(jrange)
    scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
    sizehint!(scale_targets, length(jrange))

    # stage-1 layout and run geometry
    rho_k, row_off, koff, s_total = _compressed_ftlr_stage1_layout(A, irange, jrange, plan)
    i0, i1 = first(irange), last(irange)
    run_row_prefix = Base.zeros(Int, nr + 1)
    @inbounds for (ii, i) in enumerate(irange)
        run_row_prefix[ii + 1] = run_row_prefix[ii] + plan.output_row_heights[i]
    end
    run_height = run_row_prefix[end]

    # terminal column offsets rebased to `jrange`
    tbase_cols = Base.zeros(Int, length(jrange) + 1)
    tbase_cols[1] = 1
    @inbounds for (jj, j) in enumerate(jrange)
        tbase_cols[jj + 1] = tbase_cols[jj] + run_height * plan.b_col_ranks[j]
    end
    t_total = tbase_cols[end] - 1
    output_rows = (first(tile_axis_range(A, i0, 1)):
                   last(tile_axis_range(A, i1, 1)))

    # inactive run
    if s_total == 0
        @inbounds for i in irange
            push!(scale_targets, (tile_axis_range(A, i, 1),
                                  compressed_ftlr_output_cols(B, jrange)))
        end
        return DenseAccumulationRunTasks(
            nothing, nothing, nothing, 0, nothing, false, scale_targets)
    end
    arena_reset!(arena)
    backend = get_backend(A)
    sdata = workspace_array!(arena, backend, T, s_total)
    tdata = workspace_array!(arena, backend, T, t_total)

    s1 = _compressed_ftlr_stage1_tasks(A, B, irange, jrange, plan, rho_k, koff, sdata)

    # stage-2 tasks with loop-local `Sblock` and `Tj` views
    s2 = GroupedGemmTask[]
    @inbounds for (ii, i) in enumerate(irange)
        Trows = (run_row_prefix[ii] + 1):run_row_prefix[ii + 1]
        for k in 1:qk
            rA = compressed_ftlr_storage_rank(A, i, k)
            rA == 0 && continue
            Sblock = _compressed_ftlr_sblock(sdata, koff, rho_k, plan, k, jrange)
            Srows = (row_off[ii, k] + 1):(row_off[ii, k] + rA)
            Uik = compressed_ftlr_storage_outer(A, i, k)
            for j in jrange
                rB = compressed_ftlr_storage_rank(B, k, j)
                rB == 0 && continue
                scol = compressed_ftlr_row_rank_offset(plan, k, j, jrange)
                Sview = view(Sblock, Srows, (scol + 1):(scol + rB))
                Tj = _ragged_view(tdata, tbase_cols[j - j0 + 1], run_height,
                                  plan.b_col_ranks[j])
                task = GroupedGemmTask('N', 'N', one(T), Uik, Sview, zero(T),
                                       view(Tj, Trows,
                                            (plan.b_col_k_prefix[j, k] + 1):
                                                (plan.b_col_k_prefix[j, k] + rB)))
                push!(s2, task)
            end
        end
    end

    # stage-3 column tasks
    s3 = GroupedGemmTask[]
    @inbounds for j in jrange
        gamma = plan.b_col_ranks[j]
        output_cols = tile_axis_range(B, j, 2)
        if gamma == 0
            push!(scale_targets, (output_rows, output_cols))
            continue
        end
        task = GroupedGemmTask('N', 'T', alpha,
                               _ragged_view(tdata, tbase_cols[j - j0 + 1], run_height, gamma),
                               compressed_ftlr_col_z_stack(B, j, gamma), beta,
                               view(C, output_rows, output_cols))
        push!(s3, task)
    end
    return DenseAccumulationRunTasks(
        s1, isempty(s2) ? nothing : s2, isempty(s3) ? nothing : s3, 3, tdata,
        any(1:plan.qk) do k
            compressed_ftlr_row_rank(plan, k, jrange) > 0 &&
                any(i -> compressed_ftlr_storage_rank(A, i, k) == 0, irange)
        end,
        scale_targets)
end

# symbolic analysis: reusable prepared-run bundle for repeated numerical calls

"""
    CompressedGemmAnalysis

Explicit symbolic metadata for `CompressedFTLRMatrix × CompressedFTLRMatrix → dense`.
The object owns device pointer tables and is bound to the output, operands,
workspace, logical operations, compute policy, and rank metadata used to create it.
Factor values and numerical scalars may be changed between numerical calls.
"""
mutable struct CompressedGemmAnalysis{CT,AT,BT,WT,ModeT,RAT,RBT}
    C::CT
    A::AT
    B::BT
    workspace::WT
    transA::Char
    transB::Char
    compute::ModeT
    runs::Vector{PreparedDenseAccumulationRun}
    # copied same-type rank snapshots for allocation-free mutation guards
    A_ranks::RAT
    B_ranks::RBT
    workspace_bytes::Int
    has_fallback::Bool
    closed::Bool
end

Base.close(analysis::CompressedGemmAnalysis) = close_dense_accumulation_analysis!(analysis)

"""
    analyze_compressed_gemm(C, A, B; workspace, transA='N', transB='N', compute=nothing)

Build reusable symbolic metadata and prepared descriptors. `workspace` must be
a `DenseGemmWorkspace`.
"""
function analyze_compressed_gemm(
    C::AbstractMatrix,
    A::CompressedFTLRMatrix{BackendT,T},
    B::CompressedFTLRMatrix{BackendT,T};
    workspace,
    transA::Char='N',
    transB::Char='N',
    compute=nothing,
) where {BackendT,T}
    # binding and geometry
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "symbolic compressed GEMM analysis requires a reusable DenseGemmWorkspace"))
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_compressed_ftlr_gemm(C, LA, LB, mode)

    # plan and placeholder scalars
    plan = compressed_ftlr_rank_plan(LA, LB)
    ws, arena, budget, profile =
        prepare_compressed_ftlr_workspace(LA, LB, plan, workspace)
    ws === workspace || error("internal error: symbolic analysis replaced its workspace")
    scalar_type = gemm_compute_type(mode)
    placeholder_alpha = one(scalar_type)
    placeholder_beta = zero(scalar_type)

    # scheduled prepared runs
    schedule = compressed_ftlr_column_schedule(plan, LA, LB, profile, budget)
    prepared_runs = prepare_dense_accumulation_runs(schedule, mode) do run
        run.fold === :right ?
            build_compressed_ftlr_foldright_run(
                C, LA, LB, plan, run.rows, run.cols,
                placeholder_alpha, placeholder_beta, arena) :
            build_compressed_ftlr_foldleft_run(
                C, LA, LB, plan, run.rows, run.cols,
                placeholder_alpha, placeholder_beta, arena)
    end

    # bound analysis and cleanup
    analysis = CompressedGemmAnalysis(
        C, A, B, workspace, transA, transB, mode, prepared_runs,
        copy(ranks(A)), copy(ranks(B)),
        sizeof(workspace),
        dense_accumulation_runs_have_fallback(prepared_runs),
        false)
    finalizer(analysis) do object
        try
            close_dense_accumulation_analysis!(object)
        catch
            # Device teardown may precede Julia object finalization.
        end
    end
    return analysis
end

function execute_compressed_gemm_analysis!(
    analysis::CompressedGemmAnalysis, C, A, B, workspace,
    alpha, beta, transA, transB, mode)
    validate_dense_accumulation_analysis_binding(
        analysis, C, A, B, workspace, transA, transB, mode)

    # rank snapshot guards
    ranks(A) == analysis.A_ranks ||
        throw(ArgumentError("left operand exact ranks changed after symbolic analysis"))
    ranks(B) == analysis.B_ranks ||
        throw(ArgumentError("right operand exact ranks changed after symbolic analysis"))

    backend = get_backend(A)
    return execute_prepared_dense_accumulation_runs!(
        analysis.runs, C, backend, eltype(A), alpha, beta,
        analysis.has_fallback)
end
