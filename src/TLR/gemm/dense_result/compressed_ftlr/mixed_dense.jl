"""Two-stage grouped lowering for dense × CompressedFTLR and its mirror."""

struct PreparedCompressedMixedRun
    stage1::Union{Nothing,AbstractPreparedGroupedGemm}
    stage2::Union{Nothing,AbstractPreparedGroupedGemm}
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

mutable struct CompressedMixedGemmAnalysis{CT,AT,BT,WT,ModeT}
    C::CT
    A::AT
    B::BT
    workspace::WT
    transA::Char
    transB::Char
    compute::ModeT
    side::Symbol
    ranks::Vector{Int}
    execution_ranks::Vector{Int}
    workspace_bytes::Int
    runs::Vector{PreparedCompressedMixedRun}
    closed::Bool
end

@inline function _mixed_grouped_tasks_push(tasks, task)
    tasks === nothing && return typeof(task)[task]
    push!(tasks, task)
    return tasks
end

@inline function _validate_compressed_mixed_grouped_precision(A)
    eltype(A) === Core.BFloat16 &&
        !supports_bfloat16_grouped_gemm(get_backend(A)) &&
        throw(ArgumentError(
            "CompressedFTLR BF16 grouped GEMMEx requires an NVIDIA SM80 or newer device"))
    return nothing
end

function _compressed_col_rank_plan(B)
    qk, qn = grid_size(B)
    prefix = Base.zeros(Int, qn, qk + 1)
    @inbounds for j in 1:qn, k in 1:qk
        prefix[j, k + 1] =
            prefix[j, k] + _compressed_ftlr_execution_rank(B, k, j)
    end
    totals = [prefix[j, end] for j in 1:qn]
    bases = Base.zeros(Int, qn + 1)
    @inbounds for j in 1:qn
        bases[j + 1] = bases[j] + totals[j]
    end
    return prefix, totals, bases
end

function _compressed_row_rank_plan(A)
    qm, qk = grid_size(A)
    prefix = Base.zeros(Int, qm, qk + 1)
    @inbounds for i in 1:qm, k in 1:qk
        prefix[i, k + 1] =
            prefix[i, k] + _compressed_ftlr_execution_rank(A, i, k)
    end
    totals = [prefix[i, end] for i in 1:qm]
    bases = Base.zeros(Int, qm + 1)
    @inbounds for i in 1:qm
        bases[i + 1] = bases[i] + totals[i]
    end
    return prefix, totals, bases
end

function _prepare_compressed_mixed_run(stage1, stage2, scale_targets, mode)
    prepared1 = prepared2 = nothing
    try
        prepared1 = stage1 === nothing ? nothing :
            prepare_precision_gemm_grouped(stage1, mode)
        prepared2 = stage2 === nothing ? nothing :
            prepare_precision_gemm_grouped(stage2, mode)
    catch
        prepared1 === nothing || destroy_prepared_grouped_gemm!(prepared1)
        prepared2 === nothing || destroy_prepared_grouped_gemm!(prepared2)
        rethrow()
    end
    return PreparedCompressedMixedRun(prepared1, prepared2, scale_targets)
end

function _destroy_compressed_mixed_analysis!(analysis::CompressedMixedGemmAnalysis)
    analysis.closed && return analysis
    analysis.closed = true
    for run in analysis.runs
        run.stage1 === nothing || destroy_prepared_grouped_gemm!(run.stage1)
        run.stage2 === nothing || destroy_prepared_grouped_gemm!(run.stage2)
    end
    return analysis
end

Base.close(analysis::CompressedMixedGemmAnalysis) =
    _destroy_compressed_mixed_analysis!(analysis)

function _dense_compressed_prepared_runs(C, A, B, budget, mode, arena)
    prefix, totals, bases = _compressed_col_rank_plan(B)
    total_rank = bases[end]
    total_rank == 0 && return PreparedCompressedMixedRun[
        PreparedCompressedMixedRun(
            nothing, nothing, [(1:size(C, 1), 1:size(C, 2))])]
    T = eltype(B)
    budget_elements = fld(budget, sizeof(T))
    budget_elements >= total_rank || throw(ArgumentError(
        "dense × compressed symbolic analysis requires at least " *
        "$(total_rank * sizeof(T)) workspace bytes for one fused row"))
    m = size(A, 1)
    height = clamp(fld(budget_elements, total_rank), 1, m)
    _arena_reset!(arena)
    work = _workspace_array!(arena, get_backend(B), T, height, total_rank)
    qk, qn = grid_size(B)
    runs = PreparedCompressedMixedRun[]
    for rows in Iterators.partition(1:m, height)
        h = length(rows)
        stage1 = nothing
        stage2 = nothing
        scales = Tuple{UnitRange{Int},UnitRange{Int}}[]
        @inbounds for j in 1:qn
            gamma = totals[j]
            cols = _tile_axis_range(B, j, 2)
            Cview = view(C, rows, cols)
            if gamma == 0
                push!(scales, (rows, cols))
                continue
            end
            Tj = view(work, 1:h, (bases[j] + 1):bases[j + 1])
            for k in 1:qk
                rk = _compressed_ftlr_execution_rank(B, k, j)
                rk == 0 && continue
                inner = _tile_axis_range(B, k, 1)
                Ad, opA = _dense_block(A, rows, inner)
                Tk = view(Tj, :, (prefix[j, k] + 1):prefix[j, k + 1])
                stage1 = _mixed_grouped_tasks_push(stage1, GroupedGemmTask(
                    opA, 'N', one(T), Ad,
                    compressed_ftlr_execution_outer(B, k, j), zero(T), Tk))
            end
            stage2 = _mixed_grouped_tasks_push(stage2, GroupedGemmTask(
                'N', 'T', one(T), Tj,
                _compressed_ftlr_col_z_stack(B, j, gamma), zero(T), Cview))
        end
        push!(runs, _prepare_compressed_mixed_run(stage1, stage2, scales, mode))
    end
    return runs
end

function _compressed_dense_prepared_runs(C, A, B, budget, mode, arena)
    prefix, totals, bases = _compressed_row_rank_plan(A)
    total_rank = bases[end]
    total_rank == 0 && return PreparedCompressedMixedRun[
        PreparedCompressedMixedRun(
            nothing, nothing, [(1:size(C, 1), 1:size(C, 2))])]
    T = eltype(A)
    budget_elements = fld(budget, sizeof(T))
    budget_elements >= total_rank || throw(ArgumentError(
        "compressed × dense symbolic analysis requires at least " *
        "$(total_rank * sizeof(T)) workspace bytes for one fused column"))
    n = size(B, 2)
    width = clamp(fld(budget_elements, total_rank), 1, n)
    _arena_reset!(arena)
    work = _workspace_array!(arena, get_backend(A), T, total_rank, width)
    qm, qk = grid_size(A)
    runs = PreparedCompressedMixedRun[]
    for cols in Iterators.partition(1:n, width)
        w = length(cols)
        stage1 = nothing
        stage2 = nothing
        scales = Tuple{UnitRange{Int},UnitRange{Int}}[]
        @inbounds for i in 1:qm
            rho = totals[i]
            rows = _tile_axis_range(A, i, 1)
            Cview = view(C, rows, cols)
            if rho == 0
                push!(scales, (rows, cols))
                continue
            end
            Ti = view(work, (bases[i] + 1):bases[i + 1], 1:w)
            for k in 1:qk
                rk = _compressed_ftlr_execution_rank(A, i, k)
                rk == 0 && continue
                inner = _tile_axis_range(A, k, 2)
                Bd, opB = _dense_block(B, inner, cols)
                Tk = view(Ti, (prefix[i, k] + 1):prefix[i, k + 1], :)
                stage1 = _mixed_grouped_tasks_push(stage1, GroupedGemmTask(
                    'T', opB, one(T),
                    compressed_ftlr_execution_inner(A, i, k), Bd, zero(T), Tk))
            end
            stage2 = _mixed_grouped_tasks_push(stage2, GroupedGemmTask(
                'N', 'N', one(T),
                _compressed_ftlr_row_outer_stack(A, i, rho), Ti, zero(T), Cview))
        end
        push!(runs, _prepare_compressed_mixed_run(stage1, stage2, scales, mode))
    end
    return runs
end

function _new_compressed_mixed_analysis(
    C, A, B, workspace, transA, transB, mode, side, compressed, runs)
    analysis = CompressedMixedGemmAnalysis(
        C, A, B, workspace, transA, transB, mode, side,
        Int.(ranks(compressed)), Int.(execution_ranks(compressed)),
        sizeof(workspace), runs, false)
    finalizer(analysis) do object
        try
            _destroy_compressed_mixed_analysis!(object)
        catch
        end
    end
    return analysis
end

function analyze_compressed_gemm(
    C::AbstractMatrix, A::AbstractMatrix{T},
    B::CompressedFTLRMatrix{BackendT,T};
    workspace, transA::Char='N', transB::Char='N', compute=nothing,
) where {BackendT,T}
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "symbolic mixed GEMM analysis requires a reusable DenseGemmWorkspace"))
    LA = logical_dense_operand(A, transA)
    LB = logical_operand(B, transB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C has the wrong dimensions"))
    backend = _validate_dense_backend(C, B, A)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    _validate_compressed_mixed_grouped_precision(LB)
    _, arena, budget = _prepare_single_gemm_workspace(B, workspace)
    runs = _dense_compressed_prepared_runs(C, LA, LB, budget, mode, arena)
    return _new_compressed_mixed_analysis(
        C, A, B, workspace, _normalize_tlr_op(transA), _normalize_tlr_op(transB),
        mode, :right, B, runs)
end

function analyze_compressed_gemm(
    C::AbstractMatrix, A::CompressedFTLRMatrix{BackendT,T},
    B::AbstractMatrix{T};
    workspace, transA::Char='N', transB::Char='N', compute=nothing,
) where {BackendT,T}
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "symbolic mixed GEMM analysis requires a reusable DenseGemmWorkspace"))
    LA = logical_operand(A, transA)
    LB = logical_dense_operand(B, transB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C has the wrong dimensions"))
    backend = _validate_dense_backend(C, A, B)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    _validate_compressed_mixed_grouped_precision(LA)
    _, arena, budget = _prepare_single_gemm_workspace(A, workspace)
    runs = _compressed_dense_prepared_runs(C, LA, LB, budget, mode, arena)
    return _new_compressed_mixed_analysis(
        C, A, B, workspace, _normalize_tlr_op(transA), _normalize_tlr_op(transB),
        mode, :left, A, runs)
end

function _execute_compressed_mixed_analysis!(
    analysis::CompressedMixedGemmAnalysis, C, A, B, workspace,
    alpha, beta, transA, transB, mode)
    analysis.closed && throw(ArgumentError("CompressedMixedGemmAnalysis has been closed"))
    C === analysis.C && A === analysis.A && B === analysis.B ||
        throw(ArgumentError("mixed analysis is bound to different matrix objects"))
    workspace === analysis.workspace && sizeof(workspace) == analysis.workspace_bytes ||
        throw(ArgumentError("mixed analysis workspace does not match"))
    _normalize_tlr_op(transA) == analysis.transA &&
        _normalize_tlr_op(transB) == analysis.transB ||
        throw(ArgumentError("mixed analysis transpose modes do not match"))
    typeof(mode) === typeof(analysis.compute) ||
        throw(ArgumentError("mixed analysis compute policy does not match"))
    compressed = analysis.side === :left ? A : B
    Int.(ranks(compressed)) == analysis.ranks &&
        Int.(execution_ranks(compressed)) == analysis.execution_ranks ||
        throw(ArgumentError("compressed operand ranks changed after analysis"))
    backend = get_backend(compressed)
    for run in analysis.runs
        @inbounds for (rows, cols) in run.scale_targets
            _scale_output!(view(C, rows, cols), beta)
        end
        _submit_analysis_stage(run.stage1, backend, true)
        _submit_analysis_stage(run.stage2, backend, true, alpha, beta)
    end
    return C
end

function _dense_compressed_grouped!(
    C, A::LogicalDenseOperand, B, alpha, beta, budget::Int, mode, arena)
    _validate_compressed_mixed_grouped_precision(B)
    prefix, totals, bases = _compressed_col_rank_plan(B)
    total_rank = bases[end]
    total_rank == 0 && return _scale_output!(C, beta)
    T = eltype(B)
    budget_elements = fld(budget, sizeof(T))
    # Below this point the rank reduction cannot be represented as one fused
    # terminal operand. Preserve the low-workspace contract through the scalar
    # lowering rather than silently over-allocating.
    budget_elements >= total_rank || return _dense_tlr_gemm_sequential!(
        C, A, B, alpha, beta, budget, mode, arena)

    m = size(A, 1)
    height = clamp(fld(budget_elements, total_rank), 1, m)
    _arena_reset!(arena)
    work = _workspace_array!(arena, get_backend(B), T, height, total_rank)
    qk, qn = grid_size(B)

    for rows in Iterators.partition(1:m, height)
        h = length(rows)
        stage1 = nothing
        stage2 = nothing
        scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
        @inbounds for j in 1:qn
            gamma = totals[j]
            cols = _tile_axis_range(B, j, 2)
            Cview = view(C, rows, cols)
            if gamma == 0
                push!(scale_targets, (rows, cols))
                continue
            end
            Tj = view(work, 1:h, (bases[j] + 1):bases[j + 1])
            for k in 1:qk
                rk = _compressed_ftlr_execution_rank(B, k, j)
                rk == 0 && continue
                inner = _tile_axis_range(B, k, 1)
                Ad, opA = _dense_block(A, rows, inner)
                Tk = view(Tj, :, (prefix[j, k] + 1):prefix[j, k + 1])
                task = GroupedGemmTask(
                    opA, 'N', one(T), Ad,
                    compressed_ftlr_execution_outer(B, k, j), zero(T), Tk)
                stage1 = _mixed_grouped_tasks_push(stage1, task)
            end
            task = GroupedGemmTask(
                'N', 'T', alpha, Tj,
                _compressed_ftlr_col_z_stack(B, j, gamma), beta, Cview)
            stage2 = _mixed_grouped_tasks_push(stage2, task)
        end
        @inbounds for (rs, cs) in scale_targets
            _scale_output!(view(C, rs, cs), beta)
        end
        stage1 === nothing || precision_gemm_grouped!(stage1, mode)
        stage2 === nothing || precision_gemm_grouped!(stage2, mode)
    end
    return C
end

function _compressed_dense_grouped!(
    C, A, B::LogicalDenseOperand, alpha, beta, budget::Int, mode, arena)
    _validate_compressed_mixed_grouped_precision(A)
    prefix, totals, bases = _compressed_row_rank_plan(A)
    total_rank = bases[end]
    total_rank == 0 && return _scale_output!(C, beta)
    T = eltype(A)
    budget_elements = fld(budget, sizeof(T))
    budget_elements >= total_rank || return _tlr_dense_gemm_sequential!(
        C, A, B, alpha, beta, budget, mode, arena)

    n = size(B, 2)
    width = clamp(fld(budget_elements, total_rank), 1, n)
    qm, qk = grid_size(A)
    _arena_reset!(arena)
    work = _workspace_array!(arena, get_backend(A), T, total_rank, width)

    for cols in Iterators.partition(1:n, width)
        w = length(cols)
        stage1 = nothing
        stage2 = nothing
        scale_targets = Tuple{UnitRange{Int},UnitRange{Int}}[]
        @inbounds for i in 1:qm
            rho = totals[i]
            rows = _tile_axis_range(A, i, 1)
            Cview = view(C, rows, cols)
            if rho == 0
                push!(scale_targets, (rows, cols))
                continue
            end
            Ti = view(work, (bases[i] + 1):bases[i + 1], 1:w)
            for k in 1:qk
                rk = _compressed_ftlr_execution_rank(A, i, k)
                rk == 0 && continue
                inner = _tile_axis_range(A, k, 2)
                Bd, opB = _dense_block(B, inner, cols)
                Tk = view(Ti, (prefix[i, k] + 1):prefix[i, k + 1], :)
                task = GroupedGemmTask(
                    'T', opB, one(T),
                    compressed_ftlr_execution_inner(A, i, k), Bd, zero(T), Tk)
                stage1 = _mixed_grouped_tasks_push(stage1, task)
            end
            task = GroupedGemmTask(
                'N', 'N', alpha,
                _compressed_ftlr_row_outer_stack(A, i, rho), Ti, beta, Cview)
            stage2 = _mixed_grouped_tasks_push(stage2, task)
        end
        @inbounds for (rs, cs) in scale_targets
            _scale_output!(view(C, rs, cs), beta)
        end
        stage1 === nothing || precision_gemm_grouped!(stage1, mode)
        stage2 === nothing || precision_gemm_grouped!(stage2, mode)
    end
    return C
end
