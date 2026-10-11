# Two-stage compressed/dense lowering: FoldRight for compressed × dense and
# FoldLeft for dense × compressed, sharing the dense-accumulation run lifecycle.

"""Dense block view and its transpose flag."""
@inline dense_block(A::AbstractMatrix, rows, cols) = (view(A, rows, cols), 'N')
@inline dense_block(A::Transpose, rows, cols) = (view(parent(A), cols, rows), 'T')

mutable struct CompressedMixedGemmAnalysis{CT,AT,BT,WT,ModeT,RT}
    C::CT
    A::AT
    B::BT
    workspace::WT
    transA::Char
    transB::Char
    compute::ModeT
    fold::Symbol
    # copied compressed-operand ranks
    ranks::RT
    workspace_bytes::Int
    runs::Vector{PreparedDenseAccumulationRun}
    has_fallback::Bool
    closed::Bool
end

@inline function _validate_two_stage_grouped_precision(A)
    eltype(A) === Core.BFloat16 &&
        !supports_bfloat16_grouped_gemm(get_backend(A)) &&
        throw(ArgumentError(
            "CompressedFTLR BF16 grouped GEMMEx requires an NVIDIA SM80 or newer device"))
    return nothing
end

# A fused run needs the sum of every active rank stack. Workspaces below that
# floor retain the API contract through a compressed-only tilewise fallback.
function compressed_dense_gemm_sequential!(
    C, A::AbstractTLRMatrix{T},
    B::AbstractMatrix, alpha, beta, budget::Int,
    compute, arena=nothing) where {T}
    scale_output!(C, beta)
    r = maxrank(A)
    (isempty(C) || r == 0) && return C

    # widest workspace-backed column batch
    mt, kt = grid_size(A)
    n = size(B, 2)
    batch_width = clamp(div(budget, max(r * sizeof(T), 1)), 1, n)
    arena_reset!(arena)
    work = workspace_array!(arena, get_backend(A), T, r, batch_width)

    # tilewise updates
    @inbounds for i in 1:mt, cols in Iterators.partition(1:n, batch_width)
        rows = tile_axis_range(A, i, 1)
        Cview = view(C, rows, cols)
        for k in 1:kt
            inner = tile_axis_range(A, k, 2)
            U, V = get_factors(A, i, k)
            rk = size(V, 2)
            rk == 0 && continue
            Tview = view(work, 1:rk, 1:length(cols))
            Bd, opB = dense_block(B, inner, cols)
            precision_gemm!('T', opB, one(T), V, Bd, zero(T), Tview, compute)
            precision_gemm!('N', 'N', alpha, U, Tview, one(alpha), Cview, compute)
        end
    end

    return C
end

function dense_compressed_gemm_sequential!(
    C, A::AbstractMatrix,
    B::AbstractTLRMatrix{T},
    alpha, beta, budget::Int, compute, arena=nothing) where {T}
    scale_output!(C, beta)
    r = maxrank(B)
    (isempty(C) || r == 0) && return C

    # tallest workspace-backed row batch
    kt, nt = grid_size(B)
    m = size(A, 1)
    height = clamp(div(budget, max(r * sizeof(T), 1)), 1, m)
    arena_reset!(arena)
    work = workspace_array!(arena, get_backend(B), T, height, r)

    # tilewise updates
    @inbounds for j in 1:nt, rows in Iterators.partition(1:m, height)
        cols = tile_axis_range(B, j, 2)
        Cview = view(C, rows, cols)
        for k in 1:kt
            inner = tile_axis_range(B, k, 1)
            Ad, opA = dense_block(A, rows, inner)
            U, V = get_factors(B, k, j)
            rk = size(U, 2)
            rk == 0 && continue
            Tview = view(work, 1:length(rows), 1:rk)
            precision_gemm!(opA, 'N', one(T), Ad, U, zero(T), Tview, compute)
            precision_gemm!('N', 'T', alpha, Tview, V, one(alpha), Cview, compute)
        end
    end

    return C
end

function two_stage_rank_plan(A, fold::Symbol)
    fold in (:left, :right) || throw(ArgumentError(
        "two-stage compressed fold must be :left or :right"))

    # output-rank prefixes
    qm, qn = grid_size(A)
    outputs, contraction = fold === :left ? (qn, qm) : (qm, qn)
    prefix = Base.zeros(Int, outputs, contraction + 1)
    @inbounds for output in 1:outputs, k in 1:contraction
        rank = fold === :left ?
            compressed_ftlr_storage_rank(A, k, output) :
            compressed_ftlr_storage_rank(A, output, k)
        prefix[output, k + 1] = prefix[output, k] + rank
    end

    totals = [prefix[output, end] for output in 1:outputs]
    bases = compressed_ftlr_prefix(totals)
    return (; prefix, totals, bases, total_rank=bases[end], fold)
end

"""Largest aligned dense-row run admitted by a FoldLeft workspace."""
@inline function dense_compressed_row_run_height(
    budget_elements::Int, total_rank::Int, rows::Int, ::Type{T}) where {T}
    total_rank > 0 || throw(ArgumentError("total rank must be positive"))
    rows > 0 || throw(ArgumentError("row count must be positive"))
    height = clamp(fld(budget_elements, total_rank), 1, rows)
    height == rows && return height
    alignment_rows = gemm_alignment_quantum(T)
    height < alignment_rows && return height
    return fld(height, alignment_rows) * alignment_rows
end

function two_stage_schedule(C, plan, budget::Int, ::Type{T}) where {T}
    plan.total_rank == 0 && return DenseAccumulationRun[
        DenseAccumulationRun(1:size(C, 1), 1:size(C, 2), plan.fold)]

    budget_elements = fld(budget, sizeof(T))
    budget_elements >= plan.total_rank || throw(ArgumentError(
        "two-stage compressed analysis requires at least " *
        "$(plan.total_rank * sizeof(T)) workspace bytes for one fused run"))

    # fold-specific dense-axis partitions
    if plan.fold === :left
        height = dense_compressed_row_run_height(
            budget_elements, plan.total_rank, size(C, 1), T)
        return [DenseAccumulationRun(rows, 1:length(plan.totals), :left)
                for rows in Iterators.partition(1:size(C, 1), height)]
    end
    width = clamp(fld(budget_elements, plan.total_rank), 1, size(C, 2))
    return [DenseAccumulationRun(1:length(plan.totals), cols, :right)
            for cols in Iterators.partition(1:size(C, 2), width)]
end

function _build_dense_compressed_run(C, A, B, plan, run, work)
    rows = run.rows
    h = length(rows)
    qk, _ = grid_size(B)
    stage1 = GroupedGemmTask[]
    stage2 = GroupedGemmTask[]
    scales = Tuple{UnitRange{Int},UnitRange{Int}}[]
    T = eltype(B)

    # grouped stages by output column
    @inbounds for j in run.cols
        gamma = plan.totals[j]
        cols = tile_axis_range(B, j, 2)
        Cview = view(C, rows, cols)
        if gamma == 0
            push!(scales, (rows, cols))
            continue
        end
        Tj = view(work, 1:h, (plan.bases[j] + 1):plan.bases[j + 1])
        for k in 1:qk
            rk = compressed_ftlr_storage_rank(B, k, j)
            rk == 0 && continue
            inner = tile_axis_range(B, k, 1)
            Ad, opA = dense_block(A, rows, inner)
            Tk = view(Tj, :, (plan.prefix[j, k] + 1):plan.prefix[j, k + 1])
            push!(stage1, GroupedGemmTask(
                opA, 'N', one(T), Ad,
                compressed_ftlr_storage_outer(B, k, j), zero(T), Tk))
        end
        push!(stage2, GroupedGemmTask(
            'N', 'T', one(T), Tj,
            compressed_ftlr_col_z_stack(B, j, gamma), zero(T), Cview))
    end

    return DenseAccumulationRunTasks(
        isempty(stage1) ? nothing : stage1, isempty(stage2) ? nothing : stage2,
        nothing, 2, work, false, scales)
end

function _build_compressed_dense_run(C, A, B, plan, run, work)
    cols = run.cols
    w = length(cols)
    _, qk = grid_size(A)
    stage1 = GroupedGemmTask[]
    stage2 = GroupedGemmTask[]
    scales = Tuple{UnitRange{Int},UnitRange{Int}}[]
    T = eltype(A)

    # grouped stages by output row
    @inbounds for i in run.rows
        rho = plan.totals[i]
        rows = tile_axis_range(A, i, 1)
        Cview = view(C, rows, cols)
        if rho == 0
            push!(scales, (rows, cols))
            continue
        end
        Ti = view(work, (plan.bases[i] + 1):plan.bases[i + 1], 1:w)
        for k in 1:qk
            rk = compressed_ftlr_storage_rank(A, i, k)
            rk == 0 && continue
            inner = tile_axis_range(A, k, 2)
            Bd, opB = dense_block(B, inner, cols)
            Tk = view(Ti, (plan.prefix[i, k] + 1):plan.prefix[i, k + 1], :)
            push!(stage1, GroupedGemmTask(
                'T', opB, one(T),
                compressed_ftlr_storage_inner(A, i, k), Bd, zero(T), Tk))
        end
        push!(stage2, GroupedGemmTask(
            'N', 'N', one(T),
            compressed_ftlr_row_outer_stack(A, i, rho), Ti, zero(T), Cview))
    end

    return DenseAccumulationRunTasks(
        isempty(stage1) ? nothing : stage1, isempty(stage2) ? nothing : stage2,
        nothing, 2, work, false, scales)
end

function _prepare_two_stage_runs(C, A, B, compressed, plan, budget, mode, arena)
    # run schedule and zero-rank case
    schedule = two_stage_schedule(C, plan, budget, eltype(compressed))
    plan.total_rank == 0 && return prepare_dense_accumulation_runs(schedule, mode) do _
        DenseAccumulationRunTasks(
            nothing, nothing, nothing, 0, nothing, false,
            [(1:size(C, 1), 1:size(C, 2))])
    end

    # shared numerical workspace
    arena_reset!(arena)
    backend = get_backend(compressed)
    if plan.fold === :left
        height = maximum(length(run.rows) for run in schedule)
        work = workspace_array!(arena, backend, eltype(compressed),
                                 height, plan.total_rank)
        return prepare_dense_accumulation_runs(schedule, mode) do run
            _build_dense_compressed_run(C, A, B, plan, run, work)
        end
    end
    width = maximum(length(run.cols) for run in schedule)
    work = workspace_array!(arena, backend, eltype(compressed),
                             plan.total_rank, width)
    return prepare_dense_accumulation_runs(schedule, mode) do run
        _build_compressed_dense_run(C, A, B, plan, run, work)
    end
end

Base.close(analysis::CompressedMixedGemmAnalysis) =
    close_dense_accumulation_analysis!(analysis)

function _new_compressed_mixed_analysis(
    C, A, B, workspace, transA, transB, mode, fold, compressed, runs)
    # bound analysis
    analysis = CompressedMixedGemmAnalysis(
        C, A, B, workspace, transA, transB, mode, fold,
        copy(ranks(compressed)),
        sizeof(workspace), runs, dense_accumulation_runs_have_fallback(runs), false)

    # cleanup finalizer
    finalizer(analysis) do object
        try
            close_dense_accumulation_analysis!(object)
        catch
            # Device teardown may precede Julia object finalization.
        end
    end

    return analysis
end

function analyze_compressed_gemm(
    C::AbstractMatrix, A::AbstractMatrix{T},
    B::CompressedFTLRMatrix{BackendT,T};
    workspace, transA::Char='N', transB::Char='N', compute=nothing,
) where {BackendT,T}
    # binding and geometry
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "symbolic two-stage GEMM analysis requires a reusable DenseGemmWorkspace"))
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C has the wrong dimensions"))
    backend = validate_dense_backend(C, B, A)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    _validate_two_stage_grouped_precision(LB)

    # fold-left plan and prepared runs
    _, arena, budget = prepare_dense_accumulation_workspace(B, workspace)
    plan = two_stage_rank_plan(LB, :left)
    runs = _prepare_two_stage_runs(C, LA, LB, LB, plan, budget, mode, arena)
    return _new_compressed_mixed_analysis(
        C, A, B, workspace, transA, transB,
        mode, :left, B, runs)
end

function analyze_compressed_gemm(
    C::AbstractMatrix, A::CompressedFTLRMatrix{BackendT,T},
    B::AbstractMatrix{T};
    workspace, transA::Char='N', transB::Char='N', compute=nothing,
) where {BackendT,T}
    # binding and geometry
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "symbolic two-stage GEMM analysis requires a reusable DenseGemmWorkspace"))
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C has the wrong dimensions"))
    backend = validate_dense_backend(C, A, B)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    _validate_two_stage_grouped_precision(LA)

    # fold-right plan and prepared runs
    _, arena, budget = prepare_dense_accumulation_workspace(A, workspace)
    plan = two_stage_rank_plan(LA, :right)
    runs = _prepare_two_stage_runs(C, LA, LB, LA, plan, budget, mode, arena)
    return _new_compressed_mixed_analysis(
        C, A, B, workspace, transA, transB,
        mode, :right, A, runs)
end

function execute_compressed_mixed_analysis!(
    analysis::CompressedMixedGemmAnalysis, C, A, B, workspace,
    alpha, beta, transA, transB, mode)
    validate_dense_accumulation_analysis_binding(
        analysis, C, A, B, workspace, transA, transB, mode)

    # compressed-rank guard
    compressed = analysis.fold === :right ? A : B
    ranks(compressed) == analysis.ranks ||
        throw(ArgumentError("compressed operand ranks changed after analysis"))

    return execute_prepared_dense_accumulation_runs!(
        analysis.runs, C, get_backend(compressed), eltype(compressed), alpha, beta,
        analysis.has_fallback)
end
