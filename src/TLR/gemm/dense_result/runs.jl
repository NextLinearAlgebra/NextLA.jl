"""A scheduled dense-output region and the fold used to lower it.

For compressed × compressed, `rows` and `cols` index output tile ranges. For a
two-stage specialization, the dense axis is a physical matrix range and the
compressed axis is a tile range. The fold has the same meaning in every case:
`:right` ends with the compressed left operand's outer stack, while `:left`
ends with the compressed right operand's inner stack.
"""
struct DenseResultRun
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    fold::Symbol
end

"""Unprepared grouped tasks for one dense-output run.

Two-stage products leave `stage3 === nothing` and set `terminal_stage == 2`;
compressed × compressed sets `terminal_stage == 3`. The terminal stage is the
only stage whose prepared scalars are replaced by the numerical call's
`alpha` and `beta`.
"""
struct DenseResultRunTasks{S1,S2,S3,Z}
    stage1::S1
    stage2::S2
    stage3::S3
    terminal_stage::Int
    zero_target::Z
    needs_zero::Bool
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

function DenseResultRunTasks(stage1, stage2, stage3, terminal_stage::Int,
                             zero_target, needs_zero::Bool, scale_targets)
    terminal_stage in 0:3 || throw(ArgumentError(
        "terminal stage must be between 0 and 3"))
    return DenseResultRunTasks{typeof(stage1),typeof(stage2),typeof(stage3),
                               typeof(zero_target)}(
        stage1, stage2, stage3, terminal_stage, zero_target, needs_zero,
        scale_targets)
end

"""One dense-output run with persistent grouped-GEMM descriptors."""
struct PreparedDenseResultRun
    stage1::Union{Nothing,AbstractPreparedGroupedGemm}
    stage2::Union{Nothing,AbstractPreparedGroupedGemm}
    stage3::Union{Nothing,AbstractPreparedGroupedGemm}
    terminal_stage::Int
    zero_target::Any
    needs_zero::Bool
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

@inline _prepare_dense_result_stage(::Nothing, _) = nothing
@inline _prepare_dense_result_stage(tasks, mode) =
    prepare_precision_gemm_grouped(tasks, mode)

@inline function _destroy_prepared_dense_result_stages!(stage1, stage2, stage3)
    stage1 === nothing || destroy_prepared_grouped_gemm!(stage1)
    stage2 === nothing || destroy_prepared_grouped_gemm!(stage2)
    stage3 === nothing || destroy_prepared_grouped_gemm!(stage3)
    return nothing
end

@inline _destroy_prepared_dense_result_run!(run::PreparedDenseResultRun) =
    _destroy_prepared_dense_result_stages!(run.stage1, run.stage2, run.stage3)

function _prepare_dense_result_run(tasks::DenseResultRunTasks, mode)
    stage1 = stage2 = stage3 = nothing
    try
        stage1 = _prepare_dense_result_stage(tasks.stage1, mode)
        stage2 = _prepare_dense_result_stage(tasks.stage2, mode)
        stage3 = _prepare_dense_result_stage(tasks.stage3, mode)
    catch
        _destroy_prepared_dense_result_stages!(stage1, stage2, stage3)
        rethrow()
    end
    return PreparedDenseResultRun(
        stage1, stage2, stage3, tasks.terminal_stage, tasks.zero_target,
        tasks.needs_zero, tasks.scale_targets)
end

function _prepare_dense_result_runs(build_tasks, schedule, mode)
    prepared = PreparedDenseResultRun[]
    sizehint!(prepared, length(schedule))
    try
        for run in schedule
            push!(prepared, _prepare_dense_result_run(build_tasks(run), mode))
        end
    catch
        foreach(_destroy_prepared_dense_result_run!, prepared)
        rethrow()
    end
    return prepared
end

@inline _prepared_dense_result_stages(run::PreparedDenseResultRun) =
    (run.stage1, run.stage2, run.stage3)

@inline function _prepared_dense_result_run_has_fallback(run::PreparedDenseResultRun)
    return any(stage -> stage isa PreparedGroupedGemmBundle,
               _prepared_dense_result_stages(run))
end

@inline _dense_result_runs_have_fallback(runs) =
    any(_prepared_dense_result_run_has_fallback, runs)

function _close_dense_result_analysis!(analysis)
    analysis.closed && return analysis
    analysis.closed = true
    foreach(_destroy_prepared_dense_result_run!, analysis.runs)
    return analysis
end

function _validate_dense_result_analysis_binding(
    analysis, C, A, B, workspace, transA, transB, mode)
    analysis.closed && throw(ArgumentError("dense-result analysis has been closed"))
    C === analysis.C && A === analysis.A && B === analysis.B ||
        throw(ArgumentError("analysis is bound to different matrix objects"))
    workspace === analysis.workspace && sizeof(workspace) == analysis.workspace_bytes ||
        throw(ArgumentError("analysis numerical workspace does not match"))
    _normalize_tlr_op(transA) == analysis.transA &&
        _normalize_tlr_op(transB) == analysis.transB ||
        throw(ArgumentError("analysis transpose modes do not match"))
    typeof(mode) === typeof(analysis.compute) ||
        throw(ArgumentError("analysis compute policy does not match"))
    return nothing
end

@inline function _submit_prepared_dense_result_stage(
    stage, backend, manage_pointer_mode, overrides...)
    stage === nothing && return nothing
    if manage_pointer_mode && !(stage isa PreparedGroupedGemmBundle)
        return _with_grouped_host_pointer_mode(backend) do
            precision_gemm_grouped_prepared!(stage, overrides...)
        end
    end
    return precision_gemm_grouped_prepared!(stage, overrides...)
end

function _execute_prepared_dense_result_runs_inner!(
    runs, C, backend, ::Type{T}, alpha, beta, manage_pointer_mode) where {T}
    for run in runs
        run.needs_zero && fill!(run.zero_target, zero(T))
        @inbounds for (rows, cols) in run.scale_targets
            _scale_output!(view(C, rows, cols), beta)
        end
        @inbounds for (index, stage) in enumerate(_prepared_dense_result_stages(run))
            if index == run.terminal_stage
                _submit_prepared_dense_result_stage(
                    stage, backend, manage_pointer_mode, alpha, beta)
            else
                _submit_prepared_dense_result_stage(
                    stage, backend, manage_pointer_mode)
            end
        end
    end
    return C
end

function _execute_prepared_dense_result_runs!(
    runs, C, backend, ::Type{T}, alpha, beta, has_fallback::Bool) where {T}
    if has_fallback
        return _execute_prepared_dense_result_runs_inner!(
            runs, C, backend, T, alpha, beta, true)
    end
    return _with_grouped_host_pointer_mode(backend) do
        _execute_prepared_dense_result_runs_inner!(
            runs, C, backend, T, alpha, beta, false)
    end
end
