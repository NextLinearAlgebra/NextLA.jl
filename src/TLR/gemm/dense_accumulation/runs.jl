"""A scheduled dense-output region and the fold used to lower it.

For compressed × compressed, `rows` and `cols` index output tile ranges. For a
two-stage specialization, the dense axis is a physical matrix range and the
compressed axis is a tile range. The fold has the same meaning in every case:
`:right` ends with the compressed left operand's outer stack, while `:left`
ends with the compressed right operand's inner stack.
"""
struct DenseAccumulationRun
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
struct DenseAccumulationRunTasks{S1,S2,S3,Z}
    stage1::S1
    stage2::S2
    stage3::S3
    terminal_stage::Int
    zero_target::Z
    needs_zero::Bool
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

"""One dense-output run with persistent grouped-GEMM descriptors."""
struct PreparedDenseAccumulationRun
    stage1::Union{Nothing,AbstractPreparedGroupedGemm}
    stage2::Union{Nothing,AbstractPreparedGroupedGemm}
    stage3::Union{Nothing,AbstractPreparedGroupedGemm}
    terminal_stage::Int
    zero_target::Any
    needs_zero::Bool
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

@inline function _destroy_prepared_dense_accumulation_run!(run::PreparedDenseAccumulationRun)
    run.stage1 === nothing || destroy_prepared_grouped_gemm!(run.stage1)
    run.stage2 === nothing || destroy_prepared_grouped_gemm!(run.stage2)
    run.stage3 === nothing || destroy_prepared_grouped_gemm!(run.stage3)
    return nothing
end

function _prepare_dense_accumulation_run(tasks::DenseAccumulationRunTasks, mode)
    stage1 = stage2 = stage3 = nothing
    try
        stage1 = tasks.stage1 === nothing ? nothing :
                 prepare_precision_gemm_grouped(tasks.stage1, mode)
        stage2 = tasks.stage2 === nothing ? nothing :
                 prepare_precision_gemm_grouped(tasks.stage2, mode)
        stage3 = tasks.stage3 === nothing ? nothing :
                 prepare_precision_gemm_grouped(tasks.stage3, mode)
    catch
        stage1 === nothing || destroy_prepared_grouped_gemm!(stage1)
        stage2 === nothing || destroy_prepared_grouped_gemm!(stage2)
        stage3 === nothing || destroy_prepared_grouped_gemm!(stage3)
        rethrow()
    end
    return PreparedDenseAccumulationRun(
        stage1, stage2, stage3, tasks.terminal_stage, tasks.zero_target,
        tasks.needs_zero, tasks.scale_targets)
end

function _prepare_dense_accumulation_runs(build_tasks, schedule, mode)
    prepared = PreparedDenseAccumulationRun[]
    sizehint!(prepared, length(schedule))
    try
        for run in schedule
            push!(prepared, _prepare_dense_accumulation_run(build_tasks(run), mode))
        end
    catch
        foreach(_destroy_prepared_dense_accumulation_run!, prepared)
        rethrow()
    end
    return prepared
end

@inline _dense_accumulation_runs_have_fallback(runs) =
    any(run -> any(stage -> stage isa PreparedGroupedGemmBundle,
                   (run.stage1, run.stage2, run.stage3)), runs)

function _close_dense_accumulation_analysis!(analysis)
    analysis.closed && return analysis
    analysis.closed = true
    foreach(_destroy_prepared_dense_accumulation_run!, analysis.runs)
    return analysis
end

function _validate_dense_accumulation_analysis_binding(
    analysis, C, A, B, workspace, transA, transB, mode)
    analysis.closed && throw(ArgumentError("dense-accumulation analysis has been closed"))
    C === analysis.C && A === analysis.A && B === analysis.B ||
        throw(ArgumentError("analysis is bound to different matrix objects"))
    workspace === analysis.workspace && sizeof(workspace) == analysis.workspace_bytes ||
        throw(ArgumentError("analysis numerical workspace does not match"))
    transA == analysis.transA && transB == analysis.transB ||
        throw(ArgumentError("analysis transpose modes do not match"))
    typeof(mode) === typeof(analysis.compute) ||
        throw(ArgumentError("analysis compute policy does not match"))
    return nothing
end

@inline function _submit_prepared_dense_accumulation_stage(
    stage, backend, manage_pointer_mode, overrides...)
    stage === nothing && return nothing
    if manage_pointer_mode && !(stage isa PreparedGroupedGemmBundle)
        return _with_grouped_host_pointer_mode(backend) do
            precision_gemm_grouped_prepared!(stage, overrides...)
        end
    end
    return precision_gemm_grouped_prepared!(stage, overrides...)
end

function _execute_prepared_dense_accumulation_runs_inner!(
    runs, C, backend, ::Type{T}, alpha, beta, manage_pointer_mode) where {T}
    for run in runs
        # beta pre-scale for the terminal stage's untouched output region
        run.needs_zero && fill!(run.zero_target, zero(T))
        @inbounds for (rows, cols) in run.scale_targets
            _scale_output!(view(C, rows, cols), beta)
        end

        # submit each stage, substituting alpha/beta only at the terminal one
        @inbounds for (index, stage) in enumerate((run.stage1, run.stage2, run.stage3))
            if index == run.terminal_stage
                _submit_prepared_dense_accumulation_stage(
                    stage, backend, manage_pointer_mode, alpha, beta)
            else
                _submit_prepared_dense_accumulation_stage(
                    stage, backend, manage_pointer_mode)
            end
        end
    end
    return C
end

function _execute_prepared_dense_accumulation_runs!(
    runs, C, backend, ::Type{T}, alpha, beta, has_fallback::Bool) where {T}
    if has_fallback
        return _execute_prepared_dense_accumulation_runs_inner!(
            runs, C, backend, T, alpha, beta, true)
    end
    return _with_grouped_host_pointer_mode(backend) do
        _execute_prepared_dense_accumulation_runs_inner!(
            runs, C, backend, T, alpha, beta, false)
    end
end
